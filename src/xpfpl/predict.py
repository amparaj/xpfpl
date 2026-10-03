"""Expected points for every current player over the next few gameweeks."""

import numpy as np
import pandas as pd

from xpfpl import config, models, prices
from xpfpl.data import api, archive
from xpfpl.data.history import load_matches
from xpfpl.features import build_future_frame


def predict_upcoming(horizon: int = config.HORIZON, model: str = config.MODEL,
                     bs: dict | None = None, fixtures: list[dict] | None = None,
                     press: dict | None = None) -> tuple[pd.DataFrame, list[int]]:
    """Returns (players, gameweeks): one row per player with an `xp_<gw>` column per gameweek.

    Also carries `price_delta`, the expected price change per gameweek; for the component
    model, one column per scoring component, so the dashboard can show where an xP comes from;
    `pen_order`/`pen_xp` for the penalty takers (setpieces.py); the team news behind next week's
    chance of playing (`avail`, `news_rule`, `press`, `back`: data/news.py; `press` is today's
    press conferences, read from the web when not given); and for a model with a minutes
    head (xmins, ensemble), `xmins`, `p_play` and `p_full` (60+ minutes) for the next GW; for the ensemble, how far its members disagree on the next GW
    (`xp_sd`) and the `confidence` that gives (config.CONFIDENCE_CUTS).
    With config.SIM_RUNS > 0, the next GW's simulated range (simulate.py): `pts_p10`/`pts_p50`/
    `pts_p90` and the chances of 10+ (`p_haul`) and of 2 or fewer (`p_blank`); every simulation
    for every week of the horizon goes to data/predictions/<season>/gwNN_<model>_sims.npz.

    Every run is saved to data/predictions/<season>/gwNN_<model>.csv. The next gameweek's
    deadline hasn't passed when this runs, so these are genuine pre-deadline forecasts, and
    `xpfpl scorecard` scores them once the results are in.
    """
    bs = bs or api.bootstrap()
    fixtures = fixtures or api.fixtures()
    next_gw = api.next_gameweek(bs)
    gameweeks = list(range(next_gw, min(next_gw + horizon, 39)))

    matches = load_matches()
    market, scorers = _live_odds(bs, fixtures)
    frame = build_future_frame(matches, bs, fixtures, gameweeks, api.current_season(bs), market)

    predictor = models.load(model)
    raw = pd.Series(predictor.predict(frame), index=frame.index).astype(float).clip(lower=0.0)
    # Each match's chance the player is available (data/news.py): FPL's flag, a newer press
    # conference for the next gameweek, return dates and known absences for the weeks after.
    from xpfpl.data import news
    press = news.press_now(bs) if press is None and config.PRESS_NEWS else press
    frame = frame.join(news.availability(frame, bs, next_gw, press))
    frame["xp"] = raw * frame["avail"]
    # Midweek cup/European matches (data/cups.py): next week's xP moves by what the player's own
    # minutes in the midweek match said about his place on 2025-26 onwards. Later weeks only
    # show the club's midweek matches (`cup_<gw>`): no rotation penalty showed up at club level.
    from xpfpl.data import cups
    rotation = cups.adjust(frame, next_gw, model)
    frame = frame.join(rotation[["cup_before_level", "rotation", "rotation_factor"]])
    frame["xp"] = frame["xp"] * frame["rotation_factor"]
    # The scorer market pricing a player at ~0 for the next gameweek means he's out (team news
    # the FPL flag may not show yet): cut that week's xP as the data says (config.py).
    from xpfpl.data import markets
    cut = (frame["gw"] == next_gw) & frame["code"].isin(markets.ruled_out(scorers).index)
    frame.loc[cut, "xp"] = frame.loc[cut, "xp"] * config.MARKET_OUT_XP_FACTOR
    # Penalty takers (setpieces.py): FPL's order against who took them lately. Recorded every
    # run; it only moves xP if config.PENALTY_WEIGHT is above 0.
    from xpfpl import setpieces
    frame = setpieces.apply(frame, pd.DataFrame(bs["elements"]), matches)

    ranges = _simulate(frame, raw, predictor, matches, api.current_season(bs), next_gw, model)

    # Sum fixtures within a gameweek (double gameweeks) and pivot to one column per GW.
    xp = frame.pivot_table(index="element", columns="gw", values="xp", aggfunc="sum")
    xp = xp.reindex(columns=gameweeks).fillna(0.0)  # blank gameweek -> 0
    xp.columns = [f"xp_{gw}" for gw in gameweeks]

    players = pd.DataFrame(bs["elements"]).set_index("id")
    teams = {t["id"]: t["short_name"] for t in bs["teams"]}
    out = pd.DataFrame({
        "name": players["web_name"],
        "team": players["team"],
        "team_name": players["team"].map(teams),
        "position": players["element_type"],
        "price": players["now_cost"] / 10.0,
        "status": players["status"],
        "chance": players["chance_of_playing_next_round"],
        "selected_by": pd.to_numeric(players["selected_by_percent"], errors="coerce"),
    }).join(xp).fillna({c: 0.0 for c in xp.columns})
    out["xp_total"] = out[list(xp.columns)].sum(axis=1)
    out["price_delta"] = prices.for_upcoming(frame, players).reindex(out.index).fillna(0.0)
    out = out.join(_components(predictor, frame, next_gw))
    out = out.join(_minutes(predictor, frame, next_gw))
    out = out.join(_confidence(predictor, frame, next_gw))
    out = out.join(_scorer_odds(scorers, players))
    out = out.join(_rotation(frame, next_gw, gameweeks))
    out = out.join(_penalties(frame, next_gw))
    out = out.join(_news(frame, next_gw, bs, press))
    out = out.join(ranges)
    out.index.name = "element"

    folder = config.PREDICTIONS_DIR / api.current_season(bs)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"gw{next_gw:02d}_{model}.csv"
    out.sort_values("xp_total", ascending=False).to_csv(path, encoding="utf-8")
    archive.save_prediction(path, api.current_season(bs))
    return out, gameweeks


def _simulate(frame: pd.DataFrame, raw: pd.Series, predictor, matches: pd.DataFrame, season: str,
              next_gw: int, model: str) -> pd.DataFrame:
    """Play the horizon config.SIM_RUNS times (simulate.py), each player's average matched to his
    xP, and save the simulations. Returns next week's range per element (empty if switched off).

    Whatever cut a player's xP (injury flag, midweek rotation, the market's "ruled out") is read
    as a lower chance of playing, the way it would play out on the day."""
    from xpfpl import simulate
    if config.SIM_RUNS <= 0:
        return pd.DataFrame(index=pd.Index([], name="element"))
    expectations = getattr(predictor, "expectations", None)
    minutes = expectations(frame) if expectations else None
    play = (frame["xp"] / raw.where(raw > 0)).fillna(frame["avail"])
    inp = simulate.inputs(frame, frame["xp"], minutes, play_factor=play)
    draws = simulate.by_gameweek(simulate.run(inp, simulate.history_tables(matches), sims=config.SIM_RUNS), inp)
    simulate.save(draws, season, next_gw, model)
    return simulate.summary(draws, next_gw)


def _rotation(frame: pd.DataFrame, next_gw: int, gameweeks: list[int]) -> pd.DataFrame:
    """Per element: `cup_<gw>`, the biggest midweek match his club plays in the 6 days before
    each gameweek's fixture (cups.LEVELS: 3 Champions League, 2 Europa, 1 other, 0 none), and
    next week's `rotation` group and the `rotation_factor` its xP was multiplied by."""
    level = frame.pivot_table(index="element", columns="gw", values="cup_before_level", aggfunc="max")
    level = level.reindex(columns=gameweeks).fillna(0).astype(int)
    level.columns = [f"cup_{gw}" for gw in gameweeks]
    rows = frame[frame["gw"] == next_gw].drop_duplicates("element").set_index("element")
    return level.join(rows[["rotation", "rotation_factor"]], how="outer")


def _news(frame: pd.DataFrame, next_gw: int, bs: dict, press: dict | None) -> pd.DataFrame:
    """Next gameweek's team news per element: the chance of being available it came to (`avail`,
    a double gameweek's best), the rule that set it (`news_rule`), what a press conference for
    that week said (`press`: OUT / DOUBT / IN, whether or not it won over FPL's flag) and FPL's
    return date (`back`)."""
    from xpfpl.data import news
    rows = frame[frame["gw"] == next_gw].sort_values("avail", ascending=False).drop_duplicates("element")
    out = rows.set_index("element")[["avail", "news_rule"]]
    said = news.press_table(press)["press"] if news.press_is_for_next(press, bs) else pd.Series(dtype=object)
    out["press"] = out.index.map(said)
    out["back"] = out.index.map(news.fpl(bs).set_index("id")["back"])
    return out


def _penalties(frame: pd.DataFrame, next_gw: int) -> pd.DataFrame:
    """Next gameweek's `pen_order` (FPL's) and `pen_xp` (setpieces.py, at full weight; a double
    gameweek's matches summed)."""
    rows = frame[frame["gw"] == next_gw]
    if "pen_xp" not in rows:
        return pd.DataFrame(index=pd.Index([], name="element"))
    g = rows.groupby("element")
    return pd.DataFrame({"pen_order": g["pen_order"].first(), "pen_xp": g["pen_xp"].sum()})


def _scorer_odds(scorers: pd.DataFrame | None, players: pd.DataFrame) -> pd.DataFrame:
    """Next gameweek's anytime-scorer odds per element, and whether the market has him as out."""
    from xpfpl.data import markets
    if scorers is None or not len(scorers) or "code" not in scorers:
        return pd.DataFrame(index=pd.Index([], name="element"))
    odds = scorers.groupby("code")["p_anytime"].min()
    code = players["code"]
    return pd.DataFrame({"mkt_anytime": code.map(odds),
                         "market_out": code.isin(markets.ruled_out(scorers).index)}, index=players.index)


def _live_odds(bs: dict, fixtures: list[dict]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Today's Polymarket match and scorer odds for upcoming fixtures, or (None, None) if they
    can't be fetched.

    Without them the model falls back to its own team ratings, exactly as it does in training
    for matches that had no market."""
    try:
        from xpfpl.data import markets
        market, scorers = markets.upcoming(bs, fixtures)
        return (market if len(market) else None), (scorers if len(scorers) else None)
    except Exception as exc:                 # offline, API change...: predict without odds
        print(f"  (no betting odds this run: {exc})")
        return None, None


def _minutes(predictor, frame: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Expected minutes and the chance of playing next gameweek, if the model predicts them.

    Scaled by the same injury-flag availability as xP. A double gameweek sums the minutes and
    takes the chance of playing at least once.
    """
    expectations = getattr(predictor, "expectations", None)
    rows = frame[frame["gw"] == gw]
    e = expectations(rows) if expectations else None
    if e is None:
        return pd.DataFrame(index=pd.Index([], name="element"))
    avail = rows["avail"]
    parts = pd.DataFrame({"element": rows["element"].to_numpy(),
                          "xmins": (e["xmins"] * avail).to_numpy(),
                          "p_none": (1 - (1 - e["p_none"]) * avail).to_numpy(),
                          "p_short": (1 - e["p_full"] * avail).to_numpy()})
    g = parts.groupby("element")
    return pd.DataFrame({"xmins": g["xmins"].sum(), "p_play": 1 - g["p_none"].prod(),
                         "p_full": 1 - g["p_short"].prod()})


def _confidence(predictor, frame: pd.DataFrame, gw: int) -> pd.DataFrame:
    """For an ensemble: how far its members' next-gameweek forecasts spread (`xp_sd`, before
    injury flags, a double gameweek's matches summed) and the confidence that gives.

    Tested on six held-out seasons (config.CONFIDENCE_CUTS): where the members disagree most,
    the forecast misses by more than usual for the same xP, in five of the six."""
    members = getattr(predictor, "members", None)
    if not members or len(members) < 2:
        return pd.DataFrame(index=pd.Index([], name="element"))
    rows = frame[frame["gw"] == gw]
    preds = pd.DataFrame({name: np.clip(m.predict(rows), 0, None) for name, m in members.items()}, index=rows.index)
    per = preds.groupby(rows["element"].to_numpy()).sum()
    return confidence_table(per)


def confidence_table(per_member: pd.DataFrame) -> pd.DataFrame:
    """`xp_sd` (members' standard deviation) and `confidence` (High / Medium / Low) from one
    column of next-gameweek xP per ensemble member, indexed by element."""
    sd = per_member.std(axis=1)
    rel = sd / per_member.mean(axis=1).clip(lower=1.0)
    high, low = config.CONFIDENCE_CUTS
    label = np.where(rel <= high, "High", np.where(rel <= low, "Medium", "Low"))
    return pd.DataFrame({"xp_sd": sd, "confidence": label}, index=pd.Index(per_member.index, name="element"))


def _components(predictor, frame: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Where the next gameweek's xP comes from, if the model can say (the component model can)."""
    if not hasattr(predictor, "breakdown"):
        return pd.DataFrame(index=pd.Index([], name="element"))
    rows = frame[frame["gw"] == gw]
    parts = predictor.breakdown(rows).drop(columns=["xp"])
    parts["element"] = rows["element"].to_numpy()
    return parts.groupby("element").sum().add_prefix("from_")   # doubles add up, like xP does
