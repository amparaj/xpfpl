"""Expected points for every current player over the next few gameweeks."""

import pandas as pd

from xpfpl import config, models, prices
from xpfpl.data import api, archive
from xpfpl.data.history import load_matches
from xpfpl.features import build_future_frame


def availability(status: pd.Series, chance: pd.Series, offset: pd.Series) -> pd.Series:
    """Probability a player is available `offset` gameweeks from now.

    Uses FPL's chance_of_playing (injury/suspension flags) for the next GW and assumes
    flagged players gradually recover after that. Players who left the league ('u') are 0.
    """
    p = chance.fillna(100).astype(float) / 100.0
    p = (p + 0.25 * offset).clip(upper=1.0)
    return p.where(status != "u", 0.0)


def predict_upcoming(horizon: int = config.HORIZON, model: str = config.MODEL,
                     bs: dict | None = None, fixtures: list[dict] | None = None) -> tuple[pd.DataFrame, list[int]]:
    """Returns (players, gameweeks): one row per player with an `xp_<gw>` column per gameweek.

    Also carries `price_delta`, the expected price change per gameweek; for the component
    model, one column per scoring component, so the dashboard can show where an xP comes from;
    and for a model with a minutes head (xmins, ensemble), `xmins` and `p_play` for the next GW.
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
    offset = frame["gw"] - next_gw  # 0 for the next gameweek, 1 for the one after...
    raw = pd.Series(predictor.predict(frame), index=frame.index).astype(float).clip(lower=0.0)
    frame = frame.assign(xp=raw * availability(frame["status"],
                                               frame["chance_of_playing_next_round"], offset))
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
    out = out.join(_scorer_odds(scorers, players))
    out = out.join(_rotation(frame, next_gw, gameweeks))
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
    play = (frame["xp"] / raw.where(raw > 0)).fillna(
        availability(frame["status"], frame["chance_of_playing_next_round"], frame["gw"] - next_gw))
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
    avail = availability(rows["status"], rows["chance_of_playing_next_round"], rows["gw"] * 0)
    parts = pd.DataFrame({"element": rows["element"].to_numpy(),
                          "xmins": (e["xmins"] * avail).to_numpy(),
                          "p_none": (1 - (1 - e["p_none"]) * avail).to_numpy()})
    g = parts.groupby("element")
    return pd.DataFrame({"xmins": g["xmins"].sum(), "p_play": 1 - g["p_none"].prod()})


def _components(predictor, frame: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Where the next gameweek's xP comes from, if the model can say (the component model can)."""
    if not hasattr(predictor, "breakdown"):
        return pd.DataFrame(index=pd.Index([], name="element"))
    rows = frame[frame["gw"] == gw]
    parts = predictor.breakdown(rows).drop(columns=["xp"])
    parts["element"] = rows["element"].to_numpy()
    return parts.groupby("element").sum().add_prefix("from_")   # doubles add up, like xP does
