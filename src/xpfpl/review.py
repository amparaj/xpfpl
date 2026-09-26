"""Look back at a finished gameweek: what your team scored, what the model expected, and the best
XI you could have picked from the same 15 players."""

import pandas as pd

from xpfpl import config, models
from xpfpl.data import api
from xpfpl.features import build_training_frame
from xpfpl.optimise import solve


def gameweek_points(gw: int) -> pd.DataFrame:
    """Points and minutes per player in `gw` (double gameweeks already summed by the API)."""
    live = api.event_live(gw)["elements"]
    return pd.DataFrame([{"element": e["id"], "points": e["stats"]["total_points"],
                          "minutes": e["stats"]["minutes"]} for e in live]).set_index("element")


SAVED, REBUILT = "saved", "rebuilt"     # where a past gameweek's xP came from


def saved_forecasts(season: str, model: str = config.MODEL) -> dict[int, pd.Series]:
    """GW -> xP per element, from the forecasts saved before each deadline (archive/predictions):
    `model`'s if it was saved that week, otherwise any model's. The Series is named after the model."""
    from xpfpl.data import archive

    out: dict[int, tuple[bool, pd.Series]] = {}
    for name, t in archive.predictions(season).items():
        gw, saved_model = int(name[2:4]), name.split("_", 1)[1]
        col = f"xp_{gw}"
        if col not in t or (gw in out and out[gw][0]):
            continue
        out[gw] = (saved_model == model, t.set_index("element")[col].rename(saved_model))
    return {gw: s for gw, (_, s) in out.items()}


def rebuilt_forecasts(matches: pd.DataFrame, season: str, gws: list[int], model: str = config.MODEL) -> pd.DataFrame:
    """(gw, element, xp) rebuilt from the training frame for gameweeks with no saved forecast
    (double gameweeks summed). Features only use matches before each fixture, but the model was
    refitted on this season too, so these are in-sample: a little optimistic."""
    if not gws:
        return pd.DataFrame(columns=["gw", "element", "xp"])
    frame = build_training_frame(matches)
    rows = frame[(frame["season"] == season) & frame["gw"].isin(gws)].copy()
    rows["xp"] = models.load(model).predict(rows).astype(float).clip(min=0.0)
    return rows.groupby(["gw", "element"], as_index=False)["xp"].sum()


def season_forecasts(matches: pd.DataFrame, season: str, model: str = config.MODEL) -> pd.DataFrame:
    """xP per player for every played gameweek of `season`: gw, element, xp, source (SAVED: the
    forecast saved before that deadline; REBUILT: in-sample, where none was saved)."""
    played = sorted(int(g) for g in matches.loc[matches["season"] == season, "gw"].unique())
    saved = saved_forecasts(season, model)
    parts = [pd.DataFrame({"gw": gw, "element": s.index.astype(int), "xp": s.to_numpy(dtype=float), "source": SAVED})
             for gw, s in saved.items() if gw in played]
    rebuilt = rebuilt_forecasts(matches, season, [g for g in played if g not in saved], model)
    parts.append(rebuilt.assign(source=REBUILT))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["gw", "element", "xp", "source"])
    return pd.concat(parts, ignore_index=True)


def team_forecast(picks: pd.DataFrame, xp: pd.Series) -> float:
    """What the model expected a team to score as picked: each player's xP times his FPL
    multiplier (0 on the bench, 2 for the captain, 3 with Triple Captain, 1 for the bench with
    Bench Boost). Compare it with the points before transfer penalties."""
    return float((xp.reindex(picks.index).fillna(0.0) * picks["multiplier"]).sum())


def review_picks(team_id: int, gw: int, bs: dict, points: pd.DataFrame) -> dict:
    """Your 15 picks for `gw` with actual points, plus the headline numbers for that week."""
    picks = api.entry_picks(team_id, gw)
    players = pd.DataFrame(bs["elements"]).set_index("id")
    teams = {t["id"]: t["short_name"] for t in bs["teams"]}

    df = pd.DataFrame(picks["picks"]).set_index("element")
    df["name"] = players["web_name"].reindex(df.index)
    df["team"] = players["team"].reindex(df.index)
    df["team_name"] = df["team"].map(teams)
    df["pos"] = players["element_type"].reindex(df.index)
    df["price"] = players["now_cost"].reindex(df.index) / 10.0
    df = df.join(points, how="left").fillna({"points": 0, "minutes": 0})
    df["counted"] = df["points"] * df["multiplier"]
    return {
        "picks": df.sort_values("position"),
        "history": picks["entry_history"],
        "chip": picks.get("active_chip"),
        "auto_subs": picks.get("automatic_subs", []),
    }


def hindsight_best(picks: pd.DataFrame, gw: int, chip: str | None) -> tuple[float, object]:
    """Best points available from the same 15 players with perfect hindsight (XI and captain)."""
    df = pd.DataFrame({
        "name": picks["name"], "team": picks["team"], "position": picks["pos"], "price": picks["price"],
        f"xp_{gw}": picks["points"].astype(float),
    })
    df["xp_total"] = df[f"xp_{gw}"] + 1.0  # keep everyone in the pool, even on 0 points
    plan = solve(df, [gw], current_squad={p: df.at[p, "price"] for p in df.index}, bank=0.0,
                 free_transfers=0, max_hits=0,
                 triple_captain_gw=gw if chip == "3xc" else None,
                 bench_boost_gw=gw if chip == "bboost" else None)
    best = plan.xp[gw] + (plan.bench_xp[gw] if chip == "bboost" else 0.0)
    return best, plan
