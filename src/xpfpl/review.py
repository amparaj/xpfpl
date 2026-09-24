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


def past_predictions(matches: pd.DataFrame, season: str, gw: int,
                     model: str = config.MODEL) -> pd.Series:
    """The model's pre-match xP per player for a past gameweek (summed over double gameweeks).

    Features only use matches before each fixture, but the saved model was refitted on every
    season including this one, so treat this as in-sample, i.e. a little optimistic.
    """
    frame = build_training_frame(matches)
    rows = frame[(frame["season"] == season) & (frame["gw"] == gw)].copy()
    if rows.empty:
        return pd.Series(dtype=float, name="xp")
    rows["xp"] = models.load(model).predict(rows)
    rows["xp"] = rows["xp"].astype(float).clip(lower=0.0)
    return rows.groupby("element")["xp"].sum()


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
