"""Penalty takers: who takes them now, against who took them in the matches the model learned from.

The model already credits a regular taker through his goals and xG (FPL's xG counts a penalty
at ~0.79). What it can't see is a change: a new first choice after a transfer or an injury
looks like any other player until he has scored a few, and the old taker keeps his credit for a
while. FPL's `penalties_order` (bootstrap-static, set from club announcements) says who is first
choice today; the matchstats shots (FotMob, 2025-26 on) say who took each penalty.

The adjustment per player-fixture, in points:

    (share now x his club's penalties per match - his penalties per appearance lately)
        x conversion x goal points(position) x his chance of being on the pitch

where "share now" is 1 for the first-choice taker (the second choice if the first is flagged
out) and 0 for everyone else. It is near zero for an established taker, zero for anyone who has
never taken one, and moves only the players whose role changed.

Tested on 2025-26 (out-of-sample ensemble forecasts from robustness.py, the club's most recent
taker standing in for the order, which wasn't archived per deadline): new takers beat their
forecast by +0.27 a match (se 0.22) and established takers fell short by 0.32 (se 0.21), but the
adjustment itself explained little: best weight 0.29 (se 0.28), RMSE unchanged to 4 decimals.
So `config.PENALTY_WEIGHT` is 0, and the forecast records `pen_order` and `pen_xp` (the
adjustment at full weight) for the scorecard to judge on this season's real penalty orders
(`scorecard.penalty_fit`).
"""

import numpy as np
import pandas as pd

from xpfpl import config, scoring

CONVERSION = 0.82          # penalties scored / taken, 2025-26 and 2026-27 so far (FotMob shots)
PER_MATCH = 0.12           # penalties a side is awarded per match (same data)
RECENT = 38                # appearances (and club matches) that make up "lately"
PRIOR_MATCHES = 20         # a club's own rate is pulled towards PER_MATCH by this many matches
FIT_TO_PLAY = 75           # a flagged first choice below this chance hands over to the second


def penalty_shots(seasons=None) -> pd.DataFrame:
    """Every archived penalty attempt: season, gw, code (the taker), team_code."""
    from xpfpl.data import matchstats
    parts = []
    for season in seasons or matchstats.seasons():
        shots = matchstats.load(season, "shots")
        if shots is not None and "situation" in shots:
            pens = shots[shots["situation"] == "penalty"]
            parts.append(pens.assign(season=season)[["season", "gw", "code", "team_code"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["season", "gw", "code", "team_code"])


def recent_rate(matches: pd.DataFrame, pens: pd.DataFrame) -> pd.Series:
    """Each player's (`code`) penalties per appearance over his last RECENT appearances in the
    seasons with shot data."""
    covered = set(pens["season"])
    if not covered:
        return pd.Series(dtype=float)
    played = matches[(matches["minutes"] > 0) & matches["season"].isin(covered)]
    played = played.sort_values("kickoff_time").groupby("code").tail(RECENT)
    taken = pens.groupby(["season", "gw", "code"]).size().rename("pens").reset_index()
    played = played[["season", "gw", "code"]].drop_duplicates().merge(taken, on=["season", "gw", "code"], how="left")
    g = played.groupby("code")["pens"]
    return g.sum().fillna(0) / g.size()


def club_rate(matches: pd.DataFrame, pens: pd.DataFrame) -> pd.Series:
    """Penalties per match per club (`team_code`) over its last RECENT matches in the seasons
    with shot data, pulled towards PER_MATCH by PRIOR_MATCHES matches."""
    covered = set(pens["season"])
    games = matches[matches["season"].isin(covered)].drop_duplicates(["season", "fixture", "team_code"])
    games = games.sort_values("kickoff_time").groupby("team_code").tail(RECENT)
    taken = pens.groupby(["season", "gw", "team_code"]).size().rename("pens").reset_index()
    games = games[["season", "gw", "team_code"]].merge(taken, on=["season", "gw", "team_code"], how="left")
    g = games.groupby("team_code")["pens"]
    return (g.sum().fillna(0) + PER_MATCH * PRIOR_MATCHES) / (g.size() + PRIOR_MATCHES)


def takers(elements: pd.DataFrame) -> pd.Series:
    """Per element id: 1 if he should be on penalties next match, else 0 (NaN at a club that
    lists no order). The second choice takes over when the first is flagged below FIT_TO_PLAY."""
    e = elements.set_index("id") if "id" in elements else elements
    order = pd.to_numeric(e.get("penalties_order"), errors="coerce")
    fit = e["chance_of_playing_next_round"].fillna(100).astype(float) >= FIT_TO_PLAY
    fit &= e["status"].isin(["a", "d"])
    listed = order.notna()
    share = pd.Series(np.where(e["team"].isin(e.loc[listed, "team"]), 0.0, np.nan), index=e.index)
    for _, club in e[listed].groupby("team"):
        ranked = order[club.index].sort_values()
        ready = [p for p in ranked.index if fit[p]]
        if ready:
            share[ready[0]] = 1.0
    return share


def adjustment(frame: pd.DataFrame, elements: pd.DataFrame, matches: pd.DataFrame,
               pens: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per row of a future frame: `pen_order` (FPL's order, NaN if none) and `pen_xp`, the xP the
    change in role is worth at full weight (see the module docstring)."""
    pens = penalty_shots() if pens is None else pens
    e = elements.set_index("id") if "id" in elements else elements
    now = takers(e).reindex(frame["element"]).to_numpy()
    club = e["team_code"].reindex(frame["element"]).map(club_rate(matches, pens)).fillna(PER_MATCH).to_numpy()         if "team_code" in e else np.full(len(frame), PER_MATCH)
    past = recent_rate(matches, pens).reindex(frame["code"]).fillna(0.0).to_numpy()
    on_pitch = np.clip(frame["played60_r5"].fillna(0).to_numpy(dtype=float), 0, 1) if "played60_r5" in frame \
        else np.ones(len(frame))
    goal_pts = frame["position"].astype(int).map(scoring.GOAL_POINTS).fillna(0).to_numpy()
    change = np.where(np.isnan(now), 0.0, now * club - past)
    return pd.DataFrame({
        "pen_order": pd.to_numeric(e.get("penalties_order"), errors="coerce").reindex(frame["element"]).to_numpy(),
        "pen_xp": change * CONVERSION * goal_pts * on_pitch,
    }, index=frame.index)


def apply(frame: pd.DataFrame, elements: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """`frame` with `pen_order`/`pen_xp` added and `xp` moved by config.PENALTY_WEIGHT x pen_xp
    (never below zero)."""
    try:
        adj = adjustment(frame, elements, matches)
    except Exception as exc:          # no shot archive, an API change: carry on without it
        print(f"  (no penalty-taker adjustment this run: {exc})")
        return frame.assign(pen_order=np.nan, pen_xp=0.0)
    out = frame.join(adj)
    out["xp"] = (out["xp"] + config.PENALTY_WEIGHT * out["pen_xp"]).clip(lower=0.0)
    return out
