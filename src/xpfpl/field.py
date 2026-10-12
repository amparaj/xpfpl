"""Against the field: a team's gameweek next to the average FPL manager's, in the same simulated
weeks (simulate.py). Display only: the optimiser still maximises expected points (Ash's decision,
2026-09-27, kept 2026-10-12). This shows what that choice does to your rank, for the close calls.

Ownership doesn't change a player's expected points; it changes what his points do to your rank.
A player's **effective ownership** (EO) is the share of managers whose score includes his points,
a captain counted twice:
- starts: ownership x his chance of playing, scaled so the average manager fields 11 (owners bench
  their fourth defender and fifth midfielder, not their best players: the scaling stays under his
  ownership, and players likelier to play are started first);
- captaincy: FPL only publishes it after the deadline (and only the top pick), so it's an estimate:
  armbands go in proportion to ownership squared x e^(xP / CAPTAIN_TEMPERATURE). Squared because
  managers captain the popular premium more than its xP alone says: with ownership to the first
  power GW6's estimate put B.Fernandes (38% owned, 5.9 xP) ahead of Haaland (74%, 5.3), but FPL's
  most captained was Haaland. One gameweek is all there is to check it on.

Checked on GW6: the field scored with the real points and this EO came to 52.1 against FPL's
average of 54 (one match still to play, chips not counted). The field's *expected* score is lower
than that, by this model's own xP: it rates the crowd's players lower than its own picks, so the
average margin flatters any team it picked. Read the chance of beating the field and the spread,
not just the average margin.

The field's score in a simulated week = sum of EO x points. Your **margin** = your score (captain,
auto-subs) minus that: above 0 you gain rank that week, below you lose it. `against` gives the
chance of a positive margin, its average and spread, and your **exposures**: the players where your
share (0 benched or not owned, 1 started, 2 captain) differs most from the field's, weighted by how
much their points vary. A popular player you don't own is a threat (you lose ground when he
hauls); one the field doesn't own is a differential (you gain).
"""

import numpy as np
import pandas as pd

from xpfpl import simulate

CAPTAIN_TEMPERATURE = 1.0
CAPTAIN_OWNERSHIP_POWER = 2
FIELD_STARTERS = 11.0
EXPOSURES = 6               # threats and differentials shown


def effective_ownership(players: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Per element: `own` (ownership, 0-1), `start` and `captain` (shares of all managers) and `eo`
    = start + captain. Needs `selected_by` (percent) and `xp_<gw>`; `p_play` where there is one."""
    own = (pd.to_numeric(players["selected_by"], errors="coerce").fillna(0.0) / 100).clip(0, 1)
    play = players["p_play"].astype(float).fillna(1.0).clip(0, 1) if "p_play" in players else pd.Series(1.0, index=players.index)
    xp = players[f"xp_{gw}"].astype(float).fillna(0.0)
    # Starts: ownership x chance of playing, scaled to 11 a manager without going over ownership.
    want = own * play
    lo, hi = 0.0, 1.0
    while (want * hi).clip(upper=own).sum() < FIELD_STARTERS and hi < 1e3:
        hi *= 2
    for _ in range(50):                                     # the scale k with sum(min(own, k x want)) = 11
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (want * mid).clip(upper=own).sum() < FIELD_STARTERS else (lo, mid)
    start = (want * hi).clip(upper=own)
    weight = own ** CAPTAIN_OWNERSHIP_POWER * np.exp((xp - xp.max()) / CAPTAIN_TEMPERATURE)
    captain = weight / weight.sum() if weight.sum() > 0 else weight
    return pd.DataFrame({"own": own, "start": start, "captain": captain, "eo": start + captain})


def _shares(team: dict, chip: str | None) -> dict[int, float]:
    """A team's share of each player's points: 1 started, 0 benched (1 with Bench Boost), captain 2 (3)."""
    shares = {int(e): 1.0 for e in team["lineup"]}
    shares |= {int(e): 1.0 if chip == "bboost" else 0.0 for e in team["bench"]}
    shares[int(team["captain"])] = 3.0 if chip == "3xc" else 2.0
    return shares


def against(d: simulate.Draws, gw: int, team: dict, players: pd.DataFrame, eo: pd.DataFrame | None = None,
            chip: str | None = None) -> dict | None:
    """`team` (lineup, bench, captain, vice) against the field in `gw`'s simulated weeks: the field's
    average score (`field`), the team's (`team`), the margin (`margin`, `p10`, `p50`, `p90`, `sd`),
    the chance it's positive (`p_beat`), and the biggest `threats` and `differentials` (element,
    `eo`, `share`: the team's, `exposure` = share - EO, `swing` = exposure x the sd of his points).
    None without simulations for that gameweek."""
    if gw not in d.gameweeks:
        return None
    eo = effective_ownership(players, gw) if eo is None else eo
    ids = [int(e) for e in eo.index[eo["eo"] > 1e-4] if e in set(d.elements.tolist())]
    pts = d.pts(gw, ids).astype(float)
    field = pts @ eo.loc[ids, "eo"].to_numpy()
    position = players["position"].to_dict()
    mine = simulate.team_score(d, gw, team["lineup"], team["bench"], team["captain"], team["vice"], chip, position)["points"]
    margin = mine - field
    q = np.percentile(margin, [10, 50, 90])

    shares = _shares(team, chip)
    universe = sorted(set(ids) | set(shares))
    sd = pd.Series(d.pts(gw, universe).astype(float).std(axis=0), index=universe)
    t = pd.DataFrame({"eo": eo["eo"].reindex(universe).fillna(0.0), "share": pd.Series(shares).reindex(universe).fillna(0.0)})
    t["exposure"] = t["share"] - t["eo"]
    t["swing"] = t["exposure"] * sd
    rows = lambda frame: [{"element": int(e), "eo": round(float(r.eo), 3), "share": float(r.share),     # noqa: E731
                           "exposure": round(float(r.exposure), 3), "swing": round(float(r.swing), 3)}
                          for e, r in frame.iterrows()]
    return {"gw": gw, "field": round(float(field.mean()), 2), "team": round(float(mine.mean()), 2),
            "margin": round(float(margin.mean()), 2), "sd": round(float(margin.std()), 2),
            "p10": round(float(q[0]), 1), "p50": round(float(q[1]), 1), "p90": round(float(q[2]), 1),
            "p_beat": round(float((margin > 0).mean()), 4),
            "threats": rows(t.nsmallest(EXPOSURES, "swing").query("swing < 0")),
            "differentials": rows(t.nlargest(EXPOSURES, "swing").query("swing > 0"))}


def plan_team(plan, gw: int) -> dict:
    return {"lineup": plan.lineups[gw], "bench": plan.bench[gw], "captain": plan.captains[gw], "vice": plan.vice_captains[gw]}


def compare(d: simulate.Draws, gw: int, a: dict, b: dict, players: pd.DataFrame, eo: pd.DataFrame | None = None) -> dict:
    """Two teams against the field in the same simulated weeks: each one's chance of beating it and
    the spread of its margin. Which one is safer for rank, which one gambles."""
    eo = effective_ownership(players, gw) if eo is None else eo
    x, y = against(d, gw, a, players, eo), against(d, gw, b, players, eo)
    return {"a": {k: x[k] for k in ("margin", "sd", "p_beat")}, "b": {k: y[k] for k in ("margin", "sd", "p_beat")}}
