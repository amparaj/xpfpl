"""Where a player plays: spatial metrics from the match data (data/matchstats.py), for display.

There are no heatmaps or full event data here (every pass and touch with its location). What the
source has is each shot's location, and per match zone counts: touches in the opposition box,
passes into the final third. So a player's spatial profile is built from those:

  - his shots binned into zones round the goal he attacks (`ZONES_X` x `ZONES_Y`): counts, xG and
    goals per zone, the density matrix a heatmap would show
  - per 90 minutes: touches, touches in the opposition box, final-third passes, chances created, shots
  - share of shots from inside the box, mean shot distance, xG per shot, share of headers

`similar` finds the players whose profile is closest (cosine similarity of the metrics, each
standardised within position): players who get the same kind of chances from the same places.
It is display only. The optimiser picks on xP, which already reflects where a player shoots from,
and a probe of these metrics as model features (2025-26, the ensemble's out-of-sample errors)
found about -0.005 RMSE from a linear correction, on one season: not yet enough to add them.
"""

import numpy as np
import pandas as pd

from xpfpl.data import matchstats

# Zone edges in the source's coordinates (0-100 both ways). Along: from the goal line (six-yard box
# 5.2, penalty spot 11.5, box edge 15.7, then 25 and halfway). Across: the box's and six-yard box's
# sides (20.4 / 36.5 / 63.5 / 79.6), 50 being the middle.
ZONES_X = [0.0, 5.2, 11.5, 15.7, 25.0, 50.0]
ZONES_Y = [0.0, 20.4, 36.5, 63.5, 79.6, 100.0]
IN_BOX = (15.7, 20.4, 79.6)          # along <= 15.7 and across between 20.4 and 79.6
METRES = 105 / 100                   # along the pitch, per source unit
PER_90 = {"touches": "touches_p90", "touches_opposition_box": "box_touches_p90",
          "final_third_passes": "final_third_passes_p90", "chances_created": "chances_created_p90",
          "total_shots": "shots_p90"}
# The metrics a profile is compared on (standardised within position).
SIMILARITY = ["touches_p90", "box_touches_p90", "final_third_passes_p90", "chances_created_p90", "shots_p90",
              "box_shot_share", "shot_distance", "xg_per_shot", "header_share"]
MIN_MINUTES = 270                    # three full matches before a profile is compared
MIN_SHOTS = 3                        # shot metrics need a few shots; fewer and they count as the position's average


def zone_of(along: pd.Series, across: pd.Series) -> pd.Series:
    """The zone index (row-major: along band x 5 + across band) of each shot; NaN beyond halfway."""
    ix = np.digitize(along, ZONES_X[1:-1])
    iy = np.digitize(across, ZONES_Y[1:-1])
    zone = pd.Series(ix * (len(ZONES_Y) - 1) + iy, index=along.index, dtype=float)
    return zone.where(along <= ZONES_X[-1])


def zones() -> pd.DataFrame:
    """Every zone's bounds: zone, x0, x1 (along, from the goal line), y0, y1 (across)."""
    return pd.DataFrame([{"zone": i * (len(ZONES_Y) - 1) + j, "x0": ZONES_X[i], "x1": ZONES_X[i + 1],
                          "y0": ZONES_Y[j], "y1": ZONES_Y[j + 1]}
                         for i in range(len(ZONES_X) - 1) for j in range(len(ZONES_Y) - 1)])


def shots(season: str) -> pd.DataFrame:
    """The season's shots with a known player (FPL `code`), with their zone."""
    s = matchstats.load(season, "shots")
    if s is None:
        return pd.DataFrame(columns=["code", "zone"])
    s = s.dropna(subset=["code"]).astype({"code": int})
    return s.assign(zone=zone_of(s["start_x"], s["start_y"]),
                    in_box=(s["start_x"] <= IN_BOX[0]) & s["start_y"].between(IN_BOX[1], IN_BOX[2]))


def profiles(season: str) -> pd.DataFrame:
    """One row per player (FPL `code`) who has played: minutes, per-90 rates and shot metrics."""
    stats = matchstats.load(season, "players")
    if stats is None or stats.empty:
        return pd.DataFrame()
    stats = stats[stats["minutes_played"] > 0].dropna(subset=["code"]).astype({"code": int})
    totals = stats.groupby("code").agg(minutes=("minutes_played", "sum"), matches=("match_id", "nunique"),
                                       **{c: (c, "sum") for c in PER_90 if c in stats})
    out = totals[["minutes", "matches"]].copy()
    for col, name in PER_90.items():
        if col in totals:
            out[name] = totals[col] / totals["minutes"] * 90
    s = shots(season)
    if len(s):
        per = s.groupby("code").agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("outcome", lambda o: (o == "goal").sum()),
                                    in_box=("in_box", "sum"), distance=("start_x", "mean"),
                                    headers=("body_part", lambda b: (b == "head").sum()))
        out = out.join(per)
    out[["shots", "xg", "goals", "in_box", "headers"]] = out.reindex(
        columns=["shots", "xg", "goals", "in_box", "headers"]).fillna(0)
    enough = out["shots"] >= MIN_SHOTS
    out["box_shot_share"] = (out["in_box"] / out["shots"]).where(enough)
    out["shot_distance"] = (out.get("distance", np.nan) * METRES).where(enough)
    out["xg_per_shot"] = (out["xg"] / out["shots"]).where(enough)
    out["header_share"] = (out["headers"] / out["shots"]).where(enough)
    return out.drop(columns=["distance", "in_box", "headers"], errors="ignore")


def zone_table(season: str, code: int) -> pd.DataFrame:
    """One player's shots per zone (every zone, empty ones included): shots, xG, goals, share of his shots."""
    s = shots(season)
    mine = s[s["code"] == code].dropna(subset=["zone"])
    per = mine.groupby("zone").agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("outcome", lambda o: (o == "goal").sum()))
    table = zones().set_index("zone").join(per).fillna({"shots": 0, "xg": 0.0, "goals": 0})
    table["share"] = table["shots"] / max(len(mine), 1)
    return table.reset_index()


def similar(table: pd.DataFrame, position: pd.Series, k: int = 5) -> dict[int, list[tuple[int, float]]]:
    """Per player (code), the `k` most similar players at the same position: [(code, cosine similarity)].

    Each metric is standardised within position (so a centre-back's 20 box touches a season is
    compared with other centre-backs'); a missing shot metric counts as the position's average.
    Only players with `MIN_MINUTES` are compared."""
    t = table[table["minutes"] >= MIN_MINUTES].join(position.rename("position"), how="inner")
    out: dict[int, list[tuple[int, float]]] = {}
    for _, g in t.groupby("position"):
        cols = [c for c in SIMILARITY if c in g and g[c].notna().sum() > 1]
        z = (g[cols] - g[cols].mean()) / g[cols].std(ddof=0).replace(0, 1)
        v = z.fillna(0.0).to_numpy()
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        unit = v / np.where(norm == 0, 1, norm)
        sims = unit @ unit.T
        np.fill_diagonal(sims, -np.inf)
        codes = g.index.to_numpy()
        for i, code in enumerate(codes):
            best = np.argsort(-sims[i])[:k]
            out[int(code)] = [(int(codes[j]), round(float(sims[i, j]), 3)) for j in best if np.isfinite(sims[i, j])]
    return out
