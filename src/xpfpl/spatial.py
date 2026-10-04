"""Where a player plays: spatial metrics from the match data (data/matchstats.py), for display.

There are no heatmaps or full event data here (every pass and touch with its location). What the
source has is each shot's location, and per match zone counts: touches in the opposition box,
passes into the final third. So a player's spatial profile is built from those:

  - his shots binned into zones round the goal he attacks (`ZONES_X` x `ZONES_Y`): counts, xG and
    goals per zone, the density matrix a heatmap would show
  - per 90 minutes: touches, touches in the opposition box, final-third passes, chances created, shots
  - share of shots from inside the box, mean shot distance, xG per shot, share of headers
  - per 90 minutes, what each position is judged on: tackles, interceptions, clearances, blocks,
    recoveries, aerials won, crosses, dribbles, xA; for keepers saves, xG on target faced, goals
    prevented and the share of shots on target saved
  - `keeper_shots`: every shot on target with the keeper who faced it (for his goal-frame view)
  - `games`: FPL's own match rows (defensive contribution and its target, saves, goals conceded)

`similar` finds the players whose profile is closest (cosine similarity of the position's own
metrics, `SIMILARITY`, each standardised within position): keepers by shot-stopping, defenders by
how they defend and get forward, midfielders and forwards by where they get the ball and shoot.
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
          "total_shots": "shots_p90", "xa": "xa_p90", "accurate_crosses": "crosses_p90",
          "successful_dribbles": "dribbles_p90", "tackles": "tackles_p90", "interceptions": "interceptions_p90",
          "clearances": "clearances_p90", "blocks": "blocks_p90", "recoveries": "recoveries_p90",
          "aerial_duels_won": "aerials_won_p90", "saves": "saves_p90", "goals_conceded": "conceded_p90",
          "xgot_faced": "xgot_faced_p90", "goals_prevented": "goals_prevented_p90"}
# The metrics a profile is compared on, by FPL position (standardised within it): a keeper by his
# shot-stopping, a defender by how he defends and how far he gets forward, a midfielder by where he
# gets the ball, what he makes of it and how much he wins back, a forward by his shots.
SHOT_METRICS = ["box_shot_share", "shot_distance", "xg_per_shot", "header_share"]
SIMILARITY = {
    1: ["saves_p90", "save_share", "goals_prevented_p90", "xgot_faced_p90", "touches_p90"],
    2: ["tackles_p90", "interceptions_p90", "clearances_p90", "blocks_p90", "recoveries_p90", "aerials_won_p90",
        "crosses_p90", "chances_created_p90", "final_third_passes_p90", "box_touches_p90"],
    3: ["touches_p90", "box_touches_p90", "final_third_passes_p90", "chances_created_p90", "xa_p90", "dribbles_p90",
        "shots_p90", "box_shot_share", "xg_per_shot", "tackles_p90", "recoveries_p90"],
    4: ["touches_p90", "box_touches_p90", "final_third_passes_p90", "chances_created_p90", "shots_p90", *SHOT_METRICS],
}
MIN_MINUTES = 270                    # three full matches before a profile is compared
MIN_SHOTS = 3                        # shot metrics need a few shots; fewer and they count as the position's average
MIN_ON_TARGET = 5                    # a keeper's save share needs this many shots on target faced
DC_THRESHOLD = {2: 10, 3: 12, 4: 12}  # FPL's defensive-contribution target (scoring.DC_THRESHOLD, GKPs can't)
GKP = 1


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
        # A stat that is 0 for everyone all season isn't recorded that season (2026-27's feed has
        # no tackles, dribbles or defensive contributions): missing, not zero.
        if col in totals and totals[col].abs().sum() > 0:
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
    out["shot_distance"] = (out.get("distance", pd.Series(np.nan, index=out.index)) * METRES).where(enough)
    out["xg_per_shot"] = (out["xg"] / out["shots"]).where(enough)
    out["header_share"] = (out["headers"] / out["shots"]).where(enough)
    # Keepers: the share of shots on target he saved, and goals prevented (xG on target faced
    # minus goals let in: above 0 means he stopped more than an average keeper would).
    if {"saves", "goals_conceded", "goals_prevented"} <= set(stats):
        keep = stats.groupby("code")[["saves", "goals_conceded", "goals_prevented"]].sum()
        faced = keep["saves"] + keep["goals_conceded"]
        out["goals_prevented"] = keep["goals_prevented"]
        out["save_share"] = (keep["saves"] / faced).where(faced >= MIN_ON_TARGET)
    return out.drop(columns=["distance", "in_box", "headers"], errors="ignore")


def similarity_metrics() -> list[str]:
    """Every metric any position is compared on (for the position averages)."""
    return list(dict.fromkeys(m for ms in SIMILARITY.values() for m in ms))


def games(season: str, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match he played, `season`'s, from FPL's own data (`matches`, as
    data/history.py builds it): minutes, defensive contribution (FPL's count, the one that scores:
    FotMob's matches it for only ~78% of defenders' matches) and its clearances/blocks/interceptions
    part, saves and goals conceded; plus FotMob's chances created, xA, xG on target faced and goals
    prevented where the match data has them."""
    cols = ["code", "gw", "fixture", "minutes", "defensive_contribution", "clearances_blocks_interceptions",
            "saves", "goals_conceded", "expected_goals_conceded"]
    m = matches.loc[(matches["season"] == season) & (matches["minutes"] > 0), [c for c in cols if c in matches]]
    m = m.rename(columns={"defensive_contribution": "dc", "clearances_blocks_interceptions": "cbi",
                          "goals_conceded": "conceded", "expected_goals_conceded": "xgc"})
    stats = matchstats.load(season, "players")
    if stats is not None and len(stats):
        extra = (stats.dropna(subset=["code"]).astype({"code": int})
                 .groupby(["code", "fixture"])[["chances_created", "xa", "xgot_faced", "goals_prevented"]].sum())
        m = m.join(extra, on=["code", "fixture"])
    return m.sort_values(["code", "gw", "fixture"]).reset_index(drop=True)


def dc_summary(g: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    """Per player (code): defensive contribution per 90 and the matches he reached FPL's target in
    (the 2 points), from `games`. `position` is FPL's element_type by code."""
    if g.empty or "dc" not in g:
        return pd.DataFrame(columns=["dc_p90", "dc_hits"])
    target = g["code"].map(position).map(DC_THRESHOLD)
    per = g.assign(hit=(g["dc"] >= target).astype(int)).groupby("code").agg(dc=("dc", "sum"), minutes=("minutes", "sum"),
                                                                            dc_hits=("hit", "sum"))
    return pd.DataFrame({"dc_p90": per["dc"] / per["minutes"].clip(lower=90) * 90, "dc_hits": per["dc_hits"]})


def keeper_shots(season: str, position: pd.Series) -> pd.DataFrame:
    """Every shot on target (saved or scored) with the keeper who faced it (`keeper`, an FPL code):
    the other side's goalkeeper on the pitch at that minute (`position` = element_type by code).
    Own goals aren't shots, so they aren't here."""
    s = shots(season)
    stats = matchstats.load(season, "players")
    if s.empty or stats is None or stats.empty:
        return pd.DataFrame(columns=["keeper"])
    on = s[s["outcome"].isin(["goal", "save"]) & s["team_code"].notna()].copy()
    keepers = stats.dropna(subset=["code"]).astype({"code": int})
    keepers = keepers[(keepers["minutes_played"] > 0) & (keepers["code"].map(position) == GKP)]
    pairs = on.reset_index().merge(keepers[["fixture", "code", "team_code", "start_min", "finish_min", "minutes_played"]],
                                   on="fixture", suffixes=("", "_gk"))
    pairs = pairs[pairs["team_code_gk"] != pairs["team_code"]]
    pairs["on_pitch"] = pairs["minute"].between(pairs["start_min"], pairs["finish_min"])     # added time is minute 90
    # The keeper on the pitch at that minute; failing that (clocks differ), whoever played most.
    pairs = pairs.sort_values(["on_pitch", "minutes_played"], ascending=False).drop_duplicates("index")
    out = on.join(pairs.set_index("index")["code_gk"].rename("keeper"))
    return out.dropna(subset=["keeper"]).astype({"keeper": int})


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

    Each position is compared on its own metrics (`SIMILARITY`), each standardised within the
    position (so a defender's 20 box touches a season is compared with other defenders'); a missing
    metric counts as the position's average. Only players with `MIN_MINUTES` are compared."""
    t = table[table["minutes"] >= MIN_MINUTES].join(position.rename("position"), how="inner")
    out: dict[int, list[tuple[int, float]]] = {}
    for pos, g in t.groupby("position"):
        cols = [c for c in SIMILARITY.get(int(pos), SIMILARITY[4]) if c in g and g[c].notna().sum() > 1]
        if not cols:
            continue
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
