"""Turn player-match rows into model features.

The golden rule: a row's features may only use information from *before* that match
(otherwise the model "sees the future" in training and looks great, then fails live).
Rolling stats are therefore shifted by one match for training rows. For upcoming fixtures
we use each player's/team's latest state, i.e. rolling stats including their last match.

`stale=k` shifts everything by k matches instead of one: the features a manager would have
had k gameweeks before the match. validate.py uses it to score 2- and 3-GW-ahead forecasts.

What is in here, and why:
  - rolling means over 3/5/10 matches: recent form and role.
  - per-90 rates over 20 and 38 matches: a player's underlying level. Five matches of xG are
    mostly noise; it takes 20-40 before a player's rate settles.
  - the last six matches in order (`<stat>_lag1..6`), for the GRU.
  - points at the same venue (home/away) as the next match.
  - club form (goals and xG for/against, 10 and 38 matches) and fitted attack/defence ratings
    (teams.py), turned into this fixture's expected goals and clean-sheet chance.
  - the crowd: net transfers before this deadline and ownership, a read on team news.
  - `bps_share`: where a player's BPS ranked within his match, which survives the BPS rule
    changes (2024-25 rescaled it, 2026-27 re-weighted it) better than raw BPS does.
  - role security: where a player ranks at his club and position by price and by recent
    minutes. Competition for a place is what drives rotation, so the minutes model needs it.
  - betting markets (data/markets.py): the expected goals, clean-sheet and win chances implied
    by the match odds at the deadline. Odds only exist for the next gameweek, so everywhere else
    (and before 2024-25) they fall back to our own ratings, with `mkt_known` saying which is
    which. Close to neutral on 2025-26 (the component model gains 0.015 RMSE, the rest move by
    0.002 or less); only 2024-25 had odds to learn from. The anytime-goalscorer odds are *not*
    used: on 2025-26 they ranked scorers barely better than chance (AUC 0.58 vs our model's
    0.73) and priced them ~60% too high, so they are shown in the dashboard only.

Also built but not in FEATURES (no out-of-sample gain yet; see NEW_FEATURES): finishing luck
(goals - xG, assists - xA), fixture congestion (days of rest before and after, cups included),
price moves since the last deadline and the season's start, and where a player plays (touches,
box touches, final-third passes, chances, shots per 90 and box-shot share, 2025-26 on).
"""

import numpy as np
import pandas as pd

from xpfpl import teams
from xpfpl.data.history import STAT_COLS

WINDOWS = (3, 5, 10)
TEAM_WINDOW = 10
TEAM_LONG_WINDOW = 38
PLAYER_STATS = [c for c in STAT_COLS] + ["played", "played60", "bps_share"]
TEAM_STATS = ["gf", "ga", "xgf", "xga"]

# Long windows, as rates per 90 minutes played (a sub's 20 minutes shouldn't dilute them).
LONG_WINDOWS = (20, 38)
RATE_STATS = ["total_points", "goals_scored", "assists", "expected_goals", "expected_assists",
              "threat", "creativity", "bonus", "saves", "goals_conceded",
              "clean_sheets", "defensive_contribution"]
RATE_FEATURES = [f"{s}_p90_{w}" for w in LONG_WINDOWS for s in RATE_STATS]
LONG_FEATURES = RATE_FEATURES + [f"minutes_r{w}" for w in LONG_WINDOWS] + [f"played_r{w}" for w in LONG_WINDOWS]

# The last SEQ_LEN matches, most recent first, as plain columns. Rolling means throw the order
# away ("4 points a game"); these keep it ("2, then 2, then 12"), which is what the GRU in
# models/sequence.py reads. Kept short so the extra columns stay cheap for every other model.
SEQ_STATS = ["minutes", "total_points", "goals_scored", "assists", "bonus", "bps",
             "clean_sheets", "goals_conceded", "saves", "ict_index"]
SEQ_LEN = 6
LAG_FEATURES = [f"{s}_lag{k}" for k in range(1, SEQ_LEN + 1) for s in SEQ_STATS]

VENUE_WINDOW = 10
VENUE_STATE = ["pts_home_r10", "pts_away_r10"]     # carried by the player; the fixture picks one
CROWD_FEATURES = ["transfer_flow", "ownership"]

# Finishing luck: goals (assists) minus xG (xA), over the last 10 matches and per 90 over 38. A
# player scoring well above his xG tends to come back down; one below it, up. Only matches with xG
# (2022-23 on) count, so a window reaching back before then isn't read as pure "luck".
LUCK_WINDOW, LUCK_LONG = 10, 38
LUCK_FEATURES = [f"goal_luck_r{LUCK_WINDOW}", f"assist_luck_r{LUCK_WINDOW}",
                 f"goal_luck_p90_{LUCK_LONG}", f"assist_luck_p90_{LUCK_LONG}"]
# Fixture congestion: days since the club's previous match and until its next one in any
# competition it plays (cup and European matches from data/cups.py, 2025-26 on; earlier seasons
# only know the Premier League, which `cup_era` tells the model), and the opponent's rest.
REST_CAP = 14.0
REST_FEATURES = ["rest_days", "next_days", "opp_rest_days", "cup_era"]
# Price moves (£m): since the last deadline (the week's rises and falls, i.e. the crowd's verdict
# in full) and since the season started. Both are known at the deadline.
PRICE_FEATURES = ["price_change_gw", "price_change_start"]
# Where a player plays (data/matchstats.py, 2025-26 on): per 90 over his last 10 matches with
# match data, and the share of his shots taken inside the box. `spatial_era` is 0 before then.
SPATIAL_WINDOW = 10
SPATIAL_STATS = {"touches": "touches", "touches_opposition_box": "box_touches",
                 "final_third_passes": "final_third_passes", "chances_created": "chances_created",
                 "total_shots": "shots"}
SPATIAL_FEATURES = ([f"{n}_p90_m{SPATIAL_WINDOW}" for n in SPATIAL_STATS.values()]
                    + [f"box_shot_share_m{SPATIAL_WINDOW}", "spatial_era"])

ROLLING_FEATURES = [f"{s}_r{w}" for w in WINDOWS for s in PLAYER_STATS]
TEAM_FEATURES = ["team_gf", "team_ga", "opp_gf", "opp_ga",
                 "team_xgf", "team_xga", "opp_xgf", "opp_xga",
                 "team_gf38", "team_ga38", "opp_gf38", "opp_ga38"]
FIXTURE_FEATURES = ["fx_gf", "fx_ga", "fx_cs"]    # from the attack/defence ratings
MARKET_FEATURES = ["mkt_gf", "mkt_ga", "mkt_cs", "mkt_win", "mkt_known"]
ROLE_FEATURES = ["price_rank_team", "minutes_rank_team"]
OTHER_FEATURES = [
    "was_home", "price", "experience", "venue_pts", *ROLE_FEATURES,
    "pos_1", "pos_2", "pos_3", "pos_4",
    "xg_era", "dc_era",
]
BASE_FEATURES = (ROLLING_FEATURES + LONG_FEATURES + TEAM_FEATURES + FIXTURE_FEATURES + MARKET_FEATURES
                 + CROWD_FEATURES + OTHER_FEATURES)
# Built for every row, but not fed to the models (2026-10-02): out of sample, each group moved RMSE
# by less than the noise between seeds (LightGBM, 3 seeds x 2022-23 to 2026-27 GW1-5; the ensemble,
# 2024-25 to 2026-27). LightGBM: luck +0.001, rest days +0.001, price moves -0.001, spatial
# -0.003 on 2026-27 GW1-5, the only season it can be tested on (matchstats start in 2025-26).
# Ensemble: luck +0.002 in all three seasons, rest +0.001 to +0.003, price -0.002 to +0.009,
# spatial +0.002 on 2026-27. Add a group to FEATURES to retry it; spatial is the one worth retrying when 2026-27 is complete.
NEW_FEATURES = LUCK_FEATURES + REST_FEATURES + PRICE_FEATURES + SPATIAL_FEATURES
FEATURES = BASE_FEATURES
TARGET = "total_points"

# What a player / a club carries from one match to the next. The backtest overwrites these
# from a pre-deadline snapshot; everything else about a row is known from the fixture list.
PLAYER_STATE = (ROLLING_FEATURES + LONG_FEATURES + LAG_FEATURES + VENUE_STATE + CROWD_FEATURES + ["experience"]
                + LUCK_FEATURES + PRICE_FEATURES + SPATIAL_FEATURES)
TEAM_STATE = ["gf", "ga", "xgf", "xga", "gf38", "ga38"]
LEAGUE_AVG_GOALS = 1.35


def _add_derived(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    year = m["season"].str[:4].astype(int)
    # 2026-27 BPS: clearances, blocks and interceptions pay 1 per 3 instead of 1 per 2. Rebuild
    # 2025-26 (the first season with CBI recorded next to the current scoring) on the new scale,
    # so last season's BPS form reads the same as this season's.
    cbi = m["clearances_blocks_interceptions"] if "clearances_blocks_interceptions" in m else 0
    rebuild = (year == 2025) & pd.notna(cbi)
    cbi = pd.Series(cbi, index=m.index).fillna(0)
    m["bps"] = np.where(rebuild, m["bps"] - cbi // 2 + cbi // 3, m["bps"])
    m["played"] = (m["minutes"] > 0).astype(float)
    m["played60"] = (m["minutes"] >= 60).astype(float)
    # BPS rank within the match (1 = top of the match, 0 = bottom), among players who played.
    rank = m[m["played"] > 0].groupby(["season", "fixture"])["bps"].rank(pct=True)
    m["bps_share"] = rank.reindex(m.index).fillna(0.0)
    m["gf"] = np.where(m["was_home"], m["team_h_score"], m["team_a_score"])
    m["ga"] = np.where(m["was_home"], m["team_a_score"], m["team_h_score"])
    return m


def _rolling(df: pd.DataFrame, key: str, cols: list[str], windows, shift: int, suffix=True) -> pd.DataFrame:
    """Per-`key` rolling means of `cols`, optionally shifted so row i only sees rows < i."""
    df = df.sort_values("kickoff_time")
    base = df.groupby(key)[cols].shift(shift) if shift else df[cols]
    base = base.assign(**{key: df[key]})
    out = {}
    for w in windows:
        rolled = base.groupby(key)[cols].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        for c in cols:
            out[f"{c}_r{w}" if suffix else c] = rolled[c]
    return pd.DataFrame(out, index=df.index)


def _rates(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Per-90 rates over the last 20/38 matches, plus minutes and appearance rates."""
    df = df.sort_values("kickoff_time")
    cols = RATE_STATS + ["minutes", "played"]
    base = df.groupby("code")[cols].shift(shift) if shift else df[cols]
    base = base.assign(code=df["code"])
    out = {}
    for w in LONG_WINDOWS:
        grouped = base.groupby("code")[cols]
        sums = grouped.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        means = grouped.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        minutes = np.maximum(sums["minutes"], 90.0)            # at least a match's worth
        for s in RATE_STATS:
            out[f"{s}_p90_{w}"] = sums[s] / minutes * 90.0
        out[f"minutes_r{w}"] = means["minutes"]
        out[f"played_r{w}"] = means["played"]
    return pd.DataFrame(out, index=df.index)


def _venue(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Average points over the last VENUE_WINDOW home (and away) matches, as known before each row.

    Computed on each venue's own rows (including that match), then shifted by `shift` rows of the
    player's timeline and carried forward, so an away row sees his latest home average too.
    """
    df = df.sort_values("kickoff_time")
    out = {}
    for name, home in (("pts_home_r10", True), ("pts_away_r10", False)):
        rows = df[df["was_home"] == home]
        avg = (rows.groupby("code")["total_points"].rolling(VENUE_WINDOW, min_periods=1).mean()
               .reset_index(level=0, drop=True))
        col = avg.reindex(df.index)
        col = col.groupby(df["code"]).shift(shift) if shift else col
        out[name] = col.groupby(df["code"]).ffill()
    return pd.DataFrame(out, index=df.index)


def _crowd(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Net transfers before this deadline relative to ownership, and ownership itself.

    Both are known at the deadline of the row's own gameweek, so for `stale=k` they come from
    the row k-1 matches back (what was known at that earlier deadline).
    """
    df = df.sort_values("kickoff_time")
    selected = df["selected"].astype(float)
    flow = np.log((selected + 5e3) / (selected - df["transfers_balance"].astype(float) + 5e3).clip(lower=5e3))
    own = np.log10(selected.clip(lower=0) + 1.0) / 7.0
    out = pd.DataFrame({"transfer_flow": flow, "ownership": own}, index=df.index)
    if shift > 1:
        out = out.groupby(df["code"]).shift(shift - 1)
    return out


def _luck(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Goals minus xG and assists minus xA: the mean over LUCK_WINDOW matches and per 90 over
    LUCK_LONG, counting only matches with xG (2022-23 on). 0 where there are none."""
    df = df.sort_values("kickoff_time")
    known = df["season"].str[:4].astype(int) >= 2022
    diff = pd.DataFrame({"g": (df["goals_scored"] - df["expected_goals"]).where(known),
                         "a": (df["assists"] - df["expected_assists"]).where(known),
                         "min": df["minutes"].where(known)}, index=df.index)
    base = diff.groupby(df["code"]).shift(shift) if shift else diff
    grouped = base.assign(code=df["code"]).groupby("code")
    short = grouped[["g", "a"]].rolling(LUCK_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    sums = grouped[["g", "a", "min"]].rolling(LUCK_LONG, min_periods=1).sum().reset_index(level=0, drop=True)
    minutes = np.maximum(sums["min"], 90.0)
    return pd.DataFrame({
        f"goal_luck_r{LUCK_WINDOW}": short["g"], f"assist_luck_r{LUCK_WINDOW}": short["a"],
        f"goal_luck_p90_{LUCK_LONG}": sums["g"] / minutes * 90.0,
        f"assist_luck_p90_{LUCK_LONG}": sums["a"] / minutes * 90.0,
    }, index=df.index).reindex(df.index).fillna(0.0)


def _price_moves(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Price change (£m) since the previous gameweek's deadline and since the season's start.

    A row's `value` is the price at its gameweek's deadline, so both are known then; `stale=k`
    takes them from k-1 gameweeks back. A double gameweek's two rows share one price."""
    per_gw = (df[["code", "season", "gw", "value", "kickoff_time"]].sort_values("kickoff_time")
              .drop_duplicates(["code", "season", "gw"]))
    season = per_gw.groupby(["code", "season"])["value"]
    moves = pd.DataFrame({"price_change_gw": season.diff().fillna(0) / 10.0,
                          "price_change_start": (per_gw["value"] - season.transform("first")) / 10.0},
                         index=per_gw.index)
    if shift > 1:
        moves = moves.groupby(per_gw["code"]).shift(shift - 1).fillna(0.0)
    moves[["code", "season", "gw"]] = per_gw[["code", "season", "gw"]]
    keys = df[["code", "season", "gw"]].reset_index()
    return keys.merge(moves, on=["code", "season", "gw"], how="left").set_index("index")[PRICE_FEATURES].reindex(df.index)


def club_schedule(matches: pd.DataFrame, upcoming: pd.DataFrame | None = None,
                  cup_fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every kick-off of every club (`team_code`, `kickoff_time`): Premier League matches played
    (`matches`) and to come (`upcoming`, same two columns), and cup/European ones (data/cups.py)."""
    parts = [matches[["team_code", "kickoff_time"]]]
    if upcoming is not None:
        parts.append(upcoming[["team_code", "kickoff_time"]])
    if cup_fixtures is not None and len(cup_fixtures):
        real = cup_fixtures[cup_fixtures["tournament"] != "friendlies"]
        parts.append(real[["team_code", "kickoff_time"]])
    s = pd.concat(parts, ignore_index=True).dropna()
    s["team_code"] = s["team_code"].astype(int)
    s["kickoff_time"] = pd.to_datetime(s["kickoff_time"], utc=True).astype("datetime64[ns, UTC]")
    return s.drop_duplicates().sort_values("kickoff_time", ignore_index=True)


def rest_features(rows: pd.DataFrame, schedule: pd.DataFrame, cup_seasons) -> pd.DataFrame:
    """Days since the club's previous match and until its next (any competition in `schedule`),
    and the opponent's days since its previous one, each capped at REST_CAP; plus `cup_era`."""
    out = {}
    times = pd.to_datetime(rows["kickoff_time"], utc=True).astype("datetime64[ns, UTC]")
    for name, key, direction in (("rest_days", "team_code", "backward"), ("next_days", "team_code", "forward"),
                                 ("opp_rest_days", "opp_code", "backward")):
        left = pd.DataFrame({"team_code": rows[key].fillna(-1).astype(int).to_numpy(), "t": times.to_numpy(),
                             "i": np.arange(len(rows))}).sort_values("t")
        got = pd.merge_asof(left, schedule.rename(columns={"kickoff_time": "other"}), left_on="t",
                            right_on="other", by="team_code", direction=direction, allow_exact_matches=False)
        days = (got["t"] - got["other"]).abs().dt.total_seconds() / 86400
        out[name] = pd.Series(days.clip(upper=REST_CAP).fillna(REST_CAP).to_numpy(), index=got["i"]).sort_index().to_numpy()
    out["cup_era"] = rows["season"].isin(set(cup_seasons)).astype(float).to_numpy()
    return pd.DataFrame(out, index=rows.index)


def _rest(m: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    from xpfpl.data import cups
    fixtures, _ = cups.load()
    schedule = club_schedule(m, upcoming, fixtures)
    return rest_features(m, schedule, fixtures["season"].unique() if len(fixtures) else [])


def spatial_matches() -> pd.DataFrame:
    """Per player per match with match data: minutes, the SPATIAL_STATS and shots inside the box."""
    from xpfpl import spatial
    from xpfpl.data import matchstats
    parts = []
    for season in matchstats.seasons():
        p = matchstats.load(season, "players")
        if p is None or p.empty:
            continue
        p = p.dropna(subset=["code"]).astype({"code": int})
        stats = p.groupby(["fixture", "code"])[["minutes_played", *SPATIAL_STATS]].sum()
        s = spatial.shots(season)
        box = s[s["in_box"]].groupby(["fixture", "code"]).size() if len(s) else pd.Series(dtype=float)
        stats["box_shots"] = box.reindex(stats.index).fillna(0)
        parts.append(stats.reset_index().assign(season=season))
    if not parts:
        return pd.DataFrame(columns=["season", "fixture", "code", "minutes_played", *SPATIAL_STATS, "box_shots"])
    return pd.concat(parts, ignore_index=True)


def _spatial(df: pd.DataFrame, shift: int, stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """SPATIAL_FEATURES for each row, from the player's last SPATIAL_WINDOW matches (shifted by
    `shift`). A player missing from a match's data with 0 FPL minutes didn't play (zeros); seasons
    without match data are NaN, so a window reaching back before 2025-26 only counts what's known."""
    stats = spatial_matches() if stats is None else stats
    df = df.sort_values("kickoff_time")
    cols = ["minutes_played", *SPATIAL_STATS, "box_shots"]
    joined = df[["season", "fixture", "code", "minutes"]].reset_index().merge(
        stats, on=["season", "fixture", "code"], how="left").set_index("index").reindex(df.index)
    era = df["season"].isin(set(stats["season"]))
    for c in cols:
        joined[c] = joined[c].where(joined[c].notna() | ~era | (df["minutes"] > 0), 0.0)
    base = joined[cols].groupby(df["code"]).shift(shift) if shift else joined[cols]
    sums = (base.assign(code=df["code"]).groupby("code")[cols]
            .rolling(SPATIAL_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)).reindex(df.index)
    minutes = np.maximum(sums["minutes_played"].fillna(0), 90.0)
    out = {f"{name}_p90_m{SPATIAL_WINDOW}": sums[c].fillna(0) / minutes * 90.0 for c, name in SPATIAL_STATS.items()}
    # The shot list sometimes has a shot more than the player's stats line: capped at 1.
    out[f"box_shot_share_m{SPATIAL_WINDOW}"] = ((sums["box_shots"].fillna(0) + 1)
                                                 / (sums["total_shots"].fillna(0) + 2)).clip(upper=1.0)
    out["spatial_era"] = era.astype(float)
    return pd.DataFrame(out, index=df.index)


def _lags(df: pd.DataFrame, key: str, cols: list[str], n: int, shift: int) -> pd.DataFrame:
    """Per-`key` values from the previous `n` matches: `<col>_lag1` is the most recent one.

    `shift=1` for training rows (lag1 = the match before this one), `shift=0` for the latest
    state (lag1 = the player's last completed match).
    """
    df = df.sort_values("kickoff_time")
    out = {}
    for k in range(1, n + 1):
        lagged = df.groupby(key)[cols].shift(shift + k - 1)
        for c in cols:
            out[f"{c}_lag{k}"] = lagged[c]
    return pd.DataFrame(out, index=df.index)


def _team_matches(m: pd.DataFrame) -> pd.DataFrame:
    """One row per club per fixture: goals, and xG for/against (goals before xG existed)."""
    teams_ = (m.groupby(["season", "fixture", "team_code"], sort=False)
               .agg(kickoff_time=("kickoff_time", "first"), gf=("gf", "first"), ga=("ga", "first"),
                    xgf=("expected_goals", lambda s: s.sum(min_count=1)))
               .reset_index())
    opp = teams_[["season", "fixture", "team_code", "xgf"]].rename(columns={"team_code": "o", "xgf": "xga"})
    teams_ = teams_.merge(opp, on=["season", "fixture"])
    teams_ = teams_[teams_["team_code"] != teams_["o"]].drop(columns="o")
    teams_["xgf"] = teams_["xgf"].fillna(teams_["gf"])
    teams_["xga"] = teams_["xga"].fillna(teams_["ga"])
    return teams_.reset_index(drop=True)


def _team_form(teams_: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Club form per fixture: 10-match goals/xG for and against, and 38-match goals."""
    short = _rolling(teams_, "team_code", TEAM_STATS, (TEAM_WINDOW,), shift=shift, suffix=False)
    long = _rolling(teams_, "team_code", ["gf", "ga"], (TEAM_LONG_WINDOW,), shift=shift, suffix=False)
    return teams_[["season", "fixture", "team_code"]].join(short).join(long.add_suffix("38"))


def _join_team_form(m: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Attach club and opponent form (TEAM_STATE) to each player row as team_*/opp_* columns."""
    own = form.rename(columns={c: f"team_{c}" for c in TEAM_STATE})
    opp = form.rename(columns={"team_code": "opp_code", **{c: f"opp_{c}" for c in TEAM_STATE}})
    m = m.merge(own, on=["season", "fixture", "team_code"], how="left")
    return m.merge(opp, on=["season", "fixture", "opp_code"], how="left")


def fixture_strength(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Expected goals for/against and clean-sheet chance for each row's fixture, from one
    ratings table (one row per club). Used for upcoming fixtures and by the backtest."""
    gf, ga = teams.fixture_rates(table, df["team_code"], df["opp_code"], df["was_home"])
    return pd.DataFrame({"fx_gf": gf, "fx_ga": ga, "fx_cs": np.exp(-ga)}, index=df.index)


def _historical_strength(m: pd.DataFrame, ratings: pd.DataFrame, stale: int) -> pd.DataFrame:
    """Fixture strength from the ratings known `stale - 1` gameweeks before each row's gameweek."""
    key = m[["season", "gw", "team_code", "opp_code", "was_home"]].copy()
    key["gw"] = (key["gw"] - (stale - 1)).clip(lower=1)
    r = ratings.set_index(["season", "gw", "team_code"])
    mu_home = ratings.groupby(["season", "gw"])[["mu", "home"]].first()
    idx = pd.MultiIndex.from_arrays([key["season"], key["gw"]])
    mu = mu_home.reindex(idx)
    t = r.reindex(pd.MultiIndex.from_arrays([key["season"], key["gw"], key["team_code"]]))
    o = r.reindex(pd.MultiIndex.from_arrays([key["season"], key["gw"], key["opp_code"]]))
    h = key["was_home"].astype(float).to_numpy()
    prior_a, prior_d = teams.PROMOTED_PRIOR
    ta = np.nan_to_num(t["attack"].to_numpy(), nan=prior_a)
    td = np.nan_to_num(t["defence"].to_numpy(), nan=prior_d)
    oa = np.nan_to_num(o["attack"].to_numpy(), nan=prior_a)
    od = np.nan_to_num(o["defence"].to_numpy(), nan=prior_d)
    base, home = np.nan_to_num(mu["mu"].to_numpy(), nan=0.25), np.nan_to_num(mu["home"].to_numpy(), nan=0.2)
    gf = np.exp(base + home * h + ta - od)
    ga = np.exp(base + home * (1 - h) + oa - td)
    return pd.DataFrame({"fx_gf": gf, "fx_ga": ga, "fx_cs": np.exp(-ga)}, index=m.index)


def _poisson_win(gf, ga, max_goals: int = 10) -> np.ndarray:
    """P(scoring more than the opponent) with independent Poisson goals."""
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))

    def pmf(lam):
        lam = np.clip(np.asarray(lam, float), 1e-6, None)[:, None]
        return np.exp(k[None, :] * np.log(lam) - lam - log_fact[None, :])

    pf, pa = pmf(gf), pmf(ga)
    return np.einsum("ni,nj,ij->n", pf, pa, (k[:, None] > k[None, :]).astype(float))


def market_side(market: pd.DataFrame) -> pd.DataFrame:
    """The match odds as one row per club per match: (season, fixture, team_code) -> mkt_* ."""
    sides = []
    for us, them in (("home", "away"), ("away", "home")):
        cs = 1.0 - market.get(f"{them}_over_0.5", pd.Series(np.nan, index=market.index))
        sides.append(pd.DataFrame({
            "season": market["season"], "fixture": market["fixture"], "team_code": market[f"{us}_code"],
            "mkt_gf": market[f"lam_{us}"], "mkt_ga": market[f"lam_{them}"],
            "mkt_cs": cs.fillna(np.exp(-market[f"lam_{them}"])), "mkt_win": market[f"{us}_win"],
        }))
    return pd.concat(sides, ignore_index=True).dropna(subset=["mkt_gf", "mkt_ga"])


def add_market_features(df: pd.DataFrame, market: pd.DataFrame | None, use: bool = True) -> pd.DataFrame:
    """Join the odds for each row's match, falling back to our own ratings where there are none.

    `use=False` (forecasts made further ahead than the next deadline) takes the fallback for
    every row: those odds wouldn't have existed yet.
    """
    index = df.index
    df = df.drop(columns=[c for c in MARKET_FEATURES if c in df], errors="ignore")
    if use and market is not None and len(market):
        df = df.merge(market_side(market), on=["season", "fixture", "team_code"], how="left")
    df.index = index                                   # left merges keep the row order
    odds = df.reindex(columns=["mkt_gf", "mkt_ga", "mkt_cs", "mkt_win"])    # all-NaN where no odds
    known = odds["mkt_gf"].notna() & odds["mkt_win"].notna()
    fallback_win = _poisson_win(df["fx_gf"].fillna(LEAGUE_AVG_GOALS), df["fx_ga"].fillna(LEAGUE_AVG_GOALS))
    filled = pd.DataFrame({
        "mkt_gf": odds["mkt_gf"].where(known, df["fx_gf"]), "mkt_ga": odds["mkt_ga"].where(known, df["fx_ga"]),
        "mkt_cs": odds["mkt_cs"].where(known, df["fx_cs"]), "mkt_win": odds["mkt_win"].where(known, fallback_win),
        "mkt_known": known.astype(float)}, index=df.index)
    return pd.concat([df.drop(columns=odds.columns, errors="ignore"), filled], axis=1)


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that depend on the fixture as well as the player's state (safe to recompute).

    Role ranks are within the club's players at that position listed for the fixture:
    1 = the most expensive / most-played, 0 = the least.
    """
    home = df["was_home"].astype(bool).to_numpy()
    venue = np.where(home, df["pts_home_r10"], df["pts_away_r10"])
    group = [df["season"], df["fixture"], df["team_code"], df["position"]]
    price = df["value"] if "value" in df else df["price"] * 10
    return df.assign(
        venue_pts=pd.Series(venue, index=df.index).fillna(df["total_points_r10"]),
        price_rank_team=price.groupby(group).rank(pct=True).fillna(0.5),
        minutes_rank_team=df["minutes_r5"].fillna(0).groupby(group).rank(pct=True).fillna(0.5),
    )


def _assemble(df: pd.DataFrame) -> pd.DataFrame:
    """Add the non-rolling features and fill gaps so every row has all FEATURES."""
    df = df.copy()
    df["was_home"] = df["was_home"].astype(float)
    df["price"] = df["value"] / 10.0
    for p in range(1, 5):
        df[f"pos_{p}"] = (df["position"] == p).astype(float)
    year = df["season"].str[:4].astype(int)
    df["xg_era"] = (year >= 2022).astype(float)  # FPL xG stats exist from 2022-23
    df["dc_era"] = (year >= 2025).astype(float)  # defensive contribution points from 2025-26
    df["experience"] = df["experience"].clip(upper=38) / 38.0
    for col in TEAM_FEATURES:
        df[col] = df[col].fillna(LEAGUE_AVG_GOALS)
    for col in VENUE_STATE:
        if col not in df:
            df[col] = np.nan
    df = add_row_features(df)
    missing = [c for c in FEATURES + LAG_FEATURES if c not in df.columns]
    if missing:
        df[missing] = 0.0
    cols = FEATURES + LAG_FEATURES
    df[cols] = df[cols].fillna(0.0).astype("float32")
    return df.copy()   # the many inserts above leave the frame fragmented; one copy tidies it


def build_training_frame(matches: pd.DataFrame, stale: int = 1, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per historical player-match with pre-match features and the points scored.

    `stale=1` is the normal case (form up to the previous match); `stale=k` uses form as of k
    matches before, i.e. the forecast a manager would have made k gameweeks ahead. The rows and
    their order are the same for every `stale`, only the features differ. Market odds come from
    data/processed (or `market`), and only for `stale=1`: they're priced at the deadline.
    """
    m = _add_derived(matches)
    m = m.join(_rolling(m, "code", PLAYER_STATS, WINDOWS, shift=stale))
    m = m.join(_rates(m, shift=stale))
    m = m.join(_lags(m, "code", SEQ_STATS, SEQ_LEN, shift=stale))
    m = m.join(_venue(m, shift=stale))
    m = m.join(_crowd(m, shift=stale))
    m = m.join(_luck(m, shift=stale))
    m = m.join(_price_moves(m, shift=stale))
    m = m.join(_spatial(m, shift=stale))
    m = m.join(_rest(m))
    m["experience"] = (m.sort_values("kickoff_time").groupby("code").cumcount() - (stale - 1)).clip(lower=0)
    # FPL's own xP for the previous match is the fairest reading of "FPL's forecast" for this
    # one (the value on a row is recorded after that match). A benchmark only, not a feature.
    m["fpl_xp_prev"] = m.sort_values("kickoff_time").groupby("code")["xP"].shift(stale)

    m = m.join(_historical_strength(m, teams.ratings(matches), stale))
    m = _join_team_form(m, _team_form(_team_matches(m), shift=stale))
    m["horizon"] = float(stale)
    if market is None:
        from xpfpl.data import markets
        market, _ = markets.load()
    m = add_market_features(m, market, use=stale == 1)
    return _assemble(m)


def build_horizon_frames(matches: pd.DataFrame, horizons=(1, 2, 3)) -> dict[int, pd.DataFrame]:
    """`build_training_frame` at each staleness in `horizons`: {k: frame}, rows aligned."""
    return {k: build_training_frame(matches, stale=k) for k in horizons}


def latest_player_state(matches: pd.DataFrame) -> pd.DataFrame:
    """Every PLAYER_STATE column per player `code`, including their most recent match."""
    m = _add_derived(matches).sort_values("kickoff_time")
    rolled = _rolling(m, "code", PLAYER_STATS, WINDOWS, shift=0)
    rolled = rolled.join(_rates(m, shift=0))
    rolled = rolled.join(_lags(m, "code", SEQ_STATS, SEQ_LEN, shift=0))
    rolled = rolled.join(_venue(m, shift=0))
    rolled = rolled.join(_luck(m, shift=0))
    rolled = rolled.join(_spatial(m, shift=0))
    rolled["code"] = m["code"]
    state = rolled.groupby("code").tail(1).set_index("code")  # rows are time-sorted
    state["experience"] = m.groupby("code").size()
    return state


def latest_team_state(matches: pd.DataFrame) -> pd.DataFrame:
    """TEAM_STATE per club `team_code`, including their most recent match."""
    teams_ = _team_matches(_add_derived(matches)).sort_values("kickoff_time")
    form = _team_form(teams_, shift=0)
    form["kickoff_time"] = teams_["kickoff_time"]
    return form.sort_values("kickoff_time").groupby("team_code").tail(1).set_index("team_code")[TEAM_STATE]


def crowd_now(bs: dict) -> pd.DataFrame:
    """This gameweek's transfer flow and ownership so far, from bootstrap-static (by element)."""
    players = pd.DataFrame(bs["elements"]).set_index("id")
    total = float(bs.get("total_players") or 10_000_000)
    selected = pd.to_numeric(players["selected_by_percent"], errors="coerce").fillna(0) / 100.0 * total
    balance = (players["transfers_in_event"] - players["transfers_out_event"]).astype(float)
    flow = np.log((selected + 5e3) / (selected - balance + 5e3).clip(lower=5e3))
    return pd.DataFrame({"transfer_flow": flow, "ownership": np.log10(selected + 1.0) / 7.0})


def prices_now(bs: dict) -> pd.DataFrame:
    """Each player's price change (£m) since the last deadline and since the season started (by element)."""
    players = pd.DataFrame(bs["elements"]).set_index("id")
    return pd.DataFrame({"price_change_gw": pd.to_numeric(players["cost_change_event"], errors="coerce").fillna(0) / 10.0,
                         "price_change_start": pd.to_numeric(players["cost_change_start"], errors="coerce").fillna(0) / 10.0})


def build_future_frame(matches: pd.DataFrame, bs: dict, fixtures: list[dict], gameweeks: list[int],
                       season: str, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per current player per upcoming fixture in `gameweeks`, ready for the model."""
    players = pd.DataFrame(bs["elements"])
    team_code = {t["id"]: t["code"] for t in bs["teams"]}

    rows = []
    for f in fixtures:
        if f["event"] not in gameweeks:
            continue
        for team, opp, home in ((f["team_h"], f["team_a"], True), (f["team_a"], f["team_h"], False)):
            rows.append({"gw": f["event"], "fixture": f["id"], "team": team, "opponent_team": opp,
                         "was_home": home, "kickoff_time": f["kickoff_time"]})
    fx = pd.DataFrame(rows)

    df = players.rename(columns={"id": "element", "element_type": "position", "now_cost": "value"})
    df = df[["element", "code", "web_name", "position", "team", "value", "status",
             "chance_of_playing_next_round"]].merge(fx, on="team", how="inner")
    df["team_code"] = df["team"].map(team_code)
    df["opp_code"] = df["opponent_team"].map(team_code)
    df["season"] = season

    player_state = latest_player_state(matches)
    df = df.merge(player_state, left_on="code", right_index=True, how="left")
    df = df.join(crowd_now(bs), on="element")
    df = df.join(prices_now(bs), on="element")
    df["spatial_era"] = float(season >= "2025-26")
    # Every Premier League kick-off still to come (not just the horizon's), so `next_days` sees
    # the match after the last week forecast.
    upcoming = pd.DataFrame([{"team_code": team_code.get(f[side]), "kickoff_time": f["kickoff_time"]}
                             for f in fixtures for side in ("team_h", "team_a") if f.get("kickoff_time")])
    from xpfpl.data import cups
    cup_fixtures, _ = cups.load()
    schedule = club_schedule(matches, upcoming if len(upcoming) else None, cup_fixtures)
    df = df.join(rest_features(df, schedule, set(cup_fixtures["season"]) if len(cup_fixtures) else []))

    team_state = latest_team_state(matches)
    df = df.merge(team_state.add_prefix("team_"), left_on="team_code", right_index=True, how="left")
    df = df.merge(team_state.add_prefix("opp_"), left_on="opp_code", right_index=True, how="left")
    df = df.join(fixture_strength(df, teams.latest(matches, season, min(gameweeks))))
    df["experience"] = df["experience"].fillna(0)
    # Live odds only for the next gameweek; later ones take the fallback (as in training).
    first = df["gw"] == min(gameweeks)
    df = pd.concat([add_market_features(df[first], market, use=True),
                    add_market_features(df[~first], None, use=False)]).sort_index()
    return _assemble(df)
