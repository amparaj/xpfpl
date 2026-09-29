"""What happened in each Premier League match: shots with their pitch locations, the match's
momentum, team stats and every player's match stats, for the website's match pages.

The source is FPL-Core-Insights (github.com/olbauday/FPL-Core-Insights, the same repository as
data/cups.py), which republishes FotMob's match data per FPL gameweek with FPL club and player
codes. Nothing here feeds the model: it is for looking back at a match.

  shots.csv             one row per shot: minute, player, outcome (goal / save / miss / block /
                        post / blocked-off-line), situation, body part, xG, xG on target, where it
                        was taken from and where it crossed the goal line
  momentum.csv          FotMob's momentum per minute (positive: the home side on top)
  matches.csv           the team stats: possession, xG (open play / set piece), shots, big chances,
                        passes, duels, distance covered... (kept long: match, stat, home, away)
  playermatchstats.csv  per player: touches, passes, chances created, duels, tackles, distance...
                        (`PLAYER_STATS` of its ~65 columns)

Coordinates are Opta's: the pitch is 100 x 100 whatever its size. `start_x` is measured from the
goal being attacked (the penalty spot is 11.5), `start_y` across the pitch (50 = centre).
`goal_mouth_y` runs across the goal (posts at about 45.2 and 54.8) and `goal_mouth_z` up it
(crossbar at about 38). About one shot in eight has no player recorded.

archive/matchstats/<season>/<table>/gwNN.parquet, the source's gameweek folders, each row carrying
the FPL `fixture` id (and `gw`, the fixture's FPL gameweek, which can differ for a rearranged match).
"""

import pandas as pd

from xpfpl import config
from xpfpl.data import archive, cups

ARCHIVE = config.ARCHIVE_DIR / "matchstats"
FIRST_SEASON = "2025-26"
TABLES = {"matches": "matches", "shots": "shots", "momentum": "momentum", "players": "playermatchstats"}
KEYS = {"matches": ["match_id", "stat"], "shots": ["match_id", "shot_index"], "momentum": ["match_id", "minute"],
        "players": ["match_id", "player_id"]}
# Source columns that are bookkeeping, not match data.
DROP = ["match_url", "stats_processed", "player_stats_processed", "bzzoiro_processed", "player_name"]
# The player stats kept (the source has ~65; these are the ones a match page shows).
PLAYER_STATS = ["minutes_played", "start_min", "finish_min", "goals", "assists", "xg", "xa", "xgot", "total_shots",
                "shots_on_target", "big_chances_missed", "chances_created", "touches", "touches_opposition_box",
                "accurate_passes", "accurate_passes_percent", "final_third_passes", "accurate_crosses",
                "successful_dribbles", "tackles", "tackles_won", "interceptions", "recoveries", "blocks", "clearances",
                "defensive_contributions", "duels_won", "duels_lost", "aerial_duels_won", "was_fouled",
                "fouls_committed", "dispossessed", "saves", "goals_conceded", "xgot_faced", "goals_prevented",
                "distance_covered", "top_speed", "number_of_sprints"]


def long_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """matches.csv's home_<stat> / away_<stat> columns as one row per match per stat: (match_id,
    stat, home, away). A wide file of 114 columns for ten matches is mostly Parquet metadata."""
    stats = sorted({c[5:] for c in raw if c.startswith("home_") and f"away_{c[5:]}" in raw} - {"team", "score"})
    parts = [pd.DataFrame({"match_id": raw["match_id"], "stat": s, "home": pd.to_numeric(raw[f"home_{s}"], errors="coerce"),
                           "away": pd.to_numeric(raw[f"away_{s}"], errors="coerce")}) for s in stats]
    return pd.concat(parts, ignore_index=True).dropna(subset=["home", "away"], how="all")


def fpl_fixtures(season: str) -> pd.DataFrame:
    """The season's FPL fixtures: (fixture, gw, home_code, away_code)."""
    fx = archive.fixtures(season)
    teams = pd.read_parquet(archive.FPL / season / "teams.parquet").set_index("id")["code"]
    fx = fx.drop_duplicates("id", keep="last")
    return pd.DataFrame({"fixture": fx["id"].astype(int), "gw": fx["event"].astype("Int64"),
                         "home_code": fx["team_h"].map(teams).astype(int),
                         "away_code": fx["team_a"].map(teams).astype(int)})


def link(raw: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """The source's played Premier League matches with their FPL fixture: match_id -> fixture, gw.
    A club pair meets once at home per season, so the pair identifies the fixture."""
    league = raw["tournament"].eq(cups.EPL) if "tournament" in raw else True
    played = raw.loc[league & raw["finished"].astype(bool), ["match_id", "home_team", "away_team"]]
    played = pd.DataFrame({"match_id": played["match_id"], "home_code": played["home_team"].astype(int),
                           "away_code": played["away_team"].astype(int)})
    return played.merge(fixtures, on=["home_code", "away_code"])


def fetch(season: str, gameweeks=range(1, 39), refresh: bool = False) -> int:
    """Download a season's match data into archive/matchstats. Returns the files written.

    A gameweek folder is cached in data/raw once all its matches are finished; `refresh`
    re-downloads it (FotMob's numbers are occasionally corrected after the match)."""
    if season < FIRST_SEASON or not archive.has_season(season):
        return 0
    fixtures, written = fpl_fixtures(season), 0
    for gw in gameweeks:
        raw = cups._csv(season, gw, "matches", refresh)
        if raw is None or raw.empty:
            continue
        if not raw["finished"].astype(bool).all() and not refresh:
            raw = cups._csv(season, gw, "matches", refresh=True)
        found = link(raw, fixtures)
        if found.empty:
            continue
        keys = found.set_index("match_id")[["fixture", "gw"]]
        people = cups._csv(season, gw, "players", refresh)
        codes = people.set_index("player_id")[["player_code", "team_code"]] if people is not None else None
        for table, name in TABLES.items():
            df = raw if table == "matches" else cups._csv(season, gw, name, refresh or not raw["finished"].astype(bool).all())
            if df is None or df.empty:
                continue
            df = df[df["match_id"].isin(keys.index)].drop(columns=DROP, errors="ignore")
            if table == "matches":
                df = long_stats(df)
            elif table == "players":
                df = df[["match_id", "player_id", *[c for c in PLAYER_STATS if c in df]]]
            df = df.join(keys, on="match_id")
            if table in ("shots", "players") and codes is not None:
                df = df.join(codes.rename(columns={"player_code": "code"}), on="player_id")
            written += archive.write(df.reset_index(drop=True), ARCHIVE / season / table / f"gw{gw:02d}.parquet")
    return written


def load(season: str, table: str) -> pd.DataFrame | None:
    """One table for the whole season, stacked (None if nothing is archived)."""
    parts = sorted((ARCHIVE / season / table).glob("gw*.parquet"))
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df.drop_duplicates(KEYS[table], keep="last")         # a rearranged match can sit in two folders


def gameweek(season: str, gw: int) -> dict[str, pd.DataFrame]:
    """Every table's rows for the fixtures of FPL gameweek `gw` (tables with none are left out)."""
    out = {}
    for table in TABLES:
        df = load(season, table)
        if df is not None and "gw" in df and (df["gw"] == gw).any():
            out[table] = df[df["gw"] == gw].reset_index(drop=True)
    return out
