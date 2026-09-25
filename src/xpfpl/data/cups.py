"""Midweek cup and European matches, for rotation risk.

A club with a Champions League match on Tuesday may rest its stars on Saturday, or take them off
at 60 minutes. vaastav and the FPL API only know Premier League matches, so this reads the other
competitions from FPL-Core-Insights (github.com/olbauday/FPL-Core-Insights), which files every
match an EPL club plays (Champions/Europa/Conference League, EFL Cup, FA Cup...) under the FPL
gameweek it falls before, with per-player minutes. Its club ids are FPL `team_code`s and its
players carry the FPL `player_code`, so both link straight to our `team_code` / `code`.

Coverage: 2025-26 onwards (2024-25 there holds Premier League matches only), fixtures listed
weeks ahead. Older seasons get `cup_era = 0` and zeros, the same trick as `dc_era`.

archive/cups/<season>/fixtures/gwNN.parquet   one row per EPL club per non-EPL match
archive/cups/<season>/minutes/gwNN.parquet    one row per player per played non-EPL match

`rest_features` turns these into features for each Premier League player-match: how recently
the club played another competition (and how big), whether another one follows soon after, and
how many minutes the player himself played in it.
"""

import io

import numpy as np
import pandas as pd
import requests

from xpfpl import config
from xpfpl.data import archive

SOURCE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"
FIRST_SEASON = "2025-26"
RAW_DIR = config.RAW_DIR / "coreinsights"
ARCHIVE = config.ARCHIVE_DIR / "cups"
EPL = "prem"
# How much a competition matters to a club's selection, roughly: a Champions League night is
# what gets a Saturday line-up rotated. Unknown competitions count as 1.
LEVELS = {"champions-league": 3, "europa-league": 2, "conference-league": 1, "efl-cup": 1,
          "fa-cup": 1, "uefa-super-cup": 2, "community-shield": 0, "friendlies": 0}
WINDOW_DAYS = 6.0     # a cup match counts as "before" / "after" an EPL match within this many days
FEATURES = ["cup_era", "cup_before", "cup_before_days", "cup_before_level",
            "cup_after", "cup_after_days", "cup_after_level", "cup_mins_before", "cup_started_before"]

_session = requests.Session()
_session.headers["User-Agent"] = "xpfpl (personal research project)"


def _source_season(season: str) -> str:
    """'2025-26' -> '2025-2026', the source's folder name."""
    start = int(season[:4])
    return f"{start}-{start + 1}"


def _csv(season: str, gw: int, name: str, refresh: bool) -> pd.DataFrame | None:
    """One of the source's per-gameweek CSVs, cached in data/raw (None if it isn't there)."""
    path = RAW_DIR / season / f"gw{gw:02d}" / f"{name}.csv"
    if refresh or not path.exists():
        url = f"{SOURCE}/{_source_season(season)}/By%20Gameweek/GW{gw}/{name}.csv"
        resp = _session.get(url, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(path.read_bytes()), encoding="utf-8", encoding_errors="replace")


def club_fixtures(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """The source's fixtures.csv -> one row per EPL club per non-EPL match (the other side of a
    European tie has no FPL code, so it is dropped)."""
    cups = raw[raw["tournament"] != EPL]
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        s = cups[cups[f"{side}_team"].notna()]
        sides.append(pd.DataFrame({
            "season": season,
            "gw": s["gameweek"].astype(int),
            "match_id": s["match_id"],
            "tournament": s["tournament"],
            "kickoff_time": pd.to_datetime(s["kickoff_time"], utc=True),
            "team_code": s[f"{side}_team"].astype(int),
            "opp_code": s[f"{other}_team"].astype("Int64"),
            "was_home": side == "home",
            "goals_for": s[f"{side}_score"],
            "goals_against": s[f"{other}_score"],
            "finished": s["finished"].astype(bool),
        }))
    return pd.concat(sides, ignore_index=True).sort_values(["kickoff_time", "team_code"], ignore_index=True)


def player_minutes(stats: pd.DataFrame, players: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """playermatchstats.csv -> minutes per player (by FPL `code`) in this GW's non-EPL matches."""
    cup_ids = set(fixtures["match_id"])
    s = stats[stats["match_id"].isin(cup_ids)]
    s = s.merge(players[["player_id", "player_code", "team_code"]], on="player_id", how="inner")
    return pd.DataFrame({
        "match_id": s["match_id"],
        "code": s["player_code"].astype(int),
        "team_code": s["team_code"].astype(int),
        "minutes": s["minutes_played"].fillna(0).astype(int),
        "started": s["start_min"].fillna(-1).eq(0),
    }).reset_index(drop=True)


def fetch(season: str, gameweeks=range(1, 39), refresh: bool = False) -> int:
    """Download a season's non-EPL fixtures and minutes into archive/cups. Returns files written.

    Played gameweeks are cached in data/raw; `refresh` re-downloads them (the current and future
    ones are always re-read, since kick-off times move and scores arrive)."""
    written = 0
    now = pd.Timestamp.now(tz="UTC")
    for gw in gameweeks:
        raw = _csv(season, gw, "fixtures", refresh)
        if raw is None or raw.empty:
            continue
        if not raw["finished"].astype(bool).all() and not refresh:
            raw = _csv(season, gw, "fixtures", refresh=True)
        fx = club_fixtures(raw, season)
        written += archive.write(fx, ARCHIVE / season / "fixtures" / f"gw{gw:02d}.parquet")
        played = fx[fx["finished"] & (fx["kickoff_time"] < now)]
        if played.empty:
            continue
        stats = _csv(season, gw, "playermatchstats", refresh)
        players = _csv(season, gw, "players", refresh)
        if stats is None or players is None:
            continue
        mins = player_minutes(stats, players, played)
        if len(mins):
            written += archive.write(mins, ARCHIVE / season / "minutes" / f"gw{gw:02d}.parquet")
    return written


def _stack(kind: str) -> pd.DataFrame | None:
    parts = sorted(ARCHIVE.glob(f"*/{kind}/gw*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else None


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(club fixtures, player minutes) for every archived season; empty frames if none."""
    fx = _stack("fixtures")
    mins = _stack("minutes")
    if fx is None:
        fx = pd.DataFrame(columns=["season", "gw", "match_id", "tournament", "kickoff_time", "team_code"])
    if mins is None:
        mins = pd.DataFrame(columns=["match_id", "code", "team_code", "minutes", "started"])
    fx = fx.drop_duplicates(["match_id", "team_code"], keep="last")   # a tie refiled under a later GW
    return fx, mins


def _nearest(epl: pd.DataFrame, cups: pd.DataFrame, direction: str) -> pd.DataFrame:
    """For each EPL row, the club's nearest non-EPL match `backward` (before) or `forward`."""
    left = epl[["team_code", "kickoff_time"]].reset_index().sort_values("kickoff_time")
    right = (cups[["team_code", "kickoff_time", "match_id", "tournament"]].dropna(subset=["kickoff_time"])
             .rename(columns={"kickoff_time": "cup_time"}).sort_values("cup_time"))   # TBD ties have no time
    right["team_code"] = right["team_code"].astype(int)
    left["team_code"] = left["team_code"].astype(int)
    left["kickoff_time"] = left["kickoff_time"].astype(right["cup_time"].dtype)
    got = pd.merge_asof(left, right, left_on="kickoff_time", right_on="cup_time", by="team_code",
                        direction=direction, allow_exact_matches=False,
                        tolerance=pd.Timedelta(days=WINDOW_DAYS))
    return got.set_index("index").reindex(epl.index)


def rest_features(epl: pd.DataFrame, fixtures: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Rotation features for each EPL player-match row of `epl` (needs `season`, `team_code`,
    `code`, `kickoff_time`), indexed like it:

      cup_era             1 where the season has cup data at all (else every feature is 0)
      cup_before          1 if the club played another competition in the 6 days before
      cup_before_days     days since it (6 if none), and `cup_before_level` how big (LEVELS)
      cup_after / _days / _level   the same for the next non-EPL match within 6 days after
      cup_mins_before     the player's minutes in that match / 90 (0 if unknown or not played)
      cup_started_before  1 if he started it

    `cup_mins_before` is only known once that match is played: fine for training and for the
    next gameweek (the midweek match is before the deadline), not for weeks further ahead.
    """
    out = pd.DataFrame(index=epl.index)
    out["cup_era"] = epl["season"].isin(fixtures["season"].unique()).astype(float)
    for name, direction in (("before", "backward"), ("after", "forward")):
        near = _nearest(epl, fixtures, direction)
        days = (near["kickoff_time"] - near["cup_time"]).abs().dt.total_seconds() / 86400
        out[f"cup_{name}"] = near["match_id"].notna().astype(float)
        out[f"cup_{name}_days"] = days.fillna(WINDOW_DAYS).clip(upper=WINDOW_DAYS)
        out[f"cup_{name}_level"] = near["tournament"].map(LEVELS).fillna(
            near["match_id"].notna().astype(float))
        if name == "before":
            key = pd.DataFrame({"match_id": near["match_id"], "code": epl["code"]}, index=epl.index)
            played = key.reset_index().merge(minutes[["match_id", "code", "minutes", "started"]],
                                             on=["match_id", "code"], how="left").set_index("index")
            out["cup_mins_before"] = (played["minutes"].fillna(0) / 90.0).clip(upper=4 / 3)
            out["cup_started_before"] = played["started"].fillna(False).astype(float)
    return out[FEATURES].astype("float32")
