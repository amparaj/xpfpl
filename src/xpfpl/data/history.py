"""Build one table of player-match rows across all seasons.

Past seasons come from the vaastav repo (gws/merged_gw.csv + players_raw.csv). The current
season comes from the FPL API (element-summary history), which uses the same column names.

Each row is one player in one fixture (a double gameweek gives two rows). Player `element`
ids and team ids are re-numbered every season, so rows also carry the persistent `code`
(player) and `team_code` (club) used to link seasons together.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from xpfpl import config
from xpfpl.data import api

STAT_COLS = [
    "total_points", "minutes", "goals_scored", "assists", "bonus", "bps",
    "ict_index", "influence", "creativity", "threat", "clean_sheets", "goals_conceded",
    "saves", "expected_goals", "expected_assists", "expected_goals_conceded",
    "defensive_contribution",
]

# Kept for specific jobs rather than rolled into form features:
#   starts            - started the match (from 2022-23): the expected-minutes model's target
#   transfers_balance - net transfers made *before* this GW's deadline (0 at GW1); with
#   selected            ownership, the crowd's read on team news
#   clearances_blocks_interceptions - to rebuild BPS under the 2026-27 rules (from 2025-26)
#   xP                - FPL's own expected points, recorded *after* the match (it correlates
#                       0.69 with the same match's points, 0.46 with the next). Only the
#                       previous row's value is a fair forecast: a benchmark, never a feature.
EXTRA_COLS = ["starts", "transfers_balance", "selected", "clearances_blocks_interceptions", "xP"]

MATCH_COLS = [
    "season", "gw", "kickoff_time", "fixture", "element", "code", "name", "position",
    "team", "team_code", "opponent_team", "opp_code", "was_home", "value",
    "team_h_score", "team_a_score", *STAT_COLS, *EXTRA_COLS,
]


# ---------------------------------------------------------------- vaastav (past seasons)

def _download(url: str, dest) -> None:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def _read_csv(path) -> pd.DataFrame:
    raw = path.read_bytes()
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:  # older seasons are latin-1
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1", low_memory=False)


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def load_vaastav_season(season: str, refresh: bool = False) -> pd.DataFrame:
    folder = config.RAW_DIR / "vaastav" / season
    files = {"merged_gw.csv": "gws/merged_gw.csv", "players_raw.csv": "players_raw.csv"}
    for name, remote in files.items():
        if refresh or not (folder / name).exists():
            _download(f"{config.VAASTAV_RAW}/{season}/{remote}", folder / name)

    gw = _read_csv(folder / "merged_gw.csv")
    players = _read_csv(folder / "players_raw.csv")

    gw["gw"] = gw["round"] if "round" in gw else gw["GW"]
    gw["was_home"] = _as_bool(gw["was_home"])

    players = players.rename(columns={"id": "element", "element_type": "position", "web_name": "name"})
    gw = gw.drop(columns=["name", "position", "team"], errors="ignore")
    gw = gw.merge(players[["element", "code", "position", "name"]], on="element", how="left")

    # merged_gw has no reliable team id before 2020-21, so infer it from each fixture:
    # home players' opponent is the away team and vice versa.
    away_team = gw[gw["was_home"]].groupby("fixture")["opponent_team"].first()
    home_team = gw[~gw["was_home"]].groupby("fixture")["opponent_team"].first()
    gw["team"] = np.where(gw["was_home"], gw["fixture"].map(home_team), gw["fixture"].map(away_team))

    team_code = players.groupby("team")["team_code"].first()
    gw["team_code"] = gw["team"].map(team_code)
    gw["opp_code"] = gw["opponent_team"].map(team_code)
    gw["season"] = season
    return _finalise(gw)


# ---------------------------------------------------------------- FPL API (current season)

def load_current_season(refresh: bool = False, workers: int = 8) -> pd.DataFrame:
    bs = api.bootstrap()
    fx = api.fixtures()
    season = api.current_season(bs)
    last_gw = api.last_finished_gameweek(bs)
    cache = config.RAW_DIR / "api" / season / f"gw{last_gw:02d}"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "bootstrap.json").write_text(json.dumps(bs), encoding="utf-8")
    (cache / "fixtures.json").write_text(json.dumps(fx), encoding="utf-8")

    def summary(element_id: int) -> list[dict]:
        path = cache / f"element_{element_id}.json"
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        history = api.element_summary(element_id)["history"]
        path.write_text(json.dumps(history), encoding="utf-8")
        return history

    ids = [p["id"] for p in bs["elements"]]
    print(f"Fetching {len(ids)} player histories for {season} (up to GW{last_gw})...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        histories = list(pool.map(summary, ids))

    rows = [row for h in histories for row in h]
    if not rows:
        return pd.DataFrame(columns=MATCH_COLS)
    gw = pd.DataFrame(rows)

    finished = {f["id"] for f in fx if f["finished"] or f["finished_provisional"]}
    gw = gw[gw["fixture"].isin(finished)].copy()

    players = pd.DataFrame(bs["elements"]).rename(
        columns={"id": "element", "element_type": "position", "web_name": "name"})
    gw = gw.merge(players[["element", "code", "position", "name"]], on="element", how="left")

    fixture_teams = {f["id"]: (f["team_h"], f["team_a"]) for f in fx}
    gw["team"] = [fixture_teams[f][0] if home else fixture_teams[f][1]
                  for f, home in zip(gw["fixture"], gw["was_home"])]
    team_code = {t["id"]: t["code"] for t in bs["teams"]}
    gw["team_code"] = gw["team"].map(team_code)
    gw["opp_code"] = gw["opponent_team"].map(team_code)
    gw["gw"] = gw["round"]
    gw["season"] = season
    return _finalise(gw)


# ---------------------------------------------------------------- combine

def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    for col in MATCH_COLS:
        if col not in df:
            df[col] = np.nan  # e.g. xG before 2022-23, defensive_contribution before 2025-26
    df = df[MATCH_COLS].copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["was_home"] = df["was_home"].astype(bool)
    for col in ["value", "team_h_score", "team_a_score", *STAT_COLS, *EXTRA_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["code", "team_code", "opp_code"])


def build_matches(refresh: bool = False, seasons: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for season in seasons or config.HISTORY_SEASONS:
        print(f"Loading {season} from vaastav...")
        frames.append(load_vaastav_season(season, refresh=refresh))
    frames.append(load_current_season(refresh=refresh))

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(["kickoff_time", "fixture", "element"]).reset_index(drop=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(config.MATCHES_PATH, index=False)
    print(f"Saved {len(matches):,} player-match rows to {config.MATCHES_PATH}")
    return matches


def load_matches() -> pd.DataFrame:
    if not config.MATCHES_PATH.exists():
        raise FileNotFoundError("No match data yet - run `xpfpl fetch` first.")
    return pd.read_parquet(config.MATCHES_PATH)
