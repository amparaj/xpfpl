"""Thin client for the public FPL API (no login needed for anything used here)."""

import time
from datetime import datetime

import requests

from xpfpl.config import FPL_API

_session = requests.Session()
_session.headers["User-Agent"] = "xpfpl (personal research project)"


def get(path: str, retries: int = 3) -> dict | list:
    url = f"{FPL_API}/{path.lstrip('/')}"
    for attempt in range(retries):
        try:
            resp = _session.get(url, timeout=30)
            if resp.status_code == 429:  # rate limited - back off
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def bootstrap() -> dict:
    """Players, teams, gameweeks (events), chips and game settings."""
    return get("bootstrap-static/")


def fixtures() -> list[dict]:
    return get("fixtures/")


def element_summary(element_id: int) -> dict:
    """A player's match-by-match history this season plus upcoming fixtures."""
    return get(f"element-summary/{element_id}/")


def event_live(gw: int) -> dict:
    """Every player's points and stats in a gameweek (live while it's in progress)."""
    return get(f"event/{gw}/live/")


def entry(team_id: int) -> dict:
    return get(f"entry/{team_id}/")


def entry_history(team_id: int) -> dict:
    """Per-GW points/transfers/bank for a manager, plus chips played."""
    return get(f"entry/{team_id}/history/")


def entry_picks(team_id: int, gw: int) -> dict:
    return get(f"entry/{team_id}/event/{gw}/picks/")


def entry_transfers(team_id: int) -> list[dict]:
    return get(f"entry/{team_id}/transfers/")


def current_season(bs: dict) -> str:
    """e.g. '2026-27', derived from the GW1 deadline."""
    year = int(bs["events"][0]["deadline_time"][:4])
    return f"{year}-{(year + 1) % 100:02d}"


def next_gameweek(bs: dict) -> int:
    for ev in bs["events"]:
        if ev["is_next"]:
            return ev["id"]
    raise RuntimeError("No upcoming gameweek - the season may be over.")


def last_finished_gameweek(bs: dict) -> int:
    finished = [ev["id"] for ev in bs["events"] if ev["finished"]]
    return max(finished) if finished else 0


def gameweek_status(bs: dict, fx: list[dict]) -> dict:
    """Where the current gameweek is: 'pre-season', 'live' (matches still to play or bonus not yet
    added), 'checking' (finished, FPL still confirming the data) or 'complete' (data_checked: safe
    to fetch). Also the matches played so far and the last kick-off."""
    current = next((ev for ev in bs["events"] if ev["is_current"]), None)
    if current is None:
        return {"gw": 0, "phase": "pre-season", "played": 0, "matches": 0, "last_kickoff": None}
    matches = [f for f in fx if f["event"] == current["id"]]
    kickoffs = [f["kickoff_time"] for f in matches if f["kickoff_time"]]
    phase = ("complete" if current["data_checked"] else "checking") if current["finished"] else "live"
    return {"gw": current["id"], "phase": phase,
            "played": sum(bool(f["finished"] or f["finished_provisional"]) for f in matches),
            "matches": len(matches),
            "last_kickoff": datetime.fromisoformat(max(kickoffs).replace("Z", "+00:00")) if kickoffs else None}
