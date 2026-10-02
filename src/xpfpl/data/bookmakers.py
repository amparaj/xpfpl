"""Bookmaker odds for every Premier League match since 2016-17, from Football-Data.co.uk.

Football-Data publishes one CSV per season (https://football-data.co.uk/mmz4281/<yyyy>/E0.csv,
e.g. 1617 for 2016-17), free to download, with the result and a set of bookmaker odds per
match. Two snapshots of each: "pre-closing" odds, collected shortly before the round
(Friday afternoon for a weekend, Tuesday afternoon for midweek: about when the FPL deadline
is), and closing odds at kick-off (after the deadline, so a benchmark only, never a feature).

Only the pre-closing market averages are used: home / draw / away and over / under 2.5 goals.
The column names changed over the years (BetBrain `BbAv*` to 2018-19, market `Avg*` after),
so `_pick` takes the first one a season has, falling back to Bet365. Odds become
probabilities by inverting them and scaling each set to sum to 1, which takes out the
bookmakers' margin (~5% here, against ~0% on Polymarket).

`market()` returns the same shape as `markets.build`'s market_matches (season, fixture, codes,
home_win/draw/away_win, over_2.5, lam_home/lam_away from `markets.fit_goal_rates`), so it can
stand in for Polymarket where Polymarket has no odds (everything before 2024-25).

The CSVs are archived as archive/bookmakers/<season>.parquet (only the columns read here).
"""

import io
import time

import pandas as pd
import requests

from xpfpl import config
from xpfpl.data import archive

URL = "https://football-data.co.uk/mmz4281/{code}/E0.csv"
FIRST_SEASON = "2016-17"
ARCHIVE = config.ARCHIVE_DIR / "bookmakers"

# Football-Data's club names -> FPL's persistent club `code`.
TEAM_CODES = {
    "Arsenal": 3, "Aston Villa": 7, "Bournemouth": 91, "Brentford": 94, "Brighton": 36,
    "Burnley": 90, "Cardiff": 97, "Chelsea": 8, "Coventry": 9, "Crystal Palace": 31,
    "Everton": 11, "Fulham": 54, "Huddersfield": 38, "Hull": 88, "Ipswich": 40, "Leeds": 2,
    "Leicester": 13, "Liverpool": 14, "Luton": 102, "Man City": 43, "Man United": 1,
    "Middlesbrough": 25, "Newcastle": 4, "Norwich": 45, "Nott'm Forest": 17,
    "Sheffield United": 49, "Southampton": 20, "Stoke": 110, "Sunderland": 56, "Swansea": 80,
    "Tottenham": 6, "Watford": 57, "West Brom": 35, "West Ham": 21, "Wolves": 39,
}
# Preference order per outcome: market average, BetBrain average (older seasons), Bet365.
ODDS = {
    "home_win": ["AvgH", "BbAvH", "B365H"], "draw": ["AvgD", "BbAvD", "B365D"],
    "away_win": ["AvgA", "BbAvA", "B365A"],
    "over_2.5": ["Avg>2.5", "BbAv>2.5", "B365>2.5"], "under_2.5": ["Avg<2.5", "BbAv<2.5", "B365<2.5"],
}
CLOSING = {"home_close": ["AvgCH", "PSCH", "B365CH"], "draw_close": ["AvgCD", "PSCD", "B365CD"],
           "away_close": ["AvgCA", "PSCA", "B365CA"]}
KEEP = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        *{c for cols in (*ODDS.values(), *CLOSING.values()) for c in cols}]


def _code(season: str) -> str:
    return season[2:4] + season[5:7]


def download(season: str) -> pd.DataFrame:
    """One season's CSV, trimmed to the columns used here."""
    r = requests.get(URL.format(code=_code(season)), timeout=60,
                     headers={"User-Agent": "xpfpl (personal research project)"})
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig", errors="replace")))
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    return raw[[c for c in raw.columns if c in KEEP]]


def fetch(seasons: list[str] | None = None, refresh: bool = False) -> int:
    """Archive every season's CSV. Past seasons are read once; the current one every time.
    Returns the number of files written."""
    from xpfpl.data import api
    current = api.current_season(api.bootstrap())
    seasons = seasons or [s for s in [*config.HISTORY_SEASONS, current] if s >= FIRST_SEASON]
    written = 0
    for season in seasons:
        path = ARCHIVE / f"{season}.parquet"
        if path.exists() and season != current and not refresh:
            continue
        written += archive.write(download(season), path)
        time.sleep(1)
    return written


def _pick(raw: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(float("nan"), index=raw.index)
    for c in cols:
        if c in raw:
            out = out.fillna(pd.to_numeric(raw[c], errors="coerce"))
    return out


def _fair(odds: pd.DataFrame) -> pd.DataFrame:
    p = 1 / odds.where(odds > 1)
    return p.div(p.sum(axis=1, min_count=len(odds.columns)), axis=0)


def table(season: str) -> pd.DataFrame | None:
    """One season's matches with fair probabilities: kickoff, home_code, away_code, home_win,
    draw, away_win, over_2.5 (pre-closing) and *_close (closing, a benchmark only)."""
    raw = archive.read(ARCHIVE / f"{season}.parquet")
    if raw is None or raw.empty:
        return None
    date = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")
    if "Time" in raw:
        date = date + pd.to_timedelta(raw["Time"].fillna("15:00") + ":00")
    else:
        date = date + pd.Timedelta(hours=15)
    out = pd.DataFrame({
        "season": season, "kickoff": date.dt.tz_localize("Europe/London").dt.tz_convert("UTC"),
        "home_code": raw["HomeTeam"].map(TEAM_CODES), "away_code": raw["AwayTeam"].map(TEAM_CODES),
        "home_goals": raw["FTHG"], "away_goals": raw["FTAG"]})
    result = _fair(pd.concat({k: _pick(raw, ODDS[k]) for k in ("home_win", "draw", "away_win")}, axis=1))
    goals = _fair(pd.concat({k: _pick(raw, ODDS[k]) for k in ("over_2.5", "under_2.5")}, axis=1))
    close = _fair(pd.concat({k: _pick(raw, v) for k, v in CLOSING.items()}, axis=1))
    out = pd.concat([out, result, goals[["over_2.5"]], close], axis=1)
    missing = out["home_code"].isna() | out["away_code"].isna()
    if missing.any():
        names = sorted(set(raw.loc[missing, "HomeTeam"]) | set(raw.loc[missing, "AwayTeam"]))
        raise KeyError(f"{season}: no club code for {names}")
    return out.astype({"home_code": int, "away_code": int})


def market(matches: pd.DataFrame, seasons: list[str] | None = None) -> pd.DataFrame:
    """Every archived season's odds in the shape of market_matches.parquet, mapped onto FPL
    fixtures (`matches` as from history.load_matches)."""
    from xpfpl.data import markets
    seasons = seasons or sorted(p.stem for p in ARCHIVE.glob("*.parquet"))
    frames = [t for s in seasons if (t := table(s)) is not None]
    if not frames:
        return pd.DataFrame()
    odds = pd.concat(frames, ignore_index=True).assign(volume=0.0)
    found = markets.match_fixtures(odds.drop(columns="season"), matches)
    return markets.fit_goal_rates(found).assign(source="bookmakers")
