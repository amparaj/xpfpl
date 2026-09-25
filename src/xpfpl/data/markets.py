"""Betting-market odds for Premier League matches, from Polymarket.

Polymarket lists, per match, a family of events that share a `gameId`:
  - the result: three yes/no markets (home win / draw / away win)                  from 2024-25
  - "More Markets": total-goals and team-total ladders, both teams to score ...     from 2025-26
  - "Player Props": anytime-goalscorer markets for ~16 attackers                   from Jan 2026
plus season-long markets (title, top four, relegation, top scorer, most assists, most clean
sheets). Prices are probabilities (a share pays $1 if "Yes"), and the three result prices
sum to ~1.00: prediction markets carry almost no bookmaker margin.

Two APIs: Gamma (https://gamma-api.polymarket.com) describes events and markets and gives the
current prices; CLOB (https://clob.polymarket.com) gives each market's price history, which is
how finished matches get their *pre-deadline* odds back. Everything is read at the FPL deadline
of the match's gameweek, never at kick-off: prices move on team news after the deadline, and a
manager can't act on that.

Outputs (data/processed/):
  market_matches.parquet  one row per match: result, totals, team totals, BTTS probabilities,
                          volume, and the expected goals fitted to them (`fit_goal_rates`)
  market_scorers.parquet  one row per player per match: anytime-goalscorer probability
  market_outrights.parquet  the season-long markets, one row per outcome per snapshot
Raw responses are cached in data/raw/polymarket/; a finished market's history never changes.
Played matches' events and price histories are also archived (archive/polymarket/, see
data/archive.py) and read back from there when the cache doesn't have them, so the history
survives Polymarket delisting a market.
"""

import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import torch

from xpfpl import config
from xpfpl.data import archive

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
EPL_TAG = 306
FIRST_SEASON_START = "2024-08-01T00:00:00Z"
RAW_DIR = config.RAW_DIR / "polymarket"
MATCHES_PATH = config.PROCESSED_DIR / "market_matches.parquet"
SCORERS_PATH = config.PROCESSED_DIR / "market_scorers.parquet"
OUTRIGHTS_PATH = config.PROCESSED_DIR / "market_outrights.parquet"

# Polymarket club name -> FPL's persistent club `code` (checked against vaastav's teams.csv).
TEAM_CODES = {
    "Arsenal FC": 3, "Aston Villa FC": 7, "AFC Bournemouth": 91, "Brentford FC": 94,
    "Brighton & Hove Albion FC": 36, "Burnley FC": 90, "Chelsea FC": 8, "Coventry City FC": 9,
    "Crystal Palace FC": 31, "Everton FC": 11, "Fulham FC": 54, "Hull City AFC": 88,
    "Ipswich Town FC": 40, "Leeds United FC": 2, "Leicester City FC": 13, "Liverpool FC": 14,
    "Manchester City FC": 43, "Manchester United FC": 1, "Newcastle United FC": 4,
    "Nottingham Forest FC": 17, "Southampton FC": 20, "Sunderland AFC": 56,
    "Tottenham Hotspur FC": 6, "West Ham United FC": 21, "Wolverhampton Wanderers FC": 39,
}
# The goal markets kept per match. Each is a yes/no question; the price is P(yes / over).
TOTAL_LINES = (1.5, 2.5, 3.5)
TEAM_LINES = (0.5, 1.5)
MAX_GOALS = 10                 # goals grid for the Poisson fit
DEADLINE_BEFORE_KICKOFF = pd.Timedelta(minutes=90)   # FPL deadlines are 90 min before the first match

_session = requests.Session()
_session.headers["User-Agent"] = "xpfpl (personal research project)"


def _get(url: str, params: dict | None = None, retries: int = 4):
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def _events(**params) -> list[dict]:
    """Every Gamma event matching `params`, following the 100-per-page pagination."""
    out, offset = [], 0
    while True:
        page = _get(f"{GAMMA}/events", {**params, "limit": 100, "offset": offset})
        out += page
        offset += 100
        if len(page) < 100:
            return out


def _is_epl(event: dict) -> bool:
    teams = event.get("teams") or []
    return len(teams) == 2 and all(t.get("name") in TEAM_CODES for t in teams)


def _yes_price(market: dict) -> float:
    """Current probability of the first outcome ("Yes"/"Over"): the order-book midpoint when
    there is a book, otherwise the last listed price."""
    bid, ask = market.get("bestBid"), market.get("bestAsk")
    if bid is not None and ask is not None and 0 < float(ask) - float(bid) < 0.2:
        return (float(bid) + float(ask)) / 2
    prices = json.loads(market.get("outcomePrices") or "[]")
    return float(prices[0]) if prices else float("nan")


HISTORY_DAYS = 14


def _history(token: str, end: pd.Timestamp, cache: bool) -> list[dict]:
    """Hourly prices for the HISTORY_DAYS up to `end`. Cached only when `cache` (a finished
    market's history never changes; a live one does)."""
    path = RAW_DIR / f"history_{HISTORY_DAYS}d" / f"{token}.json"
    if cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if cache and (kept := archive.price_history(path.parent.name, token)) is not None:
        return kept
    stop = int(end.timestamp())
    history = _get(f"{CLOB}/prices-history",
                   {"market": token, "startTs": stop - HISTORY_DAYS * 86400, "endTs": stop,
                    "fidelity": 60}).get("history", [])
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history), encoding="utf-8")
    return history


FINE_HOURS, FINE_MINUTES = 3, 5


def _fine_history(token: str, end: pd.Timestamp, cache: bool) -> list[dict]:
    """Prices every FINE_MINUTES over the FINE_HOURS up to `end`: the hourly history is too
    coarse for the 90-minute move. Cached like `_history`."""
    path = RAW_DIR / f"history_{FINE_HOURS}h_{FINE_MINUTES}m" / f"{token}.json"
    if cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if cache and (kept := archive.price_history(path.parent.name, token)) is not None:
        return kept
    stop = int(end.timestamp())
    history = _get(f"{CLOB}/prices-history",
                   {"market": token, "startTs": stop - FINE_HOURS * 3600, "endTs": stop,
                    "fidelity": FINE_MINUTES}).get("history", [])
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history), encoding="utf-8")
    return history


def _at(history: list[dict], when: pd.Timestamp) -> float:
    """The last price at or before `when` (NaN if the market hadn't traded yet)."""
    before = [p["p"] for p in history if p["t"] <= when.timestamp()]
    return float(before[-1]) if before else float("nan")


def _price_at(token: str, when: pd.Timestamp, cache: bool) -> float:
    """The market's price at `when`, from its hourly price history (the last point at or before)."""
    return _at(_history(token, when, cache), when)


# Also kept: each result and scorer price 90 minutes and 1, 3, 7 and 14 days before the pricing
# moment, which is what the "news in the odds" view compares against. A market that opened later
# than that takes its opening price.
MOVE_DAYS = (1.5 / 24, 1, 3, 7, 14)
MOVERS = ("home_win", "draw", "away_win", "p_anytime")


def move_column(name: str, days: float) -> str:
    """`<name>_90m` for the 90-minute move, `<name>_<n>d` for the others."""
    return f"{name}_{round(days * 24 * 60)}m" if days < 1 else f"{name}_{days}d"


# ---------------------------------------------------------------- which markets we keep

WIN = re.compile(r"^Will (.+?) win on ", re.I)
TEAM_TOTAL = re.compile(r": (.+?) O/U (\d+\.5)$")
TOTAL = re.compile(r"(?<!1st Half )(?<!2nd Half )O/U (\d+\.5)$")


def _which(text: str, event: dict) -> str | None:
    """'home' or 'away': the side of `event` whose club `text` names. Older markets use short
    names ("Tottenham", "Man City"), newer ones the full name, so pick the closest variant."""
    from difflib import SequenceMatcher

    target = _strip(text)
    best, side = 0.0, None
    for team in event["teams"]:
        variants = {team["name"], team.get("alias") or "", team["name"].replace(" FC", "").replace("AFC ", "")}
        score = max(SequenceMatcher(None, target, _strip(v)).ratio() for v in variants if v)
        if score > best:
            best, side = score, team.get("ordering")
    return side if best >= 0.6 else None


def _match_markets(group: list[dict]) -> dict[str, dict]:
    """Name -> market for the goal markets we use, from all of one match's events.

    Read from the question text rather than `sportsMarketType`, which older markets don't have:
    "Will X win on <date>?", "... end in a draw?", "... O/U 2.5", "...: X O/U 0.5", "Both Teams to
    Score". Half-time and spread markets are ignored.
    """
    keep = {}
    for event in group:
        for m in event.get("markets") or []:
            q = m.get("question", "").strip()
            kind = m.get("sportsMarketType") or ""
            if "half" in q.lower() or "half" in kind or "spread" in kind.lower():
                continue
            name = None
            if q.lower().endswith("end in a draw?"):
                name = "draw"
            elif WIN.match(q):
                side = _which(WIN.match(q).group(1), event)
                name = f"{side}_win" if side else None
            elif q.endswith("Both Teams to Score"):
                name = "btts"
            elif TEAM_TOTAL.search(q) and " vs. " not in TEAM_TOTAL.search(q).group(1):
                team, line = TEAM_TOTAL.search(q).groups()
                side = _which(team, event)
                name = f"{side}_over_{line}" if side and float(line) in TEAM_LINES else None
            elif TOTAL.search(q):
                line = float(TOTAL.search(q).group(1))
                name = f"over_{line}" if line in TOTAL_LINES else None
            if name:
                keep[name] = m
    return keep


def _scorer_markets(group: list[dict]) -> list[dict]:
    return [m for e in group for m in e.get("markets") or []
            if m.get("sportsMarketType") == "soccer_anytime_goalscorer"]


def _player_name(question: str) -> str:
    return question.split(":")[0].strip()


# ---------------------------------------------------------------- fetching

MAIN_SLUG = re.compile(r"epl-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}")


def fetch_games(since: str = FIRST_SEASON_START) -> dict[str, list[dict]]:
    """Every EPL match since `since`: its main event's slug -> all of that match's events."""
    games: dict[str, list[dict]] = {}
    for e in _events(tag_id=EPL_TAG, end_date_min=since):
        base = MAIN_SLUG.match(e.get("slug", ""))
        if base and _is_epl(e):
            games.setdefault(base.group(0), []).append(e)
    return {slug: group for slug, group in games.items() if any(e["slug"] == slug for e in group)}


def _sides(event: dict) -> tuple[str, str]:
    home, away = [t["name"] for t in sorted(event["teams"], key=lambda t: t.get("ordering") != "home")]
    return home, away


def listing(games: dict[str, list[dict]]) -> pd.DataFrame:
    """One row per match (slug, kick-off, clubs), without prices."""
    rows = []
    for slug, group in games.items():
        main = next(e for e in group if e["slug"] == slug)
        if main.get("startTime"):
            home, away = _sides(main)
            rows.append({"slug": slug, "kickoff": pd.Timestamp(main["startTime"]), "home": home,
                         "away": away, "home_code": TEAM_CODES[home], "away_code": TEAM_CODES[away],
                         "volume": sum(float(e.get("volume") or 0) for e in group)})
    return pd.DataFrame(rows)


def price_games(games: dict[str, list[dict]], deadlines: dict, cache: bool = True,
                workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every kept market of every match, priced at `deadlines[slug]` (default: 90 minutes before
    kick-off). A match whose deadline hasn't passed gets today's prices. Returns (matches, scorers)."""
    now = pd.Timestamp.now(tz="UTC")
    jobs, rows, scorer_rows = [], [], []
    for row in listing(games).to_dict("records"):
        group = games[row["slug"]]
        deadline = deadlines.get(row["slug"], row["kickoff"] - DEADLINE_BEFORE_KICKOFF)
        live = deadline > now
        row["priced_at"] = now if live else deadline
        rows.append(row)
        for name, m in _match_markets(group).items():
            jobs.append((row, name, m, live, deadline))
        for m in _scorer_markets(group):
            srow = {"slug": row["slug"], "kickoff": row["kickoff"], "player": _player_name(m["question"]),
                    "home_code": row["home_code"], "away_code": row["away_code"],
                    "volume": float(m.get("volume") or 0)}
            scorer_rows.append(srow)
            jobs.append((srow, "p_anytime", m, live, deadline))

    def price(job):
        target, name, m, live, deadline = job
        token = json.loads(m.get("clobTokenIds") or "[]")
        end = now if live else deadline
        values = {name: _yes_price(m) if live else float("nan")}
        if not token:
            return target, values
        if name in MOVERS or not live:
            history = _history(token[0], end, cache and not live)
            if not live:
                values[name] = _at(history, end)
            if name in MOVERS:
                fine = _fine_history(token[0], end, cache and not live) if history else []
                for days in MOVE_DAYS:
                    source = fine if days < 1 and fine else history
                    before = _at(source, end - pd.Timedelta(days=days)) if source else float("nan")
                    if source and np.isnan(before):
                        before = float(source[0]["p"])          # the market opened within the window
                    values[move_column(name, days)] = before
        return target, values

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for target, values in pool.map(price, jobs):
            target.update(values)
    return pd.DataFrame(rows), pd.DataFrame(scorer_rows)


def fetch_outrights() -> pd.DataFrame:
    """Today's prices for the open season-long EPL markets: one row per outcome."""
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for e in _events(tag_id=EPL_TAG, closed="false"):
        if " vs. " in e.get("title", ""):
            continue                                   # match markets are handled above
        for m in e.get("markets") or []:
            if m.get("closed"):
                continue
            token = json.loads(m.get("clobTokenIds") or "[]")
            rows.append({"snapshot": now, "event": e["title"], "outcome": m.get("groupItemTitle") or m["question"],
                         "probability": _yes_price(m), "volume": float(m.get("volume") or 0),
                         "end": e.get("endDate"), "token": token[0] if token else None,
                         "created": e.get("createdAt") or e.get("startDate")})
    return pd.DataFrame(rows)


def outright_prices_at(tokens: list[str], when: pd.Timestamp) -> dict[str, float]:
    """Each season-market outcome's price at `when` (e.g. the season's first deadline), from its
    price history. NaN where the market didn't exist yet."""
    def one(token):
        return token, _price_at(token, when, cache=False) if token else float("nan")
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, tokens))


# ---------------------------------------------------------------- odds -> expected goals

def _poisson_pmf(lam: torch.Tensor) -> torch.Tensor:
    k = torch.arange(MAX_GOALS + 1, dtype=lam.dtype)
    return torch.exp(k * torch.log(lam[:, None]) - lam[:, None] - torch.lgamma(k + 1))


def fit_goal_rates(matches: pd.DataFrame, steps: int = 300) -> pd.DataFrame:
    """Each side's expected goals, fitted to all of a match's goal markets at once.

    Independent Poisson goals for each side, with rates chosen (batched over every match, by
    gradient descent in PyTorch) so the implied probabilities of home win / draw / away win,
    over 1.5 / 2.5 / 3.5 goals, each side over 0.5 / 1.5 and both teams scoring match the
    market's as closely as possible (squared error; missing markets are skipped). With only the
    result market (2024-25) the two rates still come out, from the win/draw/loss split.
    """
    if matches.empty:
        return matches.assign(lam_home=[], lam_away=[])
    cols = ["home_win", "draw", "away_win", *[f"over_{x}" for x in TOTAL_LINES],
            *[f"{s}_over_{x}" for s in ("home", "away") for x in TEAM_LINES], "btts"]
    target = torch.tensor(matches.reindex(columns=cols).to_numpy(dtype="float64"))
    mask = ~torch.isnan(target)
    target = torch.nan_to_num(target)
    log_rates = torch.full((len(matches), 2), float(np.log(1.35)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_rates], lr=0.05)
    goals = torch.arange(MAX_GOALS + 1)
    total = goals[:, None] + goals[None, :]
    for _ in range(steps):
        opt.zero_grad()
        ph, pa = _poisson_pmf(torch.exp(log_rates[:, 0])), _poisson_pmf(torch.exp(log_rates[:, 1]))
        joint = ph[:, :, None] * pa[:, None, :]                       # [match, home goals, away goals]
        implied = [
            torch.tril(joint, -1).sum((1, 2)), torch.diagonal(joint, dim1=1, dim2=2).sum(1),
            torch.triu(joint, 1).sum((1, 2)),
            *[(joint * (total > x)).sum((1, 2)) for x in TOTAL_LINES],
            *[1 - ph[:, : int(x) + 1].sum(1) for x in TEAM_LINES],
            *[1 - pa[:, : int(x) + 1].sum(1) for x in TEAM_LINES],
            (1 - ph[:, 0]) * (1 - pa[:, 0]),
        ]
        loss = (((torch.stack(implied, 1) - target) ** 2) * mask).sum()
        loss.backward()
        opt.step()
    rates = torch.exp(log_rates).detach().numpy()
    return matches.assign(lam_home=rates[:, 0], lam_away=rates[:, 1])


# ---------------------------------------------------------------- mapping onto FPL fixtures

def _strip(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def fixture_table(matches: pd.DataFrame) -> pd.DataFrame:
    """(season, fixture, gw, kickoff_time, home_code, away_code) for every played match."""
    return (matches[matches["was_home"]].groupby(["season", "fixture"])
            .agg(gw=("gw", "first"), kickoff_time=("kickoff_time", "first"),
                 home_code=("team_code", "first"), away_code=("opp_code", "first")).reset_index())


def api_fixture_table(bs: dict, fixtures: list[dict]) -> pd.DataFrame:
    """The same for this season's fixture list from the FPL API (played or not)."""
    from xpfpl.data import api
    codes = {t["id"]: t["code"] for t in bs["teams"]}
    rows = [{"season": api.current_season(bs), "fixture": f["id"], "gw": f["event"],
             "kickoff_time": pd.Timestamp(f["kickoff_time"]), "home_code": codes[f["team_h"]],
             "away_code": codes[f["team_a"]]} for f in fixtures if f.get("event") and f.get("kickoff_time")]
    return pd.DataFrame(rows)


def match_fixtures(markets: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attach each market match to its FPL (season, gw, fixture), by home + away club and date.

    `fixtures` is `fixture_table(matches)` or `api_fixture_table(...)`; a matches frame (with
    `was_home`) is accepted too."""
    fx = fixture_table(fixtures) if "was_home" in fixtures else fixtures
    out = markets.merge(fx, on=["home_code", "away_code"], how="inner")
    # The same pairing happens once per season at each ground: keep the one within a few days.
    out = out[(out["kickoff_time"] - out["kickoff"]).abs() < pd.Timedelta(days=5)]
    # A postponed match can be listed twice (the original date and the new one): keep the busier.
    out = out.sort_values("volume", ascending=False).drop_duplicates(["season", "fixture"])
    return out.drop(columns="kickoff_time").sort_values("kickoff").reset_index(drop=True)


def match_players(scorers: pd.DataFrame, fixtures: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach each scorer market to an FPL player `code` at one of the two clubs, by name.

    `players` has one row per (season, code) with `team_code` and names (`name`, and optionally
    `first_name`/`second_name`). A market name matches if it equals a player's full name, web
    name or "first + last word"; players at other clubs are never considered.
    """
    s = scorers.merge(fixtures[["slug", "season", "gw", "fixture"]], on="slug", how="inner")
    keys = []
    for p in players.itertuples():
        names = {_strip(p.name)}
        for extra in ("first_name", "second_name"):
            if not hasattr(p, extra):
                break
        else:
            full = _strip(f"{p.first_name} {p.second_name}")
            names |= {full, f"{full.split()[0]} {full.split()[-1]}" if full else "", _strip(p.second_name)}
        for n in names - {""}:
            keys.append({"season": p.season, "team_code": p.team_code, "key": n, "code": p.code})
    lookup = pd.DataFrame(keys).drop_duplicates(["season", "team_code", "key"], keep=False)
    s["key"] = s["player"].map(_strip)
    rows = []
    for side in ("home_code", "away_code"):
        hit = s.merge(lookup, left_on=["season", side, "key"], right_on=["season", "team_code", "key"])
        rows.append(hit)
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["slug", "player"])
    return out.drop(columns=["key"])


# ---------------------------------------------------------------- the whole job

def gameweek_deadlines(matches: pd.DataFrame, bs: dict | None = None) -> pd.Series:
    """Deadline per (season, gw): the real one from bootstrap-static for this season, else 90
    minutes before the gameweek's first kick-off."""
    first = matches.groupby(["season", "gw"])["kickoff_time"].min() - DEADLINE_BEFORE_KICKOFF
    if bs:
        from xpfpl.data import api
        season = api.current_season(bs)
        for ev in bs["events"]:
            first[(season, ev["id"])] = pd.Timestamp(ev["deadline_time"])
    return first


def player_names(bs: dict | None = None) -> pd.DataFrame:
    """(season, code, team_code, name, first_name, second_name) for every season scorer markets
    exist: the archived players for past seasons, bootstrap-static for this one."""
    frames = []
    for season in config.HISTORY_SEASONS:
        if int(season[:4]) < 2025 or not archive.has_season(season):
            continue
        _, raw = archive.season_tables(season)
        frames.append(raw.assign(season=season, name=raw["web_name"]))
    if bs:
        from xpfpl.data import api
        el = pd.DataFrame(bs["elements"])
        codes = {t["id"]: t["code"] for t in bs["teams"]}
        frames.append(el.assign(season=api.current_season(bs), name=el["web_name"],
                                team_code=el["team"].map(codes)))
    cols = ["season", "code", "team_code", "name", "first_name", "second_name"]
    return pd.concat([f[cols] for f in frames], ignore_index=True) if frames else pd.DataFrame(columns=cols)


def build(matches: pd.DataFrame, bs: dict | None = None, cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch everything, price it at FPL deadlines, fit expected goals, map to fixtures and players,
    and save market_matches.parquet and market_scorers.parquet. Matches Polymarket no longer lists
    come from the archive, and every played match is archived."""
    try:
        games = {**archive.polymarket_games(), **fetch_games()}
    except requests.RequestException as err:
        print(f"Polymarket unavailable ({err}): using the archive only.")
        games = archive.polymarket_games()
    deadlines = gameweek_deadlines(matches, bs)
    found = match_fixtures(listing(games), matches)
    per_slug = {r.slug: deadlines.get((r.season, r.gw), r.kickoff - DEADLINE_BEFORE_KICKOFF)
                for r in found.itertuples()}
    market, scorers = price_games(games, per_slug, cache)
    archive.save_polymarket(games, found)
    market = match_fixtures(fit_goal_rates(market), matches)
    if len(scorers):
        scorers = match_players(scorers, market, player_names(bs))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    market.to_parquet(MATCHES_PATH, index=False)
    scorers.to_parquet(SCORERS_PATH, index=False)
    return market, scorers


def upcoming(bs: dict, fixtures: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Today's odds for every listed match that hasn't kicked off, mapped to this season's FPL
    fixtures (and its scorer markets to FPL players). Quick: nothing historical is fetched."""
    from xpfpl.data import api
    now = pd.Timestamp.now(tz="UTC")
    games = fetch_games(since=(now - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    games = {slug: g for slug, g in games.items()
             if pd.Timestamp(next(e for e in g if e["slug"] == slug).get("startTime") or now) > now}
    if not games:
        return pd.DataFrame(), pd.DataFrame()
    market, scorers = price_games(games, {}, cache=False)
    market = match_fixtures(fit_goal_rates(market), api_fixture_table(bs, fixtures))
    if len(scorers) and len(market):
        names = player_names(bs)
        scorers = match_players(scorers, market, names[names["season"] == api.current_season(bs)])
    return market, scorers


def save_outrights(table: pd.DataFrame) -> pd.DataFrame:
    """Append today's outright prices to the history file (one snapshot per fetch), and archive them."""
    archive.save_outrights(table)
    if OUTRIGHTS_PATH.exists():
        table = pd.concat([pd.read_parquet(OUTRIGHTS_PATH), table], ignore_index=True)
    OUTRIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTRIGHTS_PATH, index=False)
    return table


# A player whose anytime-goalscorer odds are this low at the deadline has almost certainly been
# ruled out: on 2025-26, 3 of the 62 priced at 6% or less played (vs ~75% of the rest), and
# the model, which had given them a 23% chance of playing, over-predicted them by ~0.5 points each.
OUT_THRESHOLD = 0.06
OUT_PLAY_RATE = 0.05          # how often those players still played


def ruled_out(scorers: pd.DataFrame | None) -> pd.Series:
    """Player codes the scorer market has priced as not playing (p <= OUT_THRESHOLD), for the
    next gameweek. Traded or not: a market maker pulling a price to ~0 is the news itself."""
    if scorers is None or not len(scorers) or "code" not in scorers:
        return pd.Series(dtype=float)
    p = scorers.groupby("code")["p_anytime"].min()
    return p[p <= OUT_THRESHOLD]


def movers(market: pd.DataFrame | None, scorers: pd.DataFrame | None,
           days: float = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How the odds moved over the `days` before the pricing moment: (matches, players).

    Matches: every match, with each result's chance then and now, sorted by the biggest move in
    any of the three. Players: change in anytime-scorer chance, largest falls first (a collapse
    towards zero is how injuries and rotation show up)."""
    matches, players = pd.DataFrame(), pd.DataFrame()
    if market is not None and len(market) and move_column("home_win", days) in market:
        matches = market.copy()
        moves = [matches[r] - matches[move_column(r, days)] for r in ("home_win", "draw", "away_win")]
        biggest = pd.concat(moves, axis=1).abs().max(axis=1)
        matches = matches.assign(biggest_move=biggest).sort_values("biggest_move", ascending=False)
    if scorers is not None and len(scorers) and move_column("p_anytime", days) in scorers:
        players = scorers.assign(before=scorers[move_column("p_anytime", days)], now=scorers["p_anytime"]).dropna(subset=["before", "now"])
        # A move away from the ~50% opening price is a new market finding its level, not news -
        # unless it ends up priced as "out".
        opening = (players["before"] - 0.5).abs() <= 0.03
        players = players[~opening | (players["now"] <= OUT_THRESHOLD)]
        players = players.assign(change=players["now"] - players["before"]).sort_values("change")
    return matches, players


def load() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(market_matches, market_scorers) as saved by `build`, or None where not fetched yet."""
    read = lambda p: pd.read_parquet(p) if p.exists() else None      # noqa: E731
    return read(MATCHES_PATH), read(SCORERS_PATH)


def match_history(slug: str, days: float = 7, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Home / draw / away prices for one match over the `days` before `end` (default: now, or
    kick-off if earlier), hourly or every 5 minutes for windows under a day: long format (time,
    outcome, probability), for the movement chart."""
    events = _get(f"{GAMMA}/events", {"slug": slug})
    if not events:
        return pd.DataFrame(columns=["time", "outcome", "probability"])
    main = events[0]
    if end is None:
        end = min(pd.Timestamp.now(tz="UTC"), pd.Timestamp(main.get("startTime") or pd.Timestamp.now(tz="UTC")))
    home, away = _sides(main)
    labels = {"home_win": f"{home} win", "draw": "Draw", "away_win": f"{away} win"}
    rows = []
    for name, m in _match_markets([main]).items():
        if name not in labels:
            continue
        token = json.loads(m.get("clobTokenIds") or "[]")
        history = _get(f"{CLOB}/prices-history", {"market": token[0],
                                                  "startTs": int(end.timestamp() - days * 86400),
                                                  "endTs": int(end.timestamp()),
                                                  "fidelity": 60 if days >= 1 else 5}).get("history", [])
        rows += [{"time": pd.to_datetime(p["t"], unit="s", utc=True), "outcome": labels[name], "probability": p["p"]}
                 for p in history]
    return pd.DataFrame(rows, columns=["time", "outcome", "probability"])


def accuracy(market: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """How well the deadline odds predicted each season, next to our own team ratings: goals
    RMSE per side, clean-sheet Brier score, and the result's log loss (lower is better for all)."""
    from xpfpl import teams
    from xpfpl.features import _historical_strength, _poisson_win, market_side

    side = matches.drop_duplicates(["season", "fixture", "team_code"]).copy()
    side["gf"] = np.where(side["was_home"], side["team_h_score"], side["team_a_score"])
    side["ga"] = np.where(side["was_home"], side["team_a_score"], side["team_h_score"])
    side = side.join(_historical_strength(side, teams.ratings(matches), 1))
    d = side.merge(market_side(market), on=["season", "fixture", "team_code"])
    d = d.assign(won=(d["gf"] > d["ga"]).astype(float), cs=(d["ga"] == 0).astype(float),
                 fx_win=_poisson_win(d["fx_gf"], d["fx_ga"]))
    rows = []
    for season, g in d.groupby("season"):
        for source, gf, cs, win in (("Market", g["mkt_gf"], g["mkt_cs"], g["mkt_win"]),
                                    ("Our ratings", g["fx_gf"], g["fx_cs"], g["fx_win"])):
            win = win.clip(0.01, 0.99)
            rows.append({"season": season, "source": source, "matches": len(g) // 2,
                         "goals_rmse": float(np.sqrt(np.mean((gf - g["gf"]) ** 2))),
                         "clean_sheet_brier": float(np.mean((cs - g["cs"]) ** 2)),
                         "win_log_loss": float(-np.mean(g["won"] * np.log(win) + (1 - g["won"]) * np.log(1 - win)))})
    return pd.DataFrame(rows)
