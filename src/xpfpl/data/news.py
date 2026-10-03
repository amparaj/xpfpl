"""Team news: who's out, who's doubtful, who's back and when, and where each piece came from.

Two sources:

  FPL (bootstrap-static), for every player:
    news               "Hamstring injury - Expected back 10 Oct", "Suspended until 17 Oct", ...
    news_added         when FPL last changed it (the "last updated" of the news)
    status, chance_of_playing_next_round / _this_round   the flag and its percentage
    scout_news_link    the article the news came from (usually the club's own team-news page)
    scout_risks        known absences ahead, e.g. a loanee who can't face his parent club in GW27

  Premier Fantasy Tools' press-conference summaries (PRESS_URL): for each club, the players the
  manager named as OUT, DOUBT or IN (fit again) before the coming gameweek, with his words. One
  page, updated before each gameweek (robots.txt allows it; one request per run). Players are
  matched to FPL's by club and name (`match_press`).

Considered and not used: Fantasy Football Scout's injury table (its terms, 6.6, forbid automated
extraction and re-use in another database; FPL's own news, source link and timestamp cover the
same ground) and NewsNow (its robots.txt shuts out AI agents). Both are linked from the pages.

The FPL API only ever shows the present, so a scheduled GitHub Action
(.github/workflows/news-snapshot.yml) runs `update` every hour and force-pushes news.json to the
`news` branch: everyone with news now, the latest press conferences, and a log of every change to
a player's news this season (dated by FPL's own `news_added`). The website and the dashboard read
it from there; `pull()` (run by `xpfpl fetch`) copies the log and each press-conference snapshot
into archive/news/<season>/.

`availability` turns all of it into each upcoming match's chance of playing for `predict`.

No torch here: the Action installs only pandas, pyarrow and requests.
"""

import html as htmllib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from xpfpl import config
from xpfpl.data import api

BRANCH = "news"
FILE = "news.json"
ARCHIVE = config.ARCHIVE_DIR / "news"
PRESS_URL = "https://www.premierfantasytools.com/premier-league-press-conferences/"
PRESS_NAME = "Premier Fantasy Tools"
# How often the hourly run re-reads the press-conference page: every run in the two days before a
# deadline (when the pressers happen), else every PRESS_EVERY_HOURS.
PRESS_EVERY_HOURS = 6
PRESS_BUSY_HOURS = 48
STATUSES = ("OUT", "DOUBT", "IN")
# FPL's status codes in words.
STATUS_WORDS = {"a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended", "u": "Unavailable",
                "n": "Not available"}
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; xpfpl personal research project; "
                          "+https://github.com/amparaj/xpfpl)"}
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                       "oct", "nov", "dec"], start=1)}


# ---------------------------------------------------------------- FPL's news

def return_date(news: str, added: str | None) -> str | None:
    """The date in "Expected back 10 Oct" / "Suspended until 17 Oct" as an ISO date, or None.

    The player is expected to be available for matches on or after it. FPL gives no year: it's
    the first such date on or after a month before the news was added."""
    m = re.search(r"(?:back|until)\s+(\d{1,2})\s+([A-Za-z]{3})", news or "")
    if not m or m.group(2).lower() not in _MONTHS:
        return None
    ref = pd.Timestamp(added) if added else pd.Timestamp.now(tz="UTC")
    ref = ref.tz_localize(None) if ref.tzinfo else ref
    day, month = int(m.group(1)), _MONTHS[m.group(2).lower()]
    for year in (ref.year, ref.year + 1):
        try:
            when = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return None
        if when >= ref - pd.Timedelta(days=31):
            return when.date().isoformat()
    return None


def kind(status: str, news: str) -> str:
    """What sort of news it is: injury, doubt, suspension, left (sold or loaned out), other."""
    text = (news or "").lower()
    if status == "s" or text.startswith("suspended"):
        return "suspension"
    if status == "u" and ("joined" in text or "departed" in text or "loan" in text or "left" in text):
        return "left"
    if status == "i" or "injury" in text or "illness" in text:
        return "injury" if status != "d" else "doubt"
    if status == "d":
        return "doubt"
    return "other"


def reason(news: str) -> str:
    """The cause, without the return part: "Hamstring injury - Expected back 10 Oct" -> "Hamstring injury",
    "Suspended until 17 Oct" -> "Suspended"."""
    text = (news or "").strip()
    if re.match(r"suspended until", text, flags=re.I):
        return "Suspended"
    return text.split(" - ")[0].strip() if " - " in text else text


def clean_url(url: str | None) -> str | None:
    """A source link without its utm_ tracking parameters."""
    if not url:
        return None
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def fpl(bs: dict) -> pd.DataFrame:
    """One row per player who has news, a flag, or a known absence ahead (`risks`)."""
    team_code = {t["id"]: t["code"] for t in bs["teams"]}
    rows = []
    for e in bs["elements"]:
        news = (e.get("news") or "").strip()
        risks = [{"gw": r.get("gameweek"), "what": r.get("property"), "notes": r.get("notes"), "url": r.get("url")}
                 for r in (e.get("scout_risks") or [])]
        if not news and e.get("status", "a") == "a" and not risks:
            continue
        rows.append({
            "id": int(e["id"]), "code": int(e["code"]), "name": e["web_name"],
            "first": e.get("first_name"), "second": e.get("second_name"),
            "team": int(e["team"]), "team_code": int(team_code.get(e["team"], 0)), "pos": int(e["element_type"]),
            "status": e.get("status", "a"), "chance": _num(e.get("chance_of_playing_next_round")),
            "chance_this": _num(e.get("chance_of_playing_this_round")),
            "news": news, "reason": reason(news), "kind": kind(e.get("status", "a"), news),
            "back": return_date(news, e.get("news_added")), "added": e.get("news_added"),
            "source": clean_url(e.get("scout_news_link")), "risks": risks,
            "selected": float(e.get("selected_by_percent") or 0), "cost": int(e.get("now_cost") or 0),
        })
    return pd.DataFrame(rows, columns=["id", "code", "name", "first", "second", "team", "team_code", "pos", "status",
                                       "chance", "chance_this", "news", "reason", "kind", "back", "added", "source",
                                       "risks", "selected", "cost"])


# ---------------------------------------------------------------- press conferences

def fetch_press(timeout: float = 30) -> str:
    resp = requests.get(PRESS_URL, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_press(page: str) -> dict:
    """The press-conference page as {"updated": ISO date or None, "clubs": [{"club", "players":
    [{"name", "status"}], "quotes": [{"text", "by"}]}]}.

    The page has a "Last updated on September 18th, 2026" line, then one block per club: an <h2>
    with the club's name, paragraphs headed OUT / DOUBT / IN with a name per line, and the
    manager's words in <blockquote>s ending "– <manager>"."""
    updated = None
    m = re.search(r'class="last-updated">\s*Last updated on ([A-Za-z]+ \d{1,2})(?:st|nd|rd|th)?,? (\d{4})', page)
    if m:
        try:
            updated = pd.Timestamp(f"{m.group(1)} {m.group(2)}").date().isoformat()
        except ValueError:
            updated = None
    start = page.find('class="last-updated"')
    body = page[start:] if start >= 0 else page
    # The article ends where the page's comments / footer start.
    end = min([i for i in (body.find("<footer"), body.find('id="comments"')) if i > 0] or [len(body)])
    body = body[:end]
    clubs = []
    parts = re.split(r"<h2[^>]*>", body)[1:]
    for part in parts:
        head, _, rest = part.partition("</h2>")
        club = _text(head)
        if not club:
            continue
        players = []
        for label, names in re.findall(r'<p>\s*<span[^>]*>\s*(OUT|DOUBT|IN)\s*</span>\s*<br\s*/?>(.*?)</p>', rest, flags=re.S):
            for name in re.split(r"<br\s*/?>", names):
                name = _text(name)
                if name and name.lower() not in ("none", "-", "n/a"):
                    players.append({"name": name, "status": label})
        quotes = []
        for q in re.findall(r"<blockquote[^>]*>(.*?)</blockquote>", rest, flags=re.S):
            text = _text(q)
            by = None
            m = re.match(r"^(.*?)\s+[–—-]\s+([^“”\"]{3,60})$", text)
            if m:
                text, by = m.group(1).strip(), m.group(2).strip()
            quotes.append({"text": text.strip("“”\" "), "by": by})
        if players or quotes:
            clubs.append({"club": club, "players": players, "quotes": quotes})
    return {"updated": updated, "clubs": clubs}


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ø", "o").replace("Ø", "O").replace("ł", "l").replace("ß", "ss").replace("đ", "d")
    return re.sub(r"[^a-z ]", " ", s.lower().replace("'", "")).split()


# Club names as the press page writes them, where they differ from FPL's.
CLUB_ALIASES = {
    "brighton and hove albion": "brighton", "brighton & hove albion": "brighton", "leeds united": "leeds",
    "manchester city": "man city", "manchester united": "man utd", "newcastle united": "newcastle",
    "nottingham forest": "nott'm forest", "tottenham hotspur": "spurs", "tottenham": "spurs",
    "afc bournemouth": "bournemouth", "wolverhampton wanderers": "wolves", "west ham united": "west ham",
    "leicester city": "leicester", "sheffield united": "sheffield utd", "luton town": "luton",
}


def club_ids(bs: dict) -> dict[str, int]:
    """Lower-case club name (FPL's and the press page's) -> FPL team id."""
    out = {}
    for t in bs["teams"]:
        out[t["name"].lower()] = t["id"]
        out[t["short_name"].lower()] = t["id"]
    for alias, name in CLUB_ALIASES.items():
        if name in out:
            out[alias] = out[name]
    return out


def _match_one(name: str, squad: list[dict]) -> dict | None:
    """The squad member called `name`, or None if no one (or more than one) fits."""
    want = _tokens(name)
    if not want:
        return None

    def full(p):
        return _tokens(f"{p.get('first_name', '')} {p.get('second_name', '')}")

    tests = [
        lambda p: full(p) == want,
        lambda p: _tokens(p["web_name"]) == want,
        lambda p: set(want) <= set(full(p)),                           # "Cristhian Mosquera" in a longer name
        lambda p: set(_tokens(p["web_name"])) <= set(want) and len(_tokens(p["web_name"])) > 0,
        lambda p: want[-1] in full(p)[1:] or want[-1] == (_tokens(p["web_name"]) or [""])[-1],
    ]
    for test in tests:
        hits = [p for p in squad if test(p)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None
    return None


def match_press(press: dict, bs: dict) -> dict:
    """`press` with FPL's team id and code on each club and FPL's id and code on each player
    matched (None where the name couldn't be matched to exactly one of the club's players)."""
    clubs = club_ids(bs)
    team_code = {t["id"]: t["code"] for t in bs["teams"]}
    by_team: dict[int, list[dict]] = {}
    for e in bs["elements"]:
        by_team.setdefault(e["team"], []).append(e)
    out = []
    for c in press.get("clubs", []):
        team = clubs.get(c["club"].lower().strip())
        squad = by_team.get(team, [])
        players = []
        for p in c["players"]:
            hit = _match_one(p["name"], squad) if squad else None
            players.append({**p, "id": hit["id"] if hit else None, "code": hit["code"] if hit else None})
        out.append({**c, "team": team, "team_code": team_code.get(team), "players": players})
    return {**press, "clubs": out}


def press_table(press: dict | None) -> pd.DataFrame:
    """One row per matched player in a (matched) press snapshot: id, code, press status and the
    page's date. A player listed twice keeps the gloomiest entry (OUT over DOUBT over IN)."""
    rows = [{"id": p["id"], "code": p["code"], "press": p["status"], "club": c["club"]}
            for c in (press or {}).get("clubs", []) for p in c["players"] if p.get("id") is not None]
    t = pd.DataFrame(rows, columns=["id", "code", "press", "club"])
    t["rank"] = t["press"].map({s: i for i, s in enumerate(STATUSES)})
    return t.sort_values("rank").drop_duplicates("id").drop(columns="rank").set_index("id")


def _deadlines(bs: dict) -> tuple[int | None, int | None, str | None, str | None]:
    current = next((ev for ev in bs["events"] if ev["is_current"]), None)
    upcoming = next((ev for ev in bs["events"] if ev["is_next"]), None)
    return ((current or {}).get("id"), (upcoming or {}).get("id"),
            (current or {}).get("deadline_time"), (upcoming or {}).get("deadline_time"))


def press_is_for_next(press: dict | None, bs: dict) -> bool:
    """Whether the press conferences were given after the last deadline, i.e. ahead of the next gameweek."""
    if not press or not press.get("updated"):
        return False
    _, upcoming, last_deadline, _ = _deadlines(bs)
    if upcoming is None:
        return False
    if not last_deadline:
        return True
    return pd.Timestamp(press["updated"]).date() > pd.Timestamp(last_deadline).date()


def press_now(bs: dict) -> dict | None:
    """Today's press conferences, matched to FPL's players, or None if the page can't be read."""
    try:
        page = fetch_press()
    except requests.RequestException as exc:
        print(f"  (press conferences not read: {exc})")
        return None
    press = match_press(parse_press(page), bs)
    press["fetched"] = pd.Timestamp.now(tz="UTC").floor("min").isoformat()
    press["fresh"] = press_is_for_next(press, bs)
    return press


# ---------------------------------------------------------------- the chance of playing each match

def _press_wins(press: dict | None, bs: dict) -> pd.Series:
    """Per element: the press status (OUT / DOUBT / IN) where it's for the next gameweek and newer
    than FPL's own news (a press conference on the same day as FPL's update loses: FPL's is timed)."""
    if not press_is_for_next(press, bs):
        return pd.Series(dtype=object)
    said = press_table(press)["press"]
    if not len(said):
        return said
    day = pd.Timestamp(press["updated"]).date()
    added = {e["id"]: e.get("news_added") for e in bs["elements"]}
    newer = [not added.get(i) or pd.Timestamp(added[i]).date() < day for i in said.index]
    return said[newer]


def availability(frame: pd.DataFrame, bs: dict, next_gw: int, press: dict | None = None) -> pd.DataFrame:
    """Each upcoming match's chance that the player is available, from everything known now.

    `frame` has one row per player per fixture (element, gw, status, chance_of_playing_next_round,
    kickoff_time). Returns, on the same index, `avail` (0-1) and `news_rule`, which rule set it:

      flag     FPL's chance of playing next round, recovering by 25 points a gameweek after it
               (the old rule, and what's left when nothing below applies); 'u' (left) = 0
      press    a press conference for the next gameweek, newer than FPL's news, said OUT (0),
               DOUBT (config.PRESS_DOUBT_CHANCE) or IN (1); later weeks recover from there
      back     FPL gives a return date ("Expected back 10 Oct"): 0 for matches before it; from it,
               config.RETURN_CHANCE for an injury, 1 for a suspension. The next gameweek keeps
               FPL's own flag, which is set with the deadline in sight.
      risk     a known absence (scout_risks, e.g. a loanee against his parent club): 0 that gameweek
    """
    offset = (frame["gw"] - next_gw).astype(float)
    chance = frame["chance_of_playing_next_round"].astype(float)
    rule = pd.Series("flag", index=frame.index, dtype=object)
    if config.PRESS_NEWS:
        said = frame["element"].map(_press_wins(press, bs))
        override = said.map({"OUT": 0.0, "DOUBT": 100.0 * config.PRESS_DOUBT_CHANCE, "IN": 100.0})
        chance = override.where(said.notna(), chance).astype(float)
        rule = rule.mask(said.notna(), "press")
    p = (chance.fillna(100.0) / 100.0 + 0.25 * offset).clip(upper=1.0)
    p = p.where(frame["status"] != "u", 0.0)

    if config.NEWS_RETURN_DATES:
        info = fpl(bs).set_index("id")
        back = frame["element"].map(info["back"])
        what = frame["element"].map(info["kind"])
        kick = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce").dt.tz_convert("Europe/London")
        day = kick.dt.strftime("%Y-%m-%d")
        use = back.notna() & (offset > 0) & (rule != "press") & frame["status"].isin(["i", "s", "d"])
        before = use & (day < back)
        after = use & (day >= back)
        p = p.mask(before, 0.0)
        p = p.mask(after & (what == "suspension"), 1.0)
        p = p.mask(after & (what != "suspension"), np.maximum(p, config.RETURN_CHANCE))
        rule = rule.mask(use, "back")

        risks = frame["element"].map(info["risks"])
        out_gw = pd.Series([isinstance(r, list) and any(x.get("gw") == g for x in r)
                            for r, g in zip(risks, frame["gw"])], index=frame.index, dtype=bool)
        p = p.mask(out_gw, 0.0)
        rule = rule.mask(out_gw, "risk")
    return pd.DataFrame({"avail": p.astype(float), "news_rule": rule}, index=frame.index)


# ---------------------------------------------------------------- the hourly log

LOG_FIELDS = ["status", "chance", "news"]


def _plain(v):
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _press_due(previous_press: dict | None, bs: dict, now: pd.Timestamp) -> bool:
    if not previous_press or not previous_press.get("fetched"):
        return True
    age = now - pd.Timestamp(previous_press["fetched"])
    _, _, _, deadline = _deadlines(bs)
    busy = deadline is not None and pd.Timedelta(0) <= pd.Timestamp(deadline) - now <= pd.Timedelta(hours=PRESS_BUSY_HOURS)
    return busy or age >= pd.Timedelta(hours=PRESS_EVERY_HOURS)


def update(previous: dict | None, bs: dict, press: dict | None = None, now: pd.Timestamp | None = None,
           read_press: bool = True) -> dict:
    """The new news.json from the `previous` one and bootstrap-static `bs`.

    `press` is today's press conferences (matched); if None and `read_press`, the page is read
    when due (`_press_due`), else the previous copy is kept. The log gets an entry for every
    player whose status, chance or news differs from the previous run's (and, on a season's first
    run, everyone with news now), timed by FPL's `news_added`."""
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").floor("min")
    season = api.current_season(bs)
    current, upcoming, last_deadline, deadline = _deadlines(bs)
    same = bool(previous) and previous.get("season") == season
    table = fpl(bs)
    old = {p["id"]: p for p in previous.get("players", [])} if same else {}
    log = list(previous.get("log", [])) if same else []

    if press is None:
        kept = previous.get("press") if same else None
        press = (press_now(bs) if read_press and _press_due(kept, bs, now) else None) or kept
    if press is not None:
        press = {**press, "fresh": press_is_for_next(press, bs)}
    said = press_table(press)["press"] if press else pd.Series(dtype=object)

    rows = []
    for p in table.to_dict("records"):
        p = {k: _plain(v) for k, v in p.items()}
        p["press"] = said.get(p["id"]) if press and press.get("fresh") else None
        rows.append(p)
    now_ids = {p["id"] for p in rows}
    first = not old
    for p in rows:
        before = old.get(p["id"])
        if first or before is None or any(before.get(k) != p.get(k) for k in LOG_FIELDS):
            if first and p["status"] == "a" and not p["news"]:
                continue
            log.append({"t": p["added"], "seen": now.isoformat(), "gw": current, "id": p["id"], "code": p["code"],
                        "name": p["name"], "team_code": p["team_code"], "pos": p["pos"], "status": p["status"],
                        "chance": p["chance"], "news": p["news"], "back": p["back"], "source": p["source"]})
    # Players whose news was cleared (fit again): they drop out of `players`, so log them as available.
    elements = {e["id"]: e for e in bs["elements"]}
    team_code = {t["id"]: t["code"] for t in bs["teams"]}
    for pid, before in old.items():
        if pid in now_ids or pid not in elements or not (before.get("news") or before.get("status") != "a"):
            continue
        e = elements[pid]
        log.append({"t": e.get("news_added"), "seen": now.isoformat(), "gw": current, "id": pid, "code": e["code"],
                    "name": e["web_name"], "team_code": team_code.get(e["team"]), "pos": e["element_type"],
                    "status": e.get("status", "a"), "chance": _num(e.get("chance_of_playing_next_round")),
                    "news": "", "back": None, "source": None})

    return {"updated": now.isoformat(), "season": season, "gw_current": current, "gw_next": upcoming,
            "last_deadline": last_deadline, "next_deadline": deadline, "players": rows, "press": press,
            "press_url": PRESS_URL, "press_name": PRESS_NAME, "log": log}


def run(folder: Path, bs: dict | None = None) -> Path:
    """`update` the news.json in `folder` (created if missing)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FILE
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    state = update(previous, bs or api.bootstrap())
    path.write_text(json.dumps(state, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- reading it back

def remote(timeout: float = 15) -> dict | None:
    """The Action's latest news.json from the `news` branch, or None (offline, no branch yet)."""
    from xpfpl.data.pricewatch import _repo
    repo = _repo()
    if not repo:
        return None
    try:
        resp = requests.get(f"https://raw.githubusercontent.com/{repo}/{BRANCH}/{FILE}", timeout=timeout)
        return resp.json() if resp.ok else None
    except (requests.RequestException, ValueError):
        return None


def current(bs: dict, read_press: bool = True) -> dict:
    """The news as it stands: the Action's log if it can be read for this season (its players
    refreshed from `bs`, its press conferences re-read if they're due), else built from `bs` alone."""
    state = remote()
    if not state or state.get("season") != api.current_season(bs):
        state = None
    return update(state, bs, read_press=read_press)


def change_log(state: dict | None) -> pd.DataFrame:
    """The news changes in a news.json, newest first, `t` (FPL's time) and `seen` as timestamps."""
    cols = ["t", "seen", "gw", "id", "code", "name", "team_code", "pos", "status", "chance", "news", "back", "source"]
    log = pd.DataFrame((state or {}).get("log", []), columns=cols)
    for c in ("t", "seen"):
        log[c] = pd.to_datetime(log[c], utc=True, errors="coerce")
    return log.sort_values(["t", "seen"], ascending=False, na_position="last", ignore_index=True)


def pull(remote_name: str = "origin") -> list[Path]:
    """Copy the `news` branch's change log and its press-conference snapshot into
    archive/news/<season>/ (log.parquet, press/<date>.json). Returns the files written."""
    import subprocess
    from xpfpl.data import archive

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True)

    if git("fetch", "--quiet", remote_name, BRANCH).returncode != 0:
        return []
    shown = git("show", f"FETCH_HEAD:{FILE}")
    if shown.returncode != 0:
        return []
    return save(json.loads(shown.stdout.decode("utf-8")))


def save(state: dict) -> list[Path]:
    """Write a news.json's log and press conferences into the archive (unchanged files skipped)."""
    from xpfpl.data import archive
    folder = ARCHIVE / state["season"]
    written = []
    log = change_log(state)
    if len(log) and archive.write(log, folder / "log.parquet"):
        written.append(folder / "log.parquet")
    press = state.get("press")
    if press and press.get("updated"):
        path = folder / "press" / f"{press['updated']}.json"
        keep = {k: press[k] for k in ("updated", "fetched", "clubs") if k in press}
        text = json.dumps(keep, ensure_ascii=False, indent=1, sort_keys=True)
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old is None or json.loads(old).get("clubs") != keep["clubs"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written


def archived_log(season: str) -> pd.DataFrame:
    path = ARCHIVE / season / "log.parquet"
    return pd.read_parquet(path) if path.exists() else change_log(None)


def archived_press(season: str) -> dict[str, dict]:
    """Every archived press-conference snapshot of a season, by its date."""
    folder = ARCHIVE / season / "press"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))} if folder.exists() else {}


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "update":
        print(f"wrote {run(Path(argv[1]))}")
    else:
        raise SystemExit("usage: python -m xpfpl.data.news update <folder>")


if __name__ == "__main__":
    main(sys.argv[1:])
