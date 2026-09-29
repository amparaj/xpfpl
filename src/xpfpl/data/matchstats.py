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
goal being attacked (the penalty spot is 11.5), `start_y` across the pitch (50 = centre), rising
towards the shooter's right (unlike Opta's own feed: right wingers average ~60, left ones ~40).
`goal_mouth_y` runs across the goal the same way (posts at about 45.2 and 54.8) and `goal_mouth_z` up it
(crossbar at about 38). In 2026-27 about one shot in eight comes without a player: `from_fotmob`
names them from FotMob's own match page (the source's source), and `attribute` works out any
left from the players' own match stats (`inferred` = True).

archive/matchstats/<season>/<table>/gwNN.parquet, the source's gameweek folders, each row carrying
the FPL `fixture` id (and `gw`, the fixture's FPL gameweek, which can differ for a rearranged match).
"""

import json
import re
import time

import pandas as pd
import requests

from xpfpl import config
from xpfpl.data import archive, cups

ARCHIVE = config.ARCHIVE_DIR / "matchstats"
FIRST_SEASON = "2025-26"
TABLES = {"matches": "matches", "shots": "shots", "momentum": "momentum", "players": "playermatchstats"}
KEYS = {"matches": ["match_id", "stat"], "shots": ["match_id", "shot_index"], "momentum": ["match_id", "minute"],
        "players": ["match_id", "player_id"], "fotmob": ["match_id", "shot"]}
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
    """The season's FPL fixtures: (fixture, gw, home_code, away_code, finished). `finished` is
    FPL's own flag: the result is in (bonus may still be provisional)."""
    fx = archive.fixtures(season)
    teams = pd.read_parquet(archive.FPL / season / "teams.parquet").set_index("id")["code"]
    fx = fx.drop_duplicates("id", keep="last")
    return pd.DataFrame({"fixture": fx["id"].astype(int), "gw": fx["event"].astype("Int64"),
                         "home_code": fx["team_h"].map(teams).astype(int),
                         "away_code": fx["team_a"].map(teams).astype(int),
                         "finished": fx["finished"].fillna(False).astype(bool)})


def link(raw: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """The source's played Premier League matches with their FPL fixture: match_id -> fixture, gw.
    A club pair meets once at home per season, so the pair identifies the fixture."""
    league = raw["tournament"].eq(cups.EPL) if "tournament" in raw else True
    played = raw.loc[league & raw["finished"].astype(bool), ["match_id", "home_team", "away_team"]]
    played = pd.DataFrame({"match_id": played["match_id"], "home_code": played["home_team"].astype(int),
                           "away_code": played["away_team"].astype(int)})
    return played.merge(fixtures, on=["home_code", "away_code"])


FOTMOB = "https://www.fotmob.com"
FOTMOB_PAUSE = 1.0      # seconds between page requests
XG_MATCH = 0.002        # xG that still agrees between FotMob's page and the source
CLOCK_SLACK = 2         # minutes the two lists' clocks may differ by


def fotmob_shots(match_id: str, url: str) -> pd.DataFrame | None:
    """FotMob's own shot list for one match, read from its public match page (FotMob's robots.txt
    allows /matches/; only /api/ is closed to crawlers): minute, added time, side, xG and the
    shooter's `code` (FotMob gives each player's Opta id, which is FPL's `code`). The source
    republishes this data but leaves the shooter off some shots; the page has them all.

    Always a download: `fetch` keeps what it reads in archive/matchstats/<season>/fotmob and
    never asks for the same match twice. None if the page can't be read."""
    time.sleep(FOTMOB_PAUSE)
    try:
        resp = cups._session.get(FOTMOB + url.split("#")[0], timeout=60)
        resp.raise_for_status()
        page = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', resp.text, re.S)
        content = json.loads(page.group(1))["props"]["pageProps"]["content"]
        home = content["lineup"]["homeTeam"]["id"]
        opta = {int(i): p.get("optaId") for i, p in content["playerStats"].items()}
        rows = [{"match_id": match_id, "shot": n, "minute": s["min"], "added_time": s.get("minAdded"),
                 "is_home": s["teamId"] == home, "xg": s["expectedGoals"],
                 "code": int(opta[s["playerId"]]) if opta.get(s["playerId"]) else None}
                for n, s in enumerate(content["shotmap"]["shots"]) if not s.get("isOwnGoal")]
    except (requests.RequestException, AttributeError, KeyError, TypeError, ValueError) as e:
        print(f"  FotMob's page for {match_id} couldn't be read ({e.__class__.__name__}): its shooters are inferred instead")
        return None
    return pd.DataFrame(rows, columns=["match_id", "shot", "minute", "added_time", "is_home", "xg", "code"])


def _read(path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def archived(season: str) -> set[int]:
    """The fixtures whose match data is already in the archive (team stats, shots and player
    stats all there): `fetch` never downloads them again."""
    have = None
    for table in ("matches", "shots", "players"):
        df = load(season, table)
        got = set() if df is None else set(df["fixture"].astype(int))
        have = got if have is None else have & got
    return have or set()


def from_fotmob(shots: pd.DataFrame, pages: dict[str, pd.DataFrame | None], people: pd.DataFrame | None) -> pd.DataFrame:
    """Name the shots the source left without a shooter from FotMob's own shot list for the match
    (`pages`: match_id -> fotmob_shots). `people` is the source's players.csv (player_code,
    player_id, team_code), to fill in its ids too.

    Shots are paired within the same side, minute and added time. FotMob revises some xG after the
    source has copied it, so xG only decides between several unnamed shots in the same minute, and
    only where it still agrees (`XG_MATCH`). Shots the source did name use up their shooter first."""
    shots = shots.copy()
    ids = (people.drop_duplicates("player_code").set_index("player_code") if people is not None
           else pd.DataFrame(columns=["player_id", "team_code"]))
    key = ["minute", "added", "is_home"]

    def name(i, code: int) -> None:
        shots.at[i, "code"] = code
        if code in ids.index:
            shots.at[i, "player_id"] = ids.at[code, "player_id"]
            shots.at[i, "team_code"] = ids.at[code, "team_code"]

    ours = shots.assign(added=shots["added_time"].fillna(0), is_home=shots["is_home"].astype(bool))
    ours = ours[ours["outcome"].ne("ownGoal")]
    for match, page in pages.items():
        if page is None or page.empty:
            continue
        page = page.assign(added=page["added_time"].fillna(0))
        mine = ours[ours["match_id"] == match]
        for k, group in mine.groupby(key):
            gap = group[group["code"].isna()]
            if gap.empty:
                continue
            theirs = page[(page["minute"] == k[0]) & (page["added"] == k[1]) & (page["is_home"] == k[2])]
            if len(theirs) != len(group) or theirs["code"].isna().any():
                continue                                    # the two lists disagree about this minute
            left = list(theirs["code"].astype(int))
            for code in group["code"].dropna().astype(int):
                if code in left:
                    left.remove(code)
            if len(left) != len(gap):
                continue
            if len(set(left)) == 1:
                pairs = [(i, left[0]) for i in gap.index]
            else:                                           # several shooters: xG has to tell them apart
                rest = theirs[theirs["code"].astype(int).isin(left)].sort_values("xg")
                pairs = list(zip(gap.sort_values("xg").index, rest["code"].astype(int)))
                if len(rest) != len(gap) or any(abs(shots.at[i, "xg"] - x) >= XG_MATCH
                                                for (i, _), x in zip(pairs, rest["xg"])):
                    continue
            for i, code in pairs:
                name(i, code)
        # The two clocks can differ by a minute: FotMob's shots on this side that no named shot
        # accounts for, each claimed by an unnamed shot within `CLOCK_SLACK` minutes when only one fits.
        for side in (True, False):
            spare = page[(page["is_home"] == side) & page["code"].notna()].copy()
            spare["t"] = spare["minute"] + spare["added"]
            named = shots[(shots["match_id"] == match) & (shots["is_home"].astype(bool) == side)
                          & shots["outcome"].ne("ownGoal") & shots["code"].notna()]
            for r in named.itertuples():
                hit = spare[spare["code"] == r.code]
                if len(hit):                                # the same player's shot nearest in time
                    t = r.minute + (0 if pd.isna(r.added_time) else r.added_time)
                    spare = spare.drop(hit.index[(hit["t"] - t).abs().argmin()])
            gap = shots[(shots["match_id"] == match) & (shots["is_home"].astype(bool) == side)
                        & shots["code"].isna() & shots["outcome"].ne("ownGoal")]
            for i in gap.index:
                t = shots.at[i, "minute"] + (0 if pd.isna(shots.at[i, "added_time"]) else shots.at[i, "added_time"])
                near = spare[(spare["t"] - t).abs() <= CLOCK_SLACK]
                if near["code"].nunique() > 1:              # several: only a shot whose xG still agrees
                    near = near[(near["xg"] - shots.at[i, "xg"]).abs() < XG_MATCH]
                if len(near) and near["code"].nunique() == 1:
                    name(i, int(near["code"].iloc[0]))
                    spare = spare.drop(near.index[(near["t"] - t).abs().argmin()])
    return shots


def _shot_kind(outcome: str) -> str:
    """FotMob's shots on target are goals and saves; everything else (miss, block, post) is off."""
    return outcome if outcome in ("goal", "save") else "off"


def _assign(shots: pd.DataFrame, owed: pd.DataFrame) -> dict:
    """Shooters for one side's unnamed shots: {shot index: player_id}, for the shots every close fit agrees on.

    `owed` is what each of the side's players' match stats have left over once their named shots
    are taken off (goals, saves, off-target shots, xG, xGOT). Every assignment that keeps within
    those counts is tried and scored on how far the xG and xGOT left over miss; a shot is named
    only if all assignments within `TOLERANCE` of the best give it to the same player."""
    rows = list(shots.itertuples())
    kinds = [_shot_kind(r.outcome) for r in rows]
    left = {p: {k: int(owed.at[p, k]) for k in ("goal", "save", "off")} for p in owed.index}
    # A kind with more unnamed shots than any player owes (the source's counts are sometimes off)
    # lets that many shots stay unassigned.
    spare = {k: max(kinds.count(k) - sum(v[k] for v in left.values()), 0) for k in ("goal", "save", "off")}
    fits, tried = [], 0

    def walk(i, chosen):
        nonlocal tried
        tried += 1
        if tried > SEARCH_LIMIT:
            raise OverflowError
        if i == len(rows):
            xg = {p: owed.at[p, "xg"] for p in left}
            xgot = {p: owed.at[p, "xgot"] for p in left}
            for r, p in zip(rows, chosen):
                if p is not None:
                    xg[p] -= r.xg
                    xgot[p] -= 0 if pd.isna(r.xgot) else r.xgot
            fits.append((sum(abs(v) for v in xg.values()) + sum(abs(v) for v in xgot.values()), tuple(chosen)))
            return
        k = kinds[i]
        for p, owes in left.items():
            if owes[k] > 0:
                owes[k] -= 1
                walk(i + 1, chosen + [p])
                owes[k] += 1
        if spare[k] > 0:
            spare[k] -= 1
            walk(i + 1, chosen + [None])
            spare[k] += 1

    try:
        walk(0, [])
    except OverflowError:
        return {}
    if not fits:
        return {}
    best = min(c for c, _ in fits)
    close = [a for c, a in fits if c <= best + TOLERANCE]
    return {r.Index: close[0][i] for i, r in enumerate(rows)
            if close[0][i] is not None and all(a[i] == close[0][i] for a in close)}


TOLERANCE = 0.01      # xG/xGOT a near-best fit may miss by (on 2025-26 with 12% of shooters hidden: 88% named, 99.2% right)
SEARCH_LIMIT = 200_000


def attribute(shots: pd.DataFrame, players: pd.DataFrame, home_code: pd.Series) -> pd.DataFrame:
    """Fill in the shooter of shots the source left without one (about one in eight in 2026-27),
    from its own per-player match stats: whatever a player's shots, goals, xG and xGOT aren't
    explained by their named shots belongs to the unnamed ones. `home_code` maps match_id to the home
    club's code. Filled rows get `inferred` = True; shots that can't be pinned down stay unnamed."""
    shots = shots.copy()
    shots["inferred"] = False
    missing = shots["player_id"].isna() & shots["code"].isna() & shots["outcome"].ne("ownGoal")
    if not missing.any() or players is None or players.empty or "team_code" not in players:
        return shots
    named = shots[shots["player_id"].notna()].assign(kind=lambda d: d["outcome"].map(_shot_kind))
    done = named.groupby(["match_id", "player_id"]).agg(
        shots=("kind", "size"), goal=("kind", lambda k: (k == "goal").sum()), save=("kind", lambda k: (k == "save").sum()),
        xg=("xg", "sum"), xgot=("xgot", "sum"))
    stats = players.set_index(["match_id", "player_id"])
    stats = stats.join(done, rsuffix="_named").fillna({c: 0 for c in ("shots", "goal", "save", "xg_named", "xgot_named")})
    owed = pd.DataFrame({
        "team_code": stats["team_code"],
        "goal": stats["goals"].fillna(0) - stats["goal"],
        "save": stats["shots_on_target"].fillna(0) - stats["goals"].fillna(0) - stats["save"],
        "off": stats["total_shots"].fillna(0) - stats["shots_on_target"].fillna(0) - (stats["shots"] - stats["goal"] - stats["save"]),
        "xg": stats["xg"].fillna(0) - stats["xg_named"], "xgot": stats["xgot"].fillna(0) - stats["xgot_named"]})
    owed[["goal", "save", "off"]] = owed[["goal", "save", "off"]].clip(lower=0)
    codes = players.drop_duplicates("player_id").set_index("player_id")[["code", "team_code"]]
    found = {}
    for (match, home), group in shots[missing].groupby(["match_id", "is_home"]):
        if match not in home_code.index or match not in owed.index.get_level_values(0):
            continue
        side = owed.loc[match]
        side = side[(side["team_code"] == home_code[match]) == bool(home)]
        side = side[side[["goal", "save", "off"]].sum(axis=1) > 0]
        if not side.empty:
            found.update(_assign(group, side))
    if found:
        hit = pd.Index(list(found))
        shots.loc[hit, "player_id"] = pd.Series(found)
        shots.loc[hit, ["code", "team_code"]] = codes.loc[shots.loc[hit, "player_id"]].to_numpy()
        shots.loc[hit, "inferred"] = True
    return shots


def fetch(season: str, gameweeks=None, refresh: bool = False) -> int:
    """Download the match data of the season's finished matches into archive/matchstats. Returns
    the files written.

    Only matches FPL has marked finished are fetched, and each only once: a match whose data is
    already in the archive is never downloaded again, so a gameweek with nothing new costs no
    requests, and one still in progress gets its finished matches now and the rest on a later
    fetch. A finished match the source hasn't published yet (it lags a day or two) is simply
    tried again next time. `refresh` re-downloads every finished match (FotMob occasionally
    corrects its numbers); FotMob's own shot lists, once archived, are kept even then."""
    if season < FIRST_SEASON or not archive.has_season(season):
        return 0
    fixtures, written = fpl_fixtures(season), 0
    done = set() if refresh else archived(season)
    wanted = fixtures[fixtures["finished"] & ~fixtures["fixture"].isin(done) & fixtures["gw"].notna()]
    if gameweeks is not None:
        wanted = wanted[wanted["gw"].isin(list(gameweeks))]
    for gw in sorted(int(g) for g in wanted["gw"].unique()):
        want = set(wanted.loc[wanted["gw"] == gw, "fixture"])
        # The source's gameweek folder, from the local cache unless that predates these matches.
        fresh = refresh
        raw = cups._csv(season, gw, "matches", fresh)
        if not fresh and (raw is None or not want <= set(link(raw, fixtures)["fixture"])):
            fresh = True
            raw = cups._csv(season, gw, "matches", fresh)
        if raw is None or raw.empty:
            continue
        found = link(raw, fixtures)
        found = found[found["fixture"].isin(want)]
        if found.empty:
            continue                                        # not published yet: next fetch
        keys = found.set_index("match_id")[["fixture", "gw"]]
        people = cups._csv(season, gw, "players", fresh)
        codes = people.set_index("player_id")[["player_code", "team_code"]] if people is not None else None
        tables = {}
        for table, name in TABLES.items():
            df = raw if table == "matches" else cups._csv(season, gw, name, fresh)
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
            tables[table] = df.reset_index(drop=True)
        if "shots" in tables:
            # Shots the source left without a shooter: FotMob's own match page first (archived,
            # so each page is read once), then the players' match stats for any it can't settle.
            shots = tables["shots"]
            path = ARCHIVE / season / "fotmob" / f"gw{gw:02d}.parquet"
            known = _read(path)
            urls = raw.set_index("match_id")["match_url"] if "match_url" in raw else pd.Series(dtype=str)
            gaps = shots.loc[shots["code"].isna() & shots["outcome"].ne("ownGoal"), "match_id"].unique()
            pages, new = {}, []
            for m in gaps:
                if known is not None and (known["match_id"] == m).any():
                    pages[m] = known[known["match_id"] == m]
                elif isinstance(urls.get(m), str):
                    pages[m] = fotmob_shots(m, urls[m])
                    if pages[m] is not None:
                        new.append(pages[m])
            if new:
                written += archive.write(pd.concat([known, *new], ignore_index=True) if known is not None
                                         else pd.concat(new, ignore_index=True), path)
            shots = from_fotmob(shots, pages, people)
            tables["shots"] = attribute(shots, tables.get("players"), found.set_index("match_id")["home_code"])
        for table, df in tables.items():
            # Added to what the gameweek's file already holds: the matches archived before stay as they were.
            path = ARCHIVE / season / table / f"gw{gw:02d}.parquet"
            old = _read(path)
            if old is not None:
                df = pd.concat([old[~old["fixture"].isin(found["fixture"])], df], ignore_index=True)
            written += archive.write(df, path)
    return written


def seasons() -> list[str]:
    """The seasons with match data in the archive, oldest first."""
    return sorted(p.name for p in ARCHIVE.glob("*") if (p / "matches").is_dir())


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
