"""A compressed, git-tracked copy of every source the pipeline reads, so the history survives
if a source goes away: vaastav's repo could disappear, the FPL API only ever holds the current
season's match-by-match history (it resets each summer), and Polymarket can delist old markets.

archive/
  fpl/<season>/gws.parquet             player-match rows as vaastav gives them (merged_gw.csv)
  fpl/<season>/gws/gwNN.parquet        ...or, for seasons archived from the FPL API, one file per GW
  fpl/<season>/players.parquet         who's who: id, code, names, position, club
  fpl/<season>/teams.parquet           clubs
  fpl/<season>/fixtures.parquet        fixtures and scores (vaastav seasons that have them)
  fpl/<season>/fixtures/gwNN.parquet   ...one file per GW for FPL API seasons, with the match stats
  fpl/<season>/events.parquet          gameweeks: deadlines, averages, chip plays (FPL API seasons)
  fpl/<season>/deadlines/gwNN.parquet  bootstrap-static's players shortly before GW NN's deadline:
                                       price, news, injury flag, ownership, transfers. Nobody else
                                       keeps these, and the API overwrites them every hour.
  predictions/<season>/gwNN_<model>.parquet  the forecasts saved before each deadline
  polymarket/<season>/events/gwNN.parquet    each played match's Polymarket events (trimmed JSON)
  polymarket/<season>/prices/gwNN.parquet    their price histories to the FPL deadline
  polymarket/outrights/<date>.parquet        season-long markets, one snapshot per day

Everything is zstd Parquet. A season, once finished, is never rewritten, and the current one grows
by a few small files a week, so the git history stays small. `write` skips a file whose contents
haven't changed, so re-running a step leaves nothing to commit.

The pipeline reads the archive first: `history.load_past_season` and `markets.build` only go to
vaastav / Polymarket for what isn't here. `python -m xpfpl.data.archive deadline` takes the
pre-deadline snapshot on its own (no torch needed), which is what the scheduled GitHub Action runs.
main only accepts pull requests, so the Action pushes to the `deadline-snapshots` branch and
`pull_snapshots` (run by `xpfpl fetch` and `xpfpl archive`) copies them into the working tree.
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd

from xpfpl import config
from xpfpl.data import api

FPL = config.ARCHIVE_DIR / "fpl"
POLYMARKET = config.ARCHIVE_DIR / "polymarket"
PREDICTIONS = config.ARCHIVE_DIR / "predictions"

# The who's-who columns of bootstrap-static's players: the rest change every hour and belong in
# the deadline snapshots.
PLAYER_KEYS = ["id", "code", "first_name", "second_name", "web_name", "element_type", "team", "team_code"]
TEAM_KEYS = ["id", "code", "name", "short_name", "pulse_id", "strength", "strength_overall_home",
             "strength_overall_away", "strength_attack_home", "strength_attack_away",
             "strength_defence_home", "strength_defence_away"]
# What `markets.py` reads from a Polymarket event and its markets; the rest is descriptions and
# display fields.
EVENT_KEYS = ["id", "slug", "title", "startTime", "endDate", "gameId", "volume", "closed", "teams"]
MARKET_KEYS = ["id", "question", "slug", "sportsMarketType", "groupItemTitle", "clobTokenIds", "outcomes",
               "outcomePrices", "bestBid", "bestAsk", "lastTradePrice", "volume", "closed"]
EMPTY = -1          # the `t` of the one row recording a price history with no trades in it


# ---------------------------------------------------------------- reading and writing

def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def frame(records) -> pd.DataFrame:
    """A table Parquet can hold: nested values (lists, dicts) as JSON text, and columns that mix
    types (e.g. numbers and text) as text."""
    df = pd.DataFrame(records)
    for col in df.columns:
        if df[col].dtype != object:
            continue
        values = df[col]
        if values.map(lambda v: isinstance(v, (dict, list))).any():
            values = values.map(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
        if len({type(v) for v in values if not _missing(v)}) > 1:
            values = values.map(lambda v: None if _missing(v) else str(v))
        df[col] = values
    return df


def write(df: pd.DataFrame, path: Path) -> bool:
    """Save `df` as zstd Parquet, unless the file already holds the same table (so an unchanged
    re-run leaves git nothing to commit). True if the file was written."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd", compression_level=9)
    data = buf.getvalue()
    if path.exists():
        if path.read_bytes() == data:
            return False
        try:
            if pd.read_parquet(path).equals(pd.read_parquet(io.BytesIO(data))):
                return False
        except Exception:        # an unreadable file: replace it
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _read_parts(single: Path, folder: Path) -> pd.DataFrame | None:
    """`single` if it exists, else every gwNN.parquet in `folder` stacked (None if neither)."""
    if single.exists():
        return pd.read_parquet(single)
    parts = sorted(folder.glob("gw*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else None


# ---------------------------------------------------------------- FPL

def has_season(season: str) -> bool:
    folder = FPL / season
    return (folder / "players.parquet").exists() and (
        (folder / "gws.parquet").exists() or any((folder / "gws").glob("gw*.parquet")))


def seasons() -> list[str]:
    return sorted(p.name for p in FPL.glob("*") if has_season(p.name))


def season_tables(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player-match rows, players) for an archived season, in the source's own columns."""
    folder = FPL / season
    return _read_parts(folder / "gws.parquet", folder / "gws"), pd.read_parquet(folder / "players.parquet")


def fixtures(season: str) -> pd.DataFrame | None:
    folder = FPL / season
    return _read_parts(folder / "fixtures.parquet", folder / "fixtures")


def save_vaastav(season: str, tables: dict[str, pd.DataFrame]) -> int:
    """A finished season from vaastav's CSVs: {"gws", "players", "teams", "fixtures"} (the last two
    only where the season has them). Returns the number of files written."""
    return sum(write(frame(df), FPL / season / f"{name}.parquet") for name, df in tables.items())


def save_api_season(season: str, bs: dict, fx: list[dict], histories: list[dict]) -> int:
    """This season from the FPL API: the player rows (element-summary history) and fixtures of
    every gameweek FPL has finished checking, one file per GW; plus players, clubs and gameweeks.
    Returns the number of files written."""
    folder = FPL / season
    checked = {ev["id"] for ev in bs["events"] if ev["finished"] and ev["data_checked"]}
    written = 0

    rows = frame([r for r in histories if r["round"] in checked])
    for gw, part in rows.groupby("round") if len(rows) else []:
        part = part.sort_values(["fixture", "element"]).reset_index(drop=True)
        written += write(part, folder / "gws" / f"gw{gw:02d}.parquet")
    matches = frame([f for f in fx if f["event"] in checked])
    for gw, part in matches.groupby("event") if len(matches) else []:
        written += write(part.sort_values("id").reset_index(drop=True), folder / "fixtures" / f"gw{gw:02d}.parquet")

    # Keep everyone ever listed: a player who leaves mid-season can drop out of bootstrap-static.
    players = frame([{k: p.get(k) for k in PLAYER_KEYS} for p in bs["elements"]])
    old = read(folder / "players.parquet")
    if old is not None:
        players = pd.concat([old[~old["id"].isin(players["id"])], players], ignore_index=True)
    written += write(players.sort_values("id").reset_index(drop=True), folder / "players.parquet")
    written += write(frame([{k: t.get(k) for k in TEAM_KEYS} for t in bs["teams"]]), folder / "teams.parquet")
    written += write(frame(bs["events"]), folder / "events.parquet")
    return written


def save_deadline(bs: dict, taken_at: pd.Timestamp | None = None) -> Path | None:
    """Every player as bootstrap-static shows them now (price, news, injury flag, ownership,
    this week's transfers), filed under the next gameweek, if its deadline is still to come. A
    later snapshot for the same gameweek replaces an earlier one: the last before the deadline
    is the one that matters. Returns the file, or None if there's no deadline ahead."""
    taken_at = taken_at or pd.Timestamp.now(tz="UTC")
    upcoming = next((ev for ev in bs["events"] if ev["is_next"]), None)
    if upcoming is None or pd.Timestamp(upcoming["deadline_time"]) <= taken_at:
        return None
    path = FPL / api.current_season(bs) / "deadlines" / f"gw{upcoming['id']:02d}.parquet"
    write(frame(bs["elements"]).assign(taken_at=taken_at), path)
    return path


def deadline_snapshots(season: str) -> dict[int, pd.DataFrame]:
    return {int(p.stem[2:]): pd.read_parquet(p) for p in sorted((FPL / season / "deadlines").glob("gw*.parquet"))}


SNAPSHOT_BRANCH = "deadline-snapshots"


def _taken_at(data: bytes) -> pd.Timestamp:
    t = pd.read_parquet(io.BytesIO(data), columns=["taken_at"])["taken_at"]
    return pd.Timestamp(t.max()) if len(t) else pd.Timestamp.min.tz_localize("UTC")


def pull_snapshots(remote: str = "origin") -> list[Path]:
    """Copy the scheduled Action's deadline snapshots into archive/. main only takes changes
    through pull requests, so the Action pushes them to the SNAPSHOT_BRANCH branch instead; this
    brings them into the working tree, and they reach main with the next pull request. Where both
    sides have a gameweek, the later snapshot wins. Returns the files written (none if offline)."""
    import subprocess

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True)

    if git("fetch", "--quiet", remote, SNAPSHOT_BRANCH).returncode != 0:
        return []                      # offline, no git, or the Action hasn't run yet
    listing = git("ls-tree", "-r", "--name-only", "FETCH_HEAD", "--", "archive/fpl").stdout.decode()
    written = []
    for name in listing.split():
        if "/deadlines/" not in name:
            continue
        theirs = git("show", f"FETCH_HEAD:{name}").stdout
        path = config.ROOT / name
        if path.exists() and (path.read_bytes() == theirs or _taken_at(path.read_bytes()) >= _taken_at(theirs)):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(theirs)
        written.append(path)
    return written


def save_prediction(csv_path: Path, season: str) -> bool:
    """A forecast `predict` saved before a deadline (data/predictions/<season>/gwNN_<model>.csv)."""
    return write(pd.read_csv(csv_path), PREDICTIONS / season / f"{csv_path.stem}.parquet")


def predictions(season: str) -> dict[str, pd.DataFrame]:
    """{"gw06_mlp": table, ...} for every archived forecast of `season`."""
    return {p.stem: pd.read_parquet(p) for p in sorted((PREDICTIONS / season).glob("gw*.parquet"))}


# ---------------------------------------------------------------- Polymarket

def _trim(event: dict) -> dict:
    out = {k: event[k] for k in EVENT_KEYS if k in event}
    out["markets"] = [{k: m[k] for k in MARKET_KEYS if k in m} for m in event.get("markets") or []]
    return out


def _tokens(group: list[dict]) -> list[str]:
    """The first ("Yes") token of every market of one match. Not only the ones `markets.py`
    prices today: which of two duplicate listings it keeps can change with Gamma's ordering."""
    tokens = (json.loads(m.get("clobTokenIds") or "[]") for e in group for m in e.get("markets") or [])
    return sorted({t[0] for t in tokens if t})


def save_polymarket(games: dict[str, list[dict]], found: pd.DataFrame) -> int:
    """Every played match's events (trimmed) and the price histories cached for its markets, one
    file of each per gameweek. `found` is `markets.match_fixtures(...)`: slug -> season, gw, fixture.
    A token whose cache file has gone keeps whatever the archive already had."""
    from xpfpl.data import markets
    windows = {"history_14d": markets.RAW_DIR / f"history_{markets.HISTORY_DAYS}d",
               "history_3h_5m": markets.RAW_DIR / f"history_{markets.FINE_HOURS}h_{markets.FINE_MINUTES}m"}
    written = 0
    for (season, gw), rows in found.groupby(["season", "gw"]):
        folder = POLYMARKET / season
        events, prices = [], []
        for r in rows.itertuples():
            group = games.get(r.slug, [])
            events += [{"season": season, "gw": int(gw), "fixture": int(r.fixture), "slug": r.slug,
                        "event_slug": e["slug"], "event": json.dumps(_trim(e), sort_keys=True)} for e in group]
            for token in _tokens(group):
                for window, cache in windows.items():
                    path = cache / f"{token}.json"
                    if path.exists():
                        points = json.loads(path.read_text(encoding="utf-8")) or [{"t": EMPTY, "p": float("nan")}]
                        prices += [{"slug": r.slug, "token": token, "window": window, "t": int(p["t"]),
                                    "p": float(p["p"])} for p in points]
        if not events:
            continue
        path = folder / "prices" / f"gw{int(gw):02d}.parquet"
        table = pd.DataFrame(prices, columns=["slug", "token", "window", "t", "p"])
        old = read(path)
        if old is not None:
            table = pd.concat([table, old[~old["token"].isin(table["token"])]], ignore_index=True)
        table = table.sort_values(["slug", "token", "window", "t"]).reset_index(drop=True)
        written += write(table, path)
        events = pd.DataFrame(events).sort_values(["fixture", "event_slug"]).reset_index(drop=True)
        written += write(events, folder / "events" / f"gw{int(gw):02d}.parquet")
    return written


def polymarket_games() -> dict[str, list[dict]]:
    """Every archived match: main slug -> its events, in the shape `markets.fetch_games` returns."""
    games: dict[str, list[dict]] = {}
    for path in sorted(POLYMARKET.glob("*/events/gw*.parquet")):
        for r in pd.read_parquet(path, columns=["slug", "event"]).itertuples():
            games.setdefault(r.slug, []).append(json.loads(r.event))
    return games


_PRICES: dict[tuple[str, str], list[dict]] | None = None


def price_history(window: str, token: str) -> list[dict] | None:
    """An archived price history ({"t", "p"} points), or None. `window` is the cache folder's
    name ("history_14d", "history_3h_5m"). All price files are loaded on the first call."""
    global _PRICES
    if _PRICES is None:
        _PRICES = {}
        files = sorted(POLYMARKET.glob("*/prices/gw*.parquet"))
        if files:
            table = pd.concat([pd.read_parquet(p, columns=["token", "window", "t", "p"]) for p in files])
            for (w, tok), g in table.groupby(["window", "token"], sort=False):
                _PRICES[(w, tok)] = [{"t": int(t), "p": float(p)} for t, p in zip(g["t"], g["p"]) if t != EMPTY]
    return _PRICES.get((window, token))


def save_outrights(table: pd.DataFrame) -> int:
    """Season-long market prices, the day's last snapshot per file."""
    if table is None or table.empty:
        return 0
    day = pd.to_datetime(table["snapshot"], utc=True).dt.strftime("%Y-%m-%d")
    written = 0
    for d, g in table.groupby(day):
        last = g[g["snapshot"] == g["snapshot"].max()]
        written += write(frame(last.to_dict("records")), POLYMARKET / "outrights" / f"{d}.parquet")
    return written


def outrights() -> pd.DataFrame | None:
    files = sorted((POLYMARKET / "outrights").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else None


# ---------------------------------------------------------------- backfill

def backfill(markets_too: bool = True) -> None:
    """Archive everything already on disk: vaastav's seasons (downloading any file that's
    missing), this season's cached API responses, the saved forecasts and the Polymarket caches
    (which needs the event listing, fetched once)."""
    from xpfpl.data import history

    for season in config.HISTORY_SEASONS:
        if not has_season(season):
            print(f"Archiving {season} from vaastav...")
            history.archive_vaastav(season)

    api_dir = config.RAW_DIR / "api"
    for season_dir in sorted(p for p in api_dir.glob("*") if p.is_dir()):
        latest = max((p for p in season_dir.glob("gw*") if (p / "bootstrap.json").exists()), default=None)
        if latest is None:
            continue
        bs = json.loads((latest / "bootstrap.json").read_text(encoding="utf-8"))
        fx = json.loads((latest / "fixtures.json").read_text(encoding="utf-8"))
        rows = [r for f in sorted(latest.glob("element_*.json")) for r in json.loads(f.read_text(encoding="utf-8"))]
        n = save_api_season(season_dir.name, bs, fx, rows)
        taken = pd.Timestamp((latest / "bootstrap.json").stat().st_mtime, unit="s", tz="UTC")
        snap = save_deadline(bs, taken)
        print(f"{season_dir.name}: {n} file(s) from the API cache ({latest.name})"
              + (f", deadline snapshot {snap.name}" if snap else ""))

    for csv in sorted(config.PREDICTIONS_DIR.glob("*/gw*_*.csv")):
        save_prediction(csv, csv.parent.name)

    if markets_too:
        from xpfpl.data import markets
        from xpfpl.data.history import load_matches
        print("Listing Polymarket's matches...")
        games = markets.fetch_games()
        found = markets.match_fixtures(markets.listing(games), load_matches())
        print(f"Polymarket: {save_polymarket(games, found)} file(s) for {len(found)} matches")
        if markets.OUTRIGHTS_PATH.exists():
            save_outrights(pd.read_parquet(markets.OUTRIGHTS_PATH))


def size_report() -> str:
    lines, total = [], 0
    for part in sorted(p for p in config.ARCHIVE_DIR.glob("*") if p.is_dir()):
        files = [f for f in part.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        total += size
        lines.append(f"  {part.name:12s} {size / 1e6:6.1f} MB in {len(files)} files")
    return "\n".join(lines + [f"  {'total':12s} {total / 1e6:6.1f} MB"])


def main(argv: list[str]) -> None:
    """`python -m xpfpl.data.archive deadline [--within-hours H]`: the pre-deadline snapshot, for
    the scheduled GitHub Action (only pandas, pyarrow and requests needed)."""
    if not argv or argv[0] != "deadline":
        raise SystemExit("usage: python -m xpfpl.data.archive deadline [--within-hours H]")
    within = float(argv[argv.index("--within-hours") + 1]) if "--within-hours" in argv else None
    bs = api.bootstrap()
    upcoming = next((ev for ev in bs["events"] if ev["is_next"]), None)
    if upcoming is None:
        print("No deadline ahead.")
        return
    left = pd.Timestamp(upcoming["deadline_time"]) - pd.Timestamp.now(tz="UTC")
    if within is not None and left > pd.Timedelta(hours=within):
        print(f"GW{upcoming['id']} deadline is {left} away: not snapshotting yet.")
        return
    path = save_deadline(bs)
    print(f"Saved {path}" if path else "The deadline has passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
