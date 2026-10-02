"""Live player prices: who has just risen or fallen, and who is closest to the next move.

FPL changes prices overnight as net transfers build up. Since 2026-27, bootstrap-static says how
far each player is towards his next change, in FPL's own numbers:

  price_change_percent       progress to the next move: +100 rises, -100 falls
  price_change_hourly_rate   how fast that is moving, in points of progress an hour
  price_change_projections   the progress expected at the next three price updates (offset 0, 1,
                             2), each with a `likelihood` from -5 (a fall all but certain) to +5
  price_change_locked_until  a new player's price can't move before this time

The API only ever shows the present, so the history has to be kept as it happens. A scheduled
GitHub Action (.github/workflows/price-snapshot.yml) runs `update` every hour and force-pushes
the result to the `prices` branch as one commit:

  prices.json            every player now, with the last HISTORY_HOURS of his progress, and every
                         price change seen this season (time = the hourly run that saw it)
  daily/<date>.parquet   one full snapshot a day (the first run after DAILY_HOUR_UTC), so FPL's
                         projections can later be scored against the moves that followed

The website reads prices.json straight from the branch (raw.githubusercontent.com, like the
odds); the dashboard reads it from there too (`remote`) and fills in today's numbers from its own
bootstrap call. `pull()` (run by `xpfpl fetch`) copies the change log and the daily snapshots into
archive/prices/<season>/ so they're kept with everything else.

No torch here: the Action installs only pandas, pyarrow and requests.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from xpfpl import config
from xpfpl.data import api

BRANCH = "prices"
FILE = "prices.json"
HISTORY_HOURS = 72
DAILY_HOUR_UTC = 22
ARCHIVE = config.ARCHIVE_DIR / "prices"
FIELDS = ["id", "code", "name", "team", "team_code", "pos", "cost", "change_event", "change_start",
          "selected", "net_event", "pct", "rate", "proj", "lik", "locked_until", "status"]


def _num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def players(bs: dict) -> pd.DataFrame:
    """One row per player: price (tenths), this GW's and the season's change, ownership (%),
    net transfers this GW and FPL's progress towards the next change (NaN where it's not given)."""
    e = pd.DataFrame(bs["elements"])
    team_code = {t["id"]: t["code"] for t in bs["teams"]}
    projections = e.get("price_change_projections", pd.Series([None] * len(e)))

    def proj(rows, key):
        out = []
        for row in rows:
            row = sorted(row or [], key=lambda p: p.get("offset", 0))[:3]
            out.append([None if p.get(key) is None else float(p[key]) for p in row])
        return out

    return pd.DataFrame({
        "id": e["id"].astype(int),
        "code": e["code"].astype(int),
        "name": e["web_name"],
        "team": e["team"].astype(int),
        "team_code": e["team"].map(team_code).astype(int),
        "pos": e["element_type"].astype(int),
        "cost": e["now_cost"].astype(int),
        "change_event": _num(e["cost_change_event"]).astype(int),
        "change_start": _num(e["cost_change_start"]).astype(int),
        "selected": _num(e["selected_by_percent"]),
        "net_event": (_num(e["transfers_in_event"]) - _num(e["transfers_out_event"])).astype(int),
        "pct": pd.to_numeric(e.get("price_change_percent"), errors="coerce"),
        "rate": pd.to_numeric(e.get("price_change_hourly_rate"), errors="coerce"),
        "proj": proj(projections, "projected_percent"),
        "lik": [[None if v is None else int(v) for v in row] for row in proj(projections, "likelihood")],
        "locked_until": e.get("price_change_locked_until", pd.Series([None] * len(e))),
        "status": e["status"],
    })


def _events(bs: dict) -> tuple[int | None, int | None, str | None]:
    current = next((ev["id"] for ev in bs["events"] if ev["is_current"]), None)
    upcoming = next((ev for ev in bs["events"] if ev["is_next"]), None)
    return current, (upcoming or {}).get("id"), (upcoming or {}).get("deadline_time")


def _plain(v):
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def update(previous: dict | None, bs: dict, now: pd.Timestamp | None = None) -> tuple[dict, pd.DataFrame | None]:
    """The new prices.json from the `previous` one and bootstrap-static `bs`, and the day's
    snapshot if one is due (else None).

    A change is any player whose price differs from the previous run's. The first run of a
    season has nothing to compare with, so it lists this gameweek's moves (`cost_change_event`)
    without a time."""
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").floor("min")
    season = api.current_season(bs)
    current, upcoming, deadline = _events(bs)
    cur = players(bs)
    same = bool(previous) and previous.get("season") == season
    old = {p["id"]: p for p in previous["players"]} if same else {}
    changes = list(previous.get("changes", [])) if same else []

    if old:
        for p in cur.itertuples():
            before = old.get(p.id)
            if before is not None and before["cost"] != p.cost:
                changes.append({"t": now.isoformat(), "gw": current, "id": p.id, "code": p.code, "name": p.name,
                                "team_code": p.team_code, "pos": p.pos, "from": int(before["cost"]), "to": int(p.cost)})
    else:
        for p in cur[cur["change_event"] != 0].itertuples():
            changes.append({"t": None, "gw": current, "id": p.id, "code": p.code, "name": p.name,
                            "team_code": p.team_code, "pos": p.pos, "from": int(p.cost - p.change_event), "to": int(p.cost)})

    # Progress over the last HISTORY_HOURS, one column per run, shared by every player.
    times = list(previous.get("times", [])) if same else []
    # A second run in the same minute replaces that minute's column rather than adding one.
    keep = [i for i, t in enumerate(times)
            if now - pd.Timedelta(hours=HISTORY_HOURS) < pd.Timestamp(t) < now]
    times = [times[i] for i in keep] + [now.isoformat()]
    rows = []
    for p in cur.to_dict("records"):
        before = old.get(p["id"], {}).get("h", [])
        history = [before[i] if i < len(before) else None for i in keep] + [_plain(p["pct"])]
        rows.append({**{k: _plain(p[k]) for k in FIELDS}, "h": history})

    daily = None
    last_daily = previous.get("last_daily") if same else None
    if now.hour >= DAILY_HOUR_UTC and last_daily != now.date().isoformat():
        daily = cur.assign(proj=cur["proj"].map(json.dumps), lik=cur["lik"].map(json.dumps), taken_at=now)
        last_daily = now.date().isoformat()

    out = {"updated": now.isoformat(), "season": season, "gw_current": current, "gw_next": upcoming,
           "next_deadline": deadline, "times": times, "players": rows, "changes": changes,
           "last_daily": last_daily}
    return out, daily


def run(folder: Path, bs: dict | None = None) -> list[Path]:
    """`update` the prices.json in `folder` (created if missing) and write any daily snapshot."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FILE
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    state, daily = update(previous, bs or api.bootstrap())
    path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    written = [path]
    if daily is not None:
        out = folder / "daily" / f"{state['last_daily']}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(out, compression="zstd", index=False)
        written.append(out)
    return written


# ---------------------------------------------------------------- reading it back

def _repo() -> str | None:
    """'owner/repo' of the git remote `origin`, for raw.githubusercontent.com URLs."""
    import re
    import subprocess
    got = subprocess.run(["git", "remote", "get-url", "origin"], cwd=config.ROOT, capture_output=True, text=True)
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", got.stdout or "")
    return m.group(1) if m else None


def remote(timeout: float = 15) -> dict | None:
    """The Action's latest prices.json from the `prices` branch, or None (offline, no branch yet)."""
    repo = _repo()
    if not repo:
        return None
    try:
        resp = requests.get(f"https://raw.githubusercontent.com/{repo}/{BRANCH}/{FILE}", timeout=timeout)
        return resp.json() if resp.ok else None
    except (requests.RequestException, ValueError):
        return None


def change_log(state: dict | None) -> pd.DataFrame:
    """The price changes in a prices.json, newest first, with `t` as a timestamp (NaT if unknown)."""
    cols = ["t", "gw", "id", "code", "name", "team_code", "pos", "from", "to"]
    log = pd.DataFrame((state or {}).get("changes", []), columns=cols)
    log["t"] = pd.to_datetime(log["t"], utc=True, errors="coerce")
    return log.sort_values(["t", "gw"], ascending=False, na_position="last", ignore_index=True)


def pull(remote_name: str = "origin") -> list[Path]:
    """Copy the `prices` branch's change log and daily snapshots into archive/prices/<season>/.
    Returns the files written (none if offline or the Action hasn't run yet)."""
    import subprocess
    from xpfpl.data import archive

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True)

    if git("fetch", "--quiet", remote_name, BRANCH).returncode != 0:
        return []
    shown = git("show", f"FETCH_HEAD:{FILE}")
    if shown.returncode != 0:
        return []
    state = json.loads(shown.stdout.decode("utf-8"))
    folder = ARCHIVE / state["season"]
    written = []
    log = change_log(state)
    if len(log) and archive.write(log, folder / "changes.parquet"):
        written.append(folder / "changes.parquet")
    listing = git("ls-tree", "-r", "--name-only", "FETCH_HEAD", "--", "daily").stdout.decode()
    for name in listing.split():
        path = folder / name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git("show", f"FETCH_HEAD:{name}").stdout)
        written.append(path)
    return written


def archived_changes(season: str) -> pd.DataFrame:
    """The change log archived by `pull` (empty if none)."""
    path = ARCHIVE / season / "changes.parquet"
    return pd.read_parquet(path) if path.exists() else change_log(None)


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "update":
        for path in run(Path(argv[1])):
            print(f"wrote {path}")
    else:
        raise SystemExit("usage: python -m xpfpl.data.pricewatch update <folder>")


if __name__ == "__main__":
    main(sys.argv[1:])
