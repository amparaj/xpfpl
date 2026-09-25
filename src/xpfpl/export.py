"""The website's data: JSON files in web/public/data/ for the static site in web/.

The site looks back at the season. It never trains or plans; it reads these files. Live betting
odds come from odds.json on the `odds` branch, which a scheduled GitHub Action
(.github/workflows/odds-snapshot.yml) refreshes every 5 minutes.

  meta.json          season, gameweeks, clubs, the club ratings for the next GW ("Our Odds")
  players.json       every player as FPL shows them now, plus the saved forecast for the next GW
  gws/gwNN.json      each played gameweek: every player's stats with the model's xP, and the fixtures
  manager.json       one FPL team's season: points, rank, picks, transfers, chips, hindsight best XI
  markets.json       Polymarket odds at each FPL deadline since 2024-25 next to what happened,
                     anytime-scorer odds, how the odds did against our ratings, season markets
  market_history/<season>/gwNN.json
                     each played match's home/draw/away prices before the deadline (movement chart)
  accuracy.json      the model reports: validation, comparison, tuning and the live scorecard

`xpfpl export` writes them; `xpfpl publish` builds the site and pushes it to the gh-pages branch.
"""

import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from xpfpl import config, models
from xpfpl.data import api, archive

# The per-match columns the site shows (FPL's own names).
GW_COLUMNS = ["element", "fixture", "opponent_team", "was_home", "minutes", "total_points", "goals_scored",
              "assists", "clean_sheets", "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
              "yellow_cards", "red_cards", "saves", "bonus", "bps", "expected_goals", "expected_assists",
              "expected_goals_conceded", "defensive_contribution", "starts", "value", "selected",
              "transfers_balance"]
PLAYER_COLUMNS = ["id", "code", "web_name", "first_name", "second_name", "team", "element_type", "now_cost",
                  "selected_by_percent", "status", "news", "chance_of_playing_next_round", "total_points",
                  "minutes", "goals_scored", "assists", "clean_sheets", "bonus", "expected_goals",
                  "expected_assists", "defensive_contribution", "starts", "form", "points_per_game",
                  "cost_change_start"]
MARKET_COLUMNS = ["season", "gw", "fixture", "slug", "kickoff", "home_code", "away_code", "volume", "home_win",
                  "draw", "away_win", "over_1.5", "over_2.5", "over_3.5", "home_over_0.5", "home_over_1.5",
                  "away_over_0.5", "away_over_1.5", "btts", "lam_home", "lam_away"]


# ---------------------------------------------------------------- JSON

def _plain(v, digits: int):
    """A JSON-safe value: NaN -> null, numpy scalars -> Python, timestamps -> ISO text."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else round(float(v), digits)
    return v


def table(df: pd.DataFrame, digits: int = 4) -> dict[str, list]:
    """Column-wise: {"col": [values...]}, which is about half the size of a list of records."""
    return {str(c): [_plain(v, digits) for v in df[c].tolist()] for c in df.columns}


def _scrub(obj):
    """NaN and infinity -> null anywhere in a nested structure (the model reports contain some)."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_scrub(obj), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- xP per played gameweek

def _saved_forecasts(season: str, model: str) -> dict[int, pd.Series]:
    """GW -> xP per element, from the forecasts saved before each deadline (`model`'s if saved,
    otherwise any model's)."""
    out: dict[int, tuple[bool, pd.Series]] = {}
    for name, t in archive.predictions(season).items():
        gw, saved_model = int(name[2:4]), name.split("_", 1)[1]
        col = f"xp_{gw}"
        if col not in t or (gw in out and out[gw][0]):
            continue
        out[gw] = (saved_model == model, t.set_index("element")[col].rename(saved_model))
    return {gw: s for gw, (_, s) in out.items()}


def _in_sample(matches: pd.DataFrame, season: str, gws: list[int], model: str) -> pd.DataFrame:
    """(gw, element) -> xP rebuilt from the training frame, as the dashboard's review does. The
    model was refitted on this season too, so it's a little optimistic."""
    if not gws:
        return pd.DataFrame(columns=["gw", "element", "xp"])
    from xpfpl.features import build_training_frame
    frame = build_training_frame(matches)
    rows = frame[(frame["season"] == season) & frame["gw"].isin(gws)].copy()
    rows["xp"] = np.clip(models.load(model).predict(rows).astype(float), 0.0, None)
    return rows.groupby(["gw", "element"], as_index=False)["xp"].sum()


# ---------------------------------------------------------------- the files

def _meta(bs: dict, fixtures: list[dict], season: str, model: str, matches: pd.DataFrame, played: list[int]) -> dict:
    from xpfpl import style, teams
    from xpfpl.data import markets

    upcoming = next((ev for ev in bs["events"] if ev["is_next"]), None)
    ratings = {}
    if upcoming is not None:
        r = teams.latest(matches, season, upcoming["id"])
        ratings = {"gw": upcoming["id"], "mu": float(r["mu"].iloc[0]), "home": float(r["home"].iloc[0]),
                   "prior": list(teams.PROMOTED_PRIOR),
                   "clubs": {str(int(c)): [round(float(a), 4), round(float(d), 4)]
                             for c, a, d in zip(r.index, r["attack"], r["defence"])}}
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=config.ROOT, capture_output=True, text=True)
    repo = remote.stdout.strip().removesuffix(".git") if remote.returncode == 0 else None
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "model": model, "model_description": models.DESCRIPTIONS.get(model, model),
        "played": played, "next_gw": upcoming["id"] if upcoming else None,
        "next_deadline": upcoming["deadline_time"] if upcoming else None,
        "teams": [{"id": t["id"], "code": t["code"], "name": t["name"], "short": t["short_name"]} for t in bs["teams"]],
        "events": [{"id": e["id"], "deadline": e["deadline_time"], "finished": e["finished"],
                    "checked": e["data_checked"], "average": e["average_entry_score"],
                    "highest": e["highest_score"], "most_captained": e["most_captained"],
                    "top_element": e["top_element"]} for e in bs["events"]],
        "chips": [{"name": c["name"], "start": c["start_event"], "stop": c["stop_event"]} for c in bs["chips"]],
        "fixtures": table(pd.DataFrame([{"id": f["id"], "gw": f["event"], "kickoff": f["kickoff_time"],
                                         "home": f["team_h"], "away": f["team_a"], "home_score": f["team_h_score"],
                                         "away_score": f["team_a_score"], "home_fdr": f["team_h_difficulty"],
                                         "away_fdr": f["team_a_difficulty"]} for f in fixtures])),
        "club_colours": {c: [bg, style.CLUB_TEXT[c]] for c, bg in style.CLUB_COLOURS.items()},
        "polymarket_teams": markets.TEAM_CODES, "out_threshold": markets.OUT_THRESHOLD,
        "ratings_next": ratings, "repo": repo,
    }


def _players(bs: dict, next_forecast: pd.Series | None) -> dict:
    el = pd.DataFrame(bs["elements"])[PLAYER_COLUMNS]
    for col in ("selected_by_percent", "expected_goals", "expected_assists", "form", "points_per_game"):
        el[col] = pd.to_numeric(el[col], errors="coerce")
    el["now_cost"] = el["now_cost"] / 10
    el["forecast"] = el["id"].map(next_forecast) if next_forecast is not None else np.nan
    return table(el, digits=3)


def _gameweek(gw: int, rows: pd.DataFrame, fx: list[dict], xp: pd.Series, source: str,
              snapshot: pd.DataFrame | None) -> dict:
    rows = rows[[c for c in GW_COLUMNS if c in rows]].copy()
    for col in ("expected_goals", "expected_assists", "expected_goals_conceded"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["value"] = rows["value"] / 10
    rows["was_home"] = rows["was_home"].astype(str).str.lower().eq("true")
    # xP is per gameweek (summed over a double), so it goes on the player's first match row only.
    first = ~rows.duplicated("element")
    rows["xp"] = np.where(first, rows["element"].map(xp), np.nan)
    if snapshot is not None:                 # what FPL said about each player before this deadline
        s = snapshot.set_index("id")
        rows["pre_chance"] = rows["element"].map(s["chance_of_playing_next_round"])
        rows["pre_news"] = rows["element"].map(s["news"])
    fixtures = [{"id": f["id"], "kickoff": f["kickoff_time"], "home": f["team_h"], "away": f["team_a"],
                 "home_score": f["team_h_score"], "away_score": f["team_a_score"]}
                for f in fx if f["event"] == gw]
    return {"gw": gw, "xp_source": source, "players": table(rows.sort_values(["fixture", "element"]), digits=3),
            "fixtures": fixtures}


def _manager(team_id: int, bs: dict, season_rows: pd.DataFrame, played: list[int]) -> dict:
    from xpfpl import review

    entry, history, transfers = api.entry(team_id), api.entry_history(team_id), api.entry_transfers(team_id)
    gws = {}
    for h in history["current"]:
        gw = h["event"]
        if gw not in played:
            continue
        rows = season_rows[season_rows["round"] == gw]
        points = rows.groupby("element").agg(points=("total_points", "sum"), minutes=("minutes", "sum"))
        r = review.review_picks(team_id, gw, bs, points)
        best, plan = review.hindsight_best(r["picks"], gw, r["chip"])
        gws[str(gw)] = {
            "picks": [{"element": int(e), "slot": int(p.position), "multiplier": int(p.multiplier),
                       "captain": bool(p.is_captain), "vice": bool(p.is_vice_captain)}
                      for e, p in r["picks"].iterrows()],
            "chip": r["chip"], "auto_subs": r["auto_subs"],
            "best": round(float(best), 1), "best_xi": [int(p) for p in plan.lineups[gw]],
            "best_captain": int(plan.captains[gw]),
        }
    return {"team_id": team_id, "name": entry["name"], "started": entry.get("started_event"),
            "history": history["current"], "chips": history["chips"], "transfers": transfers, "gameweeks": gws}


def _markets(matches: pd.DataFrame) -> dict | None:
    from xpfpl import teams
    from xpfpl.data import markets
    from xpfpl.features import _poisson_win

    market, scorers = markets.load()
    if market is None or market.empty:
        return None
    m = market[[c for c in MARKET_COLUMNS if c in market] + [c for c in market if c.endswith(("_90m", "d"))
                                                              and c.startswith(("home_win_", "draw_", "away_win_"))]]
    # What happened: goals and xG per side.
    side = matches.groupby(["season", "fixture", "was_home"]).agg(
        goals=("team_h_score", "first"), goals_away=("team_a_score", "first"),
        xg=("expected_goals", lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1))).reset_index()
    home, away = side[side["was_home"]], side[~side["was_home"]]
    result = home[["season", "fixture", "goals", "goals_away", "xg"]].rename(
        columns={"goals": "goals_home", "xg": "xg_home"}).merge(
        away[["season", "fixture", "xg"]].rename(columns={"xg": "xg_away"}), on=["season", "fixture"], how="left")
    m = m.merge(result, on=["season", "fixture"], how="left")
    # Our Odds: the club ratings as they stood before each gameweek.
    ratings = teams.ratings(matches)
    parts = []
    for (season, gw), g in m.groupby(["season", "gw"]):
        r = ratings[(ratings["season"] == season) & (ratings["gw"] == gw)].set_index("team_code")
        gf, ga = teams.fixture_rates(r, g["home_code"], g["away_code"], np.ones(len(g)))
        parts.append(g.assign(ours_home=gf, ours_away=ga, ours_home_win=_poisson_win(gf, ga),
                              ours_away_win=_poisson_win(ga, gf)))
    m = pd.concat(parts).sort_values("kickoff")

    out = {"matches": table(m, digits=4), "accuracy": table(markets.accuracy(market, matches), digits=4)}
    if scorers is not None and len(scorers) and "code" in scorers:
        actual = matches.groupby(["season", "fixture", "code"], as_index=False).agg(
            goals=("goals_scored", "sum"), minutes=("minutes", "sum"))
        s = scorers.merge(actual, on=["season", "fixture", "code"], how="left")
        keep = ["season", "gw", "fixture", "slug", "player", "code", "team_code", "volume", "p_anytime",
                *[c for c in s if c.startswith("p_anytime_")], "goals", "minutes"]
        out["scorers"] = table(s[keep], digits=4)
    snaps = archive.outrights()
    if snaps is not None and len(snaps):
        last = snaps[snaps["snapshot"] == snaps["snapshot"].max()]
        out["outrights"] = {"snapshot": _plain(pd.Timestamp(last["snapshot"].iloc[0]), 0),
                            **table(last[["event", "outcome", "probability", "volume"]], digits=4)}
    return out


def _market_histories(market: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """Each played match's result prices up to the deadline, from the archive, per (season, gw):
    slug -> {"hourly"/"fine": {"home_win"/"draw"/"away_win": [[t, p], ...]}}, the shape of
    web/src/polymarket.ts's MatchHistory, for the movement chart."""
    from xpfpl.data import markets
    games = archive.polymarket_games()
    windows = {"hourly": f"history_{markets.HISTORY_DAYS}d", "fine": f"history_{markets.FINE_HOURS}h_{markets.FINE_MINUTES}m"}
    out: dict[tuple[str, int], dict] = {}
    for r in market[["season", "gw", "slug"]].dropna().itertuples():
        main = [e for e in games.get(r.slug, []) if e.get("slug") == r.slug]
        found = {}
        for name, m in markets._match_markets(main).items():
            token = json.loads(m.get("clobTokenIds") or "[]")
            if name in ("home_win", "draw", "away_win") and token:
                for key, window in windows.items():
                    points = archive.price_history(window, token[0]) or []
                    found.setdefault(key, {})[name] = [[p["t"], round(p["p"], 4)] for p in points]
        if found:
            out.setdefault((r.season, int(r.gw)), {})[r.slug] = found
    return out


def _accuracy(season: str) -> dict:
    def load(path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return {"validation": load(config.VALIDATION_PATH), "comparison": load(config.COMPARISON_PATH),
            "tuning": load(config.TUNING_PATH),
            "scorecard": load(config.PREDICTIONS_DIR / season / "scorecard.json")}


def export(team_id: int | None = None, model: str = config.MODEL, out: Path = config.SITE_DATA_DIR) -> list[Path]:
    from xpfpl.data import markets
    from xpfpl.data.history import load_matches

    bs, fx = api.bootstrap(), api.fixtures()
    season = api.current_season(bs)
    matches = load_matches()
    rows, _ = archive.season_tables(season) if archive.has_season(season) else (pd.DataFrame(columns=["round"]), None)
    played = sorted(int(g) for g in rows["round"].unique())

    forecasts = _saved_forecasts(season, model)
    missing = [g for g in played if g not in forecasts]
    if missing:
        print(f"Rebuilding xP for GW{', GW'.join(map(str, missing))} (no forecast was saved before those deadlines)...")
    rebuilt = _in_sample(matches, season, missing, model)
    snapshots = archive.deadline_snapshots(season)

    written = [_write(_meta(bs, fx, season, model, matches, played), out / "meta.json")]
    upcoming = next((ev["id"] for ev in bs["events"] if ev["is_next"]), None)
    written.append(_write(_players(bs, forecasts.get(upcoming)), out / "players.json"))
    for gw in played:
        if gw in forecasts:
            xp, source = forecasts[gw], f"forecast ({forecasts[gw].name}, saved before the deadline)"
        else:
            xp = rebuilt[rebuilt["gw"] == gw].set_index("element")["xp"]
            source = f"rebuilt ({model}, in-sample: no forecast was saved before this deadline)"
        written.append(_write(_gameweek(gw, rows[rows["round"] == gw], fx, xp, source, snapshots.get(gw)),
                              out / "gws" / f"gw{gw:02d}.json"))
    if team_id:
        print(f"Exporting FPL team {team_id}...")
        written.append(_write(_manager(team_id, bs, rows, played), out / "manager.json"))
    market = _markets(matches)
    if market:
        written.append(_write(market, out / "markets.json"))
        saved, _ = markets.load()
        for (market_season, gw), histories in _market_histories(saved).items():
            written.append(_write(histories, out / "market_history" / market_season / f"gw{gw:02d}.json"))
    written.append(_write(_accuracy(season), out / "accuracy.json"))
    return written


# ---------------------------------------------------------------- publishing

WEB_DIR = config.ROOT / "web"


def build_site() -> Path:
    """`npm run build` in web/ (installing packages the first time): the site in web/dist/."""
    npm = "npm.cmd" if os.name == "nt" else "npm"
    if not (WEB_DIR / "node_modules").exists():
        subprocess.run([npm, "ci"], cwd=WEB_DIR, check=True)
    subprocess.run([npm, "run", "build"], cwd=WEB_DIR, check=True)
    return WEB_DIR / "dist"


def publish(dist: Path, branch: str = "gh-pages", remote: str = "origin", url: str | None = None) -> None:
    """Push `dist` to `branch` as a single commit that replaces whatever was there, so the site's
    data never piles up in the repo's history. GitHub Pages serves that branch. `url` overrides
    the remote's URL."""
    (dist / ".nojekyll").write_text("", encoding="utf-8")         # serve files as they are
    url = url or subprocess.run(["git", "remote", "get-url", remote], cwd=config.ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as git_dir:
        git = ["git", f"--git-dir={git_dir}", f"--work-tree={dist}"]
        subprocess.run([*git, "init", "-q", "-b", branch], check=True)
        subprocess.run([*git, "config", "core.autocrlf", "false"], check=True)   # publish the files byte for byte
        for key in ("user.name", "user.email"):
            value = subprocess.run(["git", "config", key], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
            if value:
                subprocess.run([*git, "config", key, value], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run([*git, "commit", "-q", "-m", f"Publish the site ({stamp})"], check=True)
        subprocess.run([*git, "push", "-q", "--force", url, f"HEAD:{branch}"], check=True)
