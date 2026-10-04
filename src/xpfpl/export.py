"""The website's data: JSON files in web/public/data/ for the static site in web/.

The site looks back at the season. It never trains or plans; it reads these files. Live betting
odds come from odds.json on the `odds` branch, which a scheduled GitHub Action
(.github/workflows/odds-snapshot.yml) refreshes.

  meta.json          season, gameweeks, clubs, the club ratings for the next GW ("Our Odds"), and
                     every cup and European match of the season (`midweek`, data/cups.py)
  players.json       every player as FPL shows them now, plus the saved forecast for the next GW
  gws/gwNN.json      each played gameweek: every player's stats with the model's xP (and minutes in
                     the midweek cup or European match before it), and the fixtures
  matches/gwNN.json  each played gameweek's matches as they happened (data/matchstats.py): every
                     shot with where it was taken and where it went, xG and xG on target; the
                     momentum by minute; the team stats; each player's match stats. For the match
                     pages (Past Gameweeks -> a result)
  profiles.json      where each player plays (spatial.py): his shots binned into zones round the goal,
                     every shot, per-90 touches / box touches / final-third passes / chances / shots,
                     shot metrics, and the players with the most similar profile
  modelteam.json     the Model's Team: a paper FPL team run on the model's own advice, each week's
                     decision (saved before the deadline, with its Monte Carlo: the team's simulated
                     score, captain odds, chip odds) and what it scored
  next.json          the forecast saved for the next gameweek, for every week of its horizon, with
                     each player's midweek rotation group and the factor it put on his xP, and his
                     simulated range for the next gameweek (simulate.py: 10th/50th/90th percentile,
                     chances of 10+ and of 2 or fewer)
  myteam.json        My Team: the team for the gameweek in progress (until it's finished) or the
                     next one (saved from the dashboard's Plan Ahead, else last week's carried over;
                     after the deadline, the team locked in) and its outlook over the same horizon
                     (myteam.outlook). The numbers are public; before the deadline the players,
                     captains, chip and transfers are encrypted with the secret in the dashboard's
                     settings (`seal`), or left out without one. Nothing about results.
  markets.json       Polymarket odds at each FPL deadline since 2024-25 next to what happened,
                     anytime-scorer odds, how the odds did against our ratings, season markets
  market_history/<season>/gwNN.json
                     each played match's home/draw/away prices before the deadline (movement chart)
  accuracy.json      the model reports: validation, comparison, tuning, the live scorecard and the
                     midweek rotation factors

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
# Bumped when what an earlier season's files hold changes (profiles.json gained keepers' shots faced and
# FPL's match rows in 2): `_past_season` rebuilds a season whose meta.json has another version.
EXPORT_VERSION = 2

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

# The forecasts saved before each deadline, and the in-sample rebuild for weeks without one, live
# in review.py (the dashboard's Gameweek Review and My Season use the same ones).


# ---------------------------------------------------------------- the files

def _meta(bs: dict, fixtures: list[dict], season: str, model: str, matches: pd.DataFrame, played: list[int]) -> dict:
    from xpfpl import style, teams
    from xpfpl.data import cups, markets

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
        "midweek": table(cups.matches(season)),
        "seasons": browsable_seasons(season),
    }


def browsable_seasons(season: str) -> list[str]:
    """The seasons the site can look back at (oldest first, this one last): those with match data
    and an archived FPL season. Earlier ones are exported to seasons/<season>/ by `_past_season`."""
    from xpfpl.data import matchstats
    return [s for s in matchstats.seasons() if s < season and archive.has_season(s)] + [season]


def _past_season(season: str, model: str, out: Path) -> list[Path]:
    """A finished season's copy of the pages that look back, in seasons/<season>/: meta.json and
    players.json (that season's clubs, players and fixtures: FPL renumbers ids every season),
    every gameweek's player rows (gws/) and match data (matches/), and the player profiles.
    xP is the forecast saved before each deadline where there is one (from 2026-27 on)."""
    from xpfpl import review, style
    from xpfpl.data import cups, markets, matchstats

    root = out / "seasons" / season
    # A finished season doesn't change: rebuilt only when something it's made from has (~35 s).
    sources = [archive.FPL / season, matchstats.ARCHIVE / season, cups.ARCHIVE / season,
               config.ARCHIVE_DIR / "predictions" / season]
    newest = max((f.stat().st_mtime for d in sources if d.exists() for f in d.rglob("*") if f.is_file()), default=0.0)
    if ((root / "meta.json").exists() and (root / "meta.json").stat().st_mtime > newest
            and json.loads((root / "meta.json").read_text(encoding="utf-8")).get("export_version") == EXPORT_VERSION):
        return sorted(root.rglob("*.json"))

    rows, people = archive.season_tables(season)
    bs, fx = archive.season_bootstrap(season), archive.fixture_list(season)
    played = sorted(int(g) for g in rows["round"].unique())
    first_kickoff = {}
    for f in fx:
        if f["event"] is not None and f["kickoff_time"]:
            first_kickoff[f["event"]] = min(first_kickoff.get(f["event"], f["kickoff_time"]), f["kickoff_time"])
    events = archive.read(archive.FPL / season / "events.parquet")
    by_event = events.set_index("id") if events is not None else None
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": season, "past": True, "export_version": EXPORT_VERSION,
        "model": model, "model_description": models.DESCRIPTIONS.get(model, model),
        "played": played, "next_gw": None, "next_deadline": None,
        "teams": [{"id": t["id"], "code": t["code"], "name": t["name"], "short": t["short_name"]} for t in bs["teams"]],
        # FPL's own gameweek summary where the season was archived from the API; the deadline otherwise
        # stands in as the first kick-off.
        "events": [{"id": g,
                    "deadline": (by_event.at[g, "deadline_time"] if by_event is not None and g in by_event.index
                                 else first_kickoff.get(g)),
                    "finished": True, "checked": True,
                    **{k: (_plain(by_event.at[g, src], 0) if by_event is not None and g in by_event.index
                           and src in by_event else None)
                       for k, src in (("average", "average_entry_score"), ("highest", "highest_score"),
                                      ("most_captained", "most_captained"), ("top_element", "top_element"))}}
                   for g in played],
        "chips": [],
        "fixtures": table(pd.DataFrame([{"id": f["id"], "gw": f["event"], "kickoff": f["kickoff_time"],
                                         "home": f["team_h"], "away": f["team_a"], "home_score": f["team_h_score"],
                                         "away_score": f["team_a_score"], "home_fdr": f["team_h_difficulty"],
                                         "away_fdr": f["team_a_difficulty"]} for f in fx])),
        "club_colours": {c: [bg, style.CLUB_TEXT[c]] for c, bg in style.CLUB_COLOURS.items()},
        "polymarket_teams": markets.TEAM_CODES, "out_threshold": markets.OUT_THRESHOLD,
        "ratings_next": {}, "repo": None,
        "midweek": table(cups.matches(season)),
    }
    written = []
    el = people.reindex(columns=PLAYER_COLUMNS)
    for col in ("selected_by_percent", "expected_goals", "expected_assists", "form", "points_per_game", "now_cost"):
        el[col] = pd.to_numeric(el[col], errors="coerce")
    el["now_cost"] = el["now_cost"] / 10
    el["forecast"] = np.nan
    written.append(_write(table(el, digits=3), root / "players.json"))
    forecasts = review.saved_forecasts(season, model)
    for gw in played:
        if gw in forecasts:
            xp, source = forecasts[gw], f"forecast ({forecasts[gw].name}, saved before the deadline)"
        else:
            xp, source = pd.Series(dtype=float), "not available: no forecast was saved before this deadline"
        written.append(_write(_gameweek(gw, rows[rows["round"] == gw], fx, xp, source, None,
                                        _midweek_minutes(season, gw, people)),
                              root / "gws" / f"gw{gw:02d}.json"))
        happened = _matches(season, gw, people)
        if happened:
            written.append(_write(happened, root / "matches" / f"gw{gw:02d}.json"))
    profiles = _profiles(season, people)
    if profiles:
        written.append(_write(profiles, root / "profiles.json"))
    written.append(_write(meta, root / "meta.json"))        # last: it marks the season as done
    return written


def _players(bs: dict, next_forecast: pd.Series | None) -> dict:
    el = pd.DataFrame(bs["elements"])[PLAYER_COLUMNS]
    for col in ("selected_by_percent", "expected_goals", "expected_assists", "form", "points_per_game"):
        el[col] = pd.to_numeric(el[col], errors="coerce")
    el["now_cost"] = el["now_cost"] / 10
    el["forecast"] = el["id"].map(next_forecast) if next_forecast is not None else np.nan
    return table(el, digits=3)


def _gameweek(gw: int, rows: pd.DataFrame, fx: list[dict], xp: pd.Series, source: str,
              snapshot: pd.DataFrame | None, midweek: pd.Series | None = None) -> dict:
    """`midweek`: minutes per element in the cup and European matches before this gameweek."""
    rows = rows[[c for c in GW_COLUMNS if c in rows]].copy()
    for col in ("expected_goals", "expected_assists", "expected_goals_conceded"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["value"] = rows["value"] / 10
    rows["was_home"] = rows["was_home"].astype(str).str.lower().eq("true")
    # xP is per gameweek (summed over a double), so it goes on the player's first match row only.
    first = ~rows.duplicated("element")
    rows["xp"] = np.where(first, rows["element"].map(xp), np.nan)
    if midweek is not None and len(midweek):
        rows["midweek_minutes"] = np.where(first, rows["element"].map(midweek), np.nan)
    if snapshot is not None:                 # what FPL said about each player before this deadline
        s = snapshot.set_index("id")
        rows["pre_chance"] = rows["element"].map(s["chance_of_playing_next_round"])
        rows["pre_news"] = rows["element"].map(s["news"])
    fixtures = [{"id": f["id"], "kickoff": f["kickoff_time"], "home": f["team_h"], "away": f["team_a"],
                 "home_score": f["team_h_score"], "away_score": f["team_a_score"]}
                for f in fx if f["event"] == gw]
    return {"gw": gw, "xp_source": source, "players": table(rows.sort_values(["fixture", "element"]), digits=3),
            "fixtures": fixtures}


SHOT_COLUMNS = ["minute", "added_time", "is_home", "element", "outcome", "situation", "body_part", "xg", "xgot",
                "start_x", "start_y", "goal_mouth_y", "goal_mouth_z", "inferred"]


def _matches(season: str, gw: int, people: pd.DataFrame | None) -> dict | None:
    """The gameweek's matches as they happened, per FPL fixture id: team stats as {stat: [home,
    away]}, shots, momentum and player stats. Players are FPL element ids (null where no shooter is
    known; `inferred` marks a shooter worked out from the match stats, matchstats.attribute; those taken
    from FotMob's match page, matchstats.from_fotmob, count as recorded)."""
    from xpfpl.data import matchstats
    tables = matchstats.gameweek(season, gw)
    if "shots" not in tables and "matches" not in tables:
        return None
    element = people.set_index("code")["id"] if people is not None else pd.Series(dtype=int)
    out = {}
    for fixture in sorted({int(f) for df in tables.values() for f in df["fixture"]}):
        part = {k: df[df["fixture"] == fixture] for k, df in tables.items()}
        match = {}
        if "matches" in part:
            match["stats"] = {r.stat: [_plain(r.home, 3), _plain(r.away, 3)] for r in part["matches"].itertuples()}
        if "shots" in part:
            s = part["shots"].sort_values(["minute", "added_time", "shot_index"], na_position="first")
            inferred = s["inferred"].fillna(False).astype(bool) if "inferred" in s else False
            s = s.assign(element=s["code"].map(element), inferred=inferred)
            match["shots"] = table(s[SHOT_COLUMNS], digits=4)
        if "momentum" in part:
            match["momentum"] = table(part["momentum"].sort_values("minute")[["minute", "value"]], digits=1)
        if "players" in part:
            pl = part["players"]
            pl = pl.assign(element=pl["code"].map(element)).dropna(subset=["element"])
            keep = ["element", *[c for c in matchstats.PLAYER_STATS if c in pl]]
            match["players"] = table(pl[keep].astype({"element": int}), digits=3)
        out[str(fixture)] = match
    return {"gw": gw, "source": "FotMob via FPL-Core-Insights", "fixtures": out}


def _profiles(season: str, people: pd.DataFrame | None, matches: pd.DataFrame | None = None) -> dict | None:
    """Every player's profile (spatial.py) by FPL element id, with position averages for context,
    his shots, his matches (FPL's defensive contribution, saves, goals conceded), every shot on
    target with the keeper who faced it, and his most similar players (on his position's metrics)."""
    from xpfpl import spatial
    prof = spatial.profiles(season)
    if prof.empty or people is None:
        return None
    if matches is None:
        from xpfpl.data.history import load_matches
        matches = load_matches()
    ids = people.drop_duplicates("code").set_index("code")
    position = ids["element_type"]
    element = ids["id"]
    games = spatial.games(season, matches)
    prof = prof.join(spatial.dc_summary(games, position)).join(ids[["id", "element_type"]], how="inner")
    similar = spatial.similar(prof.drop(columns=["id", "element_type"]), prof["element_type"])
    enough = prof[prof["minutes"] >= spatial.MIN_MINUTES]
    metrics = [c for c in [*spatial.similarity_metrics(), *spatial.SHOT_METRICS, "shot_distance", "dc_p90",
                           "conceded_p90", "goals_prevented"] if c in prof]
    metrics = list(dict.fromkeys(metrics))
    averages = {str(int(pos)): {c: _plain(g[c].mean(), 3) for c in metrics} for pos, g in enough.groupby("element_type")}
    s = spatial.shots(season)
    s = s.assign(element=s["code"].map(element)).dropna(subset=["element"])
    shot_cols = ["element", "gw", "fixture", "minute", "start_x", "start_y", "xg", "outcome", "situation", "body_part"]
    faced = spatial.keeper_shots(season, position)
    faced = faced.assign(element=faced["keeper"].map(element), shooter=faced["code"].map(element)).dropna(subset=["element"])
    faced_cols = ["element", "shooter", "gw", "fixture", "minute", "goal_mouth_y", "goal_mouth_z", "xg", "xgot", "outcome"]
    games = games.assign(element=games["code"].map(element)).dropna(subset=["element"]).drop(columns=["code"])
    return {
        "zones": table_zones(spatial.zones()), "min_minutes": spatial.MIN_MINUTES, "min_shots": spatial.MIN_SHOTS,
        "dc_threshold": {str(k): v for k, v in spatial.DC_THRESHOLD.items()},
        "players": table(prof.rename(columns={"id": "element"}).reset_index(drop=True)
                         .drop(columns=["element_type"]), digits=3),
        "averages": averages,
        "similar": {str(int(element[c])): [[int(element[o]), v] for o, v in found if o in element.index]
                    for c, found in similar.items() if c in element.index},
        "shots": table(s[shot_cols].astype({"element": int}), digits=3),
        "faced": table(faced[faced_cols].astype({"element": int}), digits=3),
        "games": table(games.astype({"element": int}), digits=3),
    }


def table_zones(z: pd.DataFrame) -> list[list[float]]:
    """Zone bounds as [zone, x0, x1, y0, y1] rows."""
    return [[int(r.zone), r.x0, r.x1, r.y0, r.y1] for r in z.itertuples()]


def _next_gameweek(bs: dict, season: str, model: str) -> dict | None:
    """The forecast saved for the next gameweek (`model`'s if saved, otherwise any model's): xP
    per player for every gameweek of its horizon, plus minutes and market odds where it has them."""
    upcoming = next((ev for ev in bs["events"] if ev["is_next"]), None)
    if upcoming is None:
        return None
    gw = upcoming["id"]
    found = archive.saved_forecast(season, gw, model)
    if found is None:
        return None
    key, t, gws = found
    # Outcome bands and the horizon's range came later than the saved forecasts' first runs: an
    # older forecast takes them from its simulations, where they're still on disk (simulate.py).
    if "p_3_5" not in t:
        from xpfpl import simulate
        draws = simulate.load(season, gw, key.split("_", 1)[1])
        if draws is not None and gw in draws.gameweeks:
            extra = simulate.summary(draws, gw).drop(columns=[c for c in t if c != "element"], errors="ignore")
            t = t.merge(extra, left_on="element", right_index=True, how="left")
    keep = ["element", *[f"xp_{g}" for g in gws], "xp_total",
            *[c for c in ("xmins", "p_play", "p_full", "mkt_anytime", "market_out", "rotation", "rotation_factor",
                          "pts_p10", "pts_p50", "pts_p90", "p_haul", "p_6_9", "p_3_5", "p_blank",
                          "total_p10", "total_p50", "total_p90", "xp_sd", "confidence",
                          "avail", "news_rule", "press", "back") if c in t]]
    return {"gw": gw, "deadline": upcoming["deadline_time"], "model": key.split("_", 1)[1], "gameweeks": gws,
            "players": table(t[keep], digits=3)}


SEAL_ITERATIONS = 600_000     # PBKDF2 rounds: ~0.5 s to unlock in a browser, slow to guess offline


def seal(obj, secret: str) -> dict:
    """`obj` as JSON, encrypted with a key derived from `secret` (PBKDF2-SHA256 -> AES-256-GCM), in
    the form web/src/seal.ts opens with the browser's WebCrypto. Only the secret opens it."""
    import base64
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(hashes.SHA256(), 32, salt, SEAL_ITERATIONS).derive(secret.strip().encode("utf-8"))
    data = AESGCM(key).encrypt(iv, json.dumps(_scrub(obj), separators=(",", ":")).encode("utf-8"), None)
    b64 = lambda b: base64.b64encode(b).decode("ascii")   # noqa: E731
    return {"v": 1, "iterations": SEAL_ITERATIONS, "salt": b64(salt), "iv": b64(iv), "data": b64(data)}


def _my_team(bs: dict, season: str, model: str) -> dict | None:
    """My Team (myteam.py): the team for the gameweek in progress (until FPL marks it finished)
    or the next one, and its outlook from the forecast saved for it. None without a team id in
    the dashboard's settings, a team, or a forecast.

    The numbers (forecast, likely range, horizon total, what the transfers are worth) are public.
    Before the deadline the rest (the players, XI, captains, chip, transfers) is `sealed` with
    the secret in the dashboard's settings, or left out without one. After the deadline FPL shows
    the team anyway, so it's `private` in plain text."""
    from xpfpl import myteam
    team_id, gw = myteam.saved_team_id(), myteam.showing_gameweek(bs)
    if team_id is None or gw is None:
        return None
    found = myteam.saved_forecast(season, gw, model)
    team = myteam.shown_team(team_id, bs) if found else None
    if team is None:
        return None
    players, gws, draws, used = found
    out = myteam.outlook(team, players, gws, draws)
    locked = myteam.deadline_passed(bs, gw)
    public = {
        "gw": gw, "deadline": next(e["deadline_time"] for e in bs["events"] if e["id"] == gw), "locked": locked,
        "model": used, "team_name": api.entry(team_id)["name"], "gameweeks": out["gameweeks"], "sims": out["sims"],
        "weeks": [{k: w[k] for k in ("gw", "xp", "points", "target", "p_target") if k in w} for w in out["weeks"]],
        "total": {"xp": out["total"]["xp"], "points": out["total"]["points"]}, "against": out["against"],
    }
    private = {"source": team["source"], "saved_at": team["saved_at"], "chip": team["chip"],
               "transfers": team["transfers"], "hits": team["hits"], "bank": team["bank"], "settings": team.get("settings"),
               "weeks": out["weeks"], "captains": out["captains"], "players": table(out["players"], digits=3)}
    secret = myteam.site_secret()
    if locked:
        return public | {"private": private, "sealed": None}
    return public | {"private": None, "sealed": seal(private, secret) if secret else None}


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

    from xpfpl.data import bookmakers
    accuracy = markets.accuracy(market, matches, bookmakers.market(matches))
    out = {"matches": table(m, digits=4), "accuracy": table(accuracy, digits=4)}
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


def _midweek_minutes(season: str, gw: int, people: pd.DataFrame | None) -> pd.Series | None:
    """Minutes per element in the cup and European matches before `gw` (data/cups.py)."""
    from xpfpl.data import cups
    return None if people is None else cups.player_minutes_before(season, gw, people)


def _accuracy(season: str, model: str) -> dict:
    def load(path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    from xpfpl import eda, robustness
    from xpfpl.data import cups
    rotation = load(cups.FACTORS_PATH) or {}
    return {"validation": load(config.VALIDATION_PATH), "comparison": load(config.COMPARISON_PATH),
            "tuning": load(config.TUNING_PATH), "robustness": robustness.site_summary(robustness.load_report()),
            "scorecard": load(config.PREDICTIONS_DIR / season / "scorecard.json"),
            "rotation": rotation.get(model), "eda": load(eda.PATH)}


def _prices(bs: dict) -> dict:
    """The price tracker (data/pricewatch.py): the hourly Action's log if it can be read, else
    today's prices and this gameweek's moves from `bs`. On github.io the page reads the Action's
    file from the `prices` branch directly; this copy is the fallback, and what a local build shows."""
    from xpfpl.data import pricewatch
    state = pricewatch.remote()
    if state and state.get("season") == api.current_season(bs):
        return state
    return pricewatch.update(None, bs)[0]


def _news(bs: dict) -> dict:
    """The team news (data/news.py): the hourly Action's news.json with today's FPL news and, when
    due, a fresh read of the press conferences. On github.io the page reads the Action's file from
    the `news` branch directly; this copy is the fallback, and what a local build shows."""
    from xpfpl.data import news
    return news.current(bs)


def export(model: str = config.MODEL, out: Path = config.SITE_DATA_DIR) -> list[Path]:
    from xpfpl.data import markets
    from xpfpl.data.history import load_matches

    bs, fx = api.bootstrap(), api.fixtures()
    season = api.current_season(bs)
    matches = load_matches()
    rows, people = archive.season_tables(season) if archive.has_season(season) else (pd.DataFrame(columns=["round"]), None)
    played = sorted(int(g) for g in rows["round"].unique())

    from xpfpl import review
    forecasts = review.saved_forecasts(season, model)
    missing = [g for g in played if g not in forecasts]
    if missing:
        print(f"Rebuilding xP for GW{', GW'.join(map(str, missing))} (no forecast was saved before those deadlines)...")
    rebuilt = review.rebuilt_forecasts(matches, season, missing, model)
    snapshots = archive.deadline_snapshots(season)

    written = [_write(_meta(bs, fx, season, model, matches, played), out / "meta.json")]
    upcoming = next((ev["id"] for ev in bs["events"] if ev["is_next"]), None)
    written.append(_write(_players(bs, forecasts.get(upcoming)), out / "players.json"))
    xp_by_gw = {}
    for gw in played:
        if gw in forecasts:
            xp, source = forecasts[gw], f"forecast ({forecasts[gw].name}, saved before the deadline)"
        else:
            xp = rebuilt[rebuilt["gw"] == gw].set_index("element")["xp"]
            source = f"rebuilt ({model}, in-sample: no forecast was saved before this deadline)"
        xp_by_gw[gw] = xp
        written.append(_write(_gameweek(gw, rows[rows["round"] == gw], fx, xp, source, snapshots.get(gw),
                                        _midweek_minutes(season, gw, people)),
                              out / "gws" / f"gw{gw:02d}.json"))
        happened = _matches(season, gw, people)
        if happened:
            written.append(_write(happened, out / "matches" / f"gw{gw:02d}.json"))
    profiles = _profiles(season, people, matches)
    if profiles:
        written.append(_write(profiles, out / "profiles.json"))
    from xpfpl import modelteam
    written.append(_write(modelteam.season_record(season, rows, bs, xp_by_gw), out / "modelteam.json"))
    ahead = _next_gameweek(bs, season, model)
    if ahead:
        written.append(_write(ahead, out / "next.json"))
    mine = _my_team(bs, season, model)
    if mine:
        written.append(_write(mine, out / "myteam.json"))
    else:                                    # don't leave a finished week's plan on the site
        (out / "myteam.json").unlink(missing_ok=True)
    market = _markets(matches)
    if market:
        written.append(_write(market, out / "markets.json"))
        saved, _ = markets.load()
        for (market_season, gw), histories in _market_histories(saved).items():
            written.append(_write(histories, out / "market_history" / market_season / f"gw{gw:02d}.json"))
    written.append(_write(_accuracy(season, model), out / "accuracy.json"))
    written.append(_write(_prices(bs), out / "prices.json"))
    written.append(_write(_news(bs), out / "news.json"))
    for past in browsable_seasons(season)[:-1]:
        written += _past_season(past, model, out)
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
