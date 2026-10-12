"""The gameweek in progress, or just played, against the forecast saved before its deadline.

From a deadline until the next one, FPL's `is_current` gameweek is the one being played. For it:
- per player, the points so far (FPL's live `explain`: the points from each scoring item), how
  far they are from his xP and why, split exactly into three parts (`decompose`): **minutes**
  (he played or didn't, against the chance the forecast gave him), **finishing** (goals and
  assists beyond or short of his xG and xA in the match) and **the rest** (clean sheets, bonus,
  defensive contribution, cards, and how good his chances were against what was expected); a few
  reasons in words (`reasons`); whether he landed inside his simulated 10th-90th range; and
  whether his fitness news changed between the forecast and the deadline (a stale forecast);
- per team (this model's, mine): points so far with FPL's automatic substitutions and vice
  captain, against the forecast and where that falls in the simulated spread (`team_live`);
- for the week: the same split over every player expected to feature, how many landed inside
  their ranges, hauls and blanks against the chances given, the bias by position, and what to
  look at before the next deadline (`lessons`).

No torch here: `python -m xpfpl.live update --site <dir> --out <dir>` runs in
.github/workflows/live-snapshot.yml every ten minutes while matches are on and writes live.json
to the `live` branch for the website's Live Gameweek page, reading the forecast from the site's
own data/current.json (or next.json if the site was last published before the deadline).
`xpfpl export` writes current.json and a live.json of its own, the dashboard's Live Gameweek tab
calls `local`, and the scorecard scores finished weeks with `decompose`.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from xpfpl import config, scoring
from xpfpl.data import api

STATS = ["minutes", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "bonus",
         "bps", "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved",
         "defensive_contribution", "starts", "expected_goals", "expected_assists"]
DONE, LIVE, TO_PLAY, BLANK = "done", "live", "to play", "blank"
FEATURE_PLAY = 0.25      # "expected to feature": the forecast gave him at least this chance of playing
MIN_TEAM = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}    # position -> (min, max) in a starting XI
SURPRISE = 3.0           # points from xP that make a result worth a line in the lessons


# ---------------------------------------------------------------- what happened

def actuals(live: dict) -> pd.DataFrame:
    """One row per player from FPL's event/<gw>/live: the stats (a double gameweek already summed)
    and `pts_<item>`, the points from each scoring item (minutes, goals_scored, bonus, ...)."""
    rows = []
    for e in live.get("elements", []):
        s = e.get("stats", {})
        row = {"element": e["id"], **{k: float(s.get(k) or 0) for k in STATS}}
        for match in e.get("explain", []):
            for item in match.get("stats", []):
                key = f"pts_{item['identifier']}"
                row[key] = row.get(key, 0.0) + float(item.get("points") or 0)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["element", *STATS]).set_index("element")
    return pd.DataFrame(rows).set_index("element").fillna(0.0)


def _done(f: dict) -> bool:
    return bool(f.get("finished") or f.get("finished_provisional"))


def club_state(fixtures: list[dict], gw: int) -> dict[int, str]:
    """Club id -> DONE (every match this gameweek over), LIVE (one started, not all over) or
    TO_PLAY. Clubs without a match are left out (BLANK)."""
    out: dict[int, list[dict]] = {}
    for f in fixtures:
        if f.get("event") == gw:
            for club in (f["team_h"], f["team_a"]):
                out.setdefault(club, []).append(f)
    return {club: DONE if all(_done(f) for f in fs) else LIVE if any(f.get("started") for f in fs) else TO_PLAY
            for club, fs in out.items()}


EVENTS = ["goals_scored", "assists", "own_goals", "penalties_saved", "penalties_missed", "yellow_cards",
          "red_cards", "saves", "bonus"]
BPS_LEADERS = 5


def provisional_bonus(bps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """(element, bonus) from a match's (element, BPS): 3, 2 and 1 for the top three, FPL's way with
    ties (a tie for first gets 3 each and the next player 1; a tie for second 2 each and no 1)."""
    ranked = sorted(bps, key=lambda x: -x[1])
    out = []
    for e, v in ranked:
        rank = 1 + sum(1 for _, w in ranked if w > v)
        if rank <= 3:
            out.append((e, 4 - rank))
    return out


def _events(f: dict) -> dict:
    """FPL's own match stats for one fixture: each event as {"h": [[element, value]], "a": [...]},
    the BPS leaders, and the bonus: FPL's once it's awarded, else worked out from the BPS
    (`bonus_provisional` true)."""
    stats = {s["identifier"]: s for s in f.get("stats") or []}
    pairs = lambda key, side: [[int(x["element"]), int(x["value"])] for x in stats.get(key, {}).get(side, [])]   # noqa: E731
    out = {key: {"h": pairs(key, "h"), "a": pairs(key, "a")} for key in EVENTS if key in stats}
    bps = [(e, v, "h") for e, v in pairs("bps", "h")] + [(e, v, "a") for e, v in pairs("bps", "a")]
    bps.sort(key=lambda x: -x[1])
    out["bps"] = [[e, v, side] for e, v, side in bps[:BPS_LEADERS]]
    if not (out.get("bonus", {}).get("h") or out.get("bonus", {}).get("a")) and bps and f.get("started"):
        sides = {e: side for e, _, side in bps}
        bonus = provisional_bonus([(e, v) for e, v, _ in bps])
        out["bonus"] = {side: [[e, b] for e, b in bonus if sides[e] == side] for side in ("h", "a")}
        out["bonus_provisional"] = True
    return out


def matches(fixtures: list[dict], gw: int) -> list[dict]:
    """The gameweek's matches with their score and state, in kick-off order, and FPL's events in
    each one (`events`: who scored, assisted, was booked, made saves, the BPS leaders and bonus)."""
    fs = sorted((f for f in fixtures if f.get("event") == gw), key=lambda f: f.get("kickoff_time") or "")
    return [{"id": f["id"], "home": f["team_h"], "away": f["team_a"], "kickoff": f.get("kickoff_time"),
             "home_score": f.get("team_h_score"), "away_score": f.get("team_a_score"), "minutes": f.get("minutes"),
             "state": DONE if _done(f) else LIVE if f.get("started") else TO_PLAY,
             "events": _events(f) if f.get("started") else {}} for f in fs]


# ---------------------------------------------------------------- forecast against result

def decompose(t: pd.DataFrame) -> pd.DataFrame:
    """Points minus xP, split exactly into three parts that add up to it.

    With p = the chance of playing the forecast gave him and xP / p his xP if he plays:
    - `minutes` = (played - p) x xP / p: he played when he might not have (+), or didn't (-);
    - `finishing` = (goals - xG) x points per goal + (assists - xA) x 3, in the match: returns
      beyond (+) or short of (-) the chances he got, mostly luck;
    - `other` = the rest: clean sheets, bonus, saves, defensive contribution, cards, and how many
      chances he got against how many his xP expected.

    Needs `xp`, `total_points`, `minutes`, `position` and, where there are any, `p_play`,
    `goals_scored`, `assists`, `expected_goals` and `expected_assists` (no xG: finishing is 0)."""
    xp = t["xp"].astype(float)
    p = (t["p_play"].astype(float).fillna(1.0) if "p_play" in t else pd.Series(1.0, index=t.index)).clip(0.05, 1.0)
    played = (t["minutes"].astype(float) > 0).astype(float)
    minutes = (played - p) * xp / p
    goal_points = t["position"].map(scoring.GOAL_POINTS).astype(float)
    col = lambda c: t[c].astype(float) if c in t else pd.Series(np.nan, index=t.index)   # noqa: E731
    finishing = ((col("goals_scored") - col("expected_goals")) * goal_points
                 + (col("assists") - col("expected_assists")) * scoring.ASSIST_POINTS) * played
    finishing = finishing.fillna(0.0)
    error = t["total_points"].astype(float) - xp
    return pd.DataFrame({"error": error, "minutes": minutes, "finishing": finishing,
                         "other": error - minutes - finishing}, index=t.index)


def _flag(status, chance) -> str:
    from xpfpl.data.news import status_label
    if status in (None, "a") and (chance is None or chance != chance or chance >= 100):
        return "fit"
    return status_label(status or "a", chance)


def news_changed(t: pd.DataFrame) -> pd.Series:
    """True where his flag at the deadline (`status_deadline`, `chance_deadline`) differs from the
    one the forecast was made with (`status`, `chance`)."""
    if not {"status", "status_deadline"} <= set(t.columns):
        return pd.Series(False, index=t.index)
    before = t["chance"].astype(float).fillna(100.0).where(t["status"].fillna("a") != "a", 100.0)
    after = t["chance_deadline"].astype(float).fillna(100.0).where(t["status_deadline"].fillna("a") != "a", 100.0)
    known = t["status_deadline"].notna()
    return known & ((t["status"].fillna("a") != t["status_deadline"].fillna("a")) | (before - after).abs().gt(0))


def reasons(r: pd.Series, final: bool = True) -> list[str]:
    """A few short sentences on why player row `r` scored what he did against his xP. With `final`
    False (his match is still on) only what has happened so far: no verdict on his minutes, his
    range or his news, which can't be judged yet."""
    out = []
    if final and r.get("news_changed"):
        out.append(f"Was {_flag(r.get('status'), r.get('chance'))} when this forecast was made, "
                   f"{_flag(r.get('status_deadline'), r.get('chance_deadline'))} at the deadline")
    p, m = r.get("p_play"), r["minutes"]
    has_p = p is not None and p == p
    if not final:
        pass
    elif m == 0:
        out.append(f"Didn't play (given {p:.0%} to play)" if has_p else "Didn't play")
    elif has_p and p < 0.5:
        out.append(f"Played {m:.0f} minutes though given only {p:.0%} to play")
    elif m < 60 and np.nan_to_num(r.get("p_full", 0.0)) >= 0.6:
        out.append(f"Only {m:.0f} minutes")
    if m > 0:
        g, xg = r["goals_scored"], r["expected_goals"]
        if g >= 1 and xg < 0.4 * g:
            out.append(f"Scored {g:.0f} from {xg:.2f} xG: finishing well beyond his chances")
        elif g >= 1 and xg >= 0.7 * g:
            out.append(f"Scored {g:.0f} from {xg:.2f} xG: chances earned")
        elif g >= 1:
            out.append(f"Scored {g:.0f} from {xg:.2f} xG")
        elif xg >= 0.6:
            out.append(f"{xg:.2f} xG without a goal: the chances were there")
        a, xa = r["assists"], r["expected_assists"]
        if a >= 1 and xa < 0.3:
            out.append(f"{a:.0f} assist{'s' if a > 1 else ''} from {xa:.2f} xA")
        elif a >= 1:
            out.append(f"{a:.0f} assist{'s' if a > 1 else ''} ({xa:.2f} xA)")
        elif xa >= 0.5:
            out.append(f"Created {xa:.2f} xA without an assist")
        if r.get("pts_clean_sheets", 0) > 0 and r["position"] <= 2:
            out.append("Clean sheet")
        elif r.get("pts_goals_conceded", 0) <= -2:
            out.append(f"Conceded {r['goals_conceded']:.0f}")
        if r.get("pts_defensive_contribution", 0) > 0:
            out.append("Defensive contribution points")
        if r.get("pts_saves", 0) >= 2:
            out.append(f"{r['saves']:.0f} saves")
        if r["bonus"] >= 2:
            out.append(f"{r['bonus']:.0f} bonus")
        if r["red_cards"]:
            out.append("Sent off")
        if r["own_goals"]:
            out.append("Own goal")
        if r["penalties_missed"]:
            out.append("Missed a penalty")
        if r["penalties_saved"]:
            out.append("Saved a penalty")
    band = r.get("band") if final else None
    if band == "above":
        out.append("Above his simulated range (fewer than 1 in 10 simulated weeks)")
    elif band == "below":
        out.append("Below his simulated range")
    return out


def verdict(r: pd.Series) -> str:
    """The main reason in a word or two: the biggest of the three parts, or the news."""
    if abs(r["error"]) < 2:
        return "As expected"
    parts = {"Minutes": r["minutes_part"], "Finishing": r["finishing"], "Chances & the rest": r["other"]}
    main = max(parts, key=lambda k: abs(parts[k]))
    if main == "Minutes" and r.get("news_changed"):
        return "News changed"
    return main


def _band(points: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    out = pd.Series(None, index=points.index, dtype=object)
    known = lo.notna() & hi.notna()
    out[known] = "inside"
    out[known & (points < lo)] = "below"
    out[known & (points > hi)] = "above"
    return out


# ---------------------------------------------------------------- a team, live

def _valid(xi: list[int], position: pd.Series) -> bool:
    counts = position.reindex(xi).value_counts()
    return all(lo <= counts.get(pos, 0) <= hi for pos, (lo, hi) in MIN_TEAM.items())


def _percentile(points: dict | None, score: float) -> float | None:
    """The share of simulated weeks that scored below `score` (from a simulation's histogram)."""
    h = (points or {}).get("histogram")
    if not h:
        return None
    start, width, shares = h["start"], h["width"], h["shares"]
    below = 0.0
    for i, share in enumerate(shares):
        lo = start + i * width
        below += share * min(max((score - lo) / width, 0.0), 1.0)
    return below


def team_live(team: dict, t: pd.DataFrame) -> dict:
    """A team's gameweek so far: `team` has `lineup` (11), `bench` (4, in order), `captain`, `vice`,
    `chip`, `forecast` (its xP as picked) and `points` (the simulated spread, with a histogram).
    FPL's automatic substitutions (a starter whose matches are all over without him playing
    makes way for the first bench player who played and keeps a valid formation) and the vice
    captain apply as soon as they're settled. Points are before transfer hits, like the forecast."""
    lineup, bench, chip = [int(e) for e in team["lineup"]], [int(e) for e in team["bench"]], team.get("chip")
    get = lambda e, c, d=0.0: float(t.at[e, c]) if e in t.index and t.at[e, c] == t.at[e, c] else d   # noqa: E731
    state = lambda e: t.at[e, "state"] if e in t.index else BLANK                                         # noqa: E731
    missed = lambda e: state(e) in (DONE, BLANK) and get(e, "minutes") == 0                              # noqa: E731
    xi, subs = list(lineup), []
    if chip != "bboost":
        for out in [e for e in lineup if missed(e)]:
            for b in bench:
                if b in xi or get(b, "minutes") == 0:
                    continue
                trial = [b if e == out else e for e in xi]
                if _valid(trial, t["position"]):
                    xi, subs = trial, subs + [{"out": out, "in": b}]
                    break
    counted = xi + (bench if chip == "bboost" else [])
    captain, vice = int(team["captain"]), team.get("vice")
    armband = captain if not missed(captain) else int(vice) if vice is not None and not missed(int(vice)) else None
    extra = 2 if chip == "3xc" else 1
    points = sum(get(e, "total_points") for e in counted) + (extra * get(armband, "total_points") if armband else 0.0)
    to_play = [e for e in counted if state(e) in (TO_PLAY, LIVE)]
    still = sum(get(e, "xp") * (1 + (extra if e == armband else 0)) for e in counted if state(e) == TO_PLAY)
    settled = not to_play
    picks = []
    for e in lineup + bench:
        role = "C" if e == captain else "V" if e == vice else "XI" if e in lineup else "Bench"
        picks.append({"element": e, "role": role, "counted": e in counted, "xp": get(e, "xp"),
                      "points": get(e, "total_points"), "minutes": get(e, "minutes"), "state": state(e),
                      "multiplier": (1 + extra if e == armband else 1) if e in counted else 0})
    forecast = team.get("forecast")
    return {"points": points, "forecast": forecast, "difference": None if forecast is None else points - forecast,
            "still_to_play": len(to_play), "still_xp": still, "projected": points + still, "settled": settled,
            "percentile": _percentile(team.get("spread"), points) if settled else None,
            "spread": {k: (team.get("spread") or {}).get(k) for k in ("p10", "p50", "p90")},
            "captain": captain, "armband": armband, "subs": subs, "chip": chip, "made_at": team.get("made_at"),
            "picks": picks}


# ---------------------------------------------------------------- the week

def _summary(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    out = {"n": int(len(t)), "xp": float(t["xp"].mean()), "points": float(t["total_points"].mean()),
           "mae": float(t["error"].abs().mean()), "bias": float(t["error"].mean()),
           "minutes": float(t["minutes_part"].mean()), "finishing": float(t["finishing"].mean()),
           "other": float(t["other"].mean())}
    if "band" in t and t["band"].notna().any():
        b = t["band"].dropna()
        out |= {"inside": float((b == "inside").mean()), "above": float((b == "above").mean()),
                "below": float((b == "below").mean())}
    if "p_haul" in t and t["p_haul"].notna().any():
        out |= {"hauls_expected": float(t["p_haul"].sum()), "hauls": int((t["total_points"] >= 10).sum()),
                "blanks_expected": float(t["p_blank"].sum()), "blanks": int((t["total_points"] <= 2).sum())}
    return out


def lessons(t: pd.DataFrame, featured: pd.Series, notable: pd.Series) -> dict[str, list[int]]:
    """Players worth a look before the next deadline (most striking first, up to eight each):
    - `stale`: his fitness news changed after the forecast was saved (re-run it nearer the deadline);
    - `backed`: beat his xP by SURPRISE+ with xG + xA of 0.6+ (the chances back it up: the model
      sees one match of it, so his next xP moves only a little);
    - `lucky`: beat it by SURPRISE+ with xG + xA under 0.3 (returns from little: don't chase this alone);
    - `unlucky`: xG + xA of 0.7+ in 60+ minutes and 3 points or fewer (the chances were there);
    - `minutes`: fit at the deadline and given 80%+ to play, but didn't start (watch his place)."""
    done = (t["state"] == DONE) & notable
    involvement = t["expected_goals"] + t["expected_assists"]
    pick = lambda mask, key: t[mask].sort_values(key, ascending=False).index[:8].astype(int).tolist()   # noqa: E731
    t = t.assign(_abs=t["error"].abs(), _inv=involvement)
    return {
        "stale": pick(t["news_changed"] & notable, "_abs"),
        "backed": pick(done & featured & (t["error"] >= SURPRISE) & (involvement >= 0.6), "error"),
        "lucky": pick(done & featured & (t["error"] >= SURPRISE) & (involvement < 0.3) & (t["finishing"] >= SURPRISE), "error"),
        "unlucky": pick(done & (t["minutes"] >= 60) & (involvement >= 0.7) & (t["total_points"] <= 3), "_inv"),
        "minutes": pick(done & (t["p_play"].fillna(0) >= 0.8) & (t["status_deadline"].fillna("a") == "a")
                        & (t["starts"] == 0), "xp"),
    }


def analyse(forecast: dict, live: dict, fixtures: list[dict], teams: dict[str, dict] | None = None,
            now: datetime | None = None, phase: str | None = None) -> dict:
    """live.json: `forecast` is the site's current.json (players as columns: element, team,
    position, xp_<gw>, p_play, pts_p10/p90, p_haul, p_blank, status/chance at the forecast and at
    the deadline), `live` FPL's event/<gw>/live, `teams` name -> team (see `team_live`), `phase`
    api.gameweek_status's ("live", "checking", "complete") for the page's status badge."""
    gw = int(forecast["gw"])
    f = pd.DataFrame(forecast["players"]).set_index("element")
    f.index = f.index.astype(int)
    f = f.rename(columns={f"xp_{gw}": "xp"})
    a = actuals(live)
    t = f.join(a, how="outer")
    t["xp"] = t["xp"].fillna(0.0)
    for c in STATS:
        t[c] = t[c].fillna(0.0) if c in t else 0.0
    t = t[t["position"].notna() & t["team"].notna()].copy()
    t["position"], t["team"] = t["position"].astype(int), t["team"].astype(int)
    states = club_state(fixtures, gw)
    t["state"] = t["team"].astype(int).map(states).fillna(BLANK)
    parts = decompose(t)
    t["error"], t["minutes_part"], t["finishing"], t["other"] = (parts["error"], parts["minutes"],
                                                               parts["finishing"], parts["other"])
    t["band"] = _band(t["total_points"], t.get("pts_p10", pd.Series(np.nan, index=t.index)),
                      t.get("pts_p90", pd.Series(np.nan, index=t.index)))
    t["news_changed"] = news_changed(t)
    for c in ("p_play", "status_deadline", "selected_by"):
        if c not in t:
            t[c] = np.nan
    featured = (t["p_play"].fillna((t["xp"] >= 1.0).astype(float)) >= FEATURE_PLAY)
    picked = {int(e) for team in (teams or {}).values() for e in [*team["lineup"], *team["bench"]]}
    notable = featured & ((t["selected_by"].fillna(0) >= 2) | (t["xp"] >= 3)) | t.index.isin(list(picked))

    done = t[(t["state"] == DONE) & featured]
    week = _summary(done)
    week["positions"] = {config.POSITIONS[p]: _summary(g) for p, g in done.groupby("position")}

    shown = t[(t["xp"] >= 0.5) | (t["minutes"] > 0) | t.index.isin(list(picked)) | t["news_changed"]].copy()
    # Reasons once his matches are over; while one is on, only what has happened so far; none before.
    shown["reasons"] = [reasons(r) if r["state"] == DONE else reasons(r, final=False) if r["state"] == LIVE else []
                        for _, r in shown.iterrows()]
    shown["verdict"] = [verdict(r) if r["state"] == DONE else None for _, r in shown.iterrows()]
    keep = ["team", "xp", "total_points", "minutes", "state", "p_play", "pts_p10", "pts_p90", "p_haul", "p_blank",
            "goals_scored", "assists", "expected_goals", "expected_assists", "bonus", "bps", "saves",
            "defensive_contribution", "error", "minutes_part",
            "finishing", "other", "band", "news_changed", "verdict", "reasons"]
    shown = shown[[c for c in keep if c in shown]].rename(columns={"total_points": "points"})

    fx = matches(fixtures, gw)
    return {
        "generated": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "gw": gw, "deadline": forecast.get("deadline"), "model": forecast.get("model"), "phase": phase,
        "forecast_made": forecast.get("made_at"),
        "matches": fx, "done": sum(m["state"] == DONE for m in fx), "total": len(fx),
        "week": week,
        "teams": {name: team_live(team, t) for name, team in (teams or {}).items()},
        "lessons": lessons(t, featured, notable),
        "stale": int((t["news_changed"] & notable).sum()),
        "players": _columns(shown.reset_index().rename(columns={"index": "element"})),
    }


def _columns(df: pd.DataFrame) -> dict[str, list]:
    def plain(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return None if not math.isfinite(v) else round(float(v), 3)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return {str(c): [plain(v) for v in df[c].tolist()] for c in df.columns}


# ---------------------------------------------------------------- the teams, from the site's files

def model_team(decision: dict | None, gw: int) -> dict | None:
    """This model's team for `gw` from its decision (archive/modelteam, or modelteam.json's `next`)."""
    if not decision or int(decision.get("gw", -1)) != gw:
        return None
    forecast = decision.get("forecast")
    if forecast is None:
        forecast = decision.get("xp", 0.0) + (decision.get("bench_xp") or 0.0 if decision.get("chip") == "bboost" else 0.0)
    return {"lineup": decision["lineup"], "bench": decision["bench"], "captain": decision["captain"],
            "vice": decision.get("vice"), "chip": decision.get("chip"), "forecast": forecast,
            "spread": (decision.get("simulation") or {}).get("points"), "made_at": decision.get("made_at")}


def my_team(mine: dict | None, gw: int) -> dict | None:
    """My team for `gw` from the site's myteam.json, once its deadline has passed (`private` is
    only in plain text then)."""
    if not mine or int(mine.get("gw", -1)) != gw or not mine.get("private"):
        return None
    week = next((w for w in mine["private"]["weeks"] if int(w["gw"]) == gw), None)
    if week is None:
        return None
    chip = week.get("chip") or mine["private"].get("chip")
    forecast = week["xp"] + ((week.get("bench_xp") or 0.0) if chip == "bboost" else 0.0)
    return {"lineup": week["lineup"], "bench": week["bench"], "captain": week["captain"], "vice": week.get("vice"),
            "chip": chip, "forecast": forecast, "spread": week.get("points")}


def stale_news(players: pd.DataFrame, bs: dict) -> pd.DataFrame:
    """Players in a saved forecast (`status`, `chance` as it was made) whose FPL flag is different
    now: the forecast is out of date for them. Columns: name, then, now."""
    if not {"status", "chance"} <= set(players.columns):
        return pd.DataFrame(columns=["name", "then", "now"])
    now = pd.DataFrame(bs["elements"]).set_index("id")[["web_name", "status", "chance_of_playing_next_round"]]
    t = players[["status", "chance"]].join(now.rename(columns={"status": "status_deadline",
                                                                "chance_of_playing_next_round": "chance_deadline"}),
                                           how="inner")
    t = t[news_changed(t)]
    return pd.DataFrame({"name": t["web_name"], "then": [_flag(s, c) for s, c in zip(t["status"], t["chance"])],
                         "now": [_flag(s, c) for s, c in zip(t["status_deadline"], t["chance_deadline"])]},
                        index=t.index)


# ---------------------------------------------------------------- run by the Action

def due(previous: dict | None, fixtures: list[dict], gw: int, now: datetime) -> bool:
    """Whether the scheduled run should fetch: no file yet or one for another gameweek; a match in
    play (kick-off to 2.5 hours after); FPL still adding bonus (matches over, not all `finished`)
    and the file 20+ minutes old; else when the file is 6+ hours old."""
    if not previous or int(previous.get("gw", -1)) != gw:
        return True
    age = (now - datetime.fromisoformat(previous["generated"])).total_seconds() / 3600
    for f in fixtures:
        if f.get("event") != gw or not f.get("kickoff_time"):
            continue
        kickoff = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
        if 0 <= (now - kickoff).total_seconds() / 3600 <= 2.5:
            return True
        if f.get("finished_provisional") and not f.get("finished") and age >= 1 / 3:
            return True
    return age >= 6


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def update(site: Path, out: Path, previous: Path | None = None) -> Path | None:
    """Write out/live.json for the gameweek in progress from the site's data files in `site`
    (current.json, else next.json if it's for that gameweek; modelteam.json; myteam.json). With
    `previous` (the branch's live.json), only when `due`. None if nothing was written."""
    bs, fixtures = api.bootstrap(), api.fixtures()
    current = next((e["id"] for e in bs["events"] if e["is_current"]), None)
    if current is None:
        print("No gameweek in progress.")
        return None
    now = datetime.now(timezone.utc)
    if previous is not None and not due(_read(previous), fixtures, current, now):
        print("Not due.")
        return None
    forecast = next((f for f in (_read(site / "current.json"), _read(site / "next.json"))
                     if f and int(f.get("gw", -1)) == current), None)
    if forecast is None:
        print(f"The site has no forecast for GW{current}.")
        return None
    players = forecast["players"]
    if "position" not in players or "team" not in players:     # a site published before these were exported
        el = pd.DataFrame(bs["elements"]).set_index("id")
        players["team"] = [int(el.at[e, "team"]) if e in el.index else None for e in players["element"]]
        players["position"] = [int(el.at[e, "element_type"]) if e in el.index else None for e in players["element"]]
    if "status_deadline" not in players and "status" in players:
        # Published before the deadline: the flags now stand in for the ones at the deadline.
        add_deadline_news(forecast, bs)
    decision = forecast.get("model_team") or (_read(site / "modelteam.json") or {}).get("next")
    teams = {"model": model_team(decision, current), "mine": my_team(_read(site / "myteam.json"), current)}
    result = analyse(forecast, api.event_live(current), fixtures, {k: v for k, v in teams.items() if v}, now,
                     api.gameweek_status(bs, fixtures)["phase"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "live.json"
    path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"GW{current}: {result['done']} of {result['total']} matches over -> {path}")
    return path


def add_deadline_news(forecast: dict, bs: dict, at_deadline: pd.DataFrame | None = None) -> None:
    """Fill `status_deadline`/`chance_deadline` in a forecast payload: from `at_deadline` (code ->
    status, chance) where given, else FPL's flags now."""
    players = forecast["players"]
    if at_deadline is None:
        now = pd.DataFrame(bs["elements"]).set_index("id")
        status, chance = now["status"], now["chance_of_playing_next_round"]
        players["status_deadline"] = [status.get(e) for e in players["element"]]
        players["chance_deadline"] = [None if chance.get(e) != chance.get(e) else chance.get(e) for e in players["element"]]
    else:
        codes = pd.DataFrame(bs["elements"]).set_index("id")["code"]
        rows = [at_deadline.loc[codes[e]] if e in codes.index and codes[e] in at_deadline.index else None
                for e in players["element"]]
        players["status_deadline"] = ["a" if r is None else r["status"] for r in rows]
        players["chance_deadline"] = [None if r is None or r["chance"] != r["chance"] else float(r["chance"]) for r in rows]


def news_at(season: str, when: str | pd.Timestamp) -> pd.DataFrame:
    """Every flagged player's status and chance at `when` from the archived news log (code -> status,
    chance); players with no entry before then were fit."""
    from xpfpl.data import news
    log = news.archived_log(season)
    if log.empty:
        return pd.DataFrame(columns=["status", "chance"])
    log = log[pd.to_datetime(log["seen"], utc=True) <= pd.Timestamp(when)].sort_values("seen", kind="stable")
    return log.groupby("code").tail(1).set_index("code")[["status", "chance"]]


# ---------------------------------------------------------------- the dashboard

def local(bs: dict, fixtures: list[dict], model: str = config.MODEL) -> dict | None:
    """live.json for the gameweek in progress from the archive and the dashboard's own team."""
    from xpfpl import modelteam, myteam
    current = next((e for e in bs["events"] if e["is_current"]), None)
    if current is None:
        return None
    season, gw = api.current_season(bs), current["id"]
    forecast = payload(bs, season, gw, model)
    if forecast is None:
        return None
    teams = {"model": model_team(modelteam.decisions(season).get(gw), gw)}
    team_id = myteam.saved_team_id()
    if team_id is not None:
        found = myteam.saved_forecast(season, gw, model)
        team = myteam.shown_team(team_id, bs) if found else None
        if team is not None:
            out = myteam.outlook(team, found[0], found[1], found[2])
            week = next((w for w in out["weeks"] if int(w["gw"]) == gw), None)
            if week:
                teams["mine"] = my_team({"gw": gw, "private": {"weeks": [week], "chip": team["chip"]}}, gw)
    return analyse(forecast, api.event_live(gw), fixtures, {k: v for k, v in teams.items() if v},
                   phase=api.gameweek_status(bs, fixtures)["phase"])


FORECAST_COLUMNS = ("xmins", "p_play", "p_full", "mkt_anytime", "market_out", "rotation", "rotation_factor",
                    "pts_p10", "pts_p50", "pts_p90", "p_haul", "p_6_9", "p_3_5", "p_blank",
                    "total_p10", "total_p50", "total_p90", "xp_sd", "confidence",
                    "avail", "news_rule", "press", "back", "excused")


def payload(bs: dict, season: str, gw: int, model: str, bands: bool = False) -> dict | None:
    """The forecast saved for `gw` as the site's next.json / current.json: per player (columns) xP
    for every gameweek of its horizon, minutes, ranges, team news, his club and position, and the
    FPL flag it was made with (`status`, `chance`); once `gw`'s deadline has passed, also the flag
    at the deadline (`status_deadline`, `chance_deadline`: the archived news log, else FPL's now)
    and this model's team for it (`model_team`). `bands` fills the outcome bands of a forecast
    saved before they were (from its simulations: needs torch)."""
    from xpfpl.data import archive
    from xpfpl import modelteam
    found = archive.saved_forecast(season, gw, model)
    if found is None:
        return None
    key, t, gws = found
    event = next(e for e in bs["events"] if e["id"] == gw)
    used = key.split("_", 1)[1]
    if "p_3_5" not in t and bands:
        # Outcome bands came later than the first saved forecasts: take them from the simulations.
        from xpfpl import simulate
        draws = simulate.load(season, gw, used)
        if draws is not None and gw in draws.gameweeks:
            extra = simulate.summary(draws, gw).drop(columns=[c for c in t if c != "element"], errors="ignore")
            t = t.merge(extra, left_on="element", right_index=True, how="left")
    keep = ["element", "team", "position", "status", "chance", "selected_by", *[f"xp_{g}" for g in gws], "xp_total",
            *[c for c in FORECAST_COLUMNS if c in t]]
    t = t[[c for c in keep if c in t]]
    made = str(t["made_at"].iloc[0]) if "made_at" in t and len(t) else None    # recorded from 2026-10-12
    out = {"gw": gw, "deadline": event["deadline_time"], "model": used, "gameweeks": gws, "made_at": made,
           "players": _columns(t)}
    if pd.Timestamp(event["deadline_time"]) <= pd.Timestamp.now(tz="UTC"):
        news = news_at(season, event["deadline_time"])
        add_deadline_news(out, bs, news if len(news) else None)
        out["model_team"] = modelteam.decisions(season).get(gw)
    return out


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m xpfpl.live")
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("update", help="write live.json for the gameweek in progress")
    up.add_argument("--site", type=Path, required=True, help="folder with the site's current.json/next.json/modelteam.json/myteam.json")
    up.add_argument("--out", type=Path, required=True)
    up.add_argument("--when-due", type=Path, default=None, help="the branch's live.json: only fetch when a refresh is due")
    args = parser.parse_args(argv)
    update(args.site, args.out, args.when_due)


if __name__ == "__main__":
    main(sys.argv[1:])
