"""Your own FPL team: read from the public API (squad, selling prices, bank, free transfers,
chips), and the team you're planning for the next gameweek (saved from the dashboard's Plan
Ahead) with its outlook over the next few weeks (xP and the Monte Carlo)."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from xpfpl import config
from xpfpl.data import api


@dataclass
class MyTeam:
    team_id: int
    name: str
    squad: dict[int, float]            # element id -> selling price (£m)
    bank: float
    free_transfers: int
    chips_used: list[dict]             # [{"name": "wildcard", "event": 3}, ...]
    pending_transfers: int             # transfers already made for the upcoming GW
    purchase: dict[int, int] = field(default_factory=dict)   # element id -> purchase price (tenths)


def selling_price(now: int, purchase: int) -> float:
    """FPL keeps half of any price rise (rounded down to £0.1m); price falls are passed on in full."""
    if now <= purchase:
        return now / 10.0
    return (purchase + (now - purchase) // 2) / 10.0


def estimate_free_transfers(history: dict, started_event: int, next_gw: int) -> int:
    """Replay the season: +1 free transfer per gameweek (max 5), used ones deducted.

    Wildcard / Free Hit weeks don't consume free transfers. This is an estimate - special
    rules (e.g. one-off FT top-ups) aren't modelled, so override with --free-transfers if needed.
    """
    chips = {c["event"]: c["name"] for c in history["chips"]}
    made = {h["event"]: h["event_transfers"] for h in history["current"]}
    ft = 1
    for gw in range(started_event + 1, next_gw):
        if chips.get(gw) not in ("wildcard", "freehit"):
            ft = max(ft - made.get(gw, 0), 0)
        ft = min(ft + 1, config.MAX_FREE_TRANSFERS)
    return ft


def load_my_team(team_id: int, bs: dict) -> MyTeam:
    next_gw = api.next_gameweek(bs)
    current_gw = next_gw - 1
    info = api.entry(team_id)
    history = api.entry_history(team_id)
    transfers = api.entry_transfers(team_id)
    prices = {p["id"]: p for p in bs["elements"]}

    picks = api.entry_picks(team_id, current_gw)
    if picks.get("active_chip") == "freehit":  # squad (and bank) revert after a Free Hit
        picks = api.entry_picks(team_id, current_gw - 1)
    squad = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0

    # Transfers already made for the upcoming deadline aren't in the picks yet.
    pending = [t for t in transfers if t["event"] == next_gw]
    for t in pending:
        squad.remove(t["element_out"])
        squad.append(t["element_in"])
        bank += (t["element_out_cost"] - t["element_in_cost"]) / 10.0

    # Purchase price = cost at the most recent transfer in, else the start-of-season price.
    bought = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        bought[t["element_in"]] = t["element_in_cost"]
    selling, purchases = {}, {}
    for pid in squad:
        now = prices[pid]["now_cost"]
        purchases[pid] = bought.get(pid, now - prices[pid]["cost_change_start"])
        selling[pid] = selling_price(now, purchases[pid])

    ft = estimate_free_transfers(history, info["started_event"], next_gw)
    return MyTeam(
        team_id=team_id,
        name=info["name"],
        squad=selling,
        bank=round(bank, 1),
        free_transfers=max(ft - len(pending), 0),
        chips_used=[{"name": c["name"], "event": c["event"]} for c in history["chips"]],
        pending_transfers=len(pending),
        purchase=purchases,
    )


# ---------------------------------------------------------------- the team you're planning to play

def saved_team_id() -> int | None:
    """The team id saved in the dashboard's sidebar (data/app_settings.json), if any."""
    try:
        team_id = json.loads(config.APP_SETTINGS_PATH.read_text(encoding="utf-8")).get("team_id")
        return int(team_id) if team_id else None
    except (OSError, ValueError, TypeError):
        return None


def site_secret() -> str | None:
    """The secret word that unlocks My Team's players on the website before the deadline:
    XPFPL_MY_TEAM_SECRET, else `my_team_secret` in the dashboard's settings. None: not published."""
    import os
    secret = os.environ.get("XPFPL_MY_TEAM_SECRET")
    if not secret:
        try:
            secret = json.loads(config.APP_SETTINGS_PATH.read_text(encoding="utf-8")).get("my_team_secret")
        except (OSError, ValueError):
            secret = None
    return secret.strip() if secret and secret.strip() else None


def plan_path(season: str, gw: int):
    return config.MY_TEAM_DIR / season / f"gw{gw:02d}.json"


def save_plan(season: str, gw: int, *, lineup: list[int], bench: list[int], captain: int, vice: int,
              chip: str | None = None, transfers_out: list[int] = (), transfers_in: list[int] = (),
              hits: int = 0, bank: float | None = None, before: list[int] | None = None,
              model: str = config.MODEL, settings: dict | None = None) -> dict:
    """Save the team you'll play in `gw` (data/myteam/<season>/gwNN.json, local only), replacing
    any earlier save for that week. `before` is the squad it was planned from: the weeks after
    a Free Hit go back to it, and the transfers are judged against it. Transfers are paired in
    the order given (sort both by position). `settings` records the Plan Ahead choices it was
    planned with (horizon, model, chip, free transfers, bank, penalties, planning, prices, always
    and never pick), so the scenario can be set up again."""
    team = {"season": season, "gw": int(gw), "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model, "chip": chip, "lineup": [int(p) for p in lineup], "bench": [int(p) for p in bench],
            "captain": int(captain), "vice": int(vice),
            "transfers": [{"out": int(o), "in": int(i)} for o, i in zip(transfers_out, transfers_in)],
            "hits": int(hits), "bank": None if bank is None else round(float(bank), 1),
            "before": None if before is None else sorted(int(p) for p in before), "settings": settings}
    path = plan_path(season, gw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(team, indent=1) + "\n", encoding="utf-8")
    return team


def saved_plan(season: str, gw: int) -> dict | None:
    try:
        return json.loads(plan_path(season, gw).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _picks_team(picks: dict) -> dict:
    order = sorted(picks["picks"], key=lambda p: p["position"])
    return {"lineup": [p["element"] for p in order[:11]], "bench": [p["element"] for p in order[11:]],
            "captain": next(p["element"] for p in order if p["is_captain"]),
            "vice": next(p["element"] for p in order if p["is_vice_captain"])}


def last_team(team_id: int, gw: int) -> dict:
    """The XI, bench, captain and vice you played in `gw`, from FPL (public once the deadline
    has passed). After a Free Hit, the week before's: that's the squad you go back to."""
    picks = api.entry_picks(team_id, gw)
    if picks.get("active_chip") == "freehit" and gw > 1:
        picks = api.entry_picks(team_id, gw - 1)
    return _picks_team(picks)


def locked_team(team_id: int, gw: int) -> dict:
    """The team locked into FPL at `gw`'s deadline (source "locked"): its picks, chip, the week's
    transfers and hits, and the squad it came from (`before`)."""
    picks = api.entry_picks(team_id, gw)
    moves = [t for t in api.entry_transfers(team_id) if t["event"] == gw]
    before = None
    if gw > 1:
        last = last_team(team_id, gw - 1)
        before = sorted(last["lineup"] + last["bench"])
    return {"gw": gw, "source": "locked", "saved_at": None, "model": None, "chip": picks.get("active_chip"),
            **_picks_team(picks), "transfers": [{"out": t["element_out"], "in": t["element_in"]} for t in moves],
            "hits": picks["entry_history"]["event_transfers_cost"] // config.HIT_COST,
            "bank": picks["entry_history"]["bank"] / 10.0, "before": before}


def showing_gameweek(bs: dict) -> int | None:
    """The gameweek My Team is about: the one in progress until FPL marks it finished, then the
    next. None once the season is over."""
    current = next((e for e in bs["events"] if e["is_current"]), None)
    if current is not None and not current["finished"]:
        return current["id"]
    return next((e["id"] for e in bs["events"] if e["is_next"]), None)


def deadline_passed(bs: dict, gw: int) -> bool:
    event = next(e for e in bs["events"] if e["id"] == gw)
    return pd.Timestamp(event["deadline_time"]) <= pd.Timestamp.now(tz="UTC")


def shown_team(team_id: int, bs: dict) -> dict | None:
    """The team My Team shows, for `showing_gameweek`:

    - before the deadline, the one last saved in Plan Ahead (source "saved"), else last week's
      team carried over (source "carried": no transfers, same XI and captain);
    - from the deadline until FPL marks the gameweek finished, the team actually locked in
      (source "locked", from FPL's public picks), whatever was saved.

    None if there's no such team (before your first gameweek, or the season is over)."""
    import requests
    gw = showing_gameweek(bs)
    if gw is None:
        return None
    season = api.current_season(bs)
    try:
        saved = saved_plan(season, gw)
        if deadline_passed(bs, gw):      # the settings of the last save, which may differ from what was locked in
            return {"season": season, **locked_team(team_id, gw), "settings": (saved or {}).get("settings")}
        if saved:
            return {**saved, "source": "saved"}
        if gw == 1:
            return None
        last = last_team(team_id, gw - 1)
    except requests.RequestException:            # no picks that week (joined later)
        return None
    return {"season": season, "gw": gw, "source": "carried", "saved_at": None, "model": None, "chip": None,
            **last, "transfers": [], "hits": 0, "bank": None, "before": None, "settings": None}


def saved_forecast(season: str, gw: int, model: str = config.MODEL):
    """(players indexed by element, gameweeks, simulate.Draws or None, model) of the forecast
    saved for `gw` (`model`'s, else any model's), or None. My Team reads this rather than a new
    forecast: after the deadline it's still the week's forecast."""
    from xpfpl import simulate
    from xpfpl.data import archive
    found = archive.saved_forecast(season, gw, model)
    if found is None:
        return None
    key, table, gameweeks = found
    used = key.split("_", 1)[1]
    return table.set_index("element"), gameweeks, simulate.load(season, gw, used), used


def _best_weeks(players: pd.DataFrame, squad: list[int], gameweeks: list[int]) -> dict[int, dict]:
    """The best XI, bench order, captain and vice from `squad` in each of `gameweeks`, no transfers."""
    from xpfpl.optimise import solve
    pool = players.loc[list(squad)]
    plan = solve(pool, gameweeks, current_squad=pool["price"].to_dict(), bank=0.0, free_transfers=0, max_hits=0)
    return {g: {"lineup": plan.lineups[g], "bench": plan.bench[g], "captain": plan.captains[g],
                "vice": plan.vice_captains[g], "chip": None} for g in gameweeks}


def _week_xp(players: pd.DataFrame, gw: int, week: dict) -> tuple[float, float]:
    """(xP counted as the week is scored: captain doubled, tripled with Triple Captain, bench
    with Bench Boost; the bench's xP)."""
    xp = players[f"xp_{gw}"]
    bench = float(xp[week["bench"]].sum())
    total = float(xp[week["lineup"]].sum()) + (2 if week["chip"] == "3xc" else 1) * float(xp[week["captain"]])
    return total + (bench if week["chip"] == "bboost" else 0.0), bench


def outlook(team: dict, players: pd.DataFrame, gameweeks: list[int], draws=None, captains: int = 5) -> dict:
    """How `team` (from `shown_team`) could do over `gameweeks`, the forecast's horizon.

    The first gameweek is played as picked. The ones after keep the same 15 (after a Free Hit,
    the squad before it) with the best XI, bench and captain from them each week: later
    transfers aren't picked yet. `players` is a forecast (`predict_upcoming`, or a saved one
    indexed by element) with position, price and `xp_<gw>`. With `draws` (simulate.Draws from
    the same forecast), every week is also simulated (auto-subs, the vice-captain, chips).

    Returns per week (`weeks`): the picks, xP counted as the week is scored, the bench's xP and
    the simulated score's spread (before transfer hits); the `total` over the horizon (after
    hits); the first week's captain options' odds (`captains`); `against`, the team against
    keeping the squad it was planned from (its best XI each week) in the same simulated weeks,
    if it made any; and `players`, the squads' xP per week and next week's simulated range."""
    from xpfpl import simulate
    first = gameweeks[0]
    squad = list(team["lineup"]) + list(team["bench"])
    later = team["before"] if team["chip"] == "freehit" and team.get("before") else squad
    weeks = {first: {k: team[k] for k in ("lineup", "bench", "captain", "vice", "chip")}}
    if len(gameweeks) > 1:
        weeks |= _best_weeks(players, later, gameweeks[1:])

    position = players["position"].astype(int).to_dict()
    everyone = set(squad) | set(later) | set(team.get("before") or [])
    simulated = draws is not None and everyone <= {int(e) for e in draws.elements}
    sim_gws = [g for g in gameweeks if simulated and g in draws.gameweeks]
    hits = config.HIT_COST * int(team.get("hits") or 0)

    def score(week: dict, gw: int) -> np.ndarray:
        return simulate.team_score(draws, gw, week["lineup"], week["bench"], week["captain"], week["vice"],
                                   week["chip"], position)["points"]

    out_weeks, sims = [], {}
    for g in gameweeks:
        w = weeks[g]
        xp, bench_xp = _week_xp(players, g, w)
        row = {"gw": int(g), "lineup": [int(p) for p in w["lineup"]], "bench": [int(p) for p in w["bench"]],
               "captain": int(w["captain"]), "vice": int(w["vice"]), "chip": w["chip"],
               "xp": round(xp, 2), "bench_xp": round(bench_xp, 2), "points": None}
        if g in sim_gws:
            sims[g] = score(w, g)
            target = int(round(xp / 10) * 10)
            row |= {"points": simulate.spread(sims[g]), "target": target,
                    "p_target": round(float((sims[g] >= target).mean()), 4)}
        out_weeks.append(row)
    all_simulated = len(sims) == len(gameweeks)

    total_xp = sum(r["xp"] for r in out_weeks) - hits
    total = {"xp": round(total_xp, 2), "hits": hits,
             "points": simulate.spread(sum(sims.values()) - hits) if all_simulated else None}

    captain_odds = []
    if first in sims:
        xp = players.loc[team["lineup"], f"xp_{first}"].sort_values(ascending=False)
        options = list(dict.fromkeys([team["captain"], *xp.index[:captains]]))
        odds = simulate.captain_odds(draws, first, options, 3 if team["chip"] == "3xc" else 2)
        captain_odds = [{"element": int(e), **{k: round(float(v), 4) for k, v in r.items()}} for e, r in odds.iterrows()]

    against = None
    before = team.get("before")
    if before and set(before) != set(squad):
        base = _best_weeks(players, before, gameweeks)
        against = {"xp": round(total_xp - sum(_week_xp(players, g, base[g])[0] for g in gameweeks), 2)}
        if all_simulated:
            diff = sum(sims.values()) - hits - sum(score(base[g], g) for g in gameweeks)
            q = np.percentile(diff, [10, 50, 90])
            against |= {"p_better": round(float((diff > 0).mean()), 4), "p_tie": round(float((diff == 0).mean()), 4),
                        "mean": round(float(diff.mean()), 2), "p10": float(q[0]), "p50": float(q[1]),
                        "p90": float(q[2])}

    cols = [f"xp_{g}" for g in gameweeks]
    extra = [c for c in ("pts_p10", "pts_p50", "pts_p90", "p_haul", "p_blank", "p_play") if c in players]
    table = players.loc[sorted(set(squad) | set(later)), cols + extra].copy()
    table["xp_total"] = table[cols].sum(axis=1)
    table.index.name = "element"
    return {"gw": int(first), "gameweeks": [int(g) for g in gameweeks], "sims": draws.sims if simulated else None,
            "weeks": out_weeks, "total": total, "captains": captain_odds, "against": against,
            "players": table.reset_index()}
