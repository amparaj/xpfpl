"""The Model's Team: a paper FPL team that follows the model's own advice, every gameweek.

Before each deadline `decide()` picks the team's transfers, XI, bench order, captain and chip
with the same optimiser, settings (config.py) and chip rules as `xpfpl recommend`, starting from
the team's own squad, bank and free transfers. The decision is saved as
archive/modelteam/<season>/gwNN.json and never changed once that deadline has passed. `recommend`
calls it, so the weekly routine keeps the record going; `xpfpl modelteam` does it on its own.

The gameweeks played before the live record started are filled in once by `replay()`, which runs
the backtest: a model trained only on earlier seasons, and each player's form as it stood at
each deadline. Those weeks are marked "replay".

`season_record()` scores every decision against what happened (auto-subs, the vice-captain, chips
and hits) for the website. A gameweek with no saved decision keeps the previous week's team and
banks the free transfer, as FPL does.
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from xpfpl import backtest, chips, config
from xpfpl.myteam import selling_price
from xpfpl.optimise import solve

FOLDER = config.ARCHIVE_DIR / "modelteam"
BUDGET = 100.0


def _path(season: str, gw: int):
    return FOLDER / season / f"gw{gw:02d}.json"


def decisions(season: str) -> dict[int, dict]:
    """GW -> the decision saved before its deadline."""
    return {int(p.stem[2:]): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((FOLDER / season).glob("gw*.json"))}


def _save(decision: dict) -> None:
    path = _path(decision["season"], decision["gw"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")


def _state(saved: dict[int, dict], gw: int) -> tuple[dict[int, int], float, int, list[dict]]:
    """(squad: element -> purchase price in tenths, bank, free transfers, chips used) going into
    `gw`. Weeks with no decision since the last one each bank a free transfer."""
    earlier = [g for g in saved if g < gw]
    used = [{"name": d["chip"], "event": g} for g, d in saved.items() if g < gw and d["chip"]]
    if not earlier:
        return {}, BUDGET, 1, used
    last = max(earlier)
    state = saved[last]["next"]
    free = min(state["free_transfers"] + (gw - 1 - last), config.MAX_FREE_TRANSFERS)
    return {int(e): int(v) for e, v in state["squad"].items()}, state["bank"], free, used


def _decision(season: str, gw: int, source: str, model: str, *, plan, chip, pool, squad, bank,
              free_transfers, selling, now_cost, after) -> dict:
    """One gameweek's choices, plus the state the next gameweek starts from."""
    position = pool["position"].to_dict()
    outs = sorted(plan.transfers_out, key=lambda p: position[p]) if squad else []
    ins = sorted(plan.transfers_in, key=lambda p: position[p]) if squad else []
    return {
        "season": season, "gw": int(gw), "source": source, "model": model,
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chip": chip, "free_transfers": int(free_transfers), "bank": round(float(bank), 1),
        "hits": 0 if chip in ("wildcard", "freehit") or not squad else int(plan.hits),
        "transfers": [{"out": int(o), "in": int(i), "sold": float(selling[o]), "bought": int(now_cost[i]) / 10.0}
                      for o, i in zip(outs, ins)],
        "squad": [int(p) for p in plan.squad], "lineup": [int(p) for p in plan.lineups[gw]],
        "bench": [int(p) for p in plan.bench[gw]], "captain": int(plan.captains[gw]),
        "vice": int(plan.vice_captains[gw]), "xp": round(float(plan.xp[gw]), 2),
        "next": {"squad": {str(p): int(v) for p, v in sorted(after["squad"].items())},
                 "bank": round(float(after["bank"]), 1), "free_transfers": int(after["free_transfers"])},
    }


# ---------------------------------------------------------------- deciding

def replay(season: str, up_to: int, model: str = config.MODEL) -> list[int]:
    """Fill in GW1..`up_to` with the backtest's decisions (source "replay"). Weeks that already
    have a decision are left alone. Returns the gameweeks written."""
    print(f"Replaying GW1-{up_to} of {season} for the Model's Team (a model trained on earlier seasons only)...")
    found: list[dict] = []

    def record(week: dict) -> None:
        week = dict(week)
        found.append(_decision(season, week.pop("gw"), "replay", model, **week))

    backtest.run(backtest.Settings(season=season, end_gw=up_to, model=model, chips=True), observer=record)
    saved = decisions(season)
    written = []
    for d in found:
        if d["gw"] not in saved:
            _save(d)
            written.append(d["gw"])
    return written


def decide(bs: dict | None = None, fixtures: list[dict] | None = None, model: str = config.MODEL,
           predicted: tuple[pd.DataFrame, list[int]] | None = None) -> dict | None:
    """The Model's Team's choices for the next gameweek, saved to the archive. None if that
    deadline has passed. `predicted` reuses a `predict_upcoming(config.HORIZON, model)` result."""
    from xpfpl.data import api
    from xpfpl.predict import predict_upcoming

    bs = bs or api.bootstrap()
    fixtures = fixtures or api.fixtures()
    season, gw = api.current_season(bs), api.next_gameweek(bs)
    event = next(e for e in bs["events"] if e["id"] == gw)
    if pd.Timestamp(event["deadline_time"]) <= pd.Timestamp.now(tz="UTC"):
        return None
    saved = decisions(season)
    if gw > 1 and not any(g < gw for g in saved):
        replay(season, gw - 1, model)
        saved = decisions(season)

    players, gws = predicted or predict_upcoming(config.HORIZON, model, bs, fixtures)
    squad, bank, free, used = _state(saved, gw)
    now_cost = (players["price"] * 10).round().astype(int).to_dict()
    selling = {p: selling_price(now_cost[p], bought) for p, bought in squad.items()}
    base = dict(current_squad=selling or None, bank=round(bank, 1), price_weight=config.PRICE_WEIGHT)
    rules = dict(free_transfers=free, max_hits=config.MAX_HITS, plan_transfers=config.PLAN_TRANSFERS)
    plan = solve(players, gws, **base, **rules, pool_size=config.POOL_SIZE if config.PLAN_TRANSFERS else None)
    chip = None
    if squad:
        advice = chips.advise(players, gws, plan, chips.available_chips(bs, used, gw), base)
        best = next((a for a in advice if a.recommended), None)
        chip = best.chip if best else None
        plan = backtest.play_chip(players, gws, base, chip, plan, **rules)
    settled = backtest.settle(squad, bank, free, plan, chip, now_cost, selling, BUDGET)
    decision = _decision(season, gw, "live", model, plan=plan, chip=chip, pool=players, squad=squad, bank=bank,
                         free_transfers=free, selling=selling, now_cost=now_cost,
                         after=dict(zip(("squad", "bank", "free_transfers"), settled)))
    _save(decision)
    return decision


def describe(decision: dict, players: pd.DataFrame) -> str:
    """One line for the console."""
    name = lambda p: players.at[p, "name"] if p in players.index else str(p)   # noqa: E731
    moves = ", ".join(f"{name(t['out'])} -> {name(t['in'])}" for t in decision["transfers"]) or "no transfers"
    chip = f", {chips.CHIP_NAMES[decision['chip']]}" if decision["chip"] else ""
    return (f"The Model's Team, GW{decision['gw']}: {moves}{chip}; captain {name(decision['captain'])}; "
            f"xP {decision['xp']:.1f}. Saved to {_path(decision['season'], decision['gw']).relative_to(config.ROOT)}")


# ---------------------------------------------------------------- scoring

def season_record(season: str, rows: pd.DataFrame, bs: dict) -> dict:
    """Every played gameweek's decision with what it scored, for the website. `rows` is the
    season's per-match table (archive.season_tables): round, element, total_points, minutes."""
    from xpfpl import review

    saved = decisions(season)
    el = pd.DataFrame(bs["elements"]).set_index("id")
    position = el["element_type"].astype(int).to_dict()
    played = sorted(int(g) for g in rows["round"].unique())
    weeks, last = [], None
    for gw in played:
        d = saved.get(gw)
        carried = d is None
        if carried:
            if last is None:
                continue
            base = last if last["chip"] != "freehit" else next(
                (saved[g] for g in sorted(saved, reverse=True) if g < last["gw"] and saved[g]["chip"] != "freehit"), last)
            d = {**base, "gw": gw, "source": "carried", "chip": None, "hits": 0, "transfers": [], "xp": None}
        week = rows[rows["round"] == gw].groupby("element").agg(points=("total_points", "sum"),
                                                                minutes=("minutes", "sum"))
        points, minutes = week["points"].to_dict(), week["minutes"].to_dict()
        plan = SimpleNamespace(lineups={gw: d["lineup"]}, bench={gw: d["bench"]}, captains={gw: d["captain"]},
                               vice_captains={gw: d["vice"]}, squad=d["squad"])
        result = backtest.score_gameweek(plan, gw, points, minutes, position, d["chip"])
        _, subs = backtest.apply_autosubs(d["lineup"], d["bench"], position, minutes)
        picks = pd.DataFrame({"name": el["web_name"], "team": el["team"], "pos": el["element_type"],
                              "price": el["now_cost"] / 10.0}).reindex(d["squad"])
        picks["points"] = [points.get(p, 0) for p in d["squad"]]
        best, best_plan = review.hindsight_best(picks, gw, d["chip"])
        weeks.append({k: v for k, v in d.items() if k not in ("next", "season")} | {
            "points": result["points"] - config.HIT_COST * d["hits"], "gross": result["points"],
            "captain_played": result["captain"], "autosubs": [[int(o), int(i)] for o, i in subs],
            "best": round(float(best), 1), "best_xi": [int(p) for p in best_plan.lineups[gw]],
            "best_captain": int(best_plan.captains[gw]),
        })
        if not carried:
            last = saved[gw]
    upcoming = [d for g, d in saved.items() if g not in played and g > (played[-1] if played else 0)]
    nxt = {k: v for k, v in upcoming[0].items() if k not in ("next", "season")} if upcoming else None
    return {"model": next(iter(saved.values()))["model"] if saved else config.MODEL,
            "gameweeks": weeks, "next": nxt}
