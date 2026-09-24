"""Read your own FPL team from the public API: squad, selling prices, bank, free transfers, chips."""

from dataclasses import dataclass

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
    selling = {}
    for pid in squad:
        now = prices[pid]["now_cost"]
        purchase = bought.get(pid, now - prices[pid]["cost_change_start"])
        selling[pid] = selling_price(now, purchase)

    ft = estimate_free_transfers(history, info["started_event"], next_gw)
    return MyTeam(
        team_id=team_id,
        name=info["name"],
        squad=selling,
        bank=round(bank, 1),
        free_transfers=max(ft - len(pending), 0),
        chips_used=[{"name": c["name"], "event": c["event"]} for c in history["chips"]],
        pending_transfers=len(pending),
    )
