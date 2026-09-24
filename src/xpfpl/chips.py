"""Simple, explainable chip rules built on top of the optimiser.

Each chip gets a "gain" in xP versus not playing it this gameweek, compared with a threshold
from config.py. Only one chip can be played per gameweek, so the biggest qualifying gain wins.
Chips come in two sets (GW1-19 and GW20-38); an unused chip about to expire is flagged.
"""

from dataclasses import dataclass

import pandas as pd

from xpfpl import config
from xpfpl.optimise import Plan, solve

CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "3xc": "Triple Captain", "bboost": "Bench Boost"}


@dataclass
class ChipAdvice:
    chip: str
    gain: float
    threshold: float
    expiring: bool
    reason: str
    better_later: bool = False  # a later GW in the horizon (same chip window) looks better

    @property
    def recommended(self) -> bool:
        return self.gain >= self.threshold and not self.better_later


def available_chips(bs: dict, chips_used: list[dict], gw: int) -> dict[str, dict]:
    """Chips playable in `gw`: {name: chip window from bootstrap-static}."""
    out = {}
    for chip in bs["chips"]:
        if not chip["start_event"] <= gw <= chip["stop_event"]:
            continue
        used = any(c["name"] == chip["name"] and chip["start_event"] <= c["event"] <= chip["stop_event"]
                   for c in chips_used)
        if not used:
            out[chip["name"]] = chip
    return out


def fixture_counts(fixtures: list[dict], gameweeks: list[int], team_ids: list[int]) -> pd.DataFrame:
    """Teams x gameweeks matrix of fixture counts (0 = blank, 2 = double)."""
    counts = pd.DataFrame(0, index=team_ids, columns=gameweeks)
    for f in fixtures:
        if f["event"] in gameweeks:
            counts.loc[f["team_h"], f["event"]] += 1
            counts.loc[f["team_a"], f["event"]] += 1
    return counts


def advise(players: pd.DataFrame, gameweeks: list[int], plan: Plan, available: dict[str, dict],
           solve_kwargs: dict) -> list[ChipAdvice]:
    gw = gameweeks[0]
    advice = []
    discount = solve_kwargs.get("discount", config.DISCOUNT)
    bench_weight = solve_kwargs.get("bench_weight", config.BENCH_WEIGHT)

    def expiring(name: str) -> bool:
        return available[name]["stop_event"] == gw

    if "3xc" in available:
        cap_xp = players.at[plan.captains[gw], f"xp_{gw}"]
        later = max((players.at[plan.captains[g], f"xp_{g}"] for g in gameweeks[1:]
                     if g <= available["3xc"]["stop_event"]), default=0.0)
        reason = f"captain xP {cap_xp:.1f}" + (f" (a later GW looks better: {later:.1f})" if later > cap_xp else "")
        advice.append(ChipAdvice("3xc", cap_xp, config.TRIPLE_CAPTAIN_MIN_XP, expiring("3xc"), reason,
                                 better_later=later > cap_xp))

    if "bboost" in available:
        bench_xp = plan.bench_xp[gw]
        later = max((plan.bench_xp[g] for g in gameweeks[1:] if g <= available["bboost"]["stop_event"]),
                    default=0.0)
        reason = f"bench xP {bench_xp:.1f}" + (f" (a later GW looks better: {later:.1f})" if later > bench_xp else "")
        advice.append(ChipAdvice("bboost", bench_xp, config.BENCH_BOOST_MIN_XP, expiring("bboost"), reason,
                                 better_later=later > bench_xp))

    if "freehit" in available:
        kwargs = {**solve_kwargs, "unlimited_transfers": True}
        fh = solve(players, [gw], **kwargs)
        gain = fh.xp[gw] - plan.xp[gw]
        advice.append(ChipAdvice("freehit", gain, config.FREE_HIT_MIN_GAIN, expiring("freehit"),
                                 f"one-week squad xP {fh.xp[gw]:.1f} vs {plan.xp[gw]:.1f}"))

    if "wildcard" in available:
        kwargs = {**solve_kwargs, "unlimited_transfers": True}
        wc = solve(players, gameweeks, **kwargs)
        # Both sides as discounted xP after hits: the solver's objectives aren't comparable,
        # because a wildcard neither pays for hits nor banks a free transfer.
        gain = wc.value(gameweeks, discount, bench_weight) - plan.value(gameweeks, discount, bench_weight)
        advice.append(ChipAdvice("wildcard", gain, config.WILDCARD_MIN_GAIN, expiring("wildcard"),
                                 f"{len(gameweeks)}-GW rebuilt squad beats current plan by {gain:.1f} (discounted xP)"))

    return sorted(advice, key=lambda a: a.gain / a.threshold, reverse=True)
