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
    later_gw: int | None = None  # the best later GW of the horizon in the same window (3xc/bboost)
    later_gain: float = 0.0  # its gain

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


def window_warning(available: dict[str, dict], gw: int, spare: int | None = None) -> str | None:
    """A warning when the unused chips are running out of gameweeks to be played in (one per GW),
    or None while there's still room. `available` comes from `available_chips`."""
    spare = config.CHIP_SPARE_WEEKS if spare is None else spare
    if not available:
        return None
    stop = min(c["stop_event"] for c in available.values())
    left = [n for n, c in available.items() if c["stop_event"] == stop]
    weeks = stop - gw + 1
    if weeks > len(left) + spare:
        return None
    names = ", ".join(CHIP_NAMES.get(n, n) for n in left)
    chips_ = f"{len(left)} chip{'s' if len(left) > 1 else ''} left ({names})"
    gws = f"GW{gw}" if weeks == 1 else f"GW{gw}-{stop}"
    if weeks < len(left):
        return (f"{chips_} but only {weeks} GW{'s' if weeks > 1 else ''} ({gws}) to play them, one per GW: "
                f"{len(left) - weeks} will expire unused. Play the best one every week.")
    if weeks == len(left):
        return f"{chips_} and exactly {weeks} GW{'s' if weeks > 1 else ''} ({gws}): play one every week from now."
    return (f"{chips_} and {weeks} GWs ({gws}) to play them, one per GW: pick their weeks now, "
            f"they expire after GW{stop}.")


def fixture_counts(fixtures: list[dict], gameweeks: list[int], team_ids: list[int]) -> pd.DataFrame:
    """Teams x gameweeks matrix of fixture counts (0 = blank, 2 = double)."""
    counts = pd.DataFrame(0, index=team_ids, columns=gameweeks)
    for f in fixtures:
        if f["event"] in gameweeks:
            counts.loc[f["team_h"], f["event"]] += 1
            counts.loc[f["team_a"], f["event"]] += 1
    return counts


def _best_later(gains: dict[int, float]) -> tuple[int | None, float]:
    if not gains:
        return None, 0.0
    g = max(gains, key=gains.get)
    return g, float(gains[g])


def fixture_weeks(fixtures: list[dict], start: int, stop: int) -> dict[int, dict]:
    """The double and blank gameweeks from `start` to `stop`: {gw: {"doubles": [team ids],
    "blanks": [team ids]}}, normal weeks left out. Only fixtures FPL has dated count."""
    teams = {t for f in fixtures for t in (f["team_h"], f["team_a"])}
    out = {}
    for g in range(start, stop + 1):
        counts = dict.fromkeys(teams, 0)
        for f in fixtures:
            if f["event"] == g:
                counts[f["team_h"]] += 1
                counts[f["team_a"]] += 1
        doubles = sorted(t for t, n in counts.items() if n > 1)
        blanks = sorted(t for t, n in counts.items() if n == 0)
        if doubles or blanks:
            out[g] = {"doubles": doubles, "blanks": blanks}
    return out


@dataclass
class Verdict:
    """What to do with one chip this gameweek, in words. Display only: `advise` makes the call."""
    chip: str
    action: str  # "play", "close" (qualifies, but only just), "last" (expires this GW),
    #              "wait" (qualifies, but another chip is played this GW), "save"
    title: str
    detail: str

    @property
    def play(self) -> bool:
        return self.action in ("play", "close", "last")


def verdicts(advice: list[ChipAdvice], available: dict[str, dict], gw: int, captain: str | None = None,
             odds: dict | None = None, weeks: dict[int, dict] | None = None,
             team_names: dict[int, str] | None = None) -> list[Verdict]:
    """Turn `advise`'s numbers into a plain call per chip: play it, play it but it's a close call,
    or save it and why. One chip a gameweek, so only the strongest that qualifies is played.
    `odds` = simulate.chip_odds, `weeks` = fixture_weeks (to say whether a double or blank is
    coming before the chip expires), `team_names` for those clubs."""
    odds, weeks, team_names = odds or {}, weeks or {}, team_names or {}
    out, chosen = [], None
    for a in advice:
        name, stop = CHIP_NAMES[a.chip], available[a.chip]["stop_event"]
        what = f"{name} on {captain}" if a.chip == "3xc" and captain else name
        unit = {"3xc": "captain xP", "bboost": "bench xP"}.get(a.chip, "xP gained")
        numbers = f"{unit} {a.gain:.1f}; this model plays it from {a.threshold:g}"
        if not (a.recommended or (a.expiring and a.gain > 0)):
            if a.better_later and a.later_gw:
                why = f"GW{a.later_gw} looks better ({a.later_gain:.1f} vs {a.gain:.1f} now)."
            else:
                why = f"Not worth it this week ({numbers})."
            out.append(Verdict(a.chip, "save", f"Save {name}", f"{why} It lasts until GW{stop}."))
            continue
        if chosen is not None:
            out.append(Verdict(a.chip, "wait", f"Not this week: {name}",
                               f"It qualifies too ({numbers}), but only one chip can be played a gameweek "
                               f"and {CHIP_NAMES[chosen]} is the stronger case. It lasts until GW{stop}."))
            continue
        chosen = a.chip
        if a.expiring and not a.recommended:
            out.append(Verdict(a.chip, "last", f"Play {what}: last chance",
                               f"It expires after GW{gw}, so anything it adds is a gain ({numbers})."))
            continue
        if a.gain < a.threshold * (1 + config.CHIP_CLOSE_CALL):
            action, title = "close", f"Close call: {what}"
            detail = (f"It only just qualifies ({numbers}). Playing it now is reasonable, "
                      f"and so is waiting for a better week.")
        else:
            action, title = "play", f"Play {what}"
            detail = f"A clear case ({numbers})."
        o = odds.get(a.chip)
        if o:
            detail += (f" In the simulated weeks it adds {o['p10']:.0f}-{o['p90']:.0f} points (middle 80%) "
                       f"and reaches {o['threshold']:g} in {o['p_clear']:.0%} of them.")
            if o.get("p_beats_later") is not None:
                detail += f" This week beats every later week of the plan in {o['p_beats_later']:.0%}"
                detail += (": a later week comes out ahead more often than not." if o["p_beats_later"] < 0.5
                           else ".")
        if a.chip in ("3xc", "bboost", "freehit") and not a.expiring:
            kind, label = ("blanks", "blank") if a.chip == "freehit" else ("doubles", "double")
            coming = {g: w[kind] for g, w in weeks.items() if gw < g <= stop and w[kind]}
            if coming:
                g = min(coming)
                clubs = ", ".join(team_names.get(t, str(t)) for t in coming[g][:6])
                more = f" and {len(coming[g]) - 6} more" if len(coming[g]) > 6 else ""
                detail += (f" GW{g} is a {label} gameweek ({clubs}{more}), before the chip expires after "
                           f"GW{stop}: {label}s are usually the best weeks for it.")
            else:
                detail += (f" No {label} gameweeks are scheduled before it expires after GW{stop}, so there is "
                           f"no obvious better week coming.")
        out.append(Verdict(a.chip, action, title, detail))
    return sorted(out, key=lambda v: not v.play)   # the chip to play first, then the rest in advise's order


def headline(verdicts_: list[Verdict], gw: int) -> tuple[str, str]:
    """One line for the top of the page: (kind, text), kind = "play" / "close" / "none"."""
    pick = next((v for v in verdicts_ if v.play), None)
    if pick is None:
        return "none", f"GW{gw}: no chip this week. Save them all."
    if pick.action == "close":
        return "close", f"GW{gw}: {pick.title[len('Close call: '):]}, but it's a close call."
    return "play", f"GW{gw}: {pick.title}."


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
        later_gw, later = _best_later({g: players.at[plan.captains[g], f"xp_{g}"] for g in gameweeks[1:]
                                       if g <= available["3xc"]["stop_event"]})
        reason = f"captain xP {cap_xp:.1f}" + (f" (a later GW looks better: {later:.1f})" if later > cap_xp else "")
        advice.append(ChipAdvice("3xc", cap_xp, config.TRIPLE_CAPTAIN_MIN_XP, expiring("3xc"), reason,
                                 better_later=later > cap_xp, later_gw=later_gw, later_gain=later))

    if "bboost" in available:
        bench_xp = plan.bench_xp[gw]
        later_gw, later = _best_later({g: plan.bench_xp[g] for g in gameweeks[1:]
                                       if g <= available["bboost"]["stop_event"]})
        reason = f"bench xP {bench_xp:.1f}" + (f" (a later GW looks better: {later:.1f})" if later > bench_xp else "")
        advice.append(ChipAdvice("bboost", bench_xp, config.BENCH_BOOST_MIN_XP, expiring("bboost"), reason,
                                 better_later=later > bench_xp, later_gw=later_gw, later_gain=later))

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
