"""Monte Carlo: thousands of simulated gameweeks behind every xP.

An xP is an average. "Salah 7.1 xP" hides whether that is a steady 6-8 or a coin flip between
2 and 15, and that difference is what a captain pick, a Triple Captain or a close transfer turns
on. This module plays each upcoming gameweek thousands of times and keeps every simulated score,
so the dashboard and the site can show a range and the chances behind it:
"a 38% chance of 10+, a 25% chance of 2 or fewer".

Each simulated gameweek, for every fixture (batched in PyTorch, like `fit_goal_rates`):

    1. each side's goals ~ Poisson(the fixture's expected goals: the market's where it has odds,
       else our club ratings, teams.py);
    2. each player: no minutes / a cameo / 60+, from the minutes model (models/minutes.py) where
       the model has one, else his recent appearance rates;
    3. each of his side's goals is his with probability (his per-90 rate x minutes on the pitch
       / the side's expected goals), and likewise the assists, so teammates' returns rise and
       fall with the side's goals and a club's total stays its Poisson draw;
    4. a clean sheet if his side conceded none (60+ minutes), -1 per 2 conceded for GKP/DEF,
       saves, defensive contribution;
    5. bonus drawn from history given his goals, assists and clean sheet; cards, own goals and
       penalties drawn from history (validate.match_tables).

Checked out of sample (`check`, in validate.py's report): on 2025-26, 79% of scores landed inside
the simulated 10th-90th percentile band (80% is right), and the chance of 2 or fewer was within
~3 points everywhere. The chance of 10+ ran high for the top players (10-20% given, 9-13% seen),
but that is the model's xP over-forecasting them that season (4.6 xP, 4.1 scored), not the
spread: the simulation is only as right as the xP it is built around.

Every player's simulated average is then matched to the model's xP (`run`): the model decides
how many points to expect, the simulation only decides how they are spread. So the numbers the
site shows next to each other always agree.

What this doesn't do, on purpose: change what the optimiser maximises. With linear scoring, the
team with the highest expected points is the same whether you average simulations or not;
Ramezani & Dinh (2026, arXiv:2505.02170) found the same for simulated forecasts. The simulations
are for reading risk, not for picking a different team.

Scores are kept as int8 with DNP (-128) marking "didn't play", which the auto-subs need.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from xpfpl import config, scoring

SIMS = config.SIM_RUNS or 5000   # simulated gameweeks per forecast (P(10+) to within ~0.7 points of %)
BATCH = 500                # simulations per PyTorch batch (keeps memory to a few hundred MB)
CALIBRATION_SIMS = 1000    # simulations per pass while matching the means to the model's xP
CAMEO_SHARE = 0.25         # a 1-59 minute appearance is on the pitch ~23 of 90 minutes
FULL_SHARE = 0.96          # a 60+ minute appearance, ~86 of 90
DC_SHAPE = 16.0            # CBIT counts vary a little more than Poisson: a gamma(16) mix on the rate
HAUL, BLANK = 10, 2        # "10+" and "2 or fewer" points, the two chances shown everywhere
ATTACK_FLOOR = 0.25       # matching to xP never cuts a player's attacking rates below a quarter
DNP = -128                 # int8 marker for "didn't play" (scores otherwise fit in -127..127)
BONUS_SEASONS = "2025-26"  # bonus and residual tables for live forecasts: the current scoring era


# ---------------------------------------------------------------- inputs

def _col(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    return frame[name].fillna(default).to_numpy(dtype="float64") if name in frame else np.full(len(frame), default)


def inputs(frame: pd.DataFrame, xp, minutes: pd.DataFrame | None = None, play_factor=None,
           dc_rate=None) -> pd.DataFrame:
    """What the simulation needs per player-fixture row of a feature frame.

    `xp` is the model's xP per row (the target the simulated mean is matched to); `minutes` the
    minutes model's `expectations(frame)` (p_cameo, p_full), if it has one; `play_factor` what
    predict.py multiplied the xP by for injury flags, midweek rotation and the market's "ruled
    out" (it scales the chance of playing). `dc_rate` overrides the per-90 CBIT rate (see
    `season_dc_rate`).
    """
    xg_era = _col(frame, "xg_era") > 0

    def rate(actual: str, expected: str) -> np.ndarray:    # per 90, goals blended with xG where it exists
        a, e = _col(frame, f"{actual}_p90_38"), _col(frame, f"{expected}_p90_38")
        return np.where(xg_era, 0.5 * a + 0.5 * e, a)

    lam_for = np.clip(_col(frame, "mkt_gf", 1.35), 0.1, 6.0)
    lam_against = np.clip(_col(frame, "mkt_ga", 1.35), 0.1, 6.0)
    # A player's rates were earned at his club's usual scoring level; this fixture moves them.
    attack = lam_for / np.clip(_col(frame, "team_gf38", 1.35), 0.5, 4.0)
    pressure = lam_against / np.clip(_col(frame, "team_ga38", 1.35), 0.5, 4.0)
    if minutes is not None:
        p_cameo, p_full = minutes["p_cameo"].to_numpy(dtype="float64"), minutes["p_full"].to_numpy(dtype="float64")
    else:
        full = np.clip(_col(frame, "played60_r5"), 0, 1)
        p_cameo, p_full = np.clip(_col(frame, "played_r5") - full, 0, 1), full
    factor = np.ones(len(frame)) if play_factor is None else np.clip(np.asarray(play_factor, dtype="float64"), 0, 1)

    own = frame["season"].astype(str) + "/" + frame["fixture"].astype(str) + "/" + frame["team_code"].astype(str)
    opp = frame["season"].astype(str) + "/" + frame["fixture"].astype(str) + "/" + frame["opp_code"].astype(str)
    codes, sides = pd.factorize(pd.concat([own, opp], ignore_index=True))
    return pd.DataFrame({
        "element": frame["element"].to_numpy() if "element" in frame else np.arange(len(frame)),
        "gw": frame["gw"].to_numpy() if "gw" in frame else 0,
        "side": codes[:len(frame)], "opp_side": codes[len(frame):],
        "position": frame["position"].astype(int).to_numpy(),
        "p_cameo": p_cameo * factor, "p_full": p_full * factor,
        "lam_for": lam_for, "lam_against": lam_against,
        "goal": rate("goals_scored", "expected_goals") * attack,
        "assist": rate("assists", "expected_assists") * attack,
        "save": _col(frame, "saves_p90_38") * pressure,
        "dc": _col(frame, "defensive_contribution_p90_38") if dc_rate is None else np.asarray(dc_rate, dtype="float64"),
        "dc_era": _col(frame, "dc_era", 1.0),
        "xp": np.clip(np.asarray(xp, dtype="float64"), 0, None),
    }, index=frame.index)


def season_dc_rate(frame: pd.DataFrame, prior_matches: float = 3.0) -> np.ndarray:
    """Each row's CBIT actions per 90 from his earlier matches in the same season, pulled towards
    his position's average by `prior_matches` matches' worth.

    For 2025-26, the first season that records CBIT: its 38-match per-90 feature still averages
    in matches from before the stat existed, which halves the simulated chance of reaching
    the threshold. Live forecasts don't need this: their 38 matches are all from 2025-26 on.
    """
    s = frame.sort_values("kickoff_time")
    dc, minutes = s["defensive_contribution"].fillna(0), s["minutes"].fillna(0)
    group = s.groupby(["season", "code"])
    before_dc = group["defensive_contribution"].cumsum().fillna(0) - dc
    before_min = minutes.groupby([s["season"], s["code"]]).cumsum() - minutes
    per_pos = dc.groupby(s["position"]).sum() / minutes.groupby(s["position"]).sum().clip(lower=1) * 90
    rate = (before_dc + s["position"].map(per_pos) * prior_matches) / (before_min / 90 + prior_matches)
    return rate.reindex(frame.index).to_numpy(dtype="float64")


def history_tables(matches: pd.DataFrame, since: str = BONUS_SEASONS) -> dict:
    """validate.match_tables (bonus given returns; cards, own goals, penalties) from the
    seasons scored like this one: defensive contributions and today's bonus system."""
    from xpfpl.validate import match_tables

    recent = matches[matches["season"] >= since]
    if recent.empty:
        recent = matches
    recent = recent.assign(played=(recent["minutes"] > 0).astype(float),
                           played60=(recent["minutes"] >= 60).astype(float),
                           dc_era=(recent["season"].str[:4].astype(int) >= 2025).astype(float))
    return match_tables(recent)


# ---------------------------------------------------------------- the simulation

def _tensor(values, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.array(values), dtype=dtype)


def _simulate(inp: pd.DataFrame, sims: int, seed: int, tables: dict, attack: np.ndarray,
              play: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(points, attacking points) per simulation and row, before matching to xP. Points are
    int8 with DNP; attacking points (goals, assists, bonus, saves) float32, for `run`."""
    gen = torch.Generator().manual_seed(seed)
    n, n_sides = len(inp), int(max(inp["side"].max(), inp["opp_side"].max())) + 1
    pos = _tensor(inp["position"], torch.long)
    side, opp = _tensor(inp["side"], torch.long), _tensor(inp["opp_side"], torch.long)
    # Each side's expected goals: the average over its players' rows, or over its opponents'
    # rows for a side with no players in the frame.
    own = np.bincount(inp["side"], inp["lam_for"], n_sides) / np.maximum(np.bincount(inp["side"], None, n_sides), 1)
    seen = np.bincount(inp["opp_side"], inp["lam_against"], n_sides)
    seen /= np.maximum(np.bincount(inp["opp_side"], None, n_sides), 1)
    lam = _tensor(np.clip(np.where(own > 0, own, seen), 0.05, None))
    p_full = _tensor(np.clip(inp["p_full"] * play, 0, 1))
    p_cameo = _tensor(np.clip(inp["p_cameo"] * play, 0, 1))
    p_cameo = torch.minimum(p_cameo, 1 - p_full)
    goal, assist = _tensor(inp["goal"] * attack), _tensor(inp["assist"] * attack)
    save, dc = _tensor(inp["save"] * attack), _tensor(inp["dc"])
    dc_ok = _tensor((inp["dc_era"] > 0) & (inp["position"] > 1), torch.bool)
    threshold = _tensor(inp["position"].map(scoring.DC_THRESHOLD))
    goal_pts = _tensor(inp["position"].map(scoring.GOAL_POINTS))
    cs_pts = _tensor(inp["position"].map(scoring.CLEAN_SHEET_POINTS))
    defensive, keeper = pos <= 2, pos == 1
    bonus_table = _tensor(tables["bonus"])                              # (pos, goals, assists, cs, bonus)
    pools = {k: _tensor(np.rint(v)) for k, v in tables["residual"].items() if len(v)}

    points, attacking = [], []
    for start in range(0, sims, BATCH):
        s = min(BATCH, sims - start)
        goals_side = torch.poisson(lam.expand(s, n_sides), generator=gen)
        scored, conceded = goals_side[:, side], goals_side[:, opp]
        u = torch.rand(s, n, generator=gen)
        full = u < p_full
        cameo = ~full & (u < p_full + p_cameo)
        played = full | cameo
        share = full * FULL_SHARE + cameo * CAMEO_SHARE
        # Each of the side's goals is his with probability (his expected goals / the side's), so
        # his average is his rate and the club's total stays the side's Poisson draw.
        g = torch.binomial(scored, (goal * share / lam[side]).clamp(max=0.95), generator=gen)
        a = torch.minimum(torch.binomial(scored, (assist * share / lam[side]).clamp(max=0.95), generator=gen),
                          scored - g)
        clean = full & (conceded == 0)
        sv = torch.poisson(save * share, generator=gen) * keeper
        mix = torch._standard_gamma(torch.full((s, n), DC_SHAPE), generator=gen) / DC_SHAPE
        dc_hit = dc_ok & (torch.poisson(dc * share * mix, generator=gen) >= threshold)
        probs = bonus_table[pos, g.clamp(max=2).long(), a.clamp(max=2).long(), clean.long()]
        bonus = (torch.rand(s, n, 1, generator=gen) > probs.cumsum(-1)[..., :3]).sum(-1).float()
        residual = torch.zeros(s, n)
        for k, pool in pools.items():
            pick = pool[torch.randint(len(pool), (s, n), generator=gen)]
            residual = torch.where(pos == k, pick, residual)
        attack_pts = goal_pts * g + scoring.ASSIST_POINTS * a + bonus + torch.floor(sv / scoring.SAVES_PER_POINT)
        pts = (played.float() + full.float() + attack_pts + cs_pts * clean
               - (defensive & full) * torch.floor(conceded / scoring.CONCEDED_PER_PENALTY)
               + scoring.DC_POINTS * dc_hit + residual)
        pts = torch.where(played, pts.clamp(-127, 127), torch.tensor(float(DNP)))
        points.append(pts.to(torch.int8).numpy())
        attacking.append((attack_pts * played).numpy())
    return np.concatenate(points), np.concatenate(attacking)


def scored(draws: np.ndarray) -> np.ndarray:
    """Simulated points with "didn't play" counted as 0."""
    return np.where(draws == DNP, 0, draws).astype(np.float32)


def run(inp: pd.DataFrame, tables: dict, sims: int = SIMS, seed: int = 0, passes: int = 2) -> np.ndarray:
    """Simulated points per simulation and row (int8, DNP for didn't play), each row's average
    matched to its `xp`.

    The match: the points that don't depend on attacking returns (appearance, clean sheets,
    conceded, defensive contribution, cards) stay as simulated, and the attacking rates are
    scaled until goals, assists, bonus and saves make up the rest. Where the model expects less
    than even that (it doubts he'll play), the chance of playing comes down too. Two passes get
    the averages within a few hundredths of the model's xP.
    """
    if not len(inp):
        return np.zeros((sims, 0), dtype=np.int8)
    xp = inp["xp"].to_numpy()
    attack, play = np.ones(len(inp)), np.ones(len(inp))
    for i in range(passes):
        pts, att = _simulate(inp, CALIBRATION_SIMS, seed + 1 + i, tables, attack, play)
        mean, a = scored(pts).mean(axis=0), att.mean(axis=0)
        base = mean - a
        # More attack when the model expects more than the rest provides (never below
        # ATTACK_FLOOR of the current rates: a doubt about returns is no reason to wipe them out)...
        attack = np.where(a > 0.02, attack * np.clip((xp - base) / np.maximum(a, 1e-6), ATTACK_FLOOR, 6.0), attack)
        # ...and a lower chance of playing for what's left over.
        expected = base + a * np.where(a > 0.02, np.clip((xp - base) / np.maximum(a, 1e-6), ATTACK_FLOOR, 6.0), 1.0)
        play = play * np.where(expected > 1e-6, np.clip(xp / np.maximum(expected, 1e-6), 0.0, 1.0), 1.0)
    return _simulate(inp, sims, seed, tables, attack, play)[0]


# ---------------------------------------------------------------- per player and gameweek

@dataclass
class Draws:
    """Simulated points for every player and gameweek of a forecast: `points[g, s, i]` is
    element `elements[i]`'s score in simulation s of `gameweeks[g]` (int8, DNP = didn't play;
    a double gameweek's two matches are summed, DNP only if he missed both)."""
    gameweeks: list[int]
    elements: np.ndarray
    points: np.ndarray

    def _cols(self, ids) -> np.ndarray:
        index = pd.Index(self.elements)
        cols = index.get_indexer(list(ids))
        if (cols < 0).any():
            raise KeyError(f"no simulations for {[p for p, c in zip(ids, cols) if c < 0]}")
        return cols

    def raw(self, gw: int, ids) -> np.ndarray:
        return self.points[self.gameweeks.index(gw)][:, self._cols(ids)]

    def pts(self, gw: int, ids) -> np.ndarray:
        """(sims, len(ids)) points, 0 where the player didn't play."""
        return scored(self.raw(gw, ids))

    @property
    def sims(self) -> int:
        return int(self.points.shape[1])


def by_gameweek(draws: np.ndarray, inp: pd.DataFrame, elements=None) -> Draws:
    """Per-row simulations -> one column per element and gameweek (doubles summed)."""
    gameweeks = sorted(int(g) for g in inp["gw"].unique())
    elements = np.asarray(sorted(inp["element"].unique()) if elements is None else elements)
    index = pd.Index(elements)
    out = np.full((len(gameweeks), draws.shape[0], len(elements)), DNP, dtype=np.int8)
    for gi, gw in enumerate(gameweeks):
        rows = np.flatnonzero(inp["gw"].to_numpy() == gw)
        cols = index.get_indexer(inp["element"].to_numpy()[rows])
        keep = cols >= 0
        rows, cols = rows[keep], cols[keep]
        total = np.zeros((draws.shape[0], len(elements)), dtype=np.int16)
        played = np.zeros((draws.shape[0], len(elements)), dtype=bool)
        np.add.at(total.T, cols, scored(draws[:, rows]).T.astype(np.int16))
        np.logical_or.at(played.T, cols, (draws[:, rows] != DNP).T)
        out[gi] = np.where(played, total.clip(-127, 127), DNP).astype(np.int8)
    return Draws(gameweeks, elements, out)


def summary(d: Draws, gw: int) -> pd.DataFrame:
    """Per element for one gameweek: the 10th/50th/90th percentile of his simulated points,
    the chance of 10+ (`p_haul`) and of 2 or fewer (`p_blank`, not playing included)."""
    pts = scored(d.points[d.gameweeks.index(gw)])
    q = np.percentile(pts, [10, 50, 90], axis=0, method="lower")
    return pd.DataFrame({"pts_p10": q[0], "pts_p50": q[1], "pts_p90": q[2],
                         "p_haul": (pts >= HAUL).mean(axis=0), "p_blank": (pts <= BLANK).mean(axis=0)},
                        index=pd.Index(d.elements, name="element"))


def path(season: str, gw: int, model: str):
    return config.PREDICTIONS_DIR / season / f"gw{gw:02d}_{model}_sims.npz"


def save(d: Draws, season: str, gw: int, model: str) -> None:
    """Next to the saved forecast (data/, not archived: a few MB a week, rebuilt by `predict`)."""
    target = path(season, gw, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, gameweeks=np.asarray(d.gameweeks), elements=d.elements, points=d.points)


def load(season: str, gw: int, model: str) -> Draws | None:
    try:
        with np.load(path(season, gw, model)) as f:
            return Draws([int(g) for g in f["gameweeks"]], f["elements"], f["points"])
    except (OSError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------- a team's score

def team_score(d: Draws, gw: int, lineup: list[int], bench: list[int], captain: int, vice: int,
               chip: str | None = None, position: dict[int, int] | None = None) -> dict[str, np.ndarray]:
    """A team's simulated gameweek score, scored as FPL does: auto-subs in bench order (keeping a
    valid formation; the keeper only for the keeper), the vice-captain if the captain didn't
    play, Triple Captain, Bench Boost. Returns per-simulation arrays: `points` (before hits),
    `captain` (the armband's points, once) and `bench` (the bench's points after auto-subs).
    `position` maps element -> position id (1-4); without it the auto-subs are skipped.
    """
    squad = list(lineup) + list(bench)
    raw = d.raw(gw, squad)
    played, pts = raw != DNP, scored(raw)
    n_xi, sims = len(lineup), raw.shape[0]
    in_xi = np.zeros((sims, len(squad)), dtype=bool)
    in_xi[:, :n_xi] = True
    if position is not None and chip != "bboost":
        pos = np.array([position[p] for p in squad])
        counts = np.stack([(in_xi & (pos == k)).sum(axis=1) for k in range(5)], axis=1)
        low = np.array([0] + [config.LINEUP_MIN[k] for k in (1, 2, 3, 4)])
        high = np.array([0] + [config.LINEUP_MAX[k] for k in (1, 2, 3, 4)])
        for b in range(n_xi, len(squad)):
            done = ~played[:, b]
            for o in range(n_xi):
                if (pos[b] == 1) != (pos[o] == 1):          # the keeper only swaps with the keeper
                    continue
                after = counts.copy()
                after[:, pos[o]] -= 1
                after[:, pos[b]] += 1
                valid = ((after >= low) & (after <= high)).all(axis=1)
                swap = ~done & in_xi[:, o] & ~played[:, o] & valid
                in_xi[swap, o], in_xi[swap, b] = False, True
                counts[swap] = after[swap]
                done |= swap
    elif chip == "bboost":
        in_xi[:] = True
    cap, vc = squad.index(captain), squad.index(vice)
    armband = np.where(played[:, cap], pts[:, cap], np.where(played[:, vc] & in_xi[:, vc], pts[:, vc], 0.0))
    extra = 2 if chip == "3xc" else 1
    total = (pts * in_xi).sum(axis=1) + extra * armband
    bench_pts = (pts * ~in_xi).sum(axis=1) if chip != "bboost" else pts[:, n_xi:].sum(axis=1)
    return {"points": total, "captain": armband, "bench": bench_pts}


def spread(values: np.ndarray, bins: int = 5) -> dict:
    """Mean, percentiles and a histogram (`bins`-point buckets) of simulated scores, for JSON."""
    q = np.percentile(values, [5, 10, 25, 50, 75, 90, 95])
    lo = int(np.floor(values.min() / bins) * bins)
    edges = np.arange(lo, values.max() + bins + 1, bins)
    counts, _ = np.histogram(values, edges)
    return {"mean": round(float(values.mean()), 2),
            **{f"p{p}": round(float(v), 1) for p, v in zip((5, 10, 25, 50, 75, 90, 95), q)},
            "histogram": {"start": lo, "width": bins, "shares": [round(c / len(values), 4) for c in counts]}}


def captain_odds(d: Draws, gw: int, candidates: list[int], multiplier: int = 2) -> pd.DataFrame:
    """Each captain option's simulated armband score (his points x `multiplier`): percentiles,
    the chance of 10+ and 2 or fewer (on his own points), and `p_best`, how often he outscores
    every other option in the same simulated week (ties shared)."""
    pts = d.pts(gw, candidates)
    top = pts.max(axis=1, keepdims=True)
    best = (pts == top) / (pts == top).sum(axis=1, keepdims=True)
    q = np.percentile(pts, [10, 50, 90], axis=0)
    return pd.DataFrame({"mean": pts.mean(axis=0), "pts_p10": q[0], "pts_p50": q[1], "pts_p90": q[2],
                         "p_haul": (pts >= HAUL).mean(axis=0), "p_blank": (pts <= BLANK).mean(axis=0),
                         "p_best": best.mean(axis=0), "armband_mean": multiplier * pts.mean(axis=0)},
                        index=pd.Index(candidates, name="element"))


def plan_score(d: Draws, plan, gameweeks: list[int], position: dict[int, int],
               discount: float = config.DISCOUNT, chip: str | None = None) -> np.ndarray:
    """A plan's simulated points over its horizon, discounted as the optimiser does, after hits:
    the simulated counterpart of `Plan.value` (without the bench weight)."""
    total = np.zeros(d.sims)
    for i, gw in enumerate(gameweeks):
        week = team_score(d, gw, plan.lineups[gw], plan.bench[gw], plan.captains[gw], plan.vice_captains[gw],
                          chip if i == 0 else None, position)
        total += discount ** i * week["points"]
    return total - config.HIT_COST * plan.total_hits


def compare(d: Draws, a, b, gameweeks: list[int], position: dict[int, int],
            discount: float = config.DISCOUNT) -> dict:
    """How often plan `a` beats plan `b` in the same simulated weeks (common random numbers:
    the same injuries, goals and clean sheets for both), and by how much."""
    diff = plan_score(d, a, gameweeks, position, discount) - plan_score(d, b, gameweeks, position, discount)
    q = np.percentile(diff, [10, 50, 90])
    return {"p_better": float((diff > 0).mean()), "p_tie": float((diff == 0).mean()),
            "mean": float(diff.mean()), "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}


def chip_odds(d: Draws, plan, gameweeks: list[int], available: dict[str, dict], position: dict[int, int]) -> dict:
    """For Triple Captain and Bench Boost: what the chip would add this gameweek in simulation
    (the captain's points once more; the bench's points), the chance that clears its config.py
    threshold, and the chance it beats playing the chip in a later gameweek of the plan (same
    chip window). The Wildcard and Free Hit are about whole squads, not one week's luck: none."""
    gw, out = gameweeks[0], {}
    thresholds = {"3xc": config.TRIPLE_CAPTAIN_MIN_XP, "bboost": config.BENCH_BOOST_MIN_XP}
    key = {"3xc": "captain", "bboost": "bench"}
    for chip in ("3xc", "bboost"):
        if chip not in available:
            continue

        def gain(g: int) -> np.ndarray:
            s = team_score(d, g, plan.lineups[g], plan.bench[g], plan.captains[g], plan.vice_captains[g],
                           None, position)
            return s[key[chip]]

        now = gain(gw)
        later = [gain(g) for g in gameweeks[1:] if g <= available[chip]["stop_event"] and g in d.gameweeks]
        best_later = np.max(later, axis=0) if later else None
        out[chip] = {"threshold": thresholds[chip], "p_clear": float((now >= thresholds[chip]).mean()),
                     "p_beats_later": None if best_later is None else float((now > best_later).mean()),
                     **{k: v for k, v in spread(now, bins=2).items() if k != "histogram"}}
    return out


# ---------------------------------------------------------------- is it right? (a held-out season)

HAUL_BINS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 1.01]
BLANK_BINS = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def _reliability(p: np.ndarray, hit: np.ndarray, bins: list[float]) -> list[dict]:
    """Rows grouped by the simulated chance; how often it actually happened in each group."""
    cut = pd.cut(p, bins, right=False)
    g = pd.DataFrame({"bin": cut, "p": p, "hit": hit}).groupby("bin", observed=True)
    return [{"bin": f"{b.left:.0%}-{min(b.right, 1):.0%}", "predicted": float(r["p"].mean()),
             "actual": float(r["hit"].mean()), "n": int(len(r)),
             "se": float(np.sqrt(max(r["hit"].mean() * (1 - r["hit"].mean()), 1e-9) / len(r)))}
            for b, r in g if len(r) >= 30]


def _pit(sim: np.ndarray, actual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised probability integral transform: where each actual score fell in its simulated
    distribution, 0-1. Flat if the ranges are right; a U shape if they are too narrow."""
    below = (sim < actual[None, :]).mean(axis=0)
    equal = (sim == actual[None, :]).mean(axis=0)
    return below + rng.random(len(actual)) * equal


def check(frame: pd.DataFrame, xp: np.ndarray, predictor, tables: dict, sims: int = 400, seed: int = 0) -> dict:
    """Score the simulated ranges on a held-out season's matches (validate.py's report).

    - `haul`/`blank`: of the players given a ~20% chance of 10+, did ~20% get 10+?
    - `pit`: where each real score fell in its simulated range, in tenths (flat = right width);
      `coverage_80`: the share inside the 10th-90th percentile band (0.8 if right).
    - `club`: the same for each club's total in each match, which tests whether teammates
      move together as much as they do in reality.
    - `mean_gap`: the simulated average's distance from the model's xP (should be ~0).
    Players getting minutes only (played in one of their previous five), as elsewhere.
    """
    expectations = getattr(predictor, "expectations", None)
    minutes = expectations(frame) if expectations else None
    first_dc = frame["season"].iloc[0] == BONUS_SEASONS         # see season_dc_rate
    inp = inputs(frame, xp, minutes, dc_rate=season_dc_rate(frame) if first_dc else None)
    draws = run(inp, tables, sims=sims, seed=seed)
    pts = scored(draws)
    y = frame["total_points"].to_numpy(dtype="float64")
    active = (frame["played_r5"] > 0).to_numpy()
    rng = np.random.default_rng(seed)
    pit = _pit(pts[:, active], y[active], rng)
    # Club totals per match: the players getting minutes, summed.
    club = pd.factorize(inp["side"].to_numpy()[active])[0]
    club_sim = np.zeros((pts.shape[0], club.max() + 1), dtype=np.float32)
    np.add.at(club_sim.T, club, pts[:, active].T)
    club_y = np.bincount(club, weights=y[active])
    club_pit = _pit(club_sim, club_y, rng)
    tenths = lambda v: [round(float(c), 4) for c in np.histogram(v, np.linspace(0, 1, 11))[0] / len(v)]  # noqa: E731
    return {
        "sims": sims, "rows": int(active.sum()), "clubs": int(len(club_y)),
        "mean_gap": float(np.mean(np.abs(pts[:, active].mean(axis=0) - np.asarray(xp)[active]))),
        "haul": _reliability((pts[:, active] >= HAUL).mean(axis=0), y[active] >= HAUL, HAUL_BINS),
        "blank": _reliability((pts[:, active] <= BLANK).mean(axis=0), y[active] <= BLANK, BLANK_BINS),
        "pit": tenths(pit), "coverage_80": float(((pit >= 0.1) & (pit <= 0.9)).mean()),
        "club_pit": tenths(club_pit), "club_coverage_80": float(((club_pit >= 0.1) & (club_pit <= 0.9)).mean()),
        "spread": {"simulated": float(pts[:, active].var(axis=0).mean() + pts[:, active].mean(axis=0).var()),
                   "actual": float(y[active].var())},
    }


def moves_label(players: pd.DataFrame, plan) -> str:
    out = ", ".join(players.at[p, "name"] for p in sorted(plan.transfers_out, key=lambda p: players.at[p, "position"]))
    into = ", ".join(players.at[p, "name"] for p in sorted(plan.transfers_in, key=lambda p: players.at[p, "position"]))
    return f"OUT {out} IN {into}" if into else "Roll the transfer(s)"


def transfer_odds(d: Draws, players: pd.DataFrame, gameweeks: list[int], plan, **solve_kwargs) -> list[dict]:
    """Does the recommended move really beat the alternatives, or is it a coin flip?

    The plan is played against, in the same simulated weeks over the horizon:
    - keeping the squad ("roll"), when the plan makes a move;
    - the next-best move: the best plan without the plan's signings, or, when the advice is to
      roll, the best move there is (the free transfer valued at nothing, so it gets used).
    `solve_kwargs` are the plan's own solve() arguments. Each entry: `label`, `moves` and
    `compare()`'s numbers, `p_better` being the chance the recommendation comes out ahead.
    """
    from xpfpl.optimise import solve

    gws = [g for g in gameweeks if g in d.gameweeks]
    position = players["position"].to_dict()
    rivals = []
    if plan.transfers_in:
        rivals.append(("Keep the squad", solve(players, gameweeks, **{**solve_kwargs, "free_transfers": 0,
                                                                     "max_hits": 0, "plan_transfers": False})))
        banned = tuple(solve_kwargs.get("banned", ())) + tuple(plan.transfers_in)
        rivals.append(("Next-best move", solve(players, gameweeks, **{**solve_kwargs, "banned": banned})))
    else:
        rivals.append(("Best move", solve(players, gameweeks, **{**solve_kwargs, "ft_value": 0.0})))
    out = []
    for label, rival in rivals:
        if set(rival.squad) == set(plan.squad) and not rival.future_transfers and not plan.future_transfers:
            continue                                        # the same team: nothing to compare
        out.append({"label": label, "moves": moves_label(players, rival),
                    **compare(d, plan, rival, gws, position)})
    return out
