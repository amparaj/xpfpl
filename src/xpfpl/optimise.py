"""Squad selection as an integer linear program (PuLP + the bundled CBC solver).

Given xP per player per gameweek, choose the 15-man squad and the starting XI + captain for
each gameweek in the horizon to maximise discounted xP, subject to FPL's rules.

Two modes:

* the default holds this week's squad for the whole horizon. The horizon still matters - it
  stops the optimiser chasing a one-week wonder - but it has to assume no future transfers,
  which makes it pessimistic about a plan like "take Haaland now, fix the defence next week".
* `plan_transfers=True` gives every gameweek its own squad, linked by buy/sell variables, with
    free transfers banking up to 5 and transfer penalties costing 4 points, exactly as the game works. The
  answer is a route through the horizon, not a single squad; only this week's transfers are
  acted on, and the rest is re-planned next week from fresh predictions.

Both modes optionally value price changes: with `price_weight > 0` and a `price_delta` column
(expected £m per gameweek, from prices.py), holding a riser is worth a little extra.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from xpfpl import config


def _solver(time_limit: int) -> pulp.LpSolver:
    """A system CBC install (COIN_CMD) if present, else the CBC binary bundled with PuLP."""
    coin = pulp.COIN_CMD(msg=False, timeLimit=time_limit)
    if coin.available():
        return coin
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)


def _run(prob: pulp.LpProblem, time_limit: int, attempts: int = 2) -> int:
    """Solve `prob`, trying again if the CBC process itself fails. It very occasionally exits
    with an error on a model it solves fine the next time (once in ~500 season replays run in
    parallel by `xpfpl tune`), and one crash shouldn't end a whole run."""
    for attempt in range(attempts):
        try:
            return prob.solve(_solver(time_limit))
        except pulp.PulpSolverError:
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")


def shortlist(players: pd.DataFrame, keep_ids=(), per_position: int = 50) -> pd.DataFrame:
    """The best `per_position` candidates in each position, plus anyone in `keep_ids`.

    The ILP is exact, but over 700 players and five gameweeks it crawls, and a player outside
    the top 50 of his position is never the answer. Used by the backtest and by multi-week
    planning, which has five times as many variables to chew through.
    """
    keep = players.index.isin(list(keep_ids))
    for pos in config.SQUAD_SIZE:
        keep |= players.index.isin(players[players["position"] == pos]["xp_total"]
                                   .nlargest(per_position).index)
    return players[keep]


@dataclass
class Plan:
    squad: list[int]
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    lineups: dict[int, list[int]]        # gw -> starting XI
    captains: dict[int, int]             # gw -> captain
    vice_captains: dict[int, int]
    bench: dict[int, list[int]]          # gw -> bench in auto-sub order (GK first)
    xp: dict[int, float]                 # gw -> xP of XI incl. captain (and chip effects)
    bench_xp: dict[int, float]
    objective: float
    budget_left: float
    notes: list[str] = field(default_factory=list)
    squads: dict[int, list[int]] = field(default_factory=dict)      # gw -> 15 players that week
    future_transfers: dict[int, dict] = field(default_factory=dict)  # gw -> {in, out, hits} (planned)

    def formation(self, players: pd.DataFrame, gw: int) -> str:
        pos = players.loc[self.lineups[gw], "position"].value_counts()
        return f"{pos.get(2, 0)}-{pos.get(3, 0)}-{pos.get(4, 0)}"

    @property
    def total_hits(self) -> int:
        """Hits this week plus any the planned route takes later."""
        return self.hits + sum(m["hits"] for m in self.future_transfers.values())

    def value(self, gameweeks: list[int], discount: float = config.DISCOUNT,
              bench_weight: float = config.BENCH_WEIGHT, hits: bool = True) -> float:
        """Discounted xP of the plan, in points.

        `objective` also carries the solver's bookkeeping (the value of a banked free transfer,
        the price term), which differs between the two modes and between chips, so anything that
        compares two plans - the chip advice, above all - compares this instead.
        """
        total = sum(discount ** i * (self.xp[gw] + bench_weight * self.bench_xp[gw])
                    for i, gw in enumerate(gameweeks))
        return total - (config.HIT_COST * self.total_hits if hits else 0.0)


def solve(players: pd.DataFrame, gameweeks: list[int], *,
          current_squad: dict[int, float] | None = None,
          bank: float = 100.0,
          free_transfers: int = 1,
          max_hits: int = 0,
          unlimited_transfers: bool = False,
          triple_captain_gw: int | None = None,
          bench_boost_gw: int | None = None,
          discount: float = config.DISCOUNT,
          bench_weight: float = config.BENCH_WEIGHT,
          ft_value: float = config.FT_VALUE,
          price_weight: float = 0.0,
          plan_transfers: bool = False,
          pool_size: int | None = None,
          must_have: list[int] | tuple[int, ...] = (),
          banned: list[int] | tuple[int, ...] = (),
          time_limit: int = 60) -> Plan:
    """
    players:       indexed by element id with columns position, team, price, xp_<gw>
                   (and price_delta if price_weight > 0).
    current_squad: {element id: selling price}. None = build a squad from scratch with `bank` (£m).
    bank:          money in the bank (on top of the current squad's selling value).
    unlimited_transfers: wildcard / free hit (no hits, free transfers not consumed).
    plan_transfers: plan a separate squad for every gameweek in the horizon (see the module docstring).
    must_have / banned: element ids forced into / kept out of the squad (your own judgement on news).
    """
    owned = current_squad or {}
    # Prune players who can't matter, to keep the problem small.
    keep = (players["xp_total"] > 0.1) | players.index.isin(list(owned)) | players.index.isin(list(must_have))
    df = players[keep]
    if pool_size:
        df = shortlist(df, list(owned) + list(must_have), pool_size)
    ids = list(df.index)
    cost = {p: owned.get(p, df.at[p, "price"]) for p in ids}     # what selling p raises
    buy_cost = {p: df.at[p, "price"] for p in ids}
    budget = bank + sum(owned.values())
    delta = df["price_delta"].fillna(0.0) if price_weight and "price_delta" in df else None

    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    weekly = plan_transfers and bool(owned) and not unlimited_transfers
    # Squad membership: one set of variables in the default mode, one per gameweek when planning.
    # PuLP 3 formats variable names from the dict keys, so tuple keys are out: one dict per week.
    if weekly:
        squad_vars = {gw: prob.add_variable_dict(f"squad_{gw}", ids, cat="Binary") for gw in gameweeks}
    else:
        shared = prob.add_variable_dict("squad", ids, cat="Binary")
        squad_vars = {gw: shared for gw in gameweeks}

    lineup, cap = {}, {}  # keyed by (player, gw)
    for gw in gameweeks:
        for p, var in prob.add_variable_dict(f"xi_{gw}", ids, cat="Binary").items():
            lineup[p, gw] = var
        for p, var in prob.add_variable_dict(f"cap_{gw}", ids, cat="Binary").items():
            cap[p, gw] = var

    # --- squad rules, for each week's squad
    for gw, x in ({gameweeks[0]: squad_vars[gameweeks[0]]} if not weekly else squad_vars).items():
        for pos, n in config.SQUAD_SIZE.items():
            prob += pulp.lpSum(x[p] for p in ids if df.at[p, "position"] == pos) == n
        for team in df["team"].unique():
            prob += pulp.lpSum(x[p] for p in ids if df.at[p, "team"] == team) <= config.MAX_PER_CLUB
        for p in must_have:
            prob += x[p] == 1
        for p in banned:
            if p in x:
                prob += x[p] == 0

    # --- lineup + captain rules, every gameweek
    for gw in gameweeks:
        x = squad_vars[gw]
        prob += pulp.lpSum(lineup[p, gw] for p in ids) == 11
        prob += pulp.lpSum(cap[p, gw] for p in ids) == 1
        for pos in config.SQUAD_SIZE:
            n = pulp.lpSum(lineup[p, gw] for p in ids if df.at[p, "position"] == pos)
            prob += n >= config.LINEUP_MIN[pos]
            prob += n <= config.LINEUP_MAX[pos]
        for p in ids:
            prob += lineup[p, gw] <= x[p]
            prob += cap[p, gw] <= lineup[p, gw]

    if weekly:
        hit_terms, ft_end = _weekly_transfers(prob, df, ids, gameweeks, squad_vars, owned,
                                              cost, buy_cost, bank, free_transfers, max_hits)
    else:
        # --- transfers, one week only: everything after this week reuses the same squad.
        x = squad_vars[gameweeks[0]]
        transfers_in = pulp.lpSum(x[p] for p in ids if p not in owned)
        hits_first = prob.add_variable("hits", lowBound=0, cat="Integer")
        saved_ft = prob.add_variable("saved_ft", lowBound=0, upBound=1)
        prob += pulp.lpSum(cost[p] * x[p] for p in ids) <= budget
        if current_squad is None or unlimited_transfers:
            prob += hits_first == 0
            prob += saved_ft == 0
        else:
            prob += hits_first >= transfers_in - free_transfers
            prob += hits_first <= max_hits
            # Rolling an unused free transfer has some value, unless already at the cap.
            prob += saved_ft <= free_transfers - transfers_in
            if free_transfers >= config.MAX_FREE_TRANSFERS:
                prob += saved_ft == 0
        ft_end = saved_ft
        hit_terms = [hits_first] + [None] * (len(gameweeks) - 1)  # no transfers planned after week one

    # --- objective
    objective = []
    for i, gw in enumerate(gameweeks):
        w = discount ** i
        x = squad_vars[gw]
        cap_extra = 2 if gw == triple_captain_gw else 1
        bench_w = 1.0 if gw == bench_boost_gw else bench_weight
        for p in ids:
            xp = df.at[p, f"xp_{gw}"]
            objective.append(w * xp * (lineup[p, gw] + cap_extra * cap[p, gw]))
            objective.append(w * bench_w * xp * (x[p] - lineup[p, gw]))
        if delta is not None:
            objective += [w * price_weight * delta.at[p] * x[p] for p in ids]
        if hit_terms[i] is not None:                     # a hit is paid for in the week it's taken
            objective.append(-w * config.HIT_COST * hit_terms[i])
    prob += pulp.lpSum(objective) + ft_value * ft_end

    status = pulp.LpStatus[_run(prob, time_limit)]
    first = gameweeks[0]
    if squad_vars[first][ids[0]].value() is None:
        raise RuntimeError(f"Solver status: {status} - check budget/squad inputs.")
    notes = [] if status == "Optimal" else [
        f"Solver stopped at '{status}' after {time_limit}s; this plan is feasible but may not be optimal."]

    squads = {gw: [p for p in ids if squad_vars[gw][p].value() > 0.5] for gw in gameweeks}
    squad = squads[first]
    plan = Plan(
        squad=squad,
        transfers_in=[p for p in squad if p not in owned] if owned else [],
        transfers_out=[p for p in owned if p not in squad],
        hits=int(round(hit_terms[0].value() or 0)),
        lineups={}, captains={}, vice_captains={}, bench={}, xp={}, bench_xp={},
        objective=pulp.value(prob.objective),
        budget_left=round(budget - sum(cost[p] for p in squad), 1) + 0.0,  # +0.0 avoids "-0.0"
        notes=notes,
        squads=squads,
    )
    if weekly:
        for i, gw in enumerate(gameweeks[1:], start=1):
            ins = [p for p in squads[gw] if p not in squads[gameweeks[i - 1]]]
            outs = [p for p in squads[gameweeks[i - 1]] if p not in squads[gw]]
            if ins or outs:
                plan.future_transfers[gw] = {
                    "in": ins, "out": outs, "hits": int(round(hit_terms[i].value() or 0))}

    for gw in gameweeks:
        col = f"xp_{gw}"
        squad_gw = squads[gw]
        xi = [p for p in squad_gw if lineup[p, gw].value() > 0.5]
        captain = next(p for p in xi if cap[p, gw].value() > 0.5)
        vice = max((p for p in xi if p != captain), key=lambda p: df.at[p, col])
        bench = [p for p in squad_gw if p not in xi]
        bench_gk = [p for p in bench if df.at[p, "position"] == 1]
        bench_out = sorted((p for p in bench if df.at[p, "position"] != 1), key=lambda p: -df.at[p, col])
        cap_mult = 3 if gw == triple_captain_gw else 2
        plan.lineups[gw] = sorted(xi, key=lambda p: (df.at[p, "position"], -df.at[p, col]))
        plan.captains[gw] = captain
        plan.vice_captains[gw] = vice
        plan.bench[gw] = bench_gk + bench_out
        plan.xp[gw] = sum(df.at[p, col] for p in xi) + (cap_mult - 1) * df.at[captain, col]
        plan.bench_xp[gw] = sum(df.at[p, col] for p in bench)
    return plan


def _weekly_transfers(prob, df, ids, gameweeks, squad_vars, owned, cost, buy_cost, bank,
                      free_transfers, max_hits):
    """Buy/sell, bank and free-transfer bookkeeping for every gameweek in the horizon.

    The accounting is the game's own:
      squad this week = squad last week + buys - sells
      bank            = last week's bank + what the sells raised - what the buys cost
      free transfers  = last week's, minus the ones used, plus one, capped at 5
    Only upper bounds are written for the free-transfer count: having more is always at least
    as good, so the solver pushes it up by itself and the constraint stays linear.
    """
    hits, ft = [], prob.add_variable("ft_0", lowBound=0, upBound=config.MAX_FREE_TRANSFERS,
                                     cat="Integer")
    prob += ft == free_transfers
    balance = bank

    for i, gw in enumerate(gameweeks):
        x = squad_vars[gw]
        previous = {p: (squad_vars[gameweeks[i - 1]][p] if i else (1 if p in owned else 0)) for p in ids}
        buy = prob.add_variable_dict(f"buy_{gw}", ids, cat="Binary")
        sell = prob.add_variable_dict(f"sell_{gw}", ids, cat="Binary")
        for p in ids:
            prob += x[p] - previous[p] == buy[p] - sell[p]
            prob += buy[p] + sell[p] <= 1          # buying and selling the same player is never useful

        moves = pulp.lpSum(buy[p] for p in ids)
        week_hits = prob.add_variable(f"hits_{gw}", lowBound=0, cat="Integer")
        prob += week_hits >= moves - ft
        prob += week_hits <= max_hits
        hits.append(week_hits)

        # Money: the bank after this week's dealing can't go negative.
        balance = (balance + pulp.lpSum(cost[p] * sell[p] for p in ids)
                   - pulp.lpSum(buy_cost[p] * buy[p] for p in ids))
        cash = prob.add_variable(f"bank_{gw}", lowBound=0)
        prob += cash == balance
        balance = cash

        nxt = prob.add_variable(f"ft_{gw}", lowBound=0, upBound=config.MAX_FREE_TRANSFERS, cat="Integer")
        prob += nxt <= ft - (moves - week_hits) + 1
        ft = nxt
    return hits, ft


def sensitivity(players: pd.DataFrame, gameweeks: list[int], sims: int = 20, noise: float = 0.25,
                seed: int = 0, **solve_kwargs) -> dict[str, pd.DataFrame]:
    """How robust is this week's recommendation to the predictions being a little wrong?

    Perturb every player's xP, re-solve `sims` times, and count how often each decision comes out on top. A move that
    wins in most runs is robust; one that only wins in the noiseless solve is a coin-flip.

    The noise is one multiplicative factor per player, shared across the horizon (a player the
    model over-rates is over-rated every week): xP * exp(N(0, noise²) - noise²/2), which keeps the
    expected xP unchanged. `noise` = 0.25 is about the spread of the models' disagreements.
    Returns {"moves": this week's transfers (or "Roll"), "captains": the GW captain}, each with
    the number and share of runs it won.
    """
    rng = np.random.default_rng(seed)
    cols = [f"xp_{g}" for g in gameweeks]
    moves, captains = Counter(), Counter()
    for _ in range(sims):
        factor = np.exp(rng.normal(0.0, noise, len(players)) - noise ** 2 / 2)
        noisy = players.copy()
        noisy[cols] = players[cols].to_numpy() * factor[:, None]
        plan = solve(noisy, gameweeks, **solve_kwargs)
        out = ", ".join(sorted(players.at[p, "name"] for p in plan.transfers_out))
        into = ", ".join(sorted(players.at[p, "name"] for p in plan.transfers_in))
        moves[f"OUT {out}  IN {into}" if into else "Roll the transfer(s)"] += 1
        captains[players.at[plan.captains[gameweeks[0]], "name"]] += 1

    def table(counter: Counter, label: str) -> pd.DataFrame:
        rows = [{label: k, "runs": v, "share": v / sims} for k, v in counter.most_common()]
        return pd.DataFrame(rows, columns=[label, "runs", "share"])

    return {"moves": table(moves, "move"), "captains": table(captains, "captain")}
