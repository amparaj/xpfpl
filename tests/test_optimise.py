import numpy as np
import pandas as pd

from xpfpl import config
from xpfpl.myteam import selling_price
from xpfpl.optimise import shortlist, solve

GWS = [6, 7]
LONG_GWS = [6, 7, 8, 9]


def make_players(n_teams: int = 20, seed: int = 0, gws: list[int] | None = None) -> pd.DataFrame:
    """Synthetic pool: per team 3 GKP, 8 DEF, 8 MID, 4 FWD with random prices and xP."""
    gws = gws or GWS
    rng = np.random.default_rng(seed)
    rows = []
    for team in range(1, n_teams + 1):
        for pos, n in ((1, 3), (2, 8), (3, 8), (4, 4)):
            for _ in range(n):
                price = round(rng.uniform(4.0, 12.0), 1)
                rows.append({"team": team, "position": pos, "price": price,
                             **{f"xp_{gw}": price * rng.uniform(0.2, 0.6) for gw in gws}})
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    df["name"] = [f"p{i}" for i in df.index]
    df["xp_total"] = df[[f"xp_{gw}" for gw in gws]].sum(axis=1)
    return df


def check_rules(players: pd.DataFrame, plan, budget: float) -> None:
    squad = players.loc[plan.squad]
    assert len(squad) == 15
    assert squad["position"].value_counts().to_dict() == config.SQUAD_SIZE
    assert squad["team"].value_counts().max() <= config.MAX_PER_CLUB
    for gw in GWS:
        xi = players.loc[plan.lineups[gw]]
        assert len(xi) == 11
        counts = xi["position"].value_counts()
        for pos in config.SQUAD_SIZE:
            assert config.LINEUP_MIN[pos] <= counts.get(pos, 0) <= config.LINEUP_MAX[pos]
        assert plan.captains[gw] in plan.lineups[gw]
        assert plan.vice_captains[gw] in plan.lineups[gw]
        assert plan.captains[gw] != plan.vice_captains[gw]
        assert players.at[plan.bench[gw][0], "position"] == 1  # bench GK in slot 1
    assert plan.budget_left >= -1e-6


def test_squad_from_scratch_follows_rules():
    players = make_players()
    plan = solve(players, GWS, bank=100.0)
    check_rules(players, plan, 100.0)


def test_transfers_respect_free_transfers_and_hits():
    players = make_players()
    start = solve(players, GWS, bank=100.0)
    # Shake up xP so the current squad is no longer optimal.
    shuffled = make_players(seed=1)
    owned = {p: players.at[p, "price"] for p in start.squad}

    plan = solve(shuffled, GWS, current_squad=owned, bank=start.budget_left,
                 free_transfers=1, max_hits=0)
    check_rules(shuffled, plan, 100.0)
    assert len(plan.transfers_in) <= 1
    assert len(plan.transfers_in) == len(plan.transfers_out)

    plan = solve(shuffled, GWS, current_squad=owned, bank=start.budget_left,
                 free_transfers=1, max_hits=2)
    assert len(plan.transfers_in) <= 3
    assert plan.hits == max(len(plan.transfers_in) - 1, 0)


def test_selling_price_keeps_half_of_rises():
    assert selling_price(now=58, purchase=55) == 5.6   # +0.3 rise -> keep 0.1
    assert selling_price(now=57, purchase=55) == 5.6   # +0.2 rise -> keep 0.1
    assert selling_price(now=53, purchase=55) == 5.3   # falls passed on in full


def test_must_have_and_banned():
    players = make_players()
    free = solve(players, GWS, bank=100.0)
    banned = free.squad[:2]
    forced = players.drop(free.squad).nsmallest(1, "xp_total").index[0]
    plan = solve(players, GWS, bank=100.0, must_have=[forced], banned=banned)
    check_rules(players, plan, 100.0)
    assert forced in plan.squad
    assert not set(banned) & set(plan.squad)


# ---------------------------------------------------------------- planning transfers week by week

def check_weekly_rules(players: pd.DataFrame, plan) -> None:
    """Every gameweek's squad must be a legal squad, not just the first one."""
    for gw, squad in plan.squads.items():
        s = players.loc[squad]
        assert len(squad) == 15
        assert s["position"].value_counts().to_dict() == config.SQUAD_SIZE
        assert s["team"].value_counts().max() <= config.MAX_PER_CLUB
        assert set(plan.lineups[gw]) <= set(squad)


def test_weekly_plan_keeps_every_squad_legal_and_affordable():
    players = make_players(seed=3, gws=LONG_GWS)
    start = solve(players, LONG_GWS, bank=100.0)
    owned = {p: players.at[p, "price"] for p in start.squad}
    later = make_players(seed=4, gws=LONG_GWS)          # same players and prices, new xP
    later["price"] = players["price"]

    plan = solve(later, LONG_GWS, current_squad=owned, bank=start.budget_left,
                 free_transfers=1, max_hits=1, plan_transfers=True, time_limit=60)

    check_weekly_rules(later, plan)
    assert set(plan.squads[LONG_GWS[0]]) == set(plan.squad)
    # Money is conserved: in every week the squad is worth no more than the bank plus what the
    # starting squad would sell for.
    budget = start.budget_left + sum(owned.values())
    for squad in plan.squads.values():
        cost = sum(owned.get(p, later.at[p, "price"]) for p in squad)
        assert cost <= budget + 1e-6


def test_weekly_plan_obeys_free_transfers_and_the_hit_allowance():
    players = make_players(seed=5, gws=LONG_GWS)
    start = solve(players, LONG_GWS, bank=100.0)
    owned = {p: players.at[p, "price"] for p in start.squad}
    later = make_players(seed=6, gws=LONG_GWS)

    plan = solve(later, LONG_GWS, current_squad=owned, bank=start.budget_left,
                 free_transfers=1, max_hits=0, plan_transfers=True, time_limit=60)

    check_weekly_rules(later, plan)
    assert plan.hits == 0
    # With one free transfer a week and no hits allowed, each week can bank at most one more.
    free = 1
    previous = set(owned)
    for gw in LONG_GWS:
        moves = len(set(plan.squads[gw]) - previous)
        assert moves <= free
        free = min(free - moves + 1, config.MAX_FREE_TRANSFERS)
        previous = set(plan.squads[gw])


def test_weekly_plan_beats_holding_one_squad():
    """Planning ahead can only help: it may always choose to make the same transfers."""
    players = make_players(seed=7, gws=LONG_GWS)
    start = solve(players, LONG_GWS, bank=100.0)
    owned = {p: players.at[p, "price"] for p in start.squad}
    later = make_players(seed=8, gws=LONG_GWS)

    common = dict(current_squad=owned, bank=start.budget_left, free_transfers=2, max_hits=1)
    held = solve(later, LONG_GWS, **common)
    planned = solve(later, LONG_GWS, **common, plan_transfers=True, time_limit=60)
    assert planned.objective >= held.objective - 1e-6


def test_future_transfers_are_reported_per_gameweek():
    players = make_players(seed=9, gws=LONG_GWS)
    start = solve(players, LONG_GWS, bank=100.0)
    owned = {p: players.at[p, "price"] for p in start.squad}
    later = make_players(seed=10, gws=LONG_GWS)

    plan = solve(later, LONG_GWS, current_squad=owned, bank=start.budget_left,
                 free_transfers=1, max_hits=0, plan_transfers=True, time_limit=60)

    for gw, moves in plan.future_transfers.items():
        assert gw in LONG_GWS[1:]
        assert len(moves["in"]) == len(moves["out"])
        assert set(moves["in"]) <= set(plan.squads[gw])
        assert not set(moves["out"]) & set(plan.squads[gw])


# ---------------------------------------------------------------- price changes

def test_price_weight_tilts_the_squad_towards_risers():
    """With a weight on price, the squad picked must be the one expected to gain more value."""
    rng = np.random.default_rng(11)
    players = make_players(seed=11)
    players["price_delta"] = rng.uniform(-0.05, 0.05, len(players))

    plain = solve(players, GWS, bank=100.0)
    priced = solve(players, GWS, bank=100.0, price_weight=5.0)

    assert (players.loc[priced.squad, "price_delta"].sum()
            > players.loc[plain.squad, "price_delta"].sum())
    # ...but not at any price: the xP given up must be small.
    assert sum(priced.xp.values()) > 0.9 * sum(plain.xp.values())


def test_price_delta_is_ignored_without_a_weight():
    players = make_players(seed=12)
    players["price_delta"] = 0.05
    plain = solve(players, GWS, bank=100.0)
    priced = solve(players, GWS, bank=100.0, price_weight=0.0)
    assert set(plain.squad) == set(priced.squad)


# ---------------------------------------------------------------- shortlisting

def test_shortlist_keeps_the_best_and_anyone_owned():
    players = make_players(seed=13)
    owned = [players["xp_total"].idxmin()]
    kept = shortlist(players, owned, per_position=3)

    assert owned[0] in kept.index
    for pos in config.SQUAD_SIZE:
        assert (kept["position"] == pos).sum() <= 3 + len(owned)
    best = players["xp_total"].idxmax()
    assert best in kept.index


def test_sensitivity_counts_every_run_once():
    from xpfpl.optimise import sensitivity
    players = make_players()
    robust = sensitivity(players, GWS, sims=4, noise=0.2, bank=100.0)
    for table in robust.values():
        assert table["runs"].sum() == 4
        assert abs(table["share"].sum() - 1.0) < 1e-9
