from types import SimpleNamespace

import numpy as np
import pandas as pd

from xpfpl import simulate
from xpfpl.simulate import DNP, Draws


def tables() -> dict:
    """Bonus never awarded, no cards: the rules alone decide the points."""
    bonus = np.zeros((5, 3, 3, 2, 4))
    bonus[..., 0] = 1.0
    return {"bonus": bonus, "residual": {k: np.zeros(10) for k in range(1, 5)}}


def frame(n_per_side: int = 11, xp: float = 4.0) -> tuple[pd.DataFrame, np.ndarray]:
    """One fixture, two sides of `n_per_side` (1 GKP, 4 DEF, 4 MID, 2 FWD), every player nailed."""
    positions = [1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4][:n_per_side]
    rows = []
    for team, opp in ((1, 2), (2, 1)):
        for i, pos in enumerate(positions):
            rows.append({"element": team * 100 + i, "gw": 6, "season": "2026-27", "fixture": 1,
                         "team_code": team, "opp_code": opp, "position": pos,
                         "mkt_gf": 1.4, "mkt_ga": 1.2, "team_gf38": 1.35, "team_ga38": 1.35,
                         "goals_scored_p90_38": 0.3 if pos >= 3 else 0.05, "expected_goals_p90_38": 0.3 if pos >= 3 else 0.05,
                         "assists_p90_38": 0.2, "expected_assists_p90_38": 0.2, "saves_p90_38": 3.0 if pos == 1 else 0,
                         "defensive_contribution_p90_38": 8.0, "xg_era": 1.0, "dc_era": 1.0,
                         "played_r5": 1.0, "played60_r5": 0.9})
    df = pd.DataFrame(rows)
    return df, np.full(len(df), xp)


def test_simulated_average_matches_xp():
    df, xp = frame()
    xp = xp + np.linspace(-1.5, 3.0, len(xp))            # some players need more attack, some less
    inp = simulate.inputs(df, xp)
    draws = simulate.run(inp, tables(), sims=4000, seed=1)
    assert draws.dtype == np.int8 and draws.shape == (4000, len(df))
    mean = simulate.scored(draws).mean(axis=0)
    assert np.abs(mean - xp).mean() < 0.15


def test_clean_sheets_are_shared_by_a_side():
    df, xp = frame()
    inp = simulate.inputs(df, xp, play_factor=np.ones(len(df)))
    draws = simulate.run(inp, tables(), sims=2000, seed=2)
    pts = simulate.scored(draws)
    defenders = [i for i in range(len(df)) if df["position"][i] == 2 and df["team_code"][i] == 1]
    # Two defenders of the same side: their points move together (the same goals conceded).
    corr = np.corrcoef(pts[:, defenders[0]], pts[:, defenders[1]])[0, 1]
    assert corr > 0.3


def test_ruled_out_player_never_plays():
    df, xp = frame()
    factor = np.ones(len(df))
    factor[0] = 0.0
    xp[0] = 0.0
    draws = simulate.run(simulate.inputs(df, xp, play_factor=factor), tables(), sims=500)
    assert (draws[:, 0] == DNP).all()


def test_by_gameweek_sums_a_double():
    rows = pd.DataFrame({"element": [7, 7, 8], "gw": [6, 6, 6]})
    draws = np.array([[3, 5, DNP], [DNP, 2, 1], [DNP, DNP, 4]], dtype=np.int8)
    d = simulate.by_gameweek(draws, rows)
    assert list(d.elements) == [7, 8]
    assert d.points[0, :, 0].tolist() == [8, 2, DNP]       # both matches, one match, neither
    assert d.points[0, :, 1].tolist() == [DNP, 1, 4]


def test_team_score_autosubs_vice_and_chips():
    # XI: GKP 1, DEF 2-5, MID 6-9, FWD 10-11; bench: GKP 12, DEF 13, MID 14, FWD 15.
    position = {1: 1, **{p: 2 for p in range(2, 6)}, **{p: 3 for p in range(6, 10)}, 10: 4, 11: 4,
                12: 1, 13: 2, 14: 3, 15: 4}
    elements = np.arange(1, 16)
    week = np.full(15, 2, dtype=np.int8)
    week[5] = 10                     # player 6 (the captain) scores 10
    missed = week.copy()
    missed[5] = DNP                  # the captain doesn't play: the vice (7) takes the armband...
    missed[1] = DNP                  # ...and DEF 2 misses: DEF 13 comes on (first outfield sub that keeps a valid XI)
    d = Draws([6], elements, np.stack([week, missed])[None, :, :])
    lineup, bench = list(range(1, 12)), [12, 13, 14, 15]
    out = simulate.team_score(d, 6, lineup, bench, captain=6, vice=7, position=position)
    assert out["points"].tolist() == [20 + 10 + 10, 18 + 2 + 2 + 2]
    assert out["captain"].tolist() == [10, 2]
    triple = simulate.team_score(d, 6, lineup, bench, captain=6, vice=7, chip="3xc", position=position)
    assert triple["points"][0] == 20 + 10 + 20
    boost = simulate.team_score(d, 6, lineup, bench, captain=6, vice=7, chip="bboost", position=position)
    assert boost["points"][0] == 40 + 8


def test_compare_counts_the_same_weeks():
    elements = np.array([1, 2])
    pts = np.array([[[5, 1], [0, 3], [4, 4]]], dtype=np.int8)       # one GW, three simulations
    d = Draws([6], elements, pts)
    plan = lambda p: SimpleNamespace(lineups={6: [p]}, bench={6: []}, captains={6: p},  # noqa: E731
                                     vice_captains={6: p}, total_hits=0)
    out = simulate.compare(d, plan(1), plan(2), [6], {1: 3, 2: 3})
    assert out["p_better"] == 1 / 3 and out["p_tie"] == 1 / 3
