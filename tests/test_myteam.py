import numpy as np
import pandas as pd

from xpfpl import config, myteam
from xpfpl.simulate import Draws

# GKP 1 and 15, DEF 2-6, MID 7-11, FWD 12-14, and a spare forward (16) to transfer in.
POSITION = {1: 1, 15: 1, **{p: 2 for p in range(2, 7)}, **{p: 3 for p in range(7, 12)}, 12: 4, 13: 4, 14: 4, 16: 4}
LINEUP = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]          # 4-4-2
BENCH = [15, 6, 11, 14]


def _bs(deadline: str = "2099-10-03T10:00:00Z", current_finished: bool = True) -> dict:
    """GW5 current (finished or not), GW6 next with the given deadline."""
    return {"events": [{"id": 1, "deadline_time": "2026-08-15T10:00:00Z", "is_current": False, "is_next": False, "finished": True},
                       {"id": 5, "deadline_time": "2026-09-26T10:00:00Z", "is_current": True, "is_next": False,
                        "finished": current_finished},
                       {"id": 6, "deadline_time": deadline, "is_current": False, "is_next": True, "finished": False}]}


def _picks(order: list[int], captain: int, vice: int, chip=None, cost: int = 0) -> dict:
    return {"active_chip": chip, "entry_history": {"event_transfers_cost": cost, "bank": 5},
            "picks": [{"element": p, "position": i + 1, "is_captain": p == captain, "is_vice_captain": p == vice}
                      for i, p in enumerate(order)]}


def _players() -> pd.DataFrame:
    ids = sorted(POSITION)
    xp6 = {p: float(p % 5 + 1) for p in ids} | {16: 9.0}
    xp7 = {p: float((p * 3) % 7 + 1) for p in ids} | {16: 2.0}
    df = pd.DataFrame({"position": POSITION, "team": {p: p for p in ids}, "price": {p: 5.0 for p in ids},
                       "xp_6": xp6, "xp_7": xp7}).loc[ids]
    df["xp_total"] = df["xp_6"] + df["xp_7"]
    df.index.name = "element"
    return df


def _draws(players: pd.DataFrame) -> Draws:
    """Ten identical simulations, each player scoring his xP (rounded) every week."""
    ids = players.index.to_numpy()
    week = lambda g: np.repeat(players[f"xp_{g}"].round().to_numpy()[None, :], 10, axis=0)   # noqa: E731
    return Draws([6, 7], ids, np.stack([week(6), week(7)]).astype(np.int8))


def _team(**extra) -> dict:
    return {"source": "saved", "lineup": LINEUP, "bench": BENCH, "captain": 12, "vice": 13, "chip": None,
            "transfers": [], "hits": 0, "before": None} | extra


def test_save_and_load_a_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MY_TEAM_DIR", tmp_path)
    saved = myteam.save_plan("2026-27", 6, lineup=LINEUP, bench=BENCH, captain=12, vice=13,
                             transfers_out=[14], transfers_in=[16], hits=1, bank=0.35, before=range(15, 0, -1))
    assert myteam.saved_plan("2026-27", 6) == saved
    assert saved["transfers"] == [{"out": 14, "in": 16}] and saved["hits"] == 1
    assert saved["before"] == list(range(1, 16))
    assert myteam.saved_plan("2026-27", 7) is None


def test_showing_gameweek_stays_on_the_live_week_until_it_is_finished():
    assert myteam.showing_gameweek(_bs(current_finished=False)) == 5
    assert myteam.showing_gameweek(_bs(current_finished=True)) == 6
    season_over = {"events": [{"id": 38, "is_current": True, "is_next": False, "finished": True}]}
    assert myteam.showing_gameweek(season_over) is None


def test_shown_team_prefers_the_saved_plan_else_carries_last_week(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MY_TEAM_DIR", tmp_path)
    picks = {5: {"active_chip": "freehit", "picks": []}, 4: _picks(LINEUP + BENCH, 12, 13)}
    monkeypatch.setattr(myteam.api, "entry_picks", lambda team_id, gw: picks[gw])
    carried = myteam.shown_team(1, _bs())
    # GW5 was a Free Hit, so the team going into GW6 is GW4's.
    assert carried["source"] == "carried" and carried["gw"] == 6
    assert (carried["lineup"], carried["bench"], carried["captain"], carried["vice"]) == (LINEUP, BENCH, 12, 13)
    assert carried["transfers"] == [] and carried["chip"] is None
    myteam.save_plan("2026-27", 6, lineup=LINEUP, bench=BENCH, captain=7, vice=8)
    saved = myteam.shown_team(1, _bs())
    assert saved["source"] == "saved" and saved["captain"] == 7


def test_shown_team_after_the_deadline_is_the_team_locked_into_fpl(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MY_TEAM_DIR", tmp_path)
    myteam.save_plan("2026-27", 6, lineup=LINEUP, bench=BENCH, captain=7, vice=8,   # changed in FPL afterwards
                     settings={"horizon": 3, "must_have": [16]})
    locked_order = LINEUP[:-1] + [16] + [15, 6, 11, 13]
    picks = {6: _picks(locked_order, 16, 12, chip="3xc", cost=4), 5: _picks(LINEUP + BENCH, 12, 13)}
    monkeypatch.setattr(myteam.api, "entry_picks", lambda team_id, gw: picks[gw])
    monkeypatch.setattr(myteam.api, "entry_transfers", lambda team_id: [
        {"event": 6, "element_out": 14, "element_in": 16}, {"event": 3, "element_out": 1, "element_in": 2}])
    team = myteam.shown_team(1, _bs(deadline="2026-01-01T10:00:00Z"))
    assert team["source"] == "locked" and team["gw"] == 6
    assert (team["captain"], team["vice"], team["chip"], team["hits"]) == (16, 12, "3xc", 1)
    assert team["transfers"] == [{"out": 14, "in": 16}]
    assert team["before"] == sorted(LINEUP + BENCH)
    assert team["settings"] == {"horizon": 3, "must_have": [16]}     # what the last save was planned with


def test_outlook_scores_the_picked_week_then_the_best_xi():
    players = _players()
    out = myteam.outlook(_team(), players, [6, 7], _draws(players))
    week6, week7 = out["weeks"]
    xp6 = players["xp_6"]
    assert week6["lineup"] == LINEUP and week6["captain"] == 12
    assert week6["xp"] == round(xp6[LINEUP].sum() + xp6[12], 2)
    assert week6["bench_xp"] == round(xp6[BENCH].sum(), 2)
    # GW7: the same 15, with the best XI and the highest xP as captain.
    assert sorted(week7["lineup"] + week7["bench"]) == sorted(LINEUP + BENCH)
    assert players.at[week7["captain"], "xp_7"] == players.loc[week7["lineup"], "xp_7"].max()
    assert out["total"]["xp"] == round(week6["xp"] + week7["xp"], 2)
    # Every simulation is the same week, everyone plays: the simulated score is the (rounded) xP.
    rounded = players["xp_6"].round()
    assert week6["points"]["mean"] == rounded[LINEUP].sum() + rounded[12]
    assert out["sims"] == 10 and out["captains"][0]["element"] == 12
    assert out["against"] is None
    assert sorted(out["players"]["element"]) == sorted(LINEUP + BENCH)


def test_outlook_judges_the_transfers_against_last_weeks_squad():
    players, before = _players(), LINEUP + BENCH
    bench = [15, 6, 11, 16]                                  # 14 out, 16 in, for a hit
    out = myteam.outlook(_team(bench=bench, hits=1, before=before), players, [6, 7], _draws(players))
    assert out["total"]["xp"] == round(sum(w["xp"] for w in out["weeks"]) - config.HIT_COST, 2)
    assert out["against"]["p_better"] in (0.0, 1.0)         # identical simulations: all or nothing
    # Whole-number xP and identical simulations: the simulated margin is the xP margin.
    assert out["against"]["mean"] == out["against"]["xp"] < 0


def test_outlook_goes_back_to_the_old_squad_after_a_free_hit():
    players, before = _players(), LINEUP + BENCH
    lineup = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 16]
    out = myteam.outlook(_team(lineup=lineup, captain=16, vice=12, chip="freehit", before=before),
                         players, [6, 7], _draws(players))
    assert 16 in out["weeks"][0]["lineup"]
    assert sorted(out["weeks"][1]["lineup"] + out["weeks"][1]["bench"]) == sorted(before)


def test_outlook_without_simulations_still_gives_xp():
    out = myteam.outlook(_team(), _players(), [6, 7])
    assert out["sims"] is None and out["captains"] == []
    assert all(w["points"] is None for w in out["weeks"]) and out["total"]["points"] is None
