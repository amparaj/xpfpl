import numpy as np
import pandas as pd
import pytest

from xpfpl import config, setpieces


def elements():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "team": [1, 1, 1, 2, 2],
        "penalties_order": [1, 2, None, None, None],
        "status": ["a", "a", "a", "a", "a"],
        "chance_of_playing_next_round": [None, None, None, None, None],
    })


def test_first_choice_takes_them_and_the_second_steps_in_when_he_is_out():
    e = elements()
    share = setpieces.takers(e)
    assert share[1] == 1 and share[2] == 0 and share[3] == 0
    assert np.isnan(share[4])                       # club 2 lists no order: no opinion
    e.loc[0, ["status", "chance_of_playing_next_round"]] = ["i", 0]
    share = setpieces.takers(e)
    assert share[1] == 0 and share[2] == 1


def test_adjustment_only_moves_players_whose_role_changed():
    matches = pd.DataFrame({
        "season": "2025-26", "gw": np.tile(np.arange(1, 11), 3), "code": np.repeat([11, 12, 13], 10),
        "minutes": 90.0, "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC")
        + pd.to_timedelta(np.tile(np.arange(10), 3) * 7, unit="D"),
    })
    # Player 12 (element 2) took a penalty every other match; 11 (element 1) none: the order now
    # says 1 is first choice, so 1 gains and 2 loses.
    pens = pd.DataFrame({"season": "2025-26", "gw": [2, 4, 6, 8, 10], "code": 12, "team_code": 1})
    frame = pd.DataFrame({"element": [1, 2, 3], "code": [11, 12, 13], "position": [4, 4, 3],
                          "played60_r5": [1.0, 1.0, 1.0], "xp": [3.0, 3.0, 3.0]})
    adj = setpieces.adjustment(frame, elements(), matches, pens)
    assert adj["pen_xp"].iloc[0] == pytest.approx(setpieces.PER_MATCH * setpieces.CONVERSION * 4)
    assert adj["pen_xp"].iloc[1] == pytest.approx(-0.5 * setpieces.CONVERSION * 4)
    assert adj["pen_xp"].iloc[2] == 0
    assert adj["pen_order"].iloc[0] == 1


def test_apply_is_recorded_but_off_by_default(monkeypatch):
    frame = pd.DataFrame({"element": [1], "code": [11], "position": [4], "played60_r5": [1.0], "xp": [3.0]})
    monkeypatch.setattr(setpieces, "penalty_shots",
                        lambda seasons=None: pd.DataFrame(columns=["season", "gw", "code", "team_code"]))
    out = setpieces.apply(frame, elements(), pd.DataFrame(columns=["season", "gw", "code", "minutes", "kickoff_time"]))
    assert out["pen_xp"].iloc[0] > 0
    assert out["xp"].iloc[0] == 3.0 + config.PENALTY_WEIGHT * out["pen_xp"].iloc[0]
