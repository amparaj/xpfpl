import numpy as np
import pandas as pd

from xpfpl import spatial


def test_zones_cover_the_attacking_half_and_shots_land_in_the_right_one():
    z = spatial.zones()
    assert len(z) == 25 and z["x0"].min() == 0 and z["x1"].max() == 50
    # A penalty (11.5 from goal, central) sits in the band that starts at the spot; a six-yard tap-in
    # in the first band; a shot from beyond halfway in no zone.
    zone = spatial.zone_of(pd.Series([11.5, 2.0, 60.0]), pd.Series([50.0, 50.0, 50.0]))
    row = z.set_index("zone").loc[zone.iloc[0]]
    assert row["x0"] == 11.5 and row["y0"] < 50 < row["y1"]
    assert z.set_index("zone").loc[zone.iloc[1], "x0"] == 0
    assert np.isnan(zone.iloc[2])


def test_similar_pairs_players_with_the_same_profile_within_position():
    table = pd.DataFrame({
        "minutes": [900, 900, 900, 900, 100],
        "box_touches_p90": [6.0, 5.8, 1.0, 1.1, 6.0], "shots_p90": [4.0, 3.9, 1.0, 0.9, 4.0],
        "box_shot_share": [0.95, 0.9, 0.3, 0.35, 0.9],
    }, index=[1, 2, 3, 4, 5])
    position = pd.Series(4, index=table.index)
    found = spatial.similar(table, position, k=1)
    assert found[1][0][0] == 2 and found[3][0][0] == 4       # box strikers together, deep ones together
    assert 5 not in found                                    # too few minutes to compare


def _fake_load(tables):
    return lambda season, table: tables.get(table)


def test_keeper_shots_go_to_the_other_sides_keeper_on_the_pitch(monkeypatch):
    # Fixture 1: club 10 (keeper 100, off at 60 for 101) v club 20 (keeper 200). Outfield 300 is club 20's.
    players = pd.DataFrame({
        "fixture": 1, "code": [100, 101, 200, 300], "team_code": [10, 10, 20, 20],
        "minutes_played": [60, 30, 90, 90], "start_min": [0, 60, 0, 0], "finish_min": [60, 90, 90, 90],
    })
    shots = pd.DataFrame({
        "fixture": 1, "code": [300, 300, 300, 100], "team_code": [20, 20, 20, 10], "minute": [20, 75, 80, 30],
        "outcome": ["save", "goal", "miss", "save"], "start_x": 10.0, "start_y": 50.0, "xg": 0.1, "body_part": "right-foot",
    })
    monkeypatch.setattr(spatial.matchstats, "load", _fake_load({"players": players, "shots": shots}))
    position = pd.Series({100: 1, 101: 1, 200: 1, 300: 4})
    faced = spatial.keeper_shots("2026-27", position)
    assert list(faced["keeper"]) == [100, 101, 200]          # the sub keeper faced the 75th-minute goal; the miss isn't on target


def test_dc_target_is_ten_for_defenders_and_twelve_for_the_rest():
    g = pd.DataFrame({"code": [1, 1, 2, 2], "minutes": [90, 90, 90, 90], "dc": [10, 9, 11, 12]})
    out = spatial.dc_summary(g, pd.Series({1: 2, 2: 3}))
    assert out.loc[1, "dc_hits"] == 1 and out.loc[2, "dc_hits"] == 1
    assert out.loc[1, "dc_p90"] == 9.5


def test_similar_compares_each_position_on_its_own_metrics():
    # Two keepers alike on saves, two unlike; their box touches (a forward's metric) are ignored.
    table = pd.DataFrame({
        "minutes": 900, "saves_p90": [4.0, 3.9, 1.0, 1.2], "save_share": [0.75, 0.74, 0.55, 0.6],
        "box_touches_p90": [9.0, 0.0, 0.0, 9.0],
    }, index=[1, 2, 3, 4])
    found = spatial.similar(table, pd.Series(1, index=table.index), k=1)
    assert found[1][0][0] == 2 and found[3][0][0] == 4


def test_a_stat_nobody_has_all_season_counts_as_missing(monkeypatch):
    players = pd.DataFrame({"match_id": ["a", "b"], "code": [1, 2], "minutes_played": [90, 90], "tackles": [0, 0],
                            "interceptions": [2, 1], "saves": 0, "goals_conceded": 0, "goals_prevented": 0.0})
    monkeypatch.setattr(spatial.matchstats, "load", _fake_load({"players": players}))
    prof = spatial.profiles("2026-27")
    assert "tackles_p90" not in prof and prof.loc[1, "interceptions_p90"] == 2.0
