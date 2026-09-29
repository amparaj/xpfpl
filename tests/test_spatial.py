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
