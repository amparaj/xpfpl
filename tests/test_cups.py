import pandas as pd

from xpfpl.data import cups


def _raw_fixtures():
    """Two EPL clubs (codes 3 and 43): a European away tie each, and their league match."""
    return pd.DataFrame({
        "gameweek": [4, 4, 4],
        "kickoff_time": ["2026-09-08T19:00:00", "2026-09-09T19:00:00", "2026-09-12T14:00:00"],
        "home_team": [None, 43.0, 3.0],
        "away_team": [3.0, None, 43.0],
        "home_score": [0.0, 2.0, 1.0],
        "away_score": [1.0, 2.0, 1.0],
        "finished": [True, True, True],
        "match_id": ["cl-napoli-vs-arsenal", "efl-man-city-vs-hull", "prem-arsenal-vs-man-city"],
        "tournament": ["champions-league", "efl-cup", "prem"],
    })


def test_club_fixtures_keeps_epl_sides_of_non_epl_matches():
    fx = cups.club_fixtures(_raw_fixtures(), "2026-27")
    assert fx["team_code"].tolist() == [3, 43]
    assert fx["was_home"].tolist() == [False, True]
    assert fx.loc[fx["team_code"] == 3, "goals_for"].item() == 1.0
    assert (fx["tournament"] != cups.EPL).all()


def test_player_minutes_maps_to_fpl_codes_and_skips_league_matches():
    fx = cups.club_fixtures(_raw_fixtures(), "2026-27")
    stats = pd.DataFrame({"player_id": [1, 2, 1], "minutes_played": [90.0, 25.0, 90.0],
                          "start_min": [0.0, 65.0, 0.0],
                          "match_id": ["cl-napoli-vs-arsenal", "cl-napoli-vs-arsenal", "prem-arsenal-vs-man-city"]})
    players = pd.DataFrame({"player_id": [1, 2], "player_code": [1001, 1002], "team_code": [3, 3]})
    mins = cups.player_minutes(stats, players, fx)
    assert mins.sort_values("code")[["code", "minutes", "started"]].values.tolist() == [[1001, 90, True], [1002, 25, False]]


def test_rest_features_before_after_and_own_minutes():
    fx = cups.club_fixtures(_raw_fixtures(), "2026-27")
    later = fx.iloc[[0]].assign(match_id="cl-arsenal-vs-lille", kickoff_time=pd.Timestamp("2026-09-16T19:00", tz="UTC"))
    fx = pd.concat([fx, later], ignore_index=True)
    mins = pd.DataFrame({"match_id": ["cl-napoli-vs-arsenal"], "code": [1001], "team_code": [3],
                         "minutes": [90], "started": [True]})
    kickoff = pd.Timestamp("2026-09-12T14:00", tz="UTC")
    epl = pd.DataFrame({"season": ["2026-27", "2026-27", "2026-27", "2019-20"], "team_code": [3, 3, 43, 3],
                        "code": [1001, 1002, 2001, 1001],
                        "kickoff_time": [kickoff, kickoff, kickoff, pd.Timestamp("2019-09-12", tz="UTC")]})
    f = cups.rest_features(epl, fx, mins)

    arsenal = f.iloc[0]
    assert arsenal["cup_era"] == 1 and arsenal["cup_before"] == 1
    assert abs(arsenal["cup_before_days"] - (3 + 19 / 24)) < 1e-4
    assert arsenal["cup_before_level"] == cups.LEVELS["champions-league"]
    assert arsenal["cup_after"] == 1 and arsenal["cup_after_level"] == 3      # Lille four days later
    assert arsenal["cup_mins_before"] == 1.0 and arsenal["cup_started_before"] == 1
    assert f.iloc[1]["cup_mins_before"] == 0 and f.iloc[1]["cup_started_before"] == 0   # didn't play
    city = f.iloc[2]
    assert city["cup_before_level"] == cups.LEVELS["efl-cup"] and city["cup_after"] == 0
    assert city["cup_after_days"] == cups.WINDOW_DAYS
    assert f.iloc[3].tolist() == [0, 0, cups.WINDOW_DAYS, 0, 0, cups.WINDOW_DAYS, 0, 0, 0]   # no data that season
    assert f.index.equals(epl.index)
