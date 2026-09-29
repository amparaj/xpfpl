import pandas as pd

from xpfpl.data import matchstats


def _raw():
    """The source's matches.csv: one league match (Arsenal 3 v Coventry 9) and one cup match."""
    return pd.DataFrame({
        "match_id": ["26-27-prem-arsenal-vs-coventry-city", "efl-arsenal-vs-hull"],
        "home_team": [3.0, 3.0], "away_team": [9.0, 88.0], "finished": [True, True],
        "tournament": ["prem", "efl-cup"],
        "home_possession": [64.0, 70.0], "away_possession": [36.0, 30.0],
        "home_expected_goals_xg": [1.88, 2.0], "away_expected_goals_xg": [0.2, None],
        "home_score": [3.0, 2.0], "away_score": [0.0, 0.0], "fotmob_id": [1, 2],
    })


def test_link_finds_the_fpl_fixture_of_league_matches_only():
    fixtures = pd.DataFrame({"fixture": [1, 2], "gw": [1, 2], "home_code": [3, 9], "away_code": [9, 3]})
    found = matchstats.link(_raw(), fixtures)
    assert found[["match_id", "fixture", "gw"]].values.tolist() == [["26-27-prem-arsenal-vs-coventry-city", 1, 1]]


def test_long_stats_pairs_home_and_away_columns():
    stats = matchstats.long_stats(_raw().iloc[:1])
    assert set(stats["stat"]) == {"possession", "expected_goals_xg"}      # not team/score, not unpaired columns
    xg = stats[stats["stat"] == "expected_goals_xg"].iloc[0]
    assert (xg["home"], xg["away"]) == (1.88, 0.2)
