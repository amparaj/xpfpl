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


def _shot(i, player, outcome, xg, xgot=None, home=True):
    return {"match_id": "m", "shot_index": i, "is_home": home, "player_id": player, "outcome": outcome,
            "xg": xg, "xgot": xgot, "code": None if player is None else 1000 + player,
            "team_code": None if player is None else (1 if home else 2)}


def _stats(player, club, shots, on_target, goals, xg, xgot):
    return {"match_id": "m", "player_id": player, "code": 1000 + player, "team_code": club, "total_shots": shots,
            "shots_on_target": on_target, "goals": goals, "xg": xg, "xgot": xgot}


def test_attribute_names_unrecorded_shots_from_the_players_match_stats():
    shots = pd.DataFrame([
        _shot(1, 1, "miss", 0.10),
        _shot(2, None, "goal", 0.40, 0.70),     # only player 2 owes a goal
        _shot(3, None, "miss", 0.05),           # players 1 and 3 each owe a miss: xG tells them apart
        _shot(4, None, "block", 0.30),
        _shot(5, None, "miss", 0.20, home=False),  # the away side's shooter, not a home player
    ])
    players = pd.DataFrame([
        _stats(1, 1, 2, 0, 0, 0.40, 0.0), _stats(2, 1, 1, 1, 1, 0.40, 0.70), _stats(3, 1, 1, 0, 0, 0.05, 0.0),
        _stats(4, 2, 1, 0, 0, 0.20, 0.0),
    ])
    out = matchstats.attribute(shots, players, pd.Series({"m": 1})).set_index("shot_index")
    assert out["player_id"].to_dict() == {1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
    assert out["inferred"].tolist() == [False, True, True, True, True]
    assert out.loc[2, "code"] == 1002 and out.loc[5, "team_code"] == 2


def test_attribute_leaves_a_shot_two_players_could_equally_have_taken():
    shots = pd.DataFrame([_shot(1, None, "miss", 0.10), _shot(2, None, "miss", 0.10)])
    players = pd.DataFrame([_stats(1, 1, 1, 0, 0, 0.10, 0.0), _stats(2, 1, 1, 0, 0, 0.10, 0.0)])
    out = matchstats.attribute(shots, players, pd.Series({"m": 1}))
    assert out["player_id"].isna().all() and not out["inferred"].any()


def _page(*rows):
    return pd.DataFrame([{"minute": m, "added_time": a, "is_home": h, "xg": x, "code": c} for m, a, h, x, c in rows])


def test_from_fotmob_names_shots_by_side_and_minute_even_when_xg_was_revised():
    shots = pd.DataFrame([
        {**_shot(1, 1, "miss", 0.10), "minute": 10, "added_time": None},
        {**_shot(2, None, "save", 0.09), "minute": 10, "added_time": None},   # FotMob now says 0.04
        {**_shot(3, None, "goal", 0.60), "minute": 83, "added_time": None},   # FotMob has the save before it at 82'
        {**_shot(4, 2, "save", 0.40), "minute": 83, "added_time": None},
    ])
    page = _page((10, None, True, 0.10, 1001), (10, None, True, 0.04, 1003),
                 (82, None, True, 0.40, 1002), (83, None, True, 0.60, 1004))
    people = pd.DataFrame({"player_code": [1003, 1004], "player_id": [3, 4], "team_code": [1, 1]})
    out = matchstats.from_fotmob(shots, {"m": page}, people).set_index("shot_index")
    assert out["code"].tolist() == [1001, 1003, 1004, 1002]
    assert out.loc[2, "player_id"] == 3 and out.loc[3, "player_id"] == 4


def test_from_fotmob_leaves_two_same_minute_shots_whose_xg_no_longer_agrees():
    shots = pd.DataFrame([{**_shot(1, None, "miss", 0.12), "minute": 49, "added_time": None},
                          {**_shot(2, None, "save", 0.13), "minute": 49, "added_time": None}])
    page = _page((49, None, True, 0.26, 1001), (49, None, True, 0.13 + 0.01, 1002))
    out = matchstats.from_fotmob(shots, {"m": page}, None)
    assert out["code"].isna().all()


def test_fetch_downloads_only_finished_matches_not_already_archived(tmp_path, monkeypatch):
    fixtures = pd.DataFrame({"fixture": [1, 2, 3], "gw": pd.array([1, 1, 2], dtype="Int64"), "home_code": [1, 3, 5],
                             "away_code": [2, 4, 6], "finished": [True, True, False]})
    monkeypatch.setattr(matchstats, "ARCHIVE", tmp_path)
    monkeypatch.setattr(matchstats, "fpl_fixtures", lambda season: fixtures)
    monkeypatch.setattr(matchstats.archive, "has_season", lambda season: True)
    asked = []

    def source(season, gw, name, refresh):
        asked.append((gw, name))
        if name != "matches":
            return None
        return pd.DataFrame({"match_id": ["a", "b"], "home_team": [1, 3], "away_team": [2, 4], "finished": [True, True],
                             "home_possession": [50, 60], "away_possession": [50, 40]})

    monkeypatch.setattr(matchstats.cups, "_csv", source)
    # Fixture 1 is archived already (all three tables), 2 is finished but new, 3 hasn't been played.
    for table in ("matches", "shots", "players"):
        (tmp_path / "2026-27" / table).mkdir(parents=True)
        pd.DataFrame({"match_id": ["a"], "fixture": [1], "gw": [1], "stat": ["possession"], "shot_index": [1],
                      "player_id": [1]}).to_parquet(tmp_path / "2026-27" / table / "gw01.parquet")
    matchstats.fetch("2026-27")
    assert {gw for gw, _ in asked} == {1}                       # nothing for GW2's unplayed match
    stats = pd.read_parquet(tmp_path / "2026-27" / "matches" / "gw01.parquet")
    assert sorted(stats["fixture"].unique()) == [1, 2]          # fixture 2 added, fixture 1 kept as it was
    asked.clear()
    for table in ("shots", "players"):                          # now all of GW1 is archived
        pd.DataFrame({"match_id": ["a", "b"], "fixture": [1, 2], "gw": [1, 1], "shot_index": [1, 1],
                      "player_id": [1, 2]}).to_parquet(
            tmp_path / "2026-27" / table / "gw01.parquet")
    matchstats.fetch("2026-27")
    assert asked == []
