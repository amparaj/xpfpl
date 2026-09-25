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


def test_unplayed_midweek_match_leaves_minutes_unknown():
    fx = cups.club_fixtures(_raw_fixtures(), "2026-27")
    epl = pd.DataFrame({"season": ["2026-27"], "team_code": [3], "code": [1001],
                        "kickoff_time": [pd.Timestamp("2026-09-12T14:00", tz="UTC")]})
    f = cups.rest_features(epl, fx, pd.DataFrame(columns=["match_id", "code", "team_code", "minutes", "started"]))
    assert f.iloc[0]["cup_before"] == 1
    assert pd.isna(f.iloc[0]["cup_mins_before"]) and pd.isna(f.iloc[0]["cup_started_before"])
    assert cups.group(f.assign(minutes_r5=90.0)).isna().all()


def test_group_splits_regulars_and_squad_players_by_midweek_minutes():
    df = pd.DataFrame({"cup_before": [1, 1, 1, 1, 1, 0],
                       "cup_mins_before": [0, 30 / 90, 1.0, 0, 0.5, 0],
                       "minutes_r5": [85, 85, 85, 10, 10, 85]})
    assert cups.group(df).tolist()[:5] == ["regular_rested", "regular_part", "regular_full",
                                           "squad_unused", "squad_played"]
    assert pd.isna(cups.group(df).iloc[5])


def _season_rows(n_weeks=4):
    """Two EPL rows per week at club 3, each after a midweek match: a squad player who played
    midweek and scored 4, and one who didn't and scored 0."""
    fx, mins, epl = [], [], []
    for w in range(n_weeks):
        cup_time = pd.Timestamp("2026-09-08T19:00", tz="UTC") + pd.Timedelta(weeks=w)
        fx.append({"season": "2026-27", "gw": w + 1, "match_id": f"cl{w}", "tournament": "champions-league",
                   "kickoff_time": cup_time, "team_code": 3})
        mins.append({"match_id": f"cl{w}", "code": 1, "team_code": 3, "minutes": 90, "started": True})
        for code, pts in ((1, 4), (2, 0)):
            epl.append({"season": "2026-27", "team_code": 3, "code": code, "minutes_r5": 10.0,
                        "kickoff_time": cup_time + pd.Timedelta(days=4), "total_points": pts, "gw": w + 1})
    return pd.DataFrame(fx), pd.DataFrame(mins), pd.DataFrame(epl)


def test_fit_and_adjust_move_xp_between_squad_players(tmp_path, monkeypatch):
    fx, mins, epl = _season_rows()
    monkeypatch.setattr(cups, "load", lambda: (fx, mins))
    monkeypatch.setattr(cups, "FACTORS_PATH", tmp_path / "cups.json")
    monkeypatch.setattr(cups, "PRIOR_XP", 0.0)
    fitted = cups.fit(epl, [2.0] * len(epl), "mlp")
    g = fitted["groups"]
    assert g["squad_played"]["rows"] == 4 and g["squad_played"]["factor"] == 2.0
    assert g["squad_unused"]["factor"] == 0.0
    assert g["regular_full"]["rows"] == 0

    out = cups.adjust(epl, next_gw=2, model="mlp")
    week2 = out[epl["gw"] == 2]
    assert week2["rotation_factor"].tolist() == [2.0, 0.0]
    assert (out.loc[epl["gw"] != 2, "rotation_factor"] == 1.0).all()     # later weeks untouched
    assert cups.factors("gbm") is None
