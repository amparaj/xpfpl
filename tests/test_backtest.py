import numpy as np
import pandas as pd
import pytest

from xpfpl import backtest, config
from xpfpl.features import FEATURES, PLAYER_STATE, ROLLING_FEATURES, TEAM_STATE


def make_season(season: str = "2099-00", gws: int = 6, teams: int = 20, seed: int = 1) -> pd.DataFrame:
    """A miniature training frame: 20 clubs, a fixture each per gameweek, 15 players per club."""
    rng = np.random.default_rng(seed)
    rows = []
    element = 0
    for team in range(1, teams + 1):
        for pos, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
            for _ in range(n):
                element += 1
                skill = rng.uniform(0.5, 6.0)
                for gw in range(1, gws + 1):
                    opponent = (team % teams) + 1
                    rows.append({
                        "season": season, "gw": gw, "element": element, "code": element,
                        "name": f"p{element}", "position": pos, "team": team, "team_code": team,
                        "opponent_team": opponent, "opp_code": opponent, "was_home": gw % 2 == 0,
                        "fixture": gw * 100 + min(team, opponent), "value": 40 + int(skill * 5),
                        "kickoff_time": pd.Timestamp("2099-08-01", tz="UTC") + pd.Timedelta(days=7 * gw),
                        "total_points": float(max(0, round(rng.normal(skill, 2)))),
                        "minutes": float(rng.choice([0, 90], p=[0.25, 0.75])),
                        "team_gf": 1.4, "team_ga": 1.3, "experience": gw,
                    })
    df = pd.DataFrame(rows)
    for col in ROLLING_FEATURES:
        df[col] = rng.uniform(0, 4, len(df))
    state = PLAYER_STATE + [f"{side}_{c}" for side in ("team", "opp") for c in TEAM_STATE]
    extra = [c for c in dict.fromkeys(state + FEATURES) if c not in df]
    df = pd.concat([df, pd.DataFrame(rng.uniform(0, 2, (len(df), len(extra))), columns=extra)], axis=1)
    df["total_points_r5"] = df.groupby("element")["total_points"].transform("mean")  # the baseline model
    df["price"] = df["value"] / 10.0
    for p in range(1, 5):
        df[f"pos_{p}"] = (df["position"] == p).astype(float)
    for col in ("opp_gf", "opp_ga", "xg_era", "dc_era"):
        df[col] = 1.0
    df["was_home"] = df["was_home"].astype(float)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df


@pytest.fixture(scope="module")
def season_frame() -> pd.DataFrame:
    backtest._RATINGS["table"] = pd.DataFrame(columns=["season", "gw", "team_code", "attack",
                                                       "defence", "mu", "home"])   # no real data
    return make_season()


def test_autosubs_replace_blanks_in_bench_order():
    position = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1, 13: 2, 14: 3, 15: 4}
    minutes = {p: 90.0 for p in position}
    minutes[5] = 0.0      # a midfielder blanked
    minutes[13] = 0.0     # the first outfield sub also blanked, so he can't come on
    xi, subs = backtest.apply_autosubs(list(range(1, 12)), [12, 13, 14, 15], position, minutes)

    assert subs == [(5, 14)]
    assert 5 not in xi and 14 in xi
    assert len(xi) == 11


def test_autosubs_keep_the_formation_legal_and_only_swap_keeper_for_keeper():
    position = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 1, 13: 4}
    minutes = {p: 90.0 for p in position}
    minutes[2] = 0.0      # a defender blanked, leaving only two: a forward can't replace him
    xi, subs = backtest.apply_autosubs(list(range(1, 12)), [12, 13], position, minutes)
    assert subs == [] and xi == list(range(1, 12))

    minutes = {p: 90.0 for p in position}
    minutes[1] = 0.0      # the keeper blanked: only the bench keeper may come on
    xi, subs = backtest.apply_autosubs(list(range(1, 12)), [12, 13], position, minutes)
    assert subs == [(1, 12)]


def test_score_gameweek_counts_the_captain_twice_and_falls_back_to_the_vice():
    class FakePlan:
        """1 GK, 4 DEF, 4 MID, 2 FWD, with a keeper and three outfield players on the bench."""
        squad = list(range(1, 16))
        lineups = {1: list(range(1, 12))}
        bench = {1: [12, 13, 14, 15]}
        captains = {1: 3}       # a defender, on 10 points
        vice_captains = {1: 4}  # a defender, on 6 points

    position = {1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4,
                12: 1, 13: 4, 14: 3, 15: 2}
    minutes = {p: 90.0 for p in range(1, 16)}
    points = {p: 2.0 for p in range(1, 16)}
    points[3], points[4] = 10.0, 6.0
    xi_points = sum(points[p] for p in range(1, 12))

    scored = backtest.score_gameweek(FakePlan(), 1, points, minutes, position, None)
    assert scored["captain"] == 3
    assert scored["points"] == xi_points + 10.0          # the captain's score, counted twice
    assert scored["bench_points"] == 8.0                 # four bench players on 2 each

    scored = backtest.score_gameweek(FakePlan(), 1, points, minutes, position, "3xc")
    assert scored["points"] == xi_points + 20.0          # Triple Captain: counted three times

    minutes[3] = 0.0  # the captain didn't play: the armband moves to the vice...
    scored = backtest.score_gameweek(FakePlan(), 1, points, minutes, position, None)
    assert scored["captain"] == 4 and scored["captain_points"] == 6.0
    # ...and a bench forward comes on for him, so his 10 points sit on the bench instead.
    assert scored["autosubs"] == 1
    assert scored["points"] == xi_points - 10.0 + 2.0 + 6.0
    assert scored["bench_points"] == 16.0

    boosted = backtest.score_gameweek(FakePlan(), 1, points, minutes, position, "bboost")
    assert boosted["points"] == scored["points"] + 16.0


def test_chip_windows_offer_two_of_each_and_retire_used_ones():
    assert set(backtest._available_chips({}, 5)) == set(backtest.CHIP_ORDER)
    assert "wildcard" not in backtest._available_chips({"wildcard": [5]}, 9)
    assert "wildcard" in backtest._available_chips({"wildcard": [5]}, 25)   # second half, second chip
    assert "wildcard" not in backtest._available_chips({"wildcard": [5, 25]}, 30)


def test_snapshot_only_looks_backwards(season_frame):
    state, teams = backtest._snapshot(season_frame, gw=3)
    assert state["gw"].max() == 3          # the GW3 row itself, whose features precede GW3
    assert len(teams) == season_frame["team_code"].nunique()


def test_replay_obeys_the_rules_and_the_budget(season_frame):
    settings = backtest.Settings(season="2099-00", start_gw=1, end_gw=4, horizon=2,
                                 model="baseline", pool_size=20, budget=100.0)
    result = backtest.run(settings, frame=season_frame, verbose=False)
    gws = result.gameweeks

    assert list(gws["gw"]) == [1, 2, 3, 4]
    assert (gws["bank"] >= -1e-9).all()
    assert gws.loc[gws["gw"] == 1, "transfers"].iloc[0] == 0   # the opening squad isn't a transfer
    assert (gws["transfers"] <= 1 + settings.max_hits).all()   # one free transfer plus the hit allowance
    assert (gws["free_transfers"] <= config.MAX_FREE_TRANSFERS).all()
    assert (gws["points"] <= gws["gross"]).all()
    assert result.summary["points"] == pytest.approx(gws["points"].sum())


def test_replay_works_with_and_without_weekly_planning(season_frame):
    """Both optimiser modes must survive a replay and obey the same accounting."""
    results = {}
    for planning in (False, True):
        settings = backtest.Settings(season="2099-00", start_gw=1, end_gw=4, horizon=3,
                                     model="baseline", pool_size=20, plan_transfers=planning)
        results[planning] = backtest.run(settings, frame=season_frame, verbose=False).gameweeks

    for planning, gws in results.items():
        assert len(gws) == 4
        assert (gws["bank"] >= -1e-9).all()
        assert (gws["free_transfers"] <= config.MAX_FREE_TRANSFERS).all()
        assert gws.loc[gws["gw"] == 1, "transfers"].iloc[0] == 0
    # Planning ahead may transfer differently, but it can never spend more than it has.
    assert results[True]["squad_value"].min() > 0


def test_price_weight_reaches_the_optimiser(season_frame):
    """With a price weight set, the replay must hand the optimiser a price_delta column."""
    settings = backtest.Settings(season="2099-00", start_gw=1, end_gw=2, horizon=1,
                                 model="baseline", pool_size=20, price_weight=2.0)
    table = {"cells": [[0.5] * 4] * 6, "overall": 0.0, "rows": 0, "seasons": ""}
    state, teams = backtest._snapshot(season_frame, gw=1)
    pool = backtest._predict_horizon(season_frame, state, teams, [1], None, table)

    assert "price_delta" in pool
    assert pool["price_delta"].tolist() == pytest.approx([0.05] * len(pool))   # 0.5 tenths -> 0.05m
    assert len(backtest.run(settings, frame=season_frame, verbose=False).gameweeks) == 2


def test_results_are_saved_and_reloaded(season_frame, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKTEST_DIR", tmp_path)
    settings = backtest.Settings(season="2099-00", start_gw=1, end_gw=2, horizon=1,
                                 model="baseline", pool_size=20)
    csv, js = backtest.run(settings, frame=season_frame, verbose=False).save()

    assert csv.exists() and js.exists()
    # The tuning report lives in the same directory and must not be mistaken for a replay.
    (tmp_path / "tuning.json").write_text('{"chosen": {"horizon": 3}}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    loaded = backtest.load_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["settings"]["season"] == "2099-00"
    assert len(loaded[0]["gameweeks"]) == 2
