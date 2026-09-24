"""The feature builder never looks ahead, at any staleness, and team ratings see strength."""

import numpy as np
import pandas as pd
import pytest

from xpfpl import features, teams
from xpfpl.data.history import MATCH_COLS


def make_matches(seasons=("2098-99", "2099-00"), n_teams: int = 6, per_team: int = 4, seed: int = 0,
                 strong: int = 1) -> pd.DataFrame:
    """A tiny league: every club plays every other home and away each season; club `strong`
    scores three goals a game, everyone else one."""
    rng = np.random.default_rng(seed)
    rows, fixture = [], 0
    start = pd.Timestamp("2098-08-01", tz="UTC")
    day = 0
    for s_i, season in enumerate(seasons):
        gw = 0
        for home in range(1, n_teams + 1):
            for away in range(1, n_teams + 1):
                if home == away:
                    continue
                gw += 1
                fixture += 1
                day += 3
                hg = 3 if home == strong else 1
                ag = 3 if away == strong else 1
                for team, opp, was_home in ((home, away, True), (away, home, False)):
                    for k in range(per_team):
                        code = team * 100 + k
                        minutes = float(rng.choice([0, 30, 90], p=[0.2, 0.2, 0.6]))
                        rows.append({
                            "season": season, "gw": gw, "kickoff_time": start + pd.Timedelta(days=day),
                            "fixture": fixture, "element": code, "code": code, "name": f"p{code}",
                            "position": 1 + k % 4, "team": team, "team_code": team, "opponent_team": opp,
                            "opp_code": opp, "was_home": was_home, "value": 50 + k,
                            "team_h_score": hg, "team_a_score": ag,
                            "total_points": float(rng.integers(0, 10)) if minutes else 0.0,
                            "minutes": minutes, "bps": float(rng.integers(0, 30)),
                            "selected": 1e5, "transfers_balance": float(rng.integers(-1000, 1000)),
                        })
    df = pd.DataFrame(rows)
    for col in MATCH_COLS:
        if col not in df:
            df[col] = 0.0
    df["expected_goals"] = np.nan
    return df[MATCH_COLS]


@pytest.fixture(autouse=True)
def ratings_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(teams, "CACHE_PATH", tmp_path / "ratings.parquet")


def test_ratings_find_the_strong_attack():
    table = teams.fit_history(make_matches())
    table = table[table["season"] == table["season"].max()]
    last = table[table["gw"] == table["gw"].max()].set_index("team_code")
    assert last["attack"].idxmax() == 1
    gf, ga = teams.fixture_rates(last, [1], [2], [True])
    assert gf[0] > 2 * ga[0]


def test_fixture_rates_fall_back_to_neutral_without_ratings():
    empty = pd.DataFrame(columns=["attack", "defence", "mu", "home"])
    gf, ga = teams.fixture_rates(empty, [1, 2], [2, 1], [True, False])
    assert np.all(np.isfinite(gf)) and np.all(gf > 0)


def test_a_rows_own_result_never_reaches_its_features():
    matches = make_matches()
    base = features.build_training_frame(matches)
    last = matches.index[-1]
    changed = matches.copy()
    changed.loc[last, ["total_points", "minutes", "bps", "transfers_balance"]] = [25.0, 90.0, 99.0, 0.0]
    after = features.build_training_frame(changed)
    row = base.index[(base["fixture"] == matches.at[last, "fixture"]) & (base["code"] == matches.at[last, "code"])]
    # transfer_flow is this gameweek's own pre-deadline transfers, known before kick-off by design.
    cols = [c for c in features.FEATURES if c != "transfer_flow"]
    pd.testing.assert_frame_equal(base.loc[row, cols], after.loc[row, cols])


def test_stale_features_are_the_previous_matchs_features():
    matches = make_matches()
    fresh = features.build_training_frame(matches, stale=1).sort_values("kickoff_time")
    stale = features.build_training_frame(matches, stale=2).sort_values("kickoff_time")
    col = "total_points_r5"
    expected = fresh.groupby("code")[col].shift(1).dropna()
    assert np.allclose(stale.loc[expected.index, col], expected)
    assert (stale["horizon"] == 2).all()
