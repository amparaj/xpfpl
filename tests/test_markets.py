"""Betting-market odds: the classifier, the expected-goals fit, fixture matching and the fallback."""

import json
import math

import numpy as np
import pandas as pd

from xpfpl.data import markets
from xpfpl.features import add_market_features


def _pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _implied(lh: float, la: float) -> dict:
    """The market probabilities two Poisson rates imply (what a perfectly consistent market shows)."""
    grid = [[_pmf(lh, i) * _pmf(la, j) for j in range(11)] for i in range(11)]
    p = lambda cond: sum(grid[i][j] for i in range(11) for j in range(11) if cond(i, j))   # noqa: E731
    return {"home_win": p(lambda i, j: i > j), "draw": p(lambda i, j: i == j), "away_win": p(lambda i, j: i < j),
            "over_2.5": p(lambda i, j: i + j > 2), "btts": p(lambda i, j: i > 0 and j > 0),
            "home_over_0.5": 1 - _pmf(lh, 0), "away_over_0.5": 1 - _pmf(la, 0)}


def test_goal_rates_are_recovered_from_consistent_prices():
    rows = pd.DataFrame([_implied(1.8, 0.9), _implied(1.1, 1.4)])
    only_result = pd.DataFrame([{k: v for k, v in _implied(2.2, 0.7).items() if k in ("home_win", "draw", "away_win")}])
    fitted = markets.fit_goal_rates(pd.concat([rows, only_result], ignore_index=True), steps=800)
    assert np.allclose(fitted[["lam_home", "lam_away"]].to_numpy()[:2], [[1.8, 0.9], [1.1, 1.4]], atol=0.03)
    assert abs(fitted.at[2, "lam_home"] - 2.2) < 0.15 and abs(fitted.at[2, "lam_away"] - 0.7) < 0.15


def _market(question: str, kind: str | None = None, line: float | None = None) -> dict:
    return {"question": question, "sportsMarketType": kind, "line": line, "clobTokenIds": json.dumps(["1", "2"])}


def test_markets_are_classified_from_old_and_new_wording():
    teams = [{"name": "Tottenham Hotspur FC", "alias": "Spurs", "ordering": "home"},
             {"name": "Manchester City FC", "alias": "Man City", "ordering": "away"}]
    old = {"slug": "epl-tot-mac-2025-01-04", "teams": teams, "markets": [
        _market("Will Tottenham win on 2025-01-04?"), _market("Will Tottenham vs. Man City end in a draw?"),
        _market("Will Man City win on 2025-01-04?")]}
    new = {"slug": "epl-tot-mac-2025-01-04-more-markets", "teams": teams, "markets": [
        _market("Tottenham Hotspur FC vs. Manchester City FC: O/U 2.5", "totals", 2.5),
        _market("Tottenham Hotspur FC vs. Manchester City FC: 1st Half O/U 1.5", "first_half_totals", 1.5),
        _market("Tottenham Hotspur FC vs. Manchester City FC: Manchester City FC O/U 0.5", "soccer_team_totals", 0.5),
        _market("Tottenham Hotspur FC vs. Manchester City FC: Both Teams to Score", "both_teams_to_score"),
        _market("Spread: Manchester City FC (-1.5)", "spreads", -1.5)]}
    kept = markets._match_markets([old, new])
    assert set(kept) == {"home_win", "draw", "away_win", "over_2.5", "away_over_0.5", "btts"}
    assert "Tottenham win" in kept["home_win"]["question"] and "Man City win" in kept["away_win"]["question"]


def test_fixtures_match_on_clubs_and_date_and_keep_one_listing():
    fixtures = pd.DataFrame({"season": ["2025-26", "2025-26"], "fixture": [10, 200], "gw": [1, 20],
                             "kickoff_time": pd.to_datetime(["2025-08-16 14:00", "2026-01-10 15:00"], utc=True),
                             "home_code": [6, 6], "away_code": [43, 43]})
    listed = pd.DataFrame({"slug": ["a", "b", "c"], "home_code": [6, 6, 6], "away_code": [43, 43, 43],
                           "kickoff": pd.to_datetime(["2025-08-16 14:00", "2025-08-18 19:00", "2026-01-10 15:00"], utc=True),
                           "volume": [10.0, 99.0, 5.0]})
    out = markets.match_fixtures(listed, fixtures)
    assert sorted(out["fixture"]) == [10, 200]
    assert out.loc[out["fixture"] == 10, "slug"].item() == "b"        # the busier of the two listings


def test_market_features_fall_back_to_our_ratings():
    rows = pd.DataFrame({"season": "2025-26", "fixture": [1, 2], "team_code": [6, 6],
                         "fx_gf": [1.2, 1.5], "fx_ga": [1.0, 1.1], "fx_cs": [0.37, 0.33]})
    market = pd.DataFrame({"season": ["2025-26"], "fixture": [1], "home_code": [6], "away_code": [43],
                           "lam_home": [2.0], "lam_away": [0.8], "home_win": [0.6], "away_win": [0.2]})
    out = add_market_features(rows, market)
    assert out["mkt_known"].tolist() == [1.0, 0.0]
    assert out.loc[0, "mkt_gf"] == 2.0 and out.loc[1, "mkt_gf"] == 1.5 and out.loc[1, "mkt_cs"] == 0.33
    assert 0 < out.loc[1, "mkt_win"] < 1
    assert add_market_features(rows, market, use=False)["mkt_known"].sum() == 0


def test_ruled_out_and_movers_ignore_opening_prices():
    scorers = pd.DataFrame({
        "player": ["Out", "Found level", "News", "Steady"], "code": [1, 2, 3, 4], "fixture": 1,
        "p_anytime": [0.02, 0.20, 0.10, 0.30], "p_anytime_3d": [0.50, 0.50, 0.40, 0.31]})
    assert list(markets.ruled_out(scorers).index) == [1]
    _, players = markets.movers(None, scorers)
    # "Found level" moved off the ~50% opening price: not news. "Out" did too, but ended up out.
    assert list(players["player"]) == ["Out", "News", "Steady"]
    assert players["change"].iloc[0] < 0

    market = pd.DataFrame({"slug": ["a", "b"], "home_win": [0.5, 0.3], "draw": [0.3, 0.3], "away_win": [0.2, 0.4],
                           "home_win_3d": [0.52, 0.45], "draw_3d": [0.28, 0.25], "away_win_3d": [0.2, 0.3]})
    matches, _ = markets.movers(market, None, days=3)
    assert list(matches["slug"]) == ["b", "a"]                   # b's home win fell 15 points
    assert abs(matches["biggest_move"].iloc[0] - 0.15) < 1e-9
