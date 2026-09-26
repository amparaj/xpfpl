import pandas as pd

from xpfpl import review


def test_team_forecast_counts_players_the_way_fpl_scores_them():
    picks = pd.DataFrame({"multiplier": [1, 2, 0, 1]}, index=[10, 11, 12, 13])
    xp = pd.Series({10: 3.0, 11: 5.0, 12: 4.0})              # 13 has no forecast: counts as 0
    assert review.team_forecast(picks, xp) == 3.0 + 2 * 5.0


def test_season_forecasts_use_the_saved_forecast_and_rebuild_the_rest(monkeypatch):
    matches = pd.DataFrame({"season": "2026-27", "gw": [1, 2, 2], "element": [7, 7, 8]})
    monkeypatch.setattr(review, "saved_forecasts", lambda season, model: {2: pd.Series({7: 4.0, 8: 1.5}, name="mlp")})
    monkeypatch.setattr(review, "rebuilt_forecasts", lambda m, season, gws, model: pd.DataFrame(
        {"gw": gws, "element": [7] * len(gws), "xp": [2.0] * len(gws)}))
    out = review.season_forecasts(matches, "2026-27", "mlp").sort_values(["gw", "element"])
    assert out[["gw", "element", "xp", "source"]].values.tolist() == [
        [1, 7, 2.0, review.REBUILT], [2, 7, 4.0, review.SAVED], [2, 8, 1.5, review.SAVED]]
