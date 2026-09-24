"""The live record scores saved pre-deadline forecasts once their gameweek is played."""

import numpy as np
import pandas as pd

from xpfpl import config, scorecard


def test_scorecard_scores_played_gameweeks_only(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path)
    season = "2099-00"
    folder = tmp_path / season
    folder.mkdir()
    rng = np.random.default_rng(0)
    elements = np.arange(1, 41)
    for gw in (3, 4):                          # GW4 hasn't been played yet
        pd.DataFrame({"element": elements, "position": 1 + elements % 4,
                      f"xp_{gw}": rng.uniform(0, 6, len(elements))}).to_csv(folder / f"gw{gw:02d}_mlp.csv", index=False)
    matches = pd.DataFrame({
        "season": season, "gw": np.repeat([1, 2, 3], len(elements)), "element": np.tile(elements, 3),
        "total_points": rng.integers(0, 10, 3 * len(elements)).astype(float),
        "minutes": rng.choice([0.0, 90.0], 3 * len(elements)),
    })
    report = scorecard.score(matches, season)
    gws = pd.DataFrame(report["gameweeks"])
    assert set(gws["gw"]) == {3}
    assert set(gws["subset"]) == {"All players", "Players getting minutes"}
    assert gws.loc[gws["subset"] == "All players", "n"].iloc[0] == len(elements)
    scorecard.save(report)
    assert scorecard.load(season)["season"] == season
    assert "Live record" in scorecard.summarise(report)
