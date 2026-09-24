import pandas as pd
import pytest

from xpfpl import backtest, config, tune
from tests.test_backtest import make_season


@pytest.fixture(scope="module")
def two_seasons() -> pd.DataFrame:
    return pd.concat([make_season("2098-99", gws=4, teams=8, seed=1),
                      make_season("2099-00", gws=4, teams=8, seed=2)], ignore_index=True)


def test_tune_picks_a_value_for_every_parameter(two_seasons, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TUNING_PATH", tmp_path / "tuning.json")
    base = backtest.Settings(model="baseline", start_gw=1, end_gw=4, pool_size=12)
    stages = [("horizon", {"horizon": [1, 2]}),
              ("transfer planning", {"plan_transfers": [False, True]})]
    chip_stages = [("triple captain", {"3xc": [3.0, 30.0]})]

    report = tune.tune(["2098-99", "2099-00"], model="baseline", base=base, frame=two_seasons,
                       stages=stages, chip_stages=chip_stages)

    assert set(report["chosen"]) == {"horizon", "plan_transfers", "3xc"}
    assert report["chosen"]["horizon"] in (1, 2)
    assert len(report["trials"]) == 2 + 2 + 2
    for trial in report["trials"]:
        assert set(trial["points_per_season"]) == {"2098-99", "2099-00"}
        assert trial["mean_points"] == pytest.approx(
            sum(trial["points_per_season"].values()) / 2)

    assert config.TUNING_PATH.exists()
    assert tune.load_report(config.TUNING_PATH)["chosen"] == report["chosen"]


def test_config_block_names_the_real_settings(two_seasons):
    lines = tune.config_block({"horizon": 4, "discount": 0.9, "3xc": 7.5, "unknown": 1})

    assert "HORIZON = 4" in lines
    assert "DISCOUNT = 0.9" in lines
    assert "TRIPLE_CAPTAIN_MIN_XP = 7.5" in lines
    assert not any("unknown" in line for line in lines)
    for line in lines:
        assert hasattr(config, line.split(" = ")[0])   # every name really is a config.py setting


def test_summary_mentions_each_stage_and_the_spread():
    report = {"seasons": ["2099-00"], "model": "mlp", "config": ["HORIZON = 5"],
              "trials": [{"stage": "horizon", "horizon": 3, "mean_points": 2000.0},
                         {"stage": "horizon", "horizon": 5, "mean_points": 2100.0}]}
    text = tune.summarise(report)

    assert "horizon" in text
    assert "spread 100.0" in text
    assert "HORIZON = 5" in text
