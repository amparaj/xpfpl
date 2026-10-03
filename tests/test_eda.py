import json

import numpy as np

from xpfpl import eda, models
from xpfpl.models.trainer import TrainConfig
from tests.test_models import make_frame


def test_labels_are_plain_words():
    assert eda.label("total_points_r5") == "Points, last 5 (avg)"
    assert eda.label("expected_goals_p90_38") == "xG per 90, last 38"
    assert eda.label("ict_index_r3") == "ICT index, last 3 (avg)"
    assert eda.label("opp_xga") == "Opponent xG against, last 10"
    assert eda.label("transfer_flow") == "Net transfers this week"


def test_correlations_find_the_feature_points_follow():
    frame = make_frame(2000)
    frame["total_points"] = frame["total_points_r5"] * 3 + np.random.default_rng(0).normal(0, 0.5, len(frame))
    rows = eda.correlations(frame, top=3)
    assert {r["position"] for r in rows} == set(eda.POSITIONS.values())
    for pos in eda.POSITIONS.values():
        best = [r for r in rows if r["position"] == pos][0]
        assert best["feature"] == "total_points_r5" and best["rho"] > 0.9


def test_spread_and_shap_by_position(tmp_path, monkeypatch):
    frame = make_frame(1500)
    spread = eda.spread(frame, sample=50)
    for r in spread:
        assert abs(sum(h["share"] for h in r["histogram"]) - 1) < 1e-3
        assert r["p5"] <= r["p25"] <= r["p50"] <= r["p75"] <= r["p95"]
        assert len(r["sample"]) == 50
    gbm = models.fit("gbm", frame, None, cfg=TrainConfig(epochs=30), quiet=True)
    rows = eda.shap(frame, gbm, top=4, rows=500)
    assert len(rows) == 4 * 4 and all(r["mean_abs"] >= 0 for r in rows)

    monkeypatch.setattr(eda, "gbm_model", lambda: gbm)
    frame["season"] = "2099-00"
    frame["played_r5"] = 1.0
    report = eda.build(frame, path=tmp_path / "eda.json")
    assert json.loads((tmp_path / "eda.json").read_text())["seasons"] == ["2099-00"]
    assert report["shap"] and report["correlations"] and report["spread"]
