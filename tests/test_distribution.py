import numpy as np
import pandas as pd
import pytest

from xpfpl import distribution, models
from xpfpl.distribution import K, LOW
from xpfpl.models.trainer import TrainConfig
from tests.test_models import make_frame


def one_hot(points: int) -> np.ndarray:
    p = np.zeros((1, K))
    p[0, points - LOW] = 1.0
    return p


def test_a_sure_and_right_forecast_scores_zero():
    s = distribution.scores(one_hot(6), [6])
    assert s["log_score"] == pytest.approx(0.0) and s["rps"] == pytest.approx(0.0)


def test_rps_rewards_being_close_and_log_score_does_not():
    near, far = distribution.scores(one_hot(7), [8]), distribution.scores(one_hot(2), [8])
    assert near["rps"] == pytest.approx(1.0) and far["rps"] == pytest.approx(6.0)
    assert near["log_score"] == far["log_score"]          # both gave the real score ~0%


def test_scores_are_proper_the_truth_beats_a_sharper_guess():
    rng = np.random.default_rng(0)
    truth = np.full(K, 1e-9)
    truth[[2 - LOW, 3 - LOW, 6 - LOW]] = [0.6, 0.3, 0.1]
    truth /= truth.sum()
    y = rng.choice(np.arange(K) + LOW, size=20000, p=truth)
    sharp = np.full(K, 1e-9)
    sharp[2 - LOW] = 1.0
    sharp /= sharp.sum()
    good = distribution.scores(np.tile(truth, (len(y), 1)), y)
    bad = distribution.scores(np.tile(sharp, (len(y), 1)), y)
    assert good["log_score"] < bad["log_score"] and good["rps"] < bad["rps"]


def test_draws_become_smoothed_chances_and_ends_take_the_tails():
    draws = np.array([[2, -10], [2, 30], [3, 30], [-128, 30]])     # -128 never happens: scored first
    pmf = distribution.from_draws(np.clip(draws, -3, None), smoothing=0.0)
    assert pmf.shape == (2, K) and np.allclose(pmf.sum(axis=1), 1)
    assert pmf[0, 2 - LOW] == 0.5 and pmf[1, K - 1] == 0.75 and pmf[1, 0] == 0.25
    assert (distribution.from_draws(draws[:, :1]) > 0).all()        # smoothed: nothing at 0%


def test_form_bands_fall_back_to_the_overall_spread():
    history = pd.DataFrame({"position": [3] * 4, "total_points_r5": [5.0] * 4, "total_points": [2, 2, 8, 13]})
    table = distribution.form_bands(history)
    rows = pd.DataFrame({"position": [3, 1], "total_points_r5": [5.0, 0.0]})
    pmf = distribution.from_bands(table, rows)
    assert np.allclose(pmf.sum(axis=1), 1)
    assert pmf[0, 2 - LOW] > pmf[0, 3 - LOW]
    assert np.allclose(pmf[1], table.mean(axis=0))                  # goalkeepers never seen


def test_dist_model_gives_chances_whose_average_is_its_xp():
    train_df, val_df = make_frame(2000), make_frame(500, seed=1)
    for df in (train_df, val_df):        # form decides between a blank and a haul
        df["total_points"] = np.where(df["total_points_r5"] > 1.5, 9.0, 1.0).astype("float32")
    predictor = models.fit("dist", train_df, val_df, cfg=TrainConfig(epochs=40, batch_size=64, patience=3), quiet=True)
    pmf = predictor.distribution(val_df)
    assert pmf.shape == (len(val_df), K) and np.allclose(pmf.sum(axis=1), 1)
    assert np.allclose(predictor.predict(val_df), pmf @ predictor.values, atol=1e-4)
    # It learned something: its chances beat the overall spread of training scores.
    counts = np.bincount(distribution.classes(train_df["total_points"]), minlength=K) + 0.5
    flat = np.tile(counts / counts.sum(), (len(val_df), 1))
    assert distribution.scores(pmf, val_df["total_points"])["log_score"] < \
        distribution.scores(flat, val_df["total_points"])["log_score"]
