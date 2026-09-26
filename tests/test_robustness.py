import numpy as np
import pandas as pd

from xpfpl import robustness
from xpfpl.features import TARGET


def _frame(n_gw=30, per_gw=200, seed=0, noise_a=0.5, noise_b=2.0):
    """Toy season: true expectation mu, points = mu + noise, a close and a poor forecast."""
    rng = np.random.default_rng(seed)
    rows = n_gw * per_gw
    mu = rng.gamma(2.0, 1.2, rows)
    return pd.DataFrame({
        "season": "2024-25", "gw": np.repeat(np.arange(1, n_gw + 1), per_gw),
        "element": np.tile(np.arange(per_gw), n_gw), "fixture": np.repeat(np.arange(n_gw), per_gw),
        "position": np.tile(np.arange(per_gw) % 4 + 1, n_gw), "played_r5": 1.0,
        TARGET: mu + rng.normal(0, 2.0, rows), "mu": mu,
        "a": mu + rng.normal(0, noise_a, rows), "b": mu + rng.normal(0, noise_b, rows),
    })


def test_isotonic_fit_is_monotone_and_recovers_the_curve():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 8, 20000)
    y = np.sqrt(x) * 2 + rng.normal(0, 1, len(x))          # a concave truth
    kx, ky = robustness.isotonic_fit(x, y)
    assert np.all(np.diff(ky) > 0) and np.all(np.diff(kx) > 0)
    grid = np.linspace(0.5, 7.5, 15)
    assert np.max(np.abs(robustness.recalibrate(grid, kx, ky) - np.sqrt(grid) * 2)) < 0.2


def test_recalibrate_keeps_the_order_and_extends_the_top_one_for_one():
    kx, ky = np.array([0.0, 2.0, 4.0]), np.array([0.5, 2.5, 3.5])
    p = np.array([1.0, 3.0, 5.0, 7.0])
    out = robustness.recalibrate(p, kx, ky)
    assert np.all(np.diff(out) > 0)
    assert out[2] == 4.5 and out[3] == 6.5                 # 3.5 + the xP beyond the top knot


def test_paired_bootstrap_separates_a_good_forecast_from_a_poor_one():
    df = _frame()
    res = robustness.paired_bootstrap(df, "a", "b", reps=300)
    assert res["rmse_diff"] < 0 and res["rmse_hi"] < 0 and res["p_a_better_rmse"] == 1.0
    assert res["spearman_diff"] > 0 and res["spearman_lo"] > 0
    same = robustness.paired_bootstrap(df, "a", "a", reps=100)
    assert same["rmse_diff"] == 0 and same["rmse_lo"] == 0


def test_calibration_line_reads_off_slope_and_intercept():
    p = np.linspace(0, 10, 101)
    line = robustness.calibration_line(2 + 0.5 * p, p)
    assert abs(line["slope"] - 0.5) < 1e-9 and abs(line["intercept"] - 2) < 1e-9


def test_winners_curse_shows_up_even_for_an_unbiased_forecast():
    df = _frame(noise_a=1.5)
    top = [r for r in robustness.winners_curse(df, "a", k=3) if r["season"] == "all"]
    assert all(r["bias"] > 0 for r in top)                   # the maximum of noisy guesses runs high
    truth = [r for r in robustness.winners_curse(df, "mu", k=3) if r["season"] == "all"]
    assert all(abs(r["bias"]) < 3 * 1.96 * r["se"] for r in truth)


def test_oracle_noisy_and_mapped_predictors():
    frame = pd.DataFrame({TARGET: [0.0, 2.0, 9.0]})

    class Const:
        def predict(self, f):
            return np.full(len(f), 4.0, dtype="float32")

    assert list(robustness.Oracle().predict(frame)) == [0.0, 2.0, 9.0]
    big = pd.DataFrame({TARGET: np.zeros(20000)})
    noisy = robustness.Noisy(Const(), 0.1, seed=0).predict(big)
    assert abs(noisy.mean() - 4.0) < 0.02 and 0.35 < noisy.std() < 0.45
    assert list(robustness.Mapped(Const(), lambda p: p / 2).predict(frame)) == [2.0, 2.0, 2.0]


def test_aligned_matches_rows_by_player_and_fixture():
    rows = pd.DataFrame({"season": "s", "element": [1, 2, 3], "fixture": [10, 10, 11]})
    other = rows.iloc[[2, 0, 1]].reset_index(drop=True)
    out = robustness._aligned(rows, other, np.array([30.0, 10.0, 20.0]))
    assert list(out) == [10.0, 20.0, 30.0]


def test_backtest_summary_gaps_and_noise_floor():
    bt = pd.DataFrame([
        {"season": s, "variant": v, "points": p}
        for s, pts in {"2023-24": {"ensemble": 2400, "baseline": 2200, "noise 0": 2390, "noise 1": 2420},
                       "2021-22": {"ensemble": 2300, "baseline": 2350, "noise 0": 2280, "noise 1": 2310}}.items()
        for v, p in pts.items()])
    out = robustness.backtest_summary(bt)
    gap = next(g for g in out["gaps"] if g["variant"] == "baseline")
    assert gap["gap_mean"] == -75 and gap["seasons_better"] == 1 and gap["seasons_worse"] == 1
    assert gap["gap_tuned_seasons"] == -200 and gap["gap_other_seasons"] == 50
    assert out["noise_floor"]["noise_range_by_season"] == {"2021-22": 30.0, "2023-24": 30.0}


def test_keyed_noise_is_the_same_for_a_player_week_whoever_asks():
    frame = pd.DataFrame({"season": "2023-24", "element": np.arange(4000) % 400,
                          "gw": np.arange(4000) // 400 + 1})
    z = robustness.keyed_normal(frame, seed=1)
    shuffled = frame.sample(frac=1.0, random_state=0)
    assert np.allclose(robustness.keyed_normal(shuffled, seed=1), z[shuffled.index])
    assert not np.allclose(robustness.keyed_normal(frame, seed=2), z)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05


def test_fit_holdout_early_stops_on_the_last_training_season(monkeypatch):
    from xpfpl import models

    calls = []

    class Fake:
        meta = {"best_epoch": 7}

    def fake_fit(name, train_df, val_df, cfg=None, quiet=False):
        calls.append((sorted(train_df["season"].unique()),
                      None if val_df is None else sorted(val_df["season"].unique()), cfg.epochs))
        return Fake()

    monkeypatch.setattr(models, "fit", fake_fit)
    history = pd.DataFrame({"season": ["2021-22", "2022-23", "2023-24"]})
    models.fit_holdout("mlp", history, quiet=True)
    assert calls[0][:2] == (["2021-22", "2022-23"], ["2023-24"])
    assert calls[1] == (["2021-22", "2022-23", "2023-24"], None, 7)


def test_latest_retune_picks_the_newest_report_checked_on_unseen_seasons(tmp_path):
    import json
    confirmed = {"generated": "2026-09-26T22:00:00", "model": "ensemble", "seasons": ["2023-24"], "replays": 4,
                 "chosen": {"horizon": 8}, "confirm_seasons": ["2025-26"],
                 "confirmation": [{"label": "tuned", "mean_points": 2200.0, "se": 10.0,
                                   "points_per_season": {"2025-26": 2200.0}, "replays": {"2025-26": [2200.0]}}]}
    (tmp_path / "tuning_old.json").write_text(json.dumps({**confirmed, "generated": "2026-09-20T00:00:00"}))
    (tmp_path / "tuning_new.json").write_text(json.dumps(confirmed))
    (tmp_path / "tuning.json").write_text(json.dumps({"generated": "2026-09-30T00:00:00", "trials": []}))
    got = robustness.latest_retune(tmp_path)
    assert got["generated"] == "2026-09-26T22:00:00"
    assert got["confirmation"] == [{"label": "tuned", "mean_points": 2200.0, "se": 10.0,
                                    "points_per_season": {"2025-26": 2200.0}}]
    assert robustness.latest_retune(tmp_path / "missing") is None
