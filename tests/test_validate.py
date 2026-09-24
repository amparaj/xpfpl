import numpy as np
import pandas as pd

from xpfpl import validate


def make_val(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """A fake held-out season: points loosely follow a hidden 'quality', with plenty of noise."""
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0, 6, n)
    df = pd.DataFrame({
        "season": "2099-00",
        "gw": rng.integers(1, 11, n),
        "position": rng.integers(1, 5, n),
        "played_r5": rng.uniform(0, 1, n).round(),
        "total_points": np.maximum(0, rng.poisson(quality)).astype(float),
    })
    preds = {"MLP": quality + rng.normal(0, 0.5, n), "Baseline": quality + rng.normal(0, 1.5, n)}
    return df, preds


def test_report_has_every_section_and_sane_numbers():
    df, preds = make_val()
    report = validate.build_report(df, preds, target="total_points", trained_on="2098-99", best_epoch=3)

    assert report["rows"] == len(df)
    assert report["active_rows"] == int((df["played_r5"] > 0).sum())
    for section in ("headline", "by_position", "by_gameweek", "calibration", "deciles", "errors", "captain"):
        assert report[section], f"{section} is empty"

    head = pd.DataFrame(report["headline"])
    assert set(head["model"]) == {"MLP", "Baseline"}
    assert set(head["subset"]) == {"All players", "Players getting minutes"}
    # The tighter predictions must score better; RMSE always punishes at least as much as MAE.
    for subset, part in head.groupby("subset"):
        scores = part.set_index("model")
        assert scores.at["MLP", "mae"] < scores.at["Baseline", "mae"], subset
        assert (part["rmse"] >= part["mae"]).all()


def test_deciles_are_ordered_and_cover_the_active_rows():
    df, preds = make_val()
    report = validate.build_report(df, preds, target="total_points", trained_on="x")
    deciles = pd.DataFrame(report["deciles"])
    active = int((df["played_r5"] > 0).sum())

    for model, part in deciles.groupby("model"):
        assert part["n"].sum() == active
        assert sorted(part["decile"]) == list(range(1, validate.DECILES + 1))
    top = deciles[(deciles["model"] == "MLP") & (deciles["decile"] == validate.DECILES)]["actual"].iloc[0]
    bottom = deciles[(deciles["model"] == "MLP") & (deciles["decile"] == 1)]["actual"].iloc[0]
    assert top > bottom  # the whole point: higher xP really does mean more points


def test_error_histogram_shares_add_up():
    df, preds = make_val()
    report = validate.build_report(df, preds, target="total_points", trained_on="x")
    errors = pd.DataFrame(report["errors"])
    assert errors["count"].sum() == int((df["played_r5"] > 0).sum())
    assert abs(errors["share"].sum() - 1.0) < 1e-9


def test_captain_test_never_beats_hindsight():
    df, preds = make_val()
    report = validate.build_report(df, preds, target="total_points", trained_on="x")
    cap = pd.DataFrame(report["captain"])
    assert (cap["MLP"] <= cap["best"]).all()
    assert (cap["typical"] <= cap["best"]).all()


def test_save_and_load_round_trip(tmp_path):
    df, preds = make_val()
    report = validate.build_report(df, preds, target="total_points", trained_on="x")
    path = tmp_path / "validation.json"
    validate.save_report(report, path)
    assert validate.load_report(path) == report
    assert validate.load_report(tmp_path / "missing.json") is None
    assert "MLP" in validate.summarise(report)


def test_r2_spearman_and_return_groups():
    df, preds = make_val()
    df["minutes"] = np.where(df["total_points"] > 0, 90.0, 0.0)
    report = validate.build_report(df, preds, target="total_points", trained_on="x")
    head = pd.DataFrame(report["headline"]).set_index(["subset", "model"])
    active = head.loc["Players getting minutes"]
    assert active.at["MLP", "r2"] > active.at["Baseline", "r2"]
    assert active.at["MLP", "spearman"] > active.at["Baseline", "spearman"]
    assert validate.r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0

    groups = pd.DataFrame(report["return_groups"])
    assert set(groups["group"]) == set(validate.RETURN_GROUPS)
    assert groups.groupby("model")["n"].sum().eq(len(df)).all()     # every row in exactly one group


def test_horizons_are_scored_on_the_same_rows():
    df, preds = make_val()
    stale = {2: {m: p + 1.0 for m, p in preds.items()}}
    report = validate.build_report(df, preds, target="total_points", trained_on="x", horizons=stale)
    hz = pd.DataFrame(report["horizons"]).set_index(["horizon", "model"])
    assert set(hz.index.get_level_values("horizon")) == {1, 2}
    assert hz.at[(2, "MLP"), "n"] == hz.at[(1, "MLP"), "n"]
    assert hz.at[(2, "MLP"), "bias"] > hz.at[(1, "MLP"), "bias"]


def test_perfect_model_ceiling_is_a_sensible_band():
    n = 400
    rng = np.random.default_rng(0)
    comp = pd.DataFrame({
        "played": rng.uniform(0.5, 1.0, n), "position": rng.integers(1, 5, n), "dc_era": 1.0,
        "goals_scored": rng.uniform(0, 0.5, n), "assists": rng.uniform(0, 0.3, n),
        "clean_sheets": rng.uniform(0, 0.4, n), "goals_conceded": rng.uniform(0.5, 2.0, n),
        "saves": rng.uniform(0, 3, n), "bonus": rng.uniform(0, 0.5, n), "dc": rng.uniform(0, 0.3, n),
    })
    comp["played60"] = comp["played"] * 0.8
    ceiling = validate.simulate_ceiling(comp, sims=50)
    assert ceiling["rmse_p5"] <= ceiling["rmse_median"] <= ceiling["rmse_p95"]
    assert 0.0 < ceiling["rmse_median"] < 5.0
    assert ceiling["r2_median"] < 0.5          # most of a week's points are luck


def test_match_tables_make_bonus_follow_the_goals():
    rng = np.random.default_rng(1)
    n = 3000
    goals = rng.poisson(0.3, n)
    history = pd.DataFrame({
        "minutes": 90.0, "position": rng.integers(1, 5, n), "goals_scored": goals,
        "assists": rng.poisson(0.2, n), "clean_sheets": rng.integers(0, 2, n), "goals_conceded": 1,
        "saves": 0, "bonus": np.where(goals > 0, 3, 0), "total_points": 2.0 + 5 * goals,
    })
    tables = validate.match_tables(history)
    probs = tables["bonus"]
    assert probs[3, 1, 0, 0, 3] > 0.8 and probs[3, 0, 0, 0, 0] > 0.8   # a goal brings the 3 bonus
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert set(tables["residual"]) <= {1, 2, 3, 4}
