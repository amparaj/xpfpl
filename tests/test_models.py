"""Every model in the registry: trains, predicts sane numbers, and survives a save/load."""

import numpy as np
import pandas as pd
import pytest

from xpfpl import models
from xpfpl.features import FEATURES, LAG_FEATURES, TARGET
from xpfpl.models.trainer import TrainConfig


def make_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Rows with the columns every model needs, and points that really do follow the features."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.uniform(0, 3, n).astype("float32") for c in FEATURES + LAG_FEATURES})
    df["position"] = rng.integers(1, 5, n)
    for p in range(1, 5):
        df[f"pos_{p}"] = (df["position"] == p).astype("float32")
    df["code"] = rng.integers(1, 40, n)
    df["team_code"] = rng.integers(1, 10, n)
    df["opp_code"] = rng.integers(1, 10, n)
    df["dc_era"] = 1.0
    df["minutes"] = rng.choice([0.0, 90.0], n, p=[0.3, 0.7])
    df["played"] = (df["minutes"] > 0).astype(float)
    df["played60"] = (df["minutes"] >= 60).astype(float)
    skill = df["total_points_r5"].to_numpy()
    df["goals_scored"] = rng.poisson(0.1 * skill)
    df["assists"] = rng.poisson(0.05 * skill)
    df["clean_sheets"] = rng.binomial(1, 0.3, n) * df["played60"]
    df["goals_conceded"] = rng.poisson(1.3, n) * df["played60"]
    df["saves"] = rng.poisson(1.0, n) * (df["position"] == 1)
    df["bonus"] = rng.binomial(3, 0.1, n)
    df["defensive_contribution"] = rng.poisson(6, n)
    df[TARGET] = (df["played"] + df["played60"] + 4 * df["goals_scored"]
                  + 3 * df["assists"] + df["bonus"]).astype("float32")
    df["season"] = "2099-00"
    return df


@pytest.fixture(scope="module")
def frames():
    return make_frame(), make_frame(300, seed=1)


@pytest.mark.parametrize("name", models.NAMES)
def test_every_model_trains_predicts_and_round_trips(name, frames, tmp_path):
    train_df, val_df = frames
    if name == "gbm":
        pytest.importorskip("lightgbm")
    predictor = models.fit(name, train_df, val_df, cfg=TrainConfig(epochs=3, patience=2), quiet=True)

    pred = predictor.predict(val_df)
    assert pred.shape == (len(val_df),)
    assert np.isfinite(pred).all()
    assert -5 < pred.mean() < 30          # points, not nonsense

    if name == "baseline":
        return
    path = tmp_path / models.path(name).name
    predictor.save(path)
    reloaded = models.module(name).Predictor.load(path)
    assert np.allclose(reloaded.predict(val_df), pred, atol=1e-4)


def test_registry_knows_where_each_model_lives():
    assert models.path("mlp").name.endswith(".pt")
    assert models.path("gbm").name.endswith(".txt")
    assert len({models.path(n) for n in models.TRAINABLE}) == len(models.TRAINABLE)
    with pytest.raises(SystemExit):
        models.module("nonesuch")


def test_component_model_breaks_an_xp_into_its_parts(frames):
    train_df, val_df = frames
    predictor = models.fit("components", train_df, None, cfg=TrainConfig(epochs=3), quiet=True)
    parts = predictor.breakdown(val_df)

    # The parts must add up to the xP, and the probabilities must be probabilities.
    assert np.allclose(parts.drop(columns=["xp"]).sum(axis=1), parts["xp"])
    components = predictor.components(val_df)
    for head in ("played", "played60", "clean_sheets", "dc"):
        assert components[head].between(0, 1).all()
    for head in ("goals_scored", "assists", "saves", "bonus"):
        assert (components[head] >= 0).all()
    # Keepers and outfielders draw on different parts of the scoring rules.
    assert (parts.loc[val_df["position"] != 1, "saves"] == 0).all()
    assert (parts.loc[val_df["position"] > 2, "conceded"] == 0).all()


def test_sequence_model_reads_the_lag_columns(frames):
    """Changing the lagged match history must change the prediction - otherwise the GRU is idle."""
    train_df, val_df = frames
    predictor = models.fit("sequence", train_df, None, cfg=TrainConfig(epochs=3), quiet=True)
    before = predictor.predict(val_df)

    hot = val_df.copy()
    for col in (c for c in LAG_FEATURES if c.startswith("total_points_lag")):
        hot[col] = 15.0
    assert not np.allclose(predictor.predict(hot), before)


def test_embeddings_cover_unknown_players(frames):
    """A player the model never saw must fall back to row 0, not blow up."""
    train_df, val_df = frames
    predictor = models.fit("embed", train_df, None, cfg=TrainConfig(epochs=2), quiet=True)
    newcomers = val_df.assign(code=999_999, team_code=999, opp_code=998)
    assert np.isfinite(predictor.predict(newcomers)).all()


def test_component_heads_with_no_training_data_predict_nothing(frames):
    """Defensive contribution didn't exist before 2025-26: an untrained head must stay silent.

    Without this, its sigmoid sits at 0.5 and hands every player a free point.
    """
    train_df, val_df = frames
    predictor = models.fit("components", train_df.assign(dc_era=0.0), None,
                           cfg=TrainConfig(epochs=3), quiet=True)

    assert predictor.meta["head_rows"]["dc"] == 0
    assert (predictor.components(val_df)["dc"] == 0).all()
    assert (predictor.breakdown(val_df)["defensive"] == 0).all()

    # With the data present, the head is trained and free to say something.
    trained = models.fit("components", train_df, None, cfg=TrainConfig(epochs=3), quiet=True)
    assert trained.meta["head_rows"]["dc"] == len(train_df)


@pytest.mark.parametrize("name", [n for n in models.TRAINABLE if n != "gbm"])
def test_training_is_reproducible(name, frames):
    """Same data, same seed, same model - including the initial weights.

    `nn.Module.__init__` draws from the global RNG, so a seed applied after the model is built
    leaves every run starting somewhere different; with early stopping that changes the epoch
    count too, and two "identical" runs then disagree by whole gameweeks in a backtest.
    """
    train_df, val_df = frames
    cfg = TrainConfig(epochs=4, patience=2)
    first = models.fit(name, train_df, val_df, cfg=cfg, quiet=True)
    second = models.fit(name, train_df, val_df, cfg=cfg, quiet=True)

    assert first.meta["best_epoch"] == second.meta["best_epoch"]
    assert np.allclose(first.predict(val_df), second.predict(val_df))
