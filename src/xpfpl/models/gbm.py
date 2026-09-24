"""LightGBM on exactly the same features: the tabular benchmark the neural nets have to beat.

Gradient-boosted trees are what most people would reach for on a table of 60 numeric columns,
so if the MLP can't match this, the PyTorch work is decoration. It needs no scaling, trains in
seconds on CPU, and `importance()` says which features are actually carrying the predictions -
useful feedback for the feature work regardless of which model ends up in front.

LightGBM is an optional dependency: `pip install -e ".[gbm]"` (or `pip install lightgbm`).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from xpfpl.models.trainer import TrainConfig

PARAMS = {
    "objective": "regression",       # squared error, to match the MLP's MSE
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,         # points are noisy; small leaves would memorise hauls
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}
ROUNDS = 2000
EARLY_STOPPING = 50


def _lightgbm():
    try:
        import lightgbm
    except ImportError as exc:                                    # pragma: no cover - env dependent
        raise SystemExit("LightGBM isn't installed. Run: pip install lightgbm") from exc
    return lightgbm


@dataclass
class Predictor:
    booster: object
    features: list[str]
    meta: dict = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].to_numpy(dtype="float32")
        return self.booster.predict(x, num_iteration=self.booster.best_iteration or None).astype("float32")

    def importance(self, top: int = 20) -> pd.Series:
        """Total gain per feature, biggest first."""
        gain = self.booster.feature_importance("gain")
        return pd.Series(gain, index=self.features).sort_values(ascending=False).head(top)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.booster.best_iteration or None)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"features": self.features, "meta": self.meta}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        lgb = _lightgbm()
        booster = lgb.Booster(model_file=str(path))
        side = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return cls(booster, side["features"], side["meta"])


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str,
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    """Boost until the validation season stops improving; with no `val_df`, run `meta_rounds`.

    The refit-on-everything step has no holdout, so it reuses the round count found here -
    the same trick as the MLP's best-epoch refit. Pass it as cfg.epochs.
    """
    lgb = _lightgbm()
    dtrain = lgb.Dataset(train_df[features].to_numpy(dtype="float32"),
                         label=train_df[target].to_numpy(dtype="float32"),
                         feature_name=list(features))
    params = {**PARAMS, "learning_rate": cfg.lr * 50, "seed": cfg.seed}

    if val_df is None:
        booster = lgb.train(params, dtrain, num_boost_round=max(cfg.epochs, 1))
        rounds, val_mse = booster.current_iteration(), None
    else:
        dval = lgb.Dataset(val_df[features].to_numpy(dtype="float32"),
                           label=val_df[target].to_numpy(dtype="float32"), reference=dtrain)
        callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=not quiet)]
        if not quiet:
            callbacks.append(lgb.log_evaluation(100))
        booster = lgb.train(params, dtrain, num_boost_round=ROUNDS, valid_sets=[dval],
                            valid_names=["val"], callbacks=callbacks)
        rounds = booster.best_iteration
        val_mse = booster.best_score["val"]["l2"]

    meta = {"kind": "gbm", "best_epoch": int(rounds), "rounds": int(rounds), "val_mse": val_mse,
            "learning_rate": params["learning_rate"]}
    return Predictor(booster, list(features), meta)
