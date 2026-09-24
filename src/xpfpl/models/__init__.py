"""The model zoo, behind one small interface.

Every model here - the plain MLP, the component model, the embedding MLP, the GRU, LightGBM
and the no-ML baseline - exposes the same three things:

    predictor = models.fit("mlp", train_df, val_df)   # train (early stopping on val_df)
    predictor.predict(frame) -> np.ndarray            # xP per row
    predictor.save(path) / Predictor.load(path)

so the CLI, predict.py, the backtest and the dashboard never care which one is in use. Add a
new model by writing a module with `train(...)` and a `Predictor`, then listing it below.

Each trained model gets its own file in models/ (`xp_mlp.pt`, `xp_gbm.txt`, ...), so several
can be trained and compared without overwriting each other.
"""

from pathlib import Path

import pandas as pd

from xpfpl import config
from xpfpl.models import baseline, components, embed, ensemble, gbm, minutes, mlp, sequence
from xpfpl.models.trainer import TrainConfig

MODULES = {
    "mlp": mlp,                  # 128-64 MLP on the rolling features (the default)
    "components": components,    # one head per scoring component, combined with FPL's rules
    "embed": embed,              # the MLP plus learned player/club embeddings
    "sequence": sequence,        # a GRU over the last six matches
    "xmins": minutes,            # expected minutes first, then points given the minutes
    "ensemble": ensemble,        # the average of mlp, gbm and xmins
    "gbm": gbm,                  # LightGBM on the same features (tabular benchmark)
    "baseline": baseline,        # average of the last 5 matches, no learning at all
}
NAMES = tuple(MODULES)
TRAINABLE = tuple(n for n in NAMES if n != "baseline")
DESCRIPTIONS = {
    "mlp": "128-64 MLP on rolling form",
    "components": "Minutes/goals/assists/clean-sheet components combined with FPL scoring",
    "embed": "MLP + player and club embeddings",
    "sequence": "GRU over the last six matches",
    "xmins": "Expected minutes (0 / cameo / 60+), then points given the minutes",
    "ensemble": "The average of the mlp, gbm and xmins predictions",
    "gbm": "Gradient-boosted decision trees for tabular data; an alternative to the neural-network models",
    "baseline": "Average of the last 5 matches (no ML)",
}
SUFFIX = {"gbm": ".txt"}         # LightGBM saves a text model, torch saves a .pt


def path(name: str) -> Path:
    """Where `name`'s trained weights live. The MLP keeps the original filename."""
    if name == "mlp":
        return config.MODEL_PATH
    return config.MODELS_DIR / f"xp_{name}{SUFFIX.get(name, '.pt')}"


def module(name: str):
    try:
        return MODULES[name]
    except KeyError:
        raise SystemExit(f"Unknown model '{name}'. Choose from: {', '.join(NAMES)}") from None


def fit(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame | None,
        features: list[str] | None = None, target: str | None = None,
        cfg: TrainConfig | None = None, quiet: bool = False):
    """Train `name` on `train_df`, early-stopping on `val_df` (None = a fixed number of epochs)."""
    from xpfpl.features import FEATURES, TARGET

    if name == "baseline":
        return baseline.Predictor()
    return module(name).train(train_df, val_df, features or FEATURES, target or TARGET,
                              cfg or TrainConfig(), quiet=quiet)


def refit_config(predictor, **overrides) -> TrainConfig:
    """The TrainConfig that refits `predictor`'s model on all the data without a holdout:
    its best epoch count (and each ensemble member's own)."""
    meta = predictor.meta
    return TrainConfig(epochs=meta.get("best_epoch") or TrainConfig().epochs,
                       member_epochs=meta.get("member_epochs"), **overrides)


def load(name: str):
    """Load a trained model from disk, with a friendly message if it hasn't been trained yet."""
    if name == "baseline":
        return baseline.Predictor()
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"No trained '{name}' model at {p} - run `xpfpl train --model {name}` "
            f"(or use --model baseline).")
    return module(name).Predictor.load(p)


def trained() -> list[str]:
    """The models that have been trained on this machine, in registry order."""
    return [n for n in NAMES if n == "baseline" or path(n).exists()]
