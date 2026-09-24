"""A small PyTorch multi-layer perceptron that predicts a player's points for one fixture.

This is deliberately plain so each PyTorch building block is visible:
  Dataset/DataLoader -> nn.Module -> loss + optimiser -> training loop -> early stopping -> save/load.

It is still the default model. The others (components, embed, sequence, gbm) are variations
on it and share its training loop; `xpfpl compare` scores them all on the same held-out season.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from xpfpl.models.trainer import TrainConfig, fit, return_weights, seed, standardise, standardiser

__all__ = ["XPNet", "TrainConfig", "Predictor", "train"]


class XPNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Predictor:
    """A trained model plus everything needed to reproduce its inputs."""
    model: XPNet
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    meta: dict = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = standardise(frame, self.features, self.mean, self.std)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(x)).numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "features": self.features,
            "mean": self.mean,
            "std": self.std,
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        model = XPNet(len(ckpt["features"]), tuple(meta["hidden"]), meta["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["features"], ckpt["mean"], ckpt["std"], meta)


def _tensors(frame: pd.DataFrame, features: list[str], target: str, mean, std, balance: bool = False):
    x = standardise(frame, features, mean, std)
    y = frame[target].to_numpy(dtype="float32", copy=True)
    w = return_weights(y) if balance else np.ones_like(y)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str,
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    """Train with early stopping on `val_df`, or for exactly cfg.epochs if `val_df` is None."""
    seed(cfg)                      # before the model is built: see trainer.seed
    mean, std = standardiser(train_df, features)
    model = XPNet(len(features), cfg.hidden, cfg.dropout)
    # MSE because we want the *mean* outcome (expected points), hauls included. Optionally
    # weighted so each return group counts equally (cfg.balance_returns; validation stays plain).
    def weighted_mse(m, b):
        return (b[2] * (m(b[0]) - b[1]) ** 2).mean()

    result = fit(model, _tensors(train_df, features, target, mean, std, cfg.balance_returns),
                 None if val_df is None else _tensors(val_df, features, target, mean, std),
                 weighted_mse, cfg, quiet=quiet)

    meta = {"kind": "mlp", "hidden": list(cfg.hidden), "dropout": cfg.dropout,
            "balance_returns": cfg.balance_returns,
            "best_epoch": result["best_epoch"], "val_mse": result["val_loss"]}
    return Predictor(model, list(features), mean, std, meta)
