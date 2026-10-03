"""A PyTorch network that forecasts the chance of every score, not just the average.

The MLP is trained on squared error, so it learns the mean and nothing else. This one has the
same body but ends in one output per score (distribution.LOW to HIGH, the ends taking the
tails), turned into probabilities with a softmax and trained with cross-entropy: the log score
distribution.py uses to judge forecasts. Its xP is the average of that distribution, so it
also slots in with the other models (`predict`), and `distribution(frame)` gives the chances.

It is the direct alternative to the Monte Carlo, which builds the same chances by simulating
every match from the xP and the scoring rules. Trees can't do this easily; a softmax head is a
few lines of PyTorch.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from xpfpl import distribution
from xpfpl.models.trainer import TrainConfig, fit, seed, standardise, standardiser

__all__ = ["DistNet", "Predictor", "train"]


class DistNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1,
                 n_classes: int = distribution.K):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                     # logits; softmax gives the chances


@dataclass
class Predictor:
    model: DistNet
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    values: np.ndarray                         # each class's average score (the ends: their tails)
    meta: dict = field(default_factory=dict)

    def distribution(self, frame: pd.DataFrame) -> np.ndarray:
        """(rows, distribution.K) chances of each score."""
        x = standardise(frame, self.features, self.mean, self.std)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(x)), dim=-1).double().numpy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.distribution(frame) @ self.values).astype("float32")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "features": self.features, "mean": self.mean,
                    "std": self.std, "values": self.values, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        model = DistNet(len(ckpt["features"]), tuple(meta["hidden"]), meta["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["features"], ckpt["mean"], ckpt["std"], ckpt["values"], meta)


def _tensors(frame: pd.DataFrame, features: list[str], target: str, mean, std):
    x = standardise(frame, features, mean, std)
    y = distribution.classes(frame[target].to_numpy())
    return torch.from_numpy(x), torch.from_numpy(y)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str,
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    seed(cfg)
    mean, std = standardiser(train_df, features)
    model = DistNet(len(features), cfg.hidden, cfg.dropout)
    loss = nn.CrossEntropyLoss()
    result = fit(model, _tensors(train_df, features, target, mean, std),
                 None if val_df is None else _tensors(val_df, features, target, mean, std),
                 lambda m, b: loss(m(b[0]), b[1]), cfg, quiet=quiet)
    # The end classes hold the tails (-3 or less, 25 or more): value them at their real average.
    y = train_df[target].to_numpy(dtype="float64")
    idx = distribution.classes(y)
    values = distribution.VALUES.astype("float64")
    for k in (0, distribution.K - 1):
        if (idx == k).any():
            values[k] = y[idx == k].mean()
    meta = {"kind": "dist", "hidden": list(cfg.hidden), "dropout": cfg.dropout,
            "best_epoch": result["best_epoch"], "val_log_score": result["val_loss"]}
    return Predictor(model, list(features), mean, std, values, meta)
