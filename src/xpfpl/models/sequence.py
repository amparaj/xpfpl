"""A GRU that reads a player's last few matches in order, instead of averaging them.

A rolling mean says "4 points a game". It cannot tell 2, 2, 12 (one lucky haul) from 5, 4, 3
(steady), or spot a player who has just come back from injury and is climbing. The lag columns
built in features.py (`<stat>_lag1` .. `_lag6`, most recent first) are reshaped here into a
(batch, 6, stats) sequence, fed oldest-first through `nn.GRU`, and the final hidden state is
concatenated with the ordinary features before the usual head.

PyTorch notes worth remembering from this one: recurrent layers want (batch, time, features)
when `batch_first=True`; `torch.flip` reverses the time axis because lag1 is the *most recent*
match; and the sequence stats are standardised per stat (not per lag column) so the same value
means the same thing at every step.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from xpfpl.features import SEQ_LEN, SEQ_STATS
from xpfpl.models.trainer import TrainConfig, fit, seed, standardise, standardiser

HIDDEN_SIZE = 32


def _sequence(frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(rows, SEQ_LEN, stats), oldest match first, standardised per stat."""
    cols = [f"{s}_lag{k}" for k in range(1, SEQ_LEN + 1) for s in SEQ_STATS]
    x = frame[cols].to_numpy(dtype="float32").reshape(len(frame), SEQ_LEN, len(SEQ_STATS))
    return ((x - mean) / std)[:, ::-1].copy()          # lag1 is the latest match; the GRU wants it last


def _seq_standardiser(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One mean/std per stat, pooled over every lag position."""
    flat = _sequence(frame, np.zeros(len(SEQ_STATS), "float32"), np.ones(len(SEQ_STATS), "float32"))
    mean, std = flat.mean(axis=(0, 1)), flat.std(axis=(0, 1))
    std[std < 1e-3] = 1.0
    return mean, std


class XPSeqNet(nn.Module):
    def __init__(self, n_features: int, n_stats: int, hidden: tuple[int, ...] = (128, 64),
                 dropout: float = 0.1, rnn_size: int = HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(n_stats, rnn_size, batch_first=True)
        layers: list[nn.Module] = []
        width = n_features + rnn_size
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        _, last = self.gru(seq)                        # last: (1, batch, rnn_size)
        return self.net(torch.cat([x, last.squeeze(0)], dim=1)).squeeze(-1)


@dataclass
class Predictor:
    model: XPSeqNet
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    seq_mean: np.ndarray
    seq_std: np.ndarray
    meta: dict = field(default_factory=dict)

    def _inputs(self, frame: pd.DataFrame):
        x = standardise(frame, self.features, self.mean, self.std)
        seq = _sequence(frame, self.seq_mean, self.seq_std)
        return torch.from_numpy(x), torch.from_numpy(seq)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x, seq = self._inputs(frame)
        self.model.eval()
        with torch.no_grad():
            return self.model(x, seq).numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "features": self.features,
                    "mean": self.mean, "std": self.std, "seq_mean": self.seq_mean,
                    "seq_std": self.seq_std, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        c = torch.load(path, map_location="cpu", weights_only=False)
        meta = c["meta"]
        model = XPSeqNet(len(c["features"]), len(SEQ_STATS), tuple(meta["hidden"]), meta["dropout"],
                         meta.get("rnn_size", HIDDEN_SIZE))
        model.load_state_dict(c["state_dict"])
        return cls(model, c["features"], c["mean"], c["std"], c["seq_mean"], c["seq_std"], meta)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str,
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    seed(cfg)                      # before the model is built: see trainer.seed
    mean, std = standardiser(train_df, features)
    seq_mean, seq_std = _seq_standardiser(train_df)
    model = XPSeqNet(len(features), len(SEQ_STATS), cfg.hidden, cfg.dropout)
    predictor = Predictor(model, list(features), mean, std, seq_mean, seq_std)
    mse = nn.MSELoss()

    def tensors(df):
        x, seq = predictor._inputs(df)
        return x, seq, torch.from_numpy(df[target].to_numpy(dtype="float32"))

    result = fit(model, tensors(train_df), None if val_df is None else tensors(val_df),
                 lambda m, b: mse(m(b[0], b[1]), b[2]), cfg, quiet=quiet)
    predictor.meta = {"kind": "sequence", "hidden": list(cfg.hidden), "dropout": cfg.dropout,
                      "rnn_size": HIDDEN_SIZE, "seq_len": SEQ_LEN,
                      "best_epoch": result["best_epoch"], "val_mse": result["val_loss"]}
    return predictor
