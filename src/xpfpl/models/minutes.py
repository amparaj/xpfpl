"""Expected minutes first, then points given the minutes: the xMins model.

Models that predict points directly lose most of their accuracy on players who don't play or
barely score: that is a question about minutes, not about how well a player plays. A single
"points" number mixes the two - will he play, and how well -
so this model answers them separately with one shared trunk and five heads:

    minutes class   softmax over {0, 1-59, 60+}                  cross-entropy
    points | 1-59   what a cameo is worth                         MSE on rows in that class
    points | 60+    what a full game is worth                     MSE on rows in that class
    minutes | 1-59  and minutes | 60+, to report expected minutes  MSE on rows in that class

    xP    = P(1-59) * points|1-59 + P(60+) * points|60+
    xMins = P(1-59) * minutes|1-59 + P(60+) * minutes|60+

The split matters because FPL points aren't proportional to minutes: 60 minutes earns the second appearance point and unlocks clean sheets, so
a nailed starter and a 50/50 rotation risk with the same "average minutes" are not worth the
same.

It can also train on several horizons at once: pass `train_df` stacking the same matches with
features as known 1, 2 and 3 gameweeks before (features.build_horizon_frames), and the
`horizon` input lets it learn how much less certain a start becomes further ahead - the
"decay over time" of expected minutes. Tried on 2025-26: it helped 3-GW-ahead forecasts
(RMSE 2.749 vs 2.773) but cost 1-GW-ahead ones (2.622 vs 2.608), and next week's forecast
decides this week's team, so the CLI trains it on 1-GW-ahead rows like every other model.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from xpfpl.features import LAG_FEATURES
from xpfpl.models.trainer import TrainConfig, fit, seed, standardise, standardiser

CLASSES = ("none", "cameo", "full")
# The regression heads are in points and (minutes / 90); the class head in nats. Rough balance.
LOSS_WEIGHTS = {"cls": 1.0, "pts": 0.25, "mins": 1.0}


def extra_features() -> list[str]:
    return LAG_FEATURES + ["horizon"]


class MinutesNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.cls = nn.Linear(width, len(CLASSES))
        self.pts = nn.Linear(width, 2)        # points | cameo, points | full game
        self.mins = nn.Linear(width, 2)       # minutes/90 | cameo, minutes/90 | full game

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        return self.cls(z), self.pts(z), torch.sigmoid(self.mins(z))

    def expectations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, pts, mins = self(x)
        p = torch.softmax(logits, dim=-1)
        return {"p_none": p[:, 0], "p_cameo": p[:, 1], "p_full": p[:, 2],
                "pts_cameo": pts[:, 0], "pts_full": pts[:, 1],
                "xmins": 90 * (p[:, 1] * mins[:, 0] + p[:, 2] * mins[:, 1]),
                "xp": p[:, 1] * pts[:, 0] + p[:, 2] * pts[:, 1]}


def minutes_class(minutes: pd.Series) -> np.ndarray:
    m = minutes.fillna(0).to_numpy()
    return np.where(m <= 0, 0, np.where(m < 60, 1, 2))


def _tensors(frame: pd.DataFrame, features: list[str], target: str, mean, std):
    x = torch.from_numpy(standardise(frame, features, mean, std))
    cls = torch.from_numpy(minutes_class(frame["minutes"]).astype("int64"))
    pts = torch.from_numpy(frame[target].to_numpy(dtype="float32"))
    mins = torch.from_numpy((frame["minutes"].fillna(0).clip(0, 90) / 90.0).to_numpy(dtype="float32"))
    return x, cls, pts, mins


def _loss(model: MinutesNet, batch) -> torch.Tensor:
    x, cls, pts, mins = batch
    logits, pts_hat, mins_hat = model(x)
    loss = LOSS_WEIGHTS["cls"] * nn.functional.cross_entropy(logits, cls)
    for i, c in enumerate((1, 2)):                     # the regression heads only see their class
        mask = (cls == c).float()
        n = mask.sum().clamp(min=1.0)
        loss = loss + LOSS_WEIGHTS["pts"] * (mask * (pts_hat[:, i] - pts) ** 2).sum() / n
        loss = loss + LOSS_WEIGHTS["mins"] * (mask * (mins_hat[:, i] - mins) ** 2).sum() / n
    return loss


@dataclass
class Predictor:
    model: MinutesNet
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    meta: dict = field(default_factory=dict)

    def expectations(self, frame: pd.DataFrame) -> pd.DataFrame:
        """P(no minutes / cameo / 60+), points given each, expected minutes and xP, per row."""
        if "horizon" not in frame:
            frame = frame.assign(horizon=1.0)
        x = torch.from_numpy(standardise(frame, self.features, self.mean, self.std))
        self.model.eval()
        with torch.no_grad():
            out = {k: v.numpy().astype("float64") for k, v in self.model.expectations(x).items()}
        return pd.DataFrame(out, index=frame.index)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.expectations(frame)["xp"].to_numpy(dtype="float32")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "features": self.features,
                    "mean": self.mean, "std": self.std, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        model = MinutesNet(len(ckpt["features"]), tuple(meta["hidden"]), meta["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["features"], ckpt["mean"], ckpt["std"], meta)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str,
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    """`train_df` may stack several horizons (a `horizon` column); `val_df` is scored as given."""
    features = list(dict.fromkeys(list(features) + extra_features()))
    if "horizon" not in train_df:
        train_df = train_df.assign(horizon=1.0)
    if val_df is not None and "horizon" not in val_df:
        val_df = val_df.assign(horizon=1.0)
    seed(cfg)
    mean, std = standardiser(train_df, features)
    model = MinutesNet(len(features), cfg.hidden, cfg.dropout)
    result = fit(model, _tensors(train_df, features, target, mean, std),
                 None if val_df is None else _tensors(val_df, features, target, mean, std),
                 _loss, cfg, quiet=quiet)
    meta = {"kind": "minutes", "hidden": list(cfg.hidden), "dropout": cfg.dropout,
            "best_epoch": result["best_epoch"], "val_loss": result["val_loss"],
            "horizons": sorted(train_df["horizon"].unique().tolist())}
    return Predictor(model, features, mean, std, meta)
