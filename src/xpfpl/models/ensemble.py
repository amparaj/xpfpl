"""The average of several models: the "wisdom of the crowd" among our own models.

A plain average of several models tends to score about as well as the best of them, because
their mistakes partly cancel. Ours are within noise of
each other on accuracy but make different mistakes (a neural net, boosted trees, a minutes
model), so averaging should cancel some of them. Each member is trained exactly as it would be
on its own; the ensemble just keeps them together and averages their predictions.

LightGBM is optional: if it isn't installed, the ensemble quietly leaves it out.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xpfpl.models.trainer import TrainConfig

MEMBERS = ("mlp", "gbm", "xmins")


def _available() -> tuple[str, ...]:
    try:
        import lightgbm  # noqa: F401
        return MEMBERS
    except ImportError:                                        # pragma: no cover - env dependent
        return tuple(m for m in MEMBERS if m != "gbm")


@dataclass
class Predictor:
    members: dict                      # name -> that model's Predictor
    meta: dict = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        preds = np.stack([m.predict(frame) for m in self.members.values()])
        return preds.mean(axis=0).astype("float32")

    def expectations(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        """Expected minutes and play probabilities, if a minutes model is a member."""
        member = self.members.get("xmins")
        return member.expectations(frame) if member is not None else None

    def save(self, path: Path) -> None:
        """One file with every member pickled inside (LightGBM boosters pickle fine)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"members": self.members, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return cls(ckpt["members"], ckpt["meta"])


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str,
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    """Fit every member. With no `val_df` (the refit), each reuses its own best epoch count."""
    from xpfpl import models

    best = cfg.member_epochs or {}
    members, epochs = {}, {}
    for name in _available():
        member_cfg = TrainConfig(**{**cfg.__dict__, "epochs": best.get(name, cfg.epochs), "member_epochs": None})
        if not quiet:
            print(f"  ensemble member: {name}")
        members[name] = models.fit(name, train_df, val_df, features, target, cfg=member_cfg, quiet=True)
        epochs[name] = members[name].meta.get("best_epoch")
    meta = {"kind": "ensemble", "members": list(members), "member_epochs": epochs,
            "best_epoch": max((e or 0) for e in epochs.values())}
    return Predictor(members, meta)
