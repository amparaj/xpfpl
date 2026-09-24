"""No-ML baseline: expected points = average points over the player's last 5 matches.

Every model should be compared against this. If the neural net can't beat it, it isn't
learning anything useful yet. It is wrapped in the same Predictor shape as the trained models
so the CLI, the backtest and `xpfpl compare` can treat it like any other choice.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Predictor:
    features: list[str] = field(default_factory=lambda: ["total_points_r5"])
    meta: dict = field(default_factory=lambda: {"kind": "baseline", "best_epoch": None})

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["total_points_r5"].to_numpy(dtype="float32")

    def save(self, path: Path) -> None:
        """Nothing to save: there are no parameters."""

    @classmethod
    def load(cls, path: Path | None = None) -> "Predictor":
        return cls()


def predict(frame: pd.DataFrame) -> np.ndarray:
    return Predictor().predict(frame)
