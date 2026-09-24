"""Predict the *components* of a score, then add them up with FPL's rules (the AIrsenal idea).

The plain MLP predicts total points in one number, which mixes up very different questions:
will he play at all, will he score, will his side keep a clean sheet. This model has one head
per question and combines the answers with xpfpl/scoring.py:

    xP = P(play) + P(60 mins)                     appearance points
       + goal points(position) * E[goals]
       + 3 * E[assists]
       + clean-sheet points(position) * P(CS)
       + E[saves]/3                               goalkeepers
       - P(60 mins) * E[conceded | 60 mins] / 2   keepers and defenders
       + E[bonus] + 2 * P(defensive contribution)
       + E[residual]                              cards, own goals, penalties

Two payoffs beyond accuracy: the numbers are readable ("4.1 xP, of which 1.8 from goals"), and
each head gets the loss that suits it - binary cross-entropy for the yes/no questions, Poisson
for counts, MSE for the residual - which is a tidy illustration of one net with many losses.

The only head trained on a subset is `goals_conceded`: goals conceded are only recorded while a
player is on the pitch, so it learns E[conceded | played 60] and is multiplied by P(60) above.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from xpfpl import scoring
from xpfpl.models.trainer import TrainConfig, fit, seed, standardise, standardiser

# head -> (target column, loss family). "rate" heads are Poisson, "prob" heads Bernoulli.
HEADS = {
    "played": ("played", "prob"),
    "played60": ("played60", "prob"),
    "goals_scored": ("goals_scored", "rate"),
    "assists": ("assists", "rate"),
    "clean_sheets": ("clean_sheets", "prob"),
    "goals_conceded": ("goals_conceded", "rate"),   # masked: only rows that played 60 minutes
    "saves": ("saves", "rate"),
    "bonus": ("bonus", "rate"),
    "dc": ("dc", "prob"),                           # masked: only the 2025-26+ rows that score it
    "residual": ("residual", "real"),
}
MASKED = {"goals_conceded": "played60", "dc": "dc_era"}
# The residual is small and noisy; at full weight it would drag the shared trunk around.
LOSS_WEIGHTS = {"residual": 0.2}


class ComponentNet(nn.Module):
    """One shared trunk, one linear head per component."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(width, 1) for name in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        return {name: head(z).squeeze(-1) for name, head in self.heads.items()}

    def expectations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw head outputs turned into the expectations scoring.py wants."""
        out = {}
        for name, raw in self(x).items():
            kind = HEADS[name][1]
            out[name] = torch.sigmoid(raw) if kind == "prob" else torch.exp(raw) if kind == "rate" else raw
        return out


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """The component columns the heads are trained on (`played`/`played60` already exist)."""
    frame = frame.copy()
    frame["dc"] = scoring.dc_awarded(frame)
    frame["residual"] = scoring.residual(frame)
    frame["clean_sheets"] = frame["clean_sheets"].fillna(0)
    return frame


def _loss(model: ComponentNet, batch) -> torch.Tensor:
    x, y, mask = batch
    raw = model(x)
    total = x.new_zeros(())
    for i, (name, (_, kind)) in enumerate(HEADS.items()):
        w = mask[:, i]
        if kind == "prob":
            per_row = nn.functional.binary_cross_entropy_with_logits(raw[name], y[:, i], reduction="none")
        elif kind == "rate":
            per_row = torch.exp(raw[name]) - y[:, i] * raw[name]          # Poisson NLL, log input
        else:
            per_row = (raw[name] - y[:, i]) ** 2
        total = total + LOSS_WEIGHTS.get(name, 1.0) * (per_row * w).sum() / w.sum().clamp(min=1.0)
    return total


def _tensors(frame: pd.DataFrame, features: list[str], mean, std):
    x = standardise(frame, features, mean, std)
    y = np.stack([frame[col].fillna(0).to_numpy(dtype="float32") for col, _ in HEADS.values()], axis=1)
    mask = np.ones_like(y)
    for i, name in enumerate(HEADS):
        if name in MASKED:
            mask[:, i] = frame[MASKED[name]].fillna(0).to_numpy(dtype="float32")
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


@dataclass
class Predictor:
    model: ComponentNet
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    meta: dict = field(default_factory=dict)

    def components(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Every predicted component for every row, before they are turned into points."""
        x = torch.from_numpy(standardise(frame, self.features, self.mean, self.std))
        self.model.eval()
        with torch.no_grad():
            out = {k: v.numpy().astype("float64") for k, v in self.model.expectations(x).items()}
        # A masked head with no training rows (defensive contribution, if training stops before
        # 2025-26) never learned anything: an untrained sigmoid sits at 0.5, which would hand
        # every player a point out of thin air. Predict none of it instead.
        for name, rows in self.meta.get("head_rows", {}).items():
            if not rows:
                out[name] = np.zeros(len(frame))
        df = pd.DataFrame(out, index=frame.index)
        df["position"] = frame["position"].to_numpy()
        df["dc_era"] = frame["dc_era"].to_numpy() if "dc_era" in frame else 1.0
        return df

    def breakdown(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Components plus the points each one contributes - the readable version of an xP."""
        c = self.components(frame)
        pos = c["position"]
        out = pd.DataFrame(index=frame.index)
        out["minutes"] = c["played"] + c["played60"]
        out["goals"] = pos.map(scoring.GOAL_POINTS).fillna(0) * c["goals_scored"]
        out["assists"] = scoring.ASSIST_POINTS * c["assists"]
        out["clean_sheet"] = pos.map(scoring.CLEAN_SHEET_POINTS).fillna(0) * c["clean_sheets"]
        out["saves"] = c["saves"] / scoring.SAVES_PER_POINT * (pos == 1)
        out["conceded"] = -(pos.isin((1, 2)) * c["played60"] * c["goals_conceded"]
                            / scoring.CONCEDED_PER_PENALTY)
        out["bonus"] = c["bonus"]
        out["defensive"] = scoring.DC_POINTS * c["dc"] * c["dc_era"]
        out["other"] = c["residual"]
        out["xp"] = out.sum(axis=1)
        return out

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.breakdown(frame)["xp"].to_numpy(dtype="float32")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "features": self.features,
                    "mean": self.mean, "std": self.std, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        model = ComponentNet(len(ckpt["features"]), tuple(meta["hidden"]), meta["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["features"], ckpt["mean"], ckpt["std"], meta)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str = "",
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    """`target` is ignored: the targets are the component columns, added here if missing."""
    train_df = add_targets(train_df) if "residual" not in train_df else train_df
    if val_df is not None and "residual" not in val_df:
        val_df = add_targets(val_df)

    seed(cfg)                      # before the model is built: see trainer.seed
    mean, std = standardiser(train_df, features)
    model = ComponentNet(len(features), cfg.hidden, cfg.dropout)
    tensors = _tensors(train_df, features, mean, std)
    result = fit(model, tensors, None if val_df is None else _tensors(val_df, features, mean, std),
                 _loss, cfg, quiet=quiet)
    head_rows = {name: int(tensors[2][:, i].sum()) for i, name in enumerate(HEADS)}
    if not quiet and any(n == 0 for n in head_rows.values()):
        empty = ", ".join(name for name, n in head_rows.items() if n == 0)
        print(f"No training rows for: {empty} - those components will be predicted as zero.")
    meta = {"kind": "components", "hidden": list(cfg.hidden), "dropout": cfg.dropout,
            "best_epoch": result["best_epoch"], "val_loss": result["val_loss"],
            "head_rows": head_rows}
    return Predictor(model, list(features), mean, std, meta)
