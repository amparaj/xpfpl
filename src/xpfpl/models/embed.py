"""The MLP plus learned embeddings for the player, his club and the opponent.

The rolling features describe a player only by what he has been doing lately. Two players with
identical form are identical to the plain MLP, even if one is Salah and the other is on loan at
a promoted side. `nn.Embedding` gives every player `code` and every club `team_code` its own
vector, learned by gradient descent along with the rest of the net, so the model can keep a
standing opinion of a player and of how good his club and his opponent are.

Why this is a good PyTorch exercise: embeddings are just a lookup table of learnable rows, and
the only fiddly part is the vocabulary - ids seen during training get a row, everyone else
(new signings, promoted clubs) shares row 0, which is what `_index` and `_lookup` handle.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from xpfpl.models.trainer import TrainConfig, fit, seed, standardise, standardiser

PLAYER_DIM = 16
TEAM_DIM = 8
MIN_MATCHES = 20      # players with fewer rows than this share the "unknown player" vector


def _index(values: pd.Series, min_count: int = 1) -> dict[int, int]:
    """id -> row in the embedding table. Row 0 is reserved for everyone not in the table."""
    counts = values.value_counts()
    keep = counts[counts >= min_count].index
    return {int(v): i + 1 for i, v in enumerate(sorted(keep))}


def _lookup(values: pd.Series, table: dict[int, int]) -> np.ndarray:
    return values.map(table).fillna(0).to_numpy(dtype="int64")


class XPEmbedNet(nn.Module):
    def __init__(self, n_features: int, n_players: int, n_teams: int,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.player = nn.Embedding(n_players, PLAYER_DIM)
        self.team = nn.Embedding(n_teams, TEAM_DIM)       # shared by "my club" and "the opponent"
        for emb in (self.player, self.team):
            nn.init.normal_(emb.weight, std=0.05)         # start near zero: form leads, identity adjusts

        layers: list[nn.Module] = []
        width = n_features + PLAYER_DIM + 2 * TEAM_DIM
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, self.player(ids[:, 0]), self.team(ids[:, 1]), self.team(ids[:, 2])], dim=1)
        return self.net(z).squeeze(-1)


@dataclass
class Predictor:
    model: XPEmbedNet
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    players: dict[int, int]
    teams: dict[int, int]
    meta: dict = field(default_factory=dict)

    def _inputs(self, frame: pd.DataFrame):
        x = standardise(frame, self.features, self.mean, self.std)
        ids = np.stack([_lookup(frame["code"], self.players),
                        _lookup(frame["team_code"], self.teams),
                        _lookup(frame["opp_code"], self.teams)], axis=1)
        return torch.from_numpy(x), torch.from_numpy(ids)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x, ids = self._inputs(frame)
        self.model.eval()
        with torch.no_grad():
            return self.model(x, ids).numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "features": self.features,
                    "mean": self.mean, "std": self.std, "players": self.players,
                    "teams": self.teams, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        c = torch.load(path, map_location="cpu", weights_only=False)
        meta = c["meta"]
        model = XPEmbedNet(len(c["features"]), len(c["players"]) + 1, len(c["teams"]) + 1,
                           tuple(meta["hidden"]), meta["dropout"])
        model.load_state_dict(c["state_dict"])
        return cls(model, c["features"], c["mean"], c["std"], c["players"], c["teams"], meta)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str], target: str,
          cfg: TrainConfig = TrainConfig(), quiet: bool = False) -> Predictor:
    seed(cfg)                      # before the model is built: see trainer.seed
    mean, std = standardiser(train_df, features)
    players = _index(train_df["code"], MIN_MATCHES)
    teams = _index(train_df["team_code"])
    model = XPEmbedNet(len(features), len(players) + 1, len(teams) + 1, cfg.hidden, cfg.dropout)
    predictor = Predictor(model, list(features), mean, std, players, teams)
    mse = nn.MSELoss()

    def tensors(df):
        x, ids = predictor._inputs(df)
        return x, ids, torch.from_numpy(df[target].to_numpy(dtype="float32"))

    result = fit(model, tensors(train_df), None if val_df is None else tensors(val_df),
                 lambda m, b: mse(m(b[0], b[1]), b[2]), cfg, quiet=quiet)
    predictor.meta = {"kind": "embed", "hidden": list(cfg.hidden), "dropout": cfg.dropout,
                      "best_epoch": result["best_epoch"], "val_mse": result["val_loss"],
                      "n_players": len(players), "n_teams": len(teams)}
    return predictor
