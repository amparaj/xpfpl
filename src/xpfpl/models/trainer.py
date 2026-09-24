"""The training loop every PyTorch model here shares: batches, early stopping, best weights.

Each model module (mlp, components, embed, sequence) builds its own tensors and loss and
hands them to `fit`, so the interesting part - the architecture - stays the only difference
between them.
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5          # stop after this many epochs without validation improvement
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    seed: int = 42
    balance_returns: bool = False   # weight blanks, tickers and hauls equally in the loss.
                                    # Tried on 2025-26: biased xP by +1.7 and hurt ranking, so off.
    member_epochs: dict | None = None   # ensemble refits: each member's own best epoch count


RETURN_BINS = (2.5, 4.5)            # blanks (<=2) / tickers (3-4) / haulers (5+)


def return_weights(points: np.ndarray) -> np.ndarray:
    """Per-row weights that give each return group the same total weight.

    Most rows are blanks, so plain MSE mostly learns to predict blanks well; this pushes the
    model towards the hauls that decide ranks. Clipped at the 95th percentile and rescaled to a
    mean of 1. The price is bias: the weighted mean is no longer an expectation.
    """
    group = np.digitize(points, RETURN_BINS)
    counts = np.bincount(group, minlength=len(RETURN_BINS) + 1).astype(float)
    w = (len(points) / (len(counts) * np.maximum(counts, 1.0)))[group]
    w = np.minimum(w, np.quantile(w, 0.95))
    return (w / w.mean()).astype("float32")


def seed(cfg: TrainConfig) -> None:
    """Make a fit reproducible.

    Call this *before* building the model: `nn.Module.__init__` draws the initial weights from
    the global RNG, so seeding afterwards leaves every run starting from different weights - and
    with early stopping that changes the epoch count, the model, and every decision downstream.
    """
    torch.manual_seed(cfg.seed)


def standardiser(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of the training features. Constant columns get std 1 so they don't blow up."""
    x = frame[features].to_numpy(dtype="float32")
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-3] = 1.0  # constant in training (e.g. stats that didn't exist yet)
    return mean, std


def standardise(frame: pd.DataFrame, features: list[str], mean, std) -> np.ndarray:
    return (frame[features].to_numpy(dtype="float32") - mean) / std


def fit(model: torch.nn.Module, train_tensors: tuple, val_tensors: tuple | None,
        loss_fn, cfg: TrainConfig, quiet: bool = False) -> dict:
    """Train `model` in place; return the best weights, epoch and validation loss.

    `loss_fn(model, batch)` gets a tuple of tensors (model inputs first, targets last) and
    returns a scalar loss, so a multi-head model can weigh several losses however it likes.
    With no `val_tensors` it simply trains for exactly cfg.epochs, which is what the final
    refit on every season does.

    Seeds the shuffling; the caller seeds the weights with `seed()` before building the model.
    """
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loader = DataLoader(TensorDataset(*train_tensors), batch_size=cfg.batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in loader:
            batch = [t.to(device) for t in batch]
            optimiser.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch[0])
            seen += len(batch[0])
        train_loss = total / max(seen, 1)

        if val_tensors is None:
            if not quiet:
                print(f"epoch {epoch:3d}  train loss {train_loss:.3f}")
            best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
            continue

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model, [t.to(device) for t in val_tensors]).item()
        if not quiet:
            print(f"epoch {epoch:3d}  train loss {train_loss:.3f}  val loss {val_loss:.3f}")

        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if not quiet:
                    print(f"Early stopping - best val loss {best_loss:.3f}")
                break

    model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_epoch": best_epoch, "val_loss": None if val_tensors is None else best_loss}
