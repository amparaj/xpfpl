"""Scoring forecasts of the whole points distribution, not just its average.

xP is an average; the Monte Carlo (simulate.py) and the `dist` model (models/dist.py) forecast
the chance of every score. RMSE can't tell two such forecasts apart if their averages agree, so
they are scored with two proper scores (a forecaster does best on them by saying what it
really believes):

- log score: minus the log of the chance given to the score that happened, averaged. It
  punishes a confident miss hard (a score given ~0% costs a lot).
- RPS (ranked probability score): the squared gaps between the forecast and the outcome's
  cumulative distributions, summed over every score. For whole-number outcomes this is the
  CRPS; it rewards being close (7 when 8 happened beats 2), which the log score ignores.

Lower is better for both. Scores run from LOW to HIGH; anything beyond goes in the end class.

The benchmark (`form_bands`) is what a forecaster with no model would say: the spread of
scores past players of the same position and similar 5-match form went on to get.
"""

import numpy as np
import pandas as pd

LOW, HIGH = -3, 25
VALUES = np.arange(LOW, HIGH + 1)
K = len(VALUES)
SMOOTHING = 0.5          # counts added to every class of a table of past scores, so no score gets 0%
SIM_SMOOTHING = 0.1      # the same for a simulated pmf (~0.3% of the mass over 1000 sims)
FORM_BANDS = [-np.inf, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, np.inf]
BENCHMARK = "Past scores by position and form"


def classes(points) -> np.ndarray:
    """Each score's class index (0 = LOW or below, K-1 = HIGH or above)."""
    return (np.clip(np.rint(np.asarray(points, dtype="float64")), LOW, HIGH) - LOW).astype(np.int64)


def from_draws(points: np.ndarray, smoothing: float = SIM_SMOOTHING) -> np.ndarray:
    """(sims, rows) simulated points -> (rows, K) probabilities, smoothed."""
    idx = classes(points)
    counts = np.stack([(idx == k).sum(axis=0) for k in range(K)], axis=1).astype("float64")
    return (counts + smoothing) / (idx.shape[0] + smoothing * K)


def scores(pmf: np.ndarray, actual) -> dict:
    """Mean log score and RPS of (rows, K) forecasts against the actual points."""
    y = classes(actual)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    log = -np.log(np.clip(pmf[np.arange(len(y)), y], 1e-12, None))
    cdf = np.cumsum(pmf, axis=1)
    hit = (np.arange(K)[None, :] >= y[:, None]).astype("float64")
    rps = ((cdf - hit) ** 2).sum(axis=1)
    return {"log_score": float(log.mean()), "rps": float(rps.mean()), "rows": int(len(y))}


def table(forecasts: dict[str, np.ndarray], actual) -> list[dict]:
    """One row per forecaster ({label: (rows, K) pmf}), best log score first."""
    rows = [{"model": label, **scores(pmf, actual)} for label, pmf in forecasts.items()]
    return sorted(rows, key=lambda r: r["log_score"])


def mean(pmf: np.ndarray) -> np.ndarray:
    return pmf @ VALUES


def form_bands(history: pd.DataFrame, target: str = "total_points", form: str = "total_points_r5") -> pd.DataFrame:
    """The benchmark: for each position x 5-match form band, the share of past rows with each
    score. Returns a (position, band) x K table of probabilities."""
    band = pd.cut(history[form].fillna(0.0), FORM_BANDS, labels=False, right=False)
    counts = pd.crosstab([history["position"].astype(int), band], classes(history[target]))
    counts = counts.reindex(columns=range(K), fill_value=0)
    return (counts + SMOOTHING).div(counts.sum(axis=1) + SMOOTHING * K, axis=0)


def from_bands(table: pd.DataFrame, frame: pd.DataFrame, form: str = "total_points_r5") -> np.ndarray:
    """Each row's benchmark forecast from `form_bands`' table (the overall spread if unseen)."""
    band = pd.cut(frame[form].fillna(0.0), FORM_BANDS, labels=False, right=False)
    keys = pd.MultiIndex.from_arrays([frame["position"].astype(int), band])
    fallback = table.mean(axis=0).to_numpy()
    out = table.reindex(keys).to_numpy(copy=True)        # pandas 3 hands back a read-only view
    missing = np.isnan(out).any(axis=1)
    out[missing] = fallback
    return out
