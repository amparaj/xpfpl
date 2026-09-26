"""Is the model robust? Out-of-sample checks of the forecasts and the backtest over many seasons.

`validate.py` scores one held-out season and `backtest.py` replays one season with one
configuration. This runs both across several seasons and asks what a single season can't answer.

Forecasts. Every test season is predicted by a model trained only on the seasons before it,
early-stopped on the season before that and then refit (`backtest.fit_model`, as the app would
have been trained at the time). From those predictions:
  - accuracy per season, and how much it moves from one season to the next
  - paired block-bootstrap intervals for the gap between two models (gameweeks resampled)
  - the early-stopping leak: the same season scored by a model early-stopped on it, which is
    what `xpfpl validate`/`compare` do
  - calibration per season, and whether a recalibration fitted on earlier seasons helps a later one
  - the winner's curse: the top-k xP in each position each week, predicted vs scored
  - bias by subgroup (position, part of the season, experience, price)
  - P(plays) calibration, from the ensemble's xmins member
  - seed stability: the same season refit with other starting weights
  - forecasts made 2 and 3 gameweeks ahead
  - a leak probe: features rebuilt from matches cut off at a deadline must equal the full build

Backtest. Per season, the app as configured against: each ensemble member, the baseline, a
perfect-foresight oracle, the pre-tuning settings, planning on, chips on, recalibrated xP,
other seeds, and the same run with a little noise on xP (how chaotic a replay is).

Stages (`xpfpl robustness --stage ...`): `forecasts` and `backtest` work per season and cache
under data/robustness/, so seasons can run in parallel processes; `report` combines them into
data/robustness/report.json.
"""

import json
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from xpfpl import backtest, config, models, validate
from xpfpl.features import TARGET

DIR = config.DATA_DIR / "robustness"
REPORT_PATH = DIR / "report.json"
MODEL = "ensemble"                   # members are read out of it, so it must be the ensemble
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SEEDS = (42, 7, 1234)                # the first is TrainConfig's default
SEED_SEASONS = ["2024-25", "2025-26"]  # refit with the other seeds (each costs a fit)
LEAKY_SEASONS = ["2025-26"]           # also fitted the way `xpfpl validate` does
LEAK_PROBES = [("2025-26", 20), ("2023-24", 8)]  # (season, GW) cut-offs for the feature leak probe
TUNED_ON = ("2023-24", "2024-25")    # the seasons `xpfpl tune` scored its candidates on
# config.py before `xpfpl tune` (CLAUDE.md, "The knobs"): planning on with one hit allowed.
PRE_TUNING = dict(horizon=5, discount=0.85, ft_value=1.5, bench_weight=0.1, max_hits=1,
                  plan_transfers=True, price_weight=0.0)
NOISE_SD = 0.1                       # lognormal noise on xP for the replay-chaos runs (~10%)
NOISE_RUNS = 5
BOOTSTRAP = 2000
TOP_K = 5
KEEP = ["season", "gw", "fixture", "element", "code", "name", "position", TARGET, "minutes",
        "played_r5", "experience", "price", "fpl_xp_prev"]
KEY = ["season", "element", "fixture"]


# ---------------------------------------------------------------- forecasts

def _model_path(season: str, seed: int, leaky: bool = False) -> Path:
    return DIR / "models" / f"{season}_{MODEL}_s{seed}{'_leaky' if leaky else ''}.pt"


def fitted(frame: pd.DataFrame, season: str, seed: int = SEEDS[0], leaky: bool = False):
    """The ensemble as it would have been trained just before `season` (cached on disk).

    `leaky=True` trains the way `xpfpl validate` does instead: early-stopped on `season` itself.
    """
    path = _model_path(season, seed, leaky)
    if path.exists():
        return models.module(MODEL).Predictor.load(path)
    if leaky:
        from xpfpl.models.trainer import TrainConfig
        history = frame[frame["season"].str[:4].astype(int) < int(season[:4])]
        predictor = models.fit(MODEL, history, frame[frame["season"] == season],
                               cfg=TrainConfig(seed=seed), quiet=True)
    else:
        predictor = backtest.fit_model(frame, season, name=MODEL, quiet=True, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(path)
    return predictor


def _aligned(rows: pd.DataFrame, other: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """`values` (one per row of `other`) reordered to line up with `rows`, by player and fixture."""
    series = pd.Series(values, index=pd.MultiIndex.from_frame(other[KEY]))
    series = series[~series.index.duplicated()]
    return series.reindex(pd.MultiIndex.from_frame(rows[KEY])).to_numpy(dtype=float)


def forecasts(frame: pd.DataFrame, season: str, seeds=(SEEDS[0],), stale: dict | None = None,
              leaky: bool = False) -> pd.DataFrame:
    """Out-of-sample predictions for every row of `season`, one column per model or variant.

    Columns: `ensemble` and its members, `baseline`, `p_play` (xmins), `ensemble_seed<s>` for
    each extra seed, `ensemble_h<k>`/`baseline_h<k>` from features as known k GWs ahead
    (`stale` maps k -> that frame), and `ensemble_leaky` when asked.
    """
    rows = frame[frame["season"] == season]
    out = rows[[c for c in KEEP if c in rows]].reset_index(drop=True)
    main = fitted(frame, season, seeds[0])
    baseline = models.load("baseline")
    cols = {MODEL: main.predict(rows), **{n: m.predict(rows) for n, m in main.members.items()},
            "baseline": baseline.predict(rows)}
    if "xmins" in main.members:
        cols["p_play"] = 1.0 - main.members["xmins"].expectations(rows)["p_none"].to_numpy()
    for s in seeds[1:]:
        cols[f"{MODEL}_seed{s}"] = fitted(frame, season, s).predict(rows)
    for k, stale_frame in (stale or {}).items():
        part = stale_frame[stale_frame["season"] == season]
        cols[f"{MODEL}_h{k}"] = _aligned(rows, part, main.predict(part))
        cols[f"baseline_h{k}"] = _aligned(rows, part, baseline.predict(part))
    if leaky:
        cols[f"{MODEL}_leaky"] = fitted(frame, season, seeds[0], leaky=True).predict(rows)
    return pd.concat([out, pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()})], axis=1)


def forecast_path(season: str) -> Path:
    return DIR / f"forecasts_{season}.parquet"


def load_forecasts(seasons=None) -> pd.DataFrame:
    """Every cached season's out-of-sample predictions, plus FPL's own xP as a benchmark."""
    paths = [forecast_path(s) for s in seasons] if seasons else sorted(DIR.glob("forecasts_*.parquet"))
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].between(1, 4)].reset_index(drop=True)   # 2024-25's assistant managers are 5
    if "fpl_xp_prev" in df:
        # Only where recorded for most rows (2020-21 on); a missing value is taken as 0, as validate does.
        recorded = df.groupby("season")["fpl_xp_prev"].transform(lambda s: s.notna().mean() > 0.5)
        df["fpl_xp"] = df["fpl_xp_prev"].fillna(0.0).where(recorded)
    return df


# ---------------------------------------------------------------- scoring the forecasts

def _active(df: pd.DataFrame) -> pd.DataFrame:
    """Players getting minutes (played in one of their last five matches), as validate.py uses."""
    return df[df["played_r5"] > 0]


def _gw_spearman(g: pd.DataFrame, col: str) -> float:
    ranks = [validate._spearman(w[TARGET].to_numpy(float), w[col].to_numpy(float))
             for _, w in g.groupby("gw")]
    ranks = [r for r in ranks if r == r]
    return float(np.mean(ranks)) if ranks else float("nan")


def accuracy(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """MAE/RMSE/R²/bias and per-GW rank correlation for each model, season by season."""
    rows = []
    for season, g in _active(df).groupby("season"):
        for col in cols:
            if col not in g or g[col].isna().all():
                continue
            h = g[g[col].notna()]
            rows.append({"season": season, "model": col,
                         **validate._scores(h[TARGET].to_numpy(float), h[col].to_numpy(float)),
                         "spearman": _gw_spearman(h, col)})
    return rows


def spread(table: list[dict], metrics=("rmse", "mae", "r2", "bias", "spearman")) -> list[dict]:
    """Across seasons: the mean, standard deviation and range of each metric for each model."""
    t = pd.DataFrame(table)
    rows = []
    for model, g in t.groupby("model", sort=False):
        row = {"model": model, "seasons": int(len(g))}
        for m in metrics:
            row.update({f"{m}_mean": float(g[m].mean()), f"{m}_sd": float(g[m].std(ddof=1)) if len(g) > 1 else 0.0,
                        f"{m}_min": float(g[m].min()), f"{m}_max": float(g[m].max())})
        rows.append(row)
    return rows


def paired_bootstrap(df: pd.DataFrame, a: str, b: str, reps: int = BOOTSTRAP, seed: int = 0) -> dict:
    """Is model `a` really better than `b`? Resample whole gameweeks (a season's week is the
    block, since errors within a week share the same matches) and recompute both RMSEs and the
    mean per-GW rank correlation. Negative `rmse_diff` / positive `spearman_diff` favour `a`."""
    g = _active(df)
    g = g[g[a].notna() & g[b].notna()]
    blocks = []
    for _, w in g.groupby(["season", "gw"]):
        y = w[TARGET].to_numpy(float)
        pa, pb = w[a].to_numpy(float), w[b].to_numpy(float)
        blocks.append((np.sum((pa - y) ** 2), np.sum((pb - y) ** 2), len(y),
                       validate._spearman(y, pa), validate._spearman(y, pb)))
    sa, sb, n, ra, rb = (np.array(x, dtype=float) for x in zip(*blocks))
    ok = ~(np.isnan(ra) | np.isnan(rb))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(reps, len(n)))
    rmse_diff = np.sqrt(sa[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
    rank_diff = np.array([np.mean((ra - rb)[i[ok[i]]]) for i in idx])
    observed = float(np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum()))
    return {"a": a, "b": b, "gameweeks": int(len(n)), "rmse_diff": observed,
            "rmse_lo": float(np.quantile(rmse_diff, 0.025)), "rmse_hi": float(np.quantile(rmse_diff, 0.975)),
            "p_a_better_rmse": float(np.mean(rmse_diff < 0)),
            "spearman_diff": float(np.mean((ra - rb)[ok])),
            "spearman_lo": float(np.quantile(rank_diff, 0.025)), "spearman_hi": float(np.quantile(rank_diff, 0.975)),
            "p_a_better_spearman": float(np.mean(rank_diff > 0))}


def calibration_line(y: np.ndarray, p: np.ndarray) -> dict:
    """Actual points regressed on xP. A calibrated forecast has slope 1 and intercept 0; a slope
    below 1 means the forecasts are too spread out (the highs too high, the lows too low)."""
    slope, intercept = np.polyfit(p, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def isotonic_fit(x: np.ndarray, y: np.ndarray, bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """A monotone map from xP to expected points (pool-adjacent-violators on quantile bins).

    Returns knots for `recalibrate`: the mean xP and fitted points of each pooled block.
    """
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max(len(edges) - 2, 0))
    size = len(edges) - 1 if len(edges) > 1 else 1
    w = np.bincount(idx, minlength=size).astype(float)
    sx, sy = np.bincount(idx, x, size), np.bincount(idx, y, size)
    keep = w > 0
    blocks: list[list[float]] = []                     # [points, weight, xP]
    for v, ww, xx in zip(sy[keep] / w[keep], w[keep], sx[keep] / w[keep]):
        blocks.append([v, ww, xx])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            v2, w2, x2 = blocks.pop()
            v1, w1, x1 = blocks.pop()
            total = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / total, total, (x1 * w1 + x2 * w2) / total])
    knots = np.array(blocks)
    return knots[:, 2], knots[:, 0]


def recalibrate(p: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray) -> np.ndarray:
    """Apply an `isotonic_fit` map. Between knots it interpolates, so the order of players is
    kept; above the top knot it adds the xP beyond it one for one, so the best players stay
    distinguishable (captaincy depends on that)."""
    p = np.asarray(p, dtype=float)
    out = np.interp(p, knots_x, knots_y)
    above = p > knots_x[-1]
    out[above] += p[above] - knots_x[-1]
    return np.clip(out, 0.0, None)


def recalibration_test(df: pd.DataFrame, col: str = MODEL) -> list[dict]:
    """Fit a linear and an isotonic recalibration on every earlier season's out-of-sample
    predictions and score them on the next one, against the raw forecast."""
    rows = []
    seasons = sorted(df["season"].unique())
    for season in seasons[1:]:
        before, now = df[df["season"] < season], df[df["season"] == season].copy()
        slope, intercept = np.polyfit(before[col], before[TARGET], 1)
        kx, ky = isotonic_fit(before[col].to_numpy(float), before[TARGET].to_numpy(float))
        variants = {"raw": now[col].to_numpy(float),
                    "linear": np.clip(intercept + slope * now[col].to_numpy(float), 0.0, None),
                    "isotonic": recalibrate(now[col].to_numpy(float), kx, ky)}
        act = (now["played_r5"] > 0).to_numpy()
        y = now[TARGET].to_numpy(float)
        for name, p in variants.items():
            top = act & (p >= 5)
            rows.append({"season": season, "method": name, **validate._scores(y[act], p[act]),
                         "spearman": _gw_spearman(now[act].assign(_p=p[act]), "_p"),
                         **calibration_line(y[act], p[act]),
                         "top_n": int(top.sum()), "top_bias": float(np.mean(p[top] - y[top])) if top.any() else float("nan")})
    return rows


def calibration(df: pd.DataFrame, col: str = MODEL) -> list[dict]:
    """The calibration bins (validate.py's) and the calibration line, per season and pooled."""
    rows = []
    g = _active(df)
    for season, w in [*g.groupby("season"), ("all", g)]:
        y, p = w[TARGET].to_numpy(float), w[col].to_numpy(float)
        rows.append({"season": season, **calibration_line(y, p),
                     "bins": validate._calibration(y, p, np.ones(len(y), dtype=bool))})
    return rows


def winners_curse(df: pd.DataFrame, col: str = MODEL, k: int = TOP_K) -> list[dict]:
    """Each week, the top `k` by xP in each position: what they were forecast vs what they scored.
    Picking the maximum of noisy forecasts selects the ones that are too high, so even an
    unbiased model looks optimistic here; the optimiser and the captaincy live in this corner."""
    rank = df.groupby(["season", "gw", "position"])[col].rank(ascending=False, method="first")
    top = df[rank <= k]
    rows = []
    for (season, pos), g in [*top.groupby(["season", "position"]), *((("all", p), g) for p, g in top.groupby("position"))]:
        y, p = g[TARGET].to_numpy(float), g[col].to_numpy(float)
        rows.append({"season": season, "position": config.POSITIONS.get(int(pos), "?"), "n": int(len(g)),
                     "predicted": float(p.mean()), "actual": float(y.mean()),
                     "bias": float(np.mean(p - y)), "se": float(np.std(y, ddof=1) / np.sqrt(len(y)))})
    return rows


def subgroups(df: pd.DataFrame, cols=(MODEL, "baseline")) -> list[dict]:
    """Bias and RMSE by position, part of the season, experience and price (pooled seasons),
    and by position within each season (a drifting bias is the warning for a rule change)."""
    g = _active(df)
    dims = {
        "position": g["position"].astype(int).map(config.POSITIONS),
        "part of season": pd.cut(g["gw"], [0, 5, 19, 100], labels=["GW1-5", "GW6-19", "GW20+"]),
        "experience": pd.cut(g["experience"] * 38, [-1, 4.5, 18.5, 100],
                             labels=["under 5 matches", "5-18 matches", "19+ matches"]),
        "price": pd.cut(g["price"], [0, 5, 7, 9, 100], labels=["under £5m", "£5-7m", "£7-9m", "£9m+"]),
        "season x position": g["season"] + " " + g["position"].astype(int).map(config.POSITIONS),
    }
    rows = []
    y = g[TARGET].to_numpy(float)
    for dim, labels in dims.items():
        labels = pd.Series(labels, index=g.index).astype(str).to_numpy()
        for value in pd.unique(labels):
            m = labels == value
            for col in cols:
                rows.append({"dimension": dim, "group": value, "model": col,
                             **validate._scores(y[m], g[col].to_numpy(float)[m])})
    return rows


def p_play_calibration(df: pd.DataFrame) -> dict:
    """Does "70% to play" mean 70%? The xmins member's P(minutes > 0) against what happened,
    with the Brier score of it and of the naive `played_r5` (share of the last five played)."""
    g = df[df["p_play"].notna()] if "p_play" in df else df.iloc[:0]
    if g.empty:
        return {}
    played = (g["minutes"] > 0).to_numpy(float)
    p = g["p_play"].to_numpy(float)
    naive = g["played_r5"].fillna(0).clip(0, 1).to_numpy(float)
    bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    table = (pd.DataFrame({"bin": bins, "p": p, "played": played}).groupby("bin", observed=True)
             .agg(predicted=("p", "mean"), actual=("played", "mean"), n=("p", "size")).reset_index())
    table["bin"] = table["bin"].astype(str)
    by_season = [{"season": s, "brier": float(np.mean((w["p_play"] - (w["minutes"] > 0)) ** 2)),
                  "brier_naive": float(np.mean((w["played_r5"].fillna(0).clip(0, 1) - (w["minutes"] > 0)) ** 2))}
                 for s, w in g.groupby("season")]
    return {"brier": float(np.mean((p - played) ** 2)), "brier_naive": float(np.mean((naive - played) ** 2)),
            "bins": table.to_dict("records"), "by_season": by_season}


def seed_stability(df: pd.DataFrame, k: int = TOP_K) -> list[dict]:
    """The same season refit from other starting weights: how far the forecasts move, whether
    the same players make each position's weekly top `k`, and whether the captain changes."""
    rows = []
    others = [c for c in df.columns if c.startswith(f"{MODEL}_seed")]
    for season, g in df.groupby("season"):
        for other in [c for c in others if g[c].notna().any()]:
            act = g[g["played_r5"] > 0]
            y = act[TARGET].to_numpy(float)
            diff = (act[other] - act[MODEL]).to_numpy(float)
            overlap, captain = [], []
            for _, w in g.groupby("gw"):
                for _, wp in w.groupby("position"):
                    a, b = set(wp.nlargest(k, MODEL)["element"]), set(wp.nlargest(k, other)["element"])
                    overlap.append(len(a & b) / k)
                captain.append(w.loc[w[MODEL].idxmax(), "element"] == w.loc[w[other].idxmax(), "element"])
            rows.append({"season": season, "seed": other.removeprefix(f"{MODEL}_seed"),
                         "rmse_base": validate._scores(y, act[MODEL].to_numpy(float))["rmse"],
                         "rmse_seed": validate._scores(y, act[other].to_numpy(float))["rmse"],
                         "xp_sd": float(np.std(diff)), "xp_mean_abs": float(np.mean(np.abs(diff))),
                         "top_overlap": float(np.mean(overlap)), "captain_same": float(np.mean(captain))})
    return rows


def horizons(df: pd.DataFrame) -> list[dict]:
    """Accuracy of the forecast made 1, 2, 3... gameweeks before the match, season by season."""
    cols = {1: [MODEL, "baseline"]}
    for c in df.columns:
        if c.startswith((f"{MODEL}_h", "baseline_h")):
            cols.setdefault(int(c.rsplit("_h", 1)[1]), []).append(c)
    rows = []
    for k, names in sorted(cols.items()):
        for r in accuracy(df, names):
            rows.append({**r, "horizon": k, "model": r["model"].split("_h")[0]})
    return rows


# ---------------------------------------------------------------- leak probe

def leak_probe(matches: pd.DataFrame, season: str, gw: int, tol: float = 1e-4, drift: float = 0.01) -> dict:
    """Rebuild the features from only the matches up to `season` GW `gw` and compare them with
    the full build on every row up to that point. A feature that differs by more than `drift`
    (or is missing on one side only) saw the future; smaller differences are listed as drift."""
    from xpfpl import teams
    from xpfpl.data import markets
    from xpfpl.features import FEATURES, LAG_FEATURES, build_training_frame

    market, _ = markets.load()
    full = build_training_frame(matches, market=market)
    year = matches["season"].str[:4].astype(int)
    cut = matches[(year < int(season[:4])) | ((matches["season"] == season) & (matches["gw"] <= gw))]
    with tempfile.TemporaryDirectory() as tmp:
        cache = teams.CACHE_PATH
        teams.CACHE_PATH = Path(tmp) / "ratings.parquet"    # never overwrite the real cache
        try:
            part = build_training_frame(cut, market=market)
        finally:
            teams.CACHE_PATH = cache
    cols = [c for c in dict.fromkeys(FEATURES + LAG_FEATURES) if c in part and c in full]
    a, b = full.set_index(KEY), part.set_index(KEY)
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    common = b.index.intersection(a.index)
    va = a.loc[common, cols].to_numpy(dtype=float)
    vb = b.loc[common, cols].to_numpy(dtype=float)
    diff = np.abs(va - vb)
    bad = (diff > tol) | (np.isnan(va) != np.isnan(vb))
    differ = [{"feature": c, "rows": int(bad[:, i].sum()), "max_diff": float(np.nanmax(diff[:, i])),
               "nan_mismatch": int((np.isnan(va[:, i]) != np.isnan(vb[:, i])).sum())}
              for i, c in enumerate(cols) if bad[:, i].any()]
    differ = sorted(differ, key=lambda r: -r["rows"])
    # The team ratings are refitted by L-BFGS with a fixed iteration budget over one parameter per
    # club in the file, so a build with fewer clubs takes a slightly different path to (nearly) the
    # same optimum. Differences that small are drift, not information: only bigger ones count.
    this = common.get_level_values("season") == season
    return {"season": season, "gw": gw, "rows_compared": int(len(common)),
            "rows_this_season": int(this.sum()), "features_compared": len(cols),
            "leaking": [r for r in differ if r["max_diff"] > drift or r["nan_mismatch"]],
            "drift": [r for r in differ if r["max_diff"] <= drift and not r["nan_mismatch"]]}


# ---------------------------------------------------------------- backtest

class Oracle:
    """Perfect foresight: every player's xP is the points they went on to score. The ceiling
    for the optimiser, the transfer rules and the budget, with forecasting taken out of it."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[TARGET].to_numpy(dtype="float32")


class Noisy:
    """Another predictor's xP times lognormal noise (mean 1). Nearly the same forecasts, so the
    spread of season totals across runs is how much a replay moves on chaos alone."""

    def __init__(self, base, sd: float, seed: int):
        self.base, self.sd, self.rng = base, sd, np.random.default_rng(seed)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.base.predict(frame)
        return p * np.exp(self.rng.normal(-self.sd ** 2 / 2, self.sd, len(p)))


class Mapped:
    """Another predictor's xP passed through a recalibration."""

    def __init__(self, base, fn):
        self.base, self.fn = base, fn

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fn(self.base.predict(frame)).astype("float32")


def backtest_path(season: str) -> Path:
    return DIR / f"backtest_{season}.json"


def backtests(frame: pd.DataFrame, season: str, earlier: pd.DataFrame | None = None,
              seeds=(SEEDS[0],), noise_runs: int = NOISE_RUNS) -> list[dict]:
    """Replay `season` once per variant. `earlier` (out-of-sample forecasts of earlier seasons)
    fits the recalibrated variants; without it they're skipped."""
    from xpfpl import prices

    main = fitted(frame, season, seeds[0])
    table = prices.fit(frame, before_season=season)
    base = backtest.Settings(season=season, model=MODEL)
    runs = [(MODEL, base, main)]
    runs += [(name, replace(base, model=name), member) for name, member in main.members.items()]
    runs += [("baseline", replace(base, model="baseline"), None), ("oracle", base, Oracle()),
             ("pre-tuning settings", replace(base, **PRE_TUNING), main),
             ("planning on", replace(base, plan_transfers=True), main),
             ("chips on", replace(base, chips=True), main)]
    if earlier is not None and len(earlier):
        slope, intercept = np.polyfit(earlier[MODEL], earlier[TARGET], 1)
        kx, ky = isotonic_fit(earlier[MODEL].to_numpy(float), earlier[TARGET].to_numpy(float))
        runs += [("recalibrated (linear)", base, Mapped(main, lambda p: np.clip(intercept + slope * p, 0.0, None))),
                 ("recalibrated (isotonic)", base, Mapped(main, lambda p: recalibrate(p, kx, ky)))]
    runs += [(f"seed {s}", base, fitted(frame, season, s)) for s in seeds[1:]]
    runs += [(f"noise {r}", base, Noisy(main, NOISE_SD, r)) for r in range(noise_runs)]

    rows = []
    for label, settings, predictor in runs:
        result = backtest.run(settings, frame=frame, predictor=predictor, price_table=table, verbose=False)
        s, g = result.summary, result.gameweeks
        rows.append({"season": season, "variant": label, "points": s["points"], "hits": s["hits"],
                     "transfers": s["transfers"], "captain_points": s["captain_points"],
                     "bench_points": s["bench_points"], "xp": s["xp"],
                     "xp_error_per_gw": s["xp_error_per_gw"], "chips": s["chips"],
                     "points_by_gw": [float(x) for x in g["points"]]})
        print(f"  {season} {label:26s} {s['points']:6.0f} points ({s['hits']} hits)", flush=True)
    return rows


def load_backtests() -> pd.DataFrame:
    rows = []
    for path in sorted(DIR.glob("backtest_*.json")):
        rows += json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> dict:
    """Season totals by variant, each variant's gap to the configured ensemble season by season,
    and the noise floor: the spread of the noisy runs and of the seeds around it."""
    if bt.empty:
        return {}
    wide = bt.pivot_table(index="variant", columns="season", values="points")
    ref = wide.loc[MODEL]
    gaps = []
    for variant, row in wide.iterrows():
        d = (row - ref).dropna()
        gaps.append({"variant": variant, "mean": float(row.mean()), "seasons": int(row.notna().sum()),
                     "gap_mean": float(d.mean()), "gap_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                     "seasons_better": int((d > 0).sum()), "seasons_worse": int((d < 0).sum()),
                     "gap_tuned_seasons": float(d[d.index.isin(TUNED_ON)].mean()) if d.index.isin(TUNED_ON).any() else None,
                     "gap_other_seasons": float(d[~d.index.isin(TUNED_ON)].mean()) if (~d.index.isin(TUNED_ON)).any() else None})
    noise = wide[wide.index.str.startswith("noise")]
    seeds = wide[wide.index.str.startswith("seed")]
    floor = {"noise_sd_by_season": {s: float(v) for s, v in noise.std(ddof=1).dropna().items()},
             "noise_range_by_season": {s: float(noise[s].max() - noise[s].min()) for s in noise.columns if noise[s].notna().any()},
             "seed_points": {s: [float(x) for x in pd.concat([seeds[s], ref[[s]]]).dropna()] for s in seeds.columns
                             if seeds[s].notna().any()}}
    sds = list(floor["noise_sd_by_season"].values())
    floor["noise_sd"] = float(np.sqrt(np.mean(np.square(sds)))) if sds else None
    return {"points": {v: {s: float(x) for s, x in row.dropna().items()} for v, row in wide.iterrows()},
            "gaps": sorted(gaps, key=lambda r: -r["gap_mean"]), "noise_floor": floor}


# ---------------------------------------------------------------- the report

COMPARE = [("ensemble", "baseline"), ("ensemble", "fpl_xp"), ("ensemble", "mlp"), ("ensemble", "gbm"),
           ("ensemble", "xmins"), ("gbm", "mlp"), ("ensemble_leaky", "ensemble")]


def build_report(df: pd.DataFrame, bt: pd.DataFrame, probes: list[dict]) -> dict:
    models_ = [c for c in (MODEL, "mlp", "gbm", "xmins", "baseline", "fpl_xp") if c in df]
    table = accuracy(df, models_)
    boot = []
    for a, b in COMPARE:
        if a in df and b in df:
            both = df[df[a].notna() & df[b].notna()]
            if both.empty:
                continue
            boot.append({"scope": "all seasons", **paired_bootstrap(both, a, b)})
            for season, g in both.groupby("season"):
                boot.append({"scope": season, **paired_bootstrap(g, a, b, reps=500)})
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seasons": sorted(df["season"].unique()),
        "rows": int(len(df)), "active_rows": int((df["played_r5"] > 0).sum()),
        "accuracy": table,
        "spread": spread(table),
        "bootstrap": boot,
        "calibration": calibration(df),
        "recalibration": recalibration_test(df),
        "winners_curse": winners_curse(df),
        "subgroups": subgroups(df),
        "p_play": p_play_calibration(df),
        "seeds": seed_stability(df),
        "horizons": horizons(df),
        "leak_probe": probes,
        "backtest": backtest_summary(bt),
        "backtest_runs": bt.drop(columns=["points_by_gw"], errors="ignore").to_dict("records") if len(bt) else [],
    }


def save_report(report: dict, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


def load_report(path: Path = REPORT_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarise(report: dict) -> str:
    """The report as text, for the end of `xpfpl robustness`."""
    lines = [f"Out-of-sample checks over {', '.join(report['seasons'])} ({report['active_rows']:,} active rows)"]
    sp = pd.DataFrame(report["spread"])
    if len(sp):
        lines.append("\nAccuracy across seasons (players getting minutes): mean, sd, range")
        for r in sp.itertuples():
            lines.append(f"  {r.model:9s} RMSE {r.rmse_mean:.3f} ±{r.rmse_sd:.3f} ({r.rmse_min:.3f}-{r.rmse_max:.3f})  "
                         f"rank corr {r.spearman_mean:.3f} ±{r.spearman_sd:.3f}  bias {r.bias_mean:+.3f}")
    boot = [b for b in report["bootstrap"] if b["scope"] == "all seasons"]
    if boot:
        lines.append("\nPaired bootstrap over gameweeks (a - b; 95% interval)")
        for b in boot:
            lines.append(f"  {b['a']:>14s} vs {b['b']:9s} RMSE {b['rmse_diff']:+.4f} ({b['rmse_lo']:+.4f} to {b['rmse_hi']:+.4f})"
                         f"  rank corr {b['spearman_diff']:+.4f} ({b['spearman_lo']:+.4f} to {b['spearman_hi']:+.4f})")
    cal = [c for c in report["calibration"] if c["season"] == "all"]
    if cal:
        lines.append(f"\nCalibration line (pooled): points = {cal[0]['intercept']:+.2f} + {cal[0]['slope']:.3f} x xP")
    rc = pd.DataFrame(report["recalibration"])
    if len(rc):
        m = rc.groupby("method")[["rmse", "spearman", "slope", "top_bias"]].mean()
        lines.append("Recalibration fitted on earlier seasons (mean over the later ones):")
        for method, r in m.iterrows():
            lines.append(f"  {method:9s} RMSE {r.rmse:.4f}  rank corr {r.spearman:.4f}  slope {r.slope:.3f}  "
                         f"bias when xP >= 5 {r.top_bias:+.2f}")
    wc = [w for w in report["winners_curse"] if w["season"] == "all"]
    if wc:
        lines.append(f"\nWinner's curse (weekly top {TOP_K} by xP per position, all seasons)")
        for w in wc:
            lines.append(f"  {w['position']}  forecast {w['predicted']:.2f}  scored {w['actual']:.2f}  "
                         f"(bias {w['bias']:+.2f} ± {1.96 * w['se']:.2f})")
    pp = report.get("p_play") or {}
    if pp:
        lines.append(f"\nP(plays): Brier {pp['brier']:.4f} vs {pp['brier_naive']:.4f} for 'share of last 5 played'")
    if report["seeds"]:
        s = pd.DataFrame(report["seeds"])
        lines.append(f"Seeds: xP moves by {s['xp_mean_abs'].mean():.3f} on average; weekly top-{TOP_K} overlap "
                     f"{s['top_overlap'].mean():.0%}; same captain {s['captain_same'].mean():.0%}; "
                     f"RMSE {s['rmse_base'].mean():.3f} vs {s['rmse_seed'].mean():.3f}")
    hz = pd.DataFrame(report["horizons"])
    if len(hz):
        t = hz.groupby(["model", "horizon"])[["rmse", "spearman"]].mean().round(3)
        lines.append("\nBy gameweeks ahead (mean over seasons):\n" + t.to_string())
    for p in report["leak_probe"]:
        state = "no leaks" if not p["leaking"] else f"LEAKS in {', '.join(r['feature'] for r in p['leaking'][:8])}"
        if p["drift"]:
            state += (f"; optimiser drift under {max(r['max_diff'] for r in p['drift']):.4f} in "
                      f"{', '.join(r['feature'] for r in p['drift'])}")
        lines.append(f"Leak probe {p['season']} GW{p['gw']}: {p['rows_compared']:,} rows x "
                     f"{p['features_compared']} features, {state}")
    bt = report.get("backtest") or {}
    if bt:
        lines.append("\nBacktest (season points; gap to the configured ensemble)")
        for g in bt["gaps"]:
            if g["variant"].startswith(("noise", "seed")):
                continue
            lines.append(f"  {g['variant']:26s} mean {g['mean']:6.0f}  gap {g['gap_mean']:+6.0f} ± {g['gap_sd']:4.0f}  "
                         f"better in {g['seasons_better']}/{g['seasons']}")
        floor = bt["noise_floor"]
        if floor.get("noise_sd") is not None:
            lines.append(f"  noise floor: season total sd {floor['noise_sd']:.0f} points with ~{NOISE_SD:.0%} xP noise")
    return "\n".join(lines)
