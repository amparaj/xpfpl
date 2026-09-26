"""The live record: every saved pre-deadline forecast, scored against what then happened.

Backtests and held-out seasons say how the model *would* have done. This says how it *is*
doing, week by week, with forecasts nobody can quietly revise: `predict_upcoming` saves each
run to data/predictions/<season>/gwNN_<model>.csv before that gameweek's deadline, and once the
gameweek is in `matches.parquet` (after `xpfpl fetch`) it is scored here - a public-style
accuracy record that can't be revised after the fact.

Per gameweek and model: MAE, RMSE, R², rank correlation and bias, over every player forecast
and over the players who'd been getting minutes. Bias by position is the early warning for
rule changes the history can't teach (e.g. the 2026-27 bonus-points changes). Forecasts saved
with Monte Carlo ranges (simulate.py) are also scored on those (`ranges`): how many scores
landed inside the 10th-90th percentile band, and the average chance of 10+ / of 2 or fewer
against how often it happened.
"""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from xpfpl import config
from xpfpl.validate import _scores, _spearman

FILE = re.compile(r"gw(\d+)_(\w+)\.csv$")
ACTIVE_LOOKBACK = 5            # "getting minutes" = played in one of the previous 5 gameweeks


def _actuals(matches: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """Points and minutes per player in `gw` (a double gameweek sums both matches)."""
    rows = matches[(matches["season"] == season) & (matches["gw"] == gw)]
    return rows.groupby("element")[["total_points", "minutes"]].sum()


def _active(matches: pd.DataFrame, season: str, gw: int) -> pd.Index:
    recent = matches[(matches["season"] == season) & matches["gw"].between(gw - ACTIVE_LOOKBACK, gw - 1)]
    return pd.Index(recent.loc[recent["minutes"] > 0, "element"].unique())


def score(matches: pd.DataFrame, season: str) -> dict:
    """Score every saved forecast for `season` whose gameweek has been played."""
    folder = config.PREDICTIONS_DIR / season
    played = set(matches.loc[matches["season"] == season, "gw"].unique())
    rows, positions, ranges = [], [], []
    for path in sorted(folder.glob("gw*_*.csv")) if folder.exists() else []:
        match = FILE.search(path.name)
        if not match:
            continue
        gw, model = int(match.group(1)), match.group(2)
        column = f"xp_{gw}"
        if gw not in played:
            continue
        pred = pd.read_csv(path, index_col="element")
        if column not in pred:
            continue
        actual = _actuals(matches, season, gw).reindex(pred.index).fillna(0.0)   # blank GW -> 0
        y = actual["total_points"].to_numpy(dtype=float)
        p = pred[column].to_numpy(dtype=float)
        active = pred.index.isin(_active(matches, season, gw))
        for subset, mask in (("All players", np.ones(len(y), dtype=bool)), ("Players getting minutes", active)):
            rows.append({"gw": gw, "model": model, "subset": subset, **_scores(y[mask], p[mask]),
                         "spearman": _spearman(y[mask], p[mask])})
        if {"pts_p10", "pts_p90", "p_haul", "p_blank"} <= set(pred.columns):
            r, got = pred[active], y[active]
            ranges.append({"gw": gw, "model": model, "n": int(active.sum()),
                           "below": float((got < r["pts_p10"]).mean()), "above": float((got > r["pts_p90"]).mean()),
                           "inside": float(((got >= r["pts_p10"]) & (got <= r["pts_p90"])).mean()),
                           "p_haul": float(r["p_haul"].mean()), "haul": float((got >= 10).mean()),
                           "p_blank": float(r["p_blank"].mean()), "blank": float((got <= 2).mean())})
        for pos, name in config.POSITIONS.items():
            mask = active & (pred["position"].to_numpy() == pos)
            if mask.any():
                positions.append({"gw": gw, "model": model, "position": name,
                                  "predicted": float(p[mask].mean()), "actual": float(y[mask].mean()),
                                  "n": int(mask.sum())})
    return {"generated": datetime.now().isoformat(timespec="seconds"), "season": season,
            "gameweeks": rows, "positions": positions, "ranges": ranges}


def save(report: dict) -> None:
    path = config.PREDICTIONS_DIR / report["season"] / "scorecard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")


def load(season: str) -> dict | None:
    try:
        return json.loads((config.PREDICTIONS_DIR / season / "scorecard.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarise(report: dict) -> str:
    gws = pd.DataFrame(report["gameweeks"])
    if gws.empty:
        return (f"No scored forecasts yet for {report['season']}: run `xpfpl predict` (or `recommend`) "
                f"before each deadline, and `xpfpl fetch` after the gameweek.")
    lines = [f"Live record, {report['season']} (forecasts saved before each deadline):"]
    active = gws[gws["subset"] == "Players getting minutes"]
    table = active[["gw", "model", "n", "mae", "rmse", "r2", "spearman", "bias"]].sort_values(["model", "gw"])
    lines.append(table.round(3).to_string(index=False))
    season = active.groupby("model")[["mae", "rmse", "r2", "spearman", "bias"]].mean()
    lines.append("\nSeason so far (mean of gameweeks):\n" + season.round(3).to_string())
    pos = pd.DataFrame(report["positions"])
    if len(pos):
        bias = pos.assign(bias=pos["predicted"] - pos["actual"]).pivot_table(
            index="model", columns="position", values="bias", aggfunc="mean")
        lines.append("\nBias by position (xP minus points, + = over-predicting):\n" + bias.round(2).to_string())
    ranges = pd.DataFrame(report.get("ranges", []))
    if len(ranges):
        lines.append("\nMonte Carlo ranges (scores are whole numbers, so inside the band should be 80% or a bit more):\n"
                     + ranges[["gw", "model", "n", "below", "inside", "above", "p_haul", "haul", "p_blank", "blank"]]
                     .sort_values(["model", "gw"]).round(3).to_string(index=False))
    return "\n".join(lines)
