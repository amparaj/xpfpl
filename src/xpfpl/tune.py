"""Set the tuning parameters in config.py from evidence instead of taste.

Every setting in config.py - how far ahead to look, how much to discount later gameweeks, what
a banked free transfer is worth, when a chip is worth playing - changes the season total, so
the backtest is the right judge. Trying every combination is not affordable (five parameters with
four values each is 1024 seasons of football), so this does coordinate descent: take one small
group of related parameters at a time, score each candidate over several seasons, keep the winner,
move on. A stage is only ever a handful of runs, and each parameter is tuned against the best
version of the ones before it.

    xpfpl tune --seasons 2022-23,2023-24,2024-25

Scoring is the mean points per gameweek across the seasons, so no single season's quirks (an
easy title run, a COVID restart) decide a parameter on their own. Every trial is written to
data/backtests/tuning.json, and the run ends by printing the config.py block to paste in -
deliberately not written automatically, so the numbers get a human glance first.

One caveat worth remembering when reading the output: the seasons used for tuning are the same
ones used to report the backtest score, so the tuned numbers flatter themselves a little. The
gaps between candidates are what matters, not the absolute totals.
"""

import json
from dataclasses import asdict, replace
from datetime import datetime

import pandas as pd

from xpfpl import config
from xpfpl.backtest import Settings, predictors, run
from xpfpl.data.history import load_matches
from xpfpl.features import build_training_frame
from xpfpl import prices

# Each stage is a group of parameters tried together, in the order they matter. Parameters in the same
# stage interact (a longer horizon wants a gentler discount), so their values are crossed.
STAGES: list[tuple[str, dict[str, list]]] = [
    ("horizon and discount", {"horizon": [3, 5, 8], "discount": [0.8, 0.9]}),
    ("transfer planning", {"plan_transfers": [False, True], "max_hits": [0, 1, 2]}),
    ("free transfer value", {"ft_value": [0.5, 1.5, 3.0]}),
    ("bench weight", {"bench_weight": [0.05, 0.1, 0.2]}),
    ("price changes", {"price_weight": [0.0, 1.0, 3.0]}),
]
# Chip thresholds are tuned one chip at a time, with chips switched on.
CHIP_STAGES: list[tuple[str, dict[str, list]]] = [
    ("triple captain", {"3xc": [6.0, 9.0, 12.0]}),
    ("bench boost", {"bboost": [8.0, 12.0, 16.0]}),
    ("free hit", {"freehit": [8.0, 12.0, 18.0]}),
    ("wildcard", {"wildcard": [12.0, 20.0, 30.0]}),
]
CONFIG_NAMES = {"horizon": "HORIZON", "discount": "DISCOUNT", "bench_weight": "BENCH_WEIGHT",
                "ft_value": "FT_VALUE", "max_hits": "MAX_HITS", "plan_transfers": "PLAN_TRANSFERS",
                "price_weight": "PRICE_WEIGHT", "3xc": "TRIPLE_CAPTAIN_MIN_XP",
                "bboost": "BENCH_BOOST_MIN_XP", "freehit": "FREE_HIT_MIN_GAIN",
                "wildcard": "WILDCARD_MIN_GAIN"}


def _combos(grid: dict[str, list]) -> list[dict]:
    from itertools import product
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def _score(base: Settings, choice: dict, seasons: list[str], frame, fitted, tables,
           chips_on: bool, verbose: bool) -> dict:
    """Replay every season with one candidate setting and average the points per gameweek."""
    thresholds = {k: v for k, v in choice.items() if k in ("3xc", "bboost", "freehit", "wildcard")}
    settings = {k: v for k, v in choice.items() if k not in thresholds}
    per_season = {}
    for season in seasons:
        trial_settings = replace(base, season=season, chips=chips_on, **settings)
        if thresholds:
            trial_settings.thresholds = {**base.thresholds, **thresholds}
        result = run(trial_settings, frame=frame, predictor=fitted[season],
                     price_table=tables[season], verbose=verbose)
        per_season[season] = result.summary["points"]
    points = list(per_season.values())
    return {**choice, "points_per_season": per_season,
            "mean_points": sum(points) / len(points), "worst_season": min(points)}


def tune(seasons: list[str], model: str = config.MODEL, base: Settings | None = None,
         frame=None, stages=None, chip_stages=None, verbose: bool = False,
         fitted: dict | None = None) -> dict:
    """Run the coordinate descent and return the report (also written to config.TUNING_PATH).

    `fitted` lets a caller pass in {season: predictor} instead of training them here, which is
    how several stages can be run as separate processes against identical models.
    """
    frame = build_training_frame(load_matches()) if frame is None else frame
    base = base or Settings(model=model)
    stages = STAGES if stages is None else stages
    chip_stages = CHIP_STAGES if chip_stages is None else chip_stages

    if fitted is None:
        print(f"Training one {model} model per season ({', '.join(seasons)})...")
        fitted = {s: predictors(frame, s, [model], quiet=True)[model] for s in seasons}
    tables = {s: prices.fit(frame, before_season=s) for s in seasons}

    chosen: dict = {}
    trials: list[dict] = []
    for name, grid in [(n, g) for n, g in stages] + [(n, g) for n, g in chip_stages]:
        chips_on = any(k in ("3xc", "bboost", "freehit", "wildcard") for k in grid)
        candidates = _combos(grid)
        print(f"\n--- {name} ({len(candidates)} candidates x {len(seasons)} seasons) ---")
        scored = []
        for choice in candidates:
            settings = replace(base, **{k: v for k, v in chosen.items() if k in asdict(base)})
            settings.thresholds = {**base.thresholds,
                                   **{k: v for k, v in chosen.items() if k in base.thresholds}}
            row = _score(settings, choice, seasons, frame, fitted, tables, chips_on, verbose)
            row["stage"] = name
            scored.append(row)
            trials.append(row)
            values = ", ".join(f"{k}={v}" for k, v in choice.items())
            print(f"  {values:40s} {row['mean_points']:7.1f} points "
                  f"({'  '.join(f'{s}: {p:.0f}' for s, p in row['points_per_season'].items())})")
        best = max(scored, key=lambda r: r["mean_points"])
        chosen.update({k: v for k, v in best.items() if k in grid})
        print(f"  -> {', '.join(f'{k}={best[k]}' for k in grid)} "
              f"({best['mean_points']:.1f} points a season)")

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seasons": seasons,
        "model": model,
        "chosen": chosen,
        "trials": trials,
        "config": config_block(chosen),
    }
    config.TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.TUNING_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def config_block(chosen: dict) -> list[str]:
    """The config.py lines the tuning run recommends."""
    return [f"{CONFIG_NAMES[k]} = {v!r}" for k, v in chosen.items() if k in CONFIG_NAMES]


def load_report(path=None) -> dict | None:
    path = path or config.TUNING_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarise(report: dict) -> str:
    """The tuning run as a table of stages, for the end of `xpfpl tune` and for the dashboard."""
    trials = pd.DataFrame(report["trials"])
    lines = [f"Tuned on {', '.join(report['seasons'])} with the {report['model']} model "
             f"({len(trials)} backtested seasons of candidates)."]
    for stage, rows in trials.groupby("stage", sort=False):
        best = rows.loc[rows["mean_points"].idxmax()]
        parameters = [c for c in rows.columns if c in CONFIG_NAMES and rows[c].notna().any()]
        spread = rows["mean_points"].max() - rows["mean_points"].min()
        lines.append(f"  {stage:24s} {', '.join(f'{k}={best[k]}' for k in parameters):32s} "
                     f"{best['mean_points']:7.1f} points  (spread {spread:.1f})")
    lines.append("\nPaste into src/xpfpl/config.py:")
    lines += [f"  {line}" for line in report["config"]]
    return "\n".join(lines)
