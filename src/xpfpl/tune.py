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
gaps between candidates are what matters, not the absolute totals. `confirm_seasons` answers
that: the winner, today's config and the pre-tuning settings are replayed on seasons the search
never saw.

A single replay is chaotic: 10% noise on xP moves a season by ~84 points (robustness.py), more
than most gaps between candidates. So each candidate is scored on `replays` runs per season -
the clean forecast plus `replays - 1` with keyed noise (robustness.Noisy), the same noise for
every candidate, which makes the comparison paired - and the runs can go to `workers` processes.
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor
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


CHIPS = ("3xc", "bboost", "freehit", "wildcard")
NOISE_SD = 0.1
_WORKER: dict = {}          # per process: frame, predictors and price tables


def _trial(base: Settings, choice: dict, season: str, chips_on: bool) -> Settings:
    """`base` with one candidate's settings (and chip thresholds) applied, for one season."""
    thresholds = {k: v for k, v in choice.items() if k in CHIPS}
    settings = replace(base, season=season, chips=chips_on,
                       **{k: v for k, v in choice.items() if k not in CHIPS})
    settings.thresholds = {**base.thresholds, **thresholds}
    return settings


def _predictor(fitted, season: str, replay: int, noise_sd: float):
    """Replay 0 is the clean forecast; replay r > 0 adds keyed noise number r."""
    from xpfpl.robustness import Noisy

    base = fitted[season]
    if replay == 0 or base is None:
        return base
    return Noisy(base, noise_sd, seed=replay, keyed=True)


def _init_worker(seasons: list[str], model: str) -> None:
    """Each worker builds the features and loads the cached per-season models once."""
    import torch

    from xpfpl import robustness

    torch.set_num_threads(1)            # one core per worker: the pool is the parallelism
    frame = build_training_frame(load_matches())
    _WORKER.update(frame=frame,
                   fitted={s: robustness.fitted(None, s, name=model) for s in seasons},
                   tables={s: prices.fit(frame, before_season=s) for s in seasons})


def _replay_in_worker(settings: Settings, replay: int, noise_sd: float) -> float:
    w = _WORKER
    predictor = _predictor(w["fitted"], settings.season, replay, noise_sd)
    return run(settings, frame=w["frame"], predictor=predictor,
               price_table=w["tables"][settings.season], verbose=False).summary["points"]


class _Runner:
    """Runs backtests in this process or in a pool of worker processes."""

    def __init__(self, frame, fitted, tables, seasons, model, workers: int, noise_sd: float,
                 verbose: bool = False):
        self.frame, self.fitted, self.tables = frame, fitted, tables
        self.noise_sd, self.verbose = noise_sd, verbose
        self.pool = (ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(seasons, model))
                     if workers > 1 else None)

    def points(self, jobs: list[tuple[Settings, int]]) -> list[float]:
        if self.pool is not None:
            futures = [self.pool.submit(_replay_in_worker, s, r, self.noise_sd) for s, r in jobs]
            return [f.result() for f in futures]
        return [run(s, frame=self.frame, predictor=_predictor(self.fitted, s.season, r, self.noise_sd),
                    price_table=self.tables[s.season], verbose=self.verbose).summary["points"]
                for s, r in jobs]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)   # after an error, drop the queued replays


def _summarise_candidate(choice: dict, points: dict[str, list[float]]) -> dict:
    """Mean over seasons of the mean over replays, and its standard error from the replay spread."""
    means = {s: sum(v) / len(v) for s, v in points.items()}
    n = len(points)
    var = [pd.Series(v).var(ddof=1) / len(v) for v in points.values() if len(v) > 1]
    se = math.sqrt(sum(var)) / n if var else None
    return {**choice, "points_per_season": means, "replays": points,
            "mean_points": sum(means.values()) / n, "worst_season": min(means.values()), "se": se}


def _score_all(runner: _Runner, base: Settings, candidates: list[dict], seasons: list[str],
               chips_on: bool, replays: int) -> list[dict]:
    """Every candidate on every season and replay, submitted together so a pool stays busy."""
    jobs, keys = [], []
    for i, choice in enumerate(candidates):
        for season in seasons:
            for r in range(replays):
                jobs.append((_trial(base, choice, season, chips_on), r))
                keys.append((i, season))
    results = runner.points(jobs)
    points: list[dict[str, list[float]]] = [{s: [] for s in seasons} for _ in candidates]
    for (i, season), value in zip(keys, results):
        points[i][season].append(float(value))
    return [_summarise_candidate(choice, p) for choice, p in zip(candidates, points)]


def _fitted_models(frame, seasons: list[str], model: str, workers: int) -> dict:
    """One model per season. With workers, fit through robustness.py's cache so they can load it."""
    if workers > 1:
        from xpfpl import robustness
        return {s: robustness.fitted(frame, s, name=model) for s in seasons}
    return {s: predictors(frame, s, [model], quiet=True)[model] for s in seasons}


def tune(seasons: list[str], model: str = config.MODEL, base: Settings | None = None,
         frame=None, stages=None, chip_stages=None, verbose: bool = False,
         fitted: dict | None = None, replays: int = 1, workers: int = 1,
         noise_sd: float = NOISE_SD, confirm_seasons: list[str] | None = None,
         chosen: dict | None = None, trials: list[dict] | None = None) -> dict:
    """Run the coordinate descent and return the report (also written to config.TUNING_PATH).

    `fitted` lets a caller pass in {season: predictor} instead of training them here (in-process
    runs only). `replays` runs per season per candidate (the clean forecast plus keyed-noise
    copies) are averaged; `workers` > 1 spreads them over processes, which build their own
    features and load the models from robustness.py's cache. `confirm_seasons`: afterwards,
    replay the winner, today's config and the pre-tuning settings on these unseen seasons.
    `chosen`/`trials` resume an interrupted run: the winners and trials of the stages it
    finished (pass only the stages still to run).
    """
    from xpfpl.robustness import PRE_TUNING

    frame = build_training_frame(load_matches()) if frame is None else frame
    base = base or Settings(model=model)
    stages = STAGES if stages is None else stages
    chip_stages = CHIP_STAGES if chip_stages is None else chip_stages
    every = list(dict.fromkeys(seasons + (confirm_seasons or [])))

    if fitted is None:
        print(f"Training (or loading) one {model} model per season ({', '.join(every)})...")
        fitted = _fitted_models(frame, every, model, workers)
    tables = {s: prices.fit(frame, before_season=s) for s in every} if workers <= 1 else {}
    runner = _Runner(frame, fitted, tables, every, model, workers, noise_sd, verbose)

    chosen = dict(chosen or {})
    trials = list(trials or [])
    confirmation: list[dict] = []
    try:
        for name, grid in [(n, g) for n, g in stages] + [(n, g) for n, g in chip_stages]:
            chips_on = any(k in CHIPS for k in grid)
            candidates = _combos(grid)
            print(f"\n--- {name} ({len(candidates)} candidates x {len(seasons)} seasons x "
                  f"{replays} replays) ---", flush=True)
            settings = replace(base, **{k: v for k, v in chosen.items() if k in asdict(base)})
            settings.thresholds = {**base.thresholds, **{k: v for k, v in chosen.items() if k in CHIPS}}
            scored = _score_all(runner, settings, candidates, seasons, chips_on, replays)
            for row in scored:
                row["stage"] = name
                trials.append(row)
                values = ", ".join(f"{k}={row[k]}" for k in grid)
                se = f" ± {row['se']:.0f}" if row["se"] is not None else ""
                print(f"  {values:40s} {row['mean_points']:7.1f}{se} points "
                      f"({'  '.join(f'{s}: {p:.0f}' for s, p in row['points_per_season'].items())})",
                      flush=True)
            best = max(scored, key=lambda r: r["mean_points"])
            chosen.update({k: best[k] for k in grid})
            print(f"  -> {', '.join(f'{k}={best[k]}' for k in grid)} "
                  f"({best['mean_points']:.1f} points a season)", flush=True)

        if confirm_seasons:
            named = {"tuned": chosen, "current config": {}, "pre-tuning": PRE_TUNING}
            chips_on = bool(chip_stages)
            print(f"\n--- confirmation on unseen seasons ({', '.join(confirm_seasons)}; "
                  f"chips {'on' if chips_on else 'off'}) ---", flush=True)
            scored = _score_all(runner, base, list(named.values()), confirm_seasons, chips_on, replays)
            for label, row in zip(named, scored):
                confirmation.append({"label": label, **row})
                print(f"  {label:16s} {row['mean_points']:7.1f} ± {row['se'] or 0:.0f} points "
                      f"({'  '.join(f'{s}: {p:.0f}' for s, p in row['points_per_season'].items())})",
                      flush=True)
    finally:
        runner.close()

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seasons": seasons,
        "model": model,
        "replays": replays,
        "noise_sd": noise_sd,
        "chosen": chosen,
        "trials": trials,
        "confirm_seasons": confirm_seasons or [],
        "confirmation": confirmation,
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
