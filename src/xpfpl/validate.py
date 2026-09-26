"""How good is the model? Scores it on a held-out season and writes a report the dashboard reads.

`xpfpl train` already predicts the validation season with a model that never saw it, so the
report is built from those predictions and saved to models/validation.json. Everything here is
plain numbers in, dict out: no plotting, no Streamlit, so the CLI and the app can share it.

The report answers five questions, in the order a manager would ask them:
  1. How far off is a typical prediction?          -> headline MAE/RMSE/R², vs. the no-ML baseline
  2. Is it equally wrong everywhere?               -> by position, by gameweek, by return group
                                                      (zeros/blanks/tickers/haulers)
  3. Does "6 xP" really mean six points?           -> calibration curve
  4. Does it pick the right players?               -> ranking (Spearman per GW), deciles, captain test
  5. How good could it possibly be?                -> a "perfect model" ceiling: simulate
                                                      each match from the component model's
                                                      probabilities and score the true expectation
                                                      against those simulated outcomes
  6. Are the Monte Carlo ranges right?             -> of the players given a 20% chance of 10+,
                                                      did 20% get there (simulate.check)
Optionally it also scores forecasts made 2 and 3 gameweeks ahead (`horizons`), since the
optimiser leans on those too.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from xpfpl import config

# Predicted-xP bins for the calibration curve. Above ~8 xP there are too few rows to be useful.
CALIBRATION_BINS = [0, 1, 2, 3, 4, 5, 6, 8, 100]
CALIBRATION_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-8", "8+"]
ERROR_BINS = [-100, -10, -6, -4, -2, -1, 1, 2, 4, 6, 10, 100]
DECILES = 10
REFERENCE = "Baseline"      # the no-ML comparison every report carries
FPL_XP = "FPL xP"           # FPL's own expected points, where recorded (2020-21 on)


# Return groups, by what actually happened. Errors on "zeros" are about minutes;
# errors on "haulers" are about spotting the big scores, which is where rank is won.
RETURN_GROUPS = ["Zeros (didn't play)", "Blanks (1-2 pts)", "Tickers (3-4 pts)", "Haulers (5+ pts)"]


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Share of the variance in points the predictions explain. Even a simulated perfect model
    only reaches ~0.2: most of a week's points are luck nobody can forecast."""
    sst = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else float("nan")


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    """MAE, RMSE, R² and bias (positive = the model over-predicts) for one set of rows."""
    if len(y) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "bias": float("nan")}
    err = p - y
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "r2": r2(y, p), "bias": float(np.mean(err))}


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    """Rank correlation: 1 = the players are in exactly the right order, 0 = no better than random."""
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(pd.Series(p).rank().corr(pd.Series(y).rank()))


def _ranking(df: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray], mask: np.ndarray) -> list[dict]:
    """Spearman correlation between xP and points, within each gameweek.

    Ranking is what the optimiser actually consumes: it only needs the order of the players to
    be right, not the numbers. Scored per gameweek (the decision unit) and then averaged.
    """
    rows = []
    gw = df["gw"].to_numpy()
    for g in np.unique(gw[mask]):
        sub = mask & (gw == g)
        for model, p in preds.items():
            rows.append({"gw": int(g), "model": model, "spearman": _spearman(y[sub], p[sub])})
    return rows


def _headline(df: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray], active: np.ndarray) -> list[dict]:
    rows = []
    everyone = np.ones(len(y), dtype=bool)
    for model, p in preds.items():
        for subset, mask in (("All players", everyone), ("Players getting minutes", active)):
            ranks = pd.DataFrame(_ranking(df, y, {model: p}, mask))
            spearman = float(ranks["spearman"].mean()) if len(ranks) else float("nan")
            rows.append({"model": model, "subset": subset, **_scores(y[mask], p[mask]), "spearman": spearman})
    return rows


def _return_groups(df: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray]) -> list[dict]:
    """Errors split by the outcome, over every row. Needs `minutes`."""
    if "minutes" not in df:
        return []
    played = df["minutes"].to_numpy() > 0
    groups = np.select([~played, y <= 2, y <= 4], RETURN_GROUPS[:3], RETURN_GROUPS[3])
    rows = []
    for group in RETURN_GROUPS:
        mask = groups == group
        for model, p in preds.items():
            rows.append({"group": group, "model": model, **_scores(y[mask], p[mask])})
    return rows


def _horizons(y: np.ndarray, horizons: dict[int, dict[str, np.ndarray]], active: np.ndarray) -> list[dict]:
    """Scores for predictions made k gameweeks before the match, on the same rows."""
    return [{"horizon": int(k), "model": model, **_scores(y[active], p[active])}
            for k, preds in sorted(horizons.items()) for model, p in preds.items()]


def match_tables(frame: pd.DataFrame) -> dict:
    """What the ceiling simulation borrows from history (rows where the player got on):

    - `bonus`: P(0/1/2/3 bonus) by position, goals (0/1/2+), assists (0/1/2+) and clean sheet.
      Bonus follows the returns - a goal usually brings bonus - and simulating it independently
      of them leaves the simulated world far too calm.
    - `residual`: a sample of each position's points the rules don't explain (cards, own goals,
      penalties), drawn as-is.
    """
    from xpfpl import scoring

    played = frame[frame["minutes"].fillna(0) > 0]
    g = played["goals_scored"].clip(upper=2).astype(int).to_numpy()
    a = played["assists"].clip(upper=2).astype(int).to_numpy()
    cs = (played["clean_sheets"].fillna(0) > 0).astype(int).to_numpy()
    pos = played["position"].astype(int).to_numpy()
    b = played["bonus"].clip(0, 3).astype(int).to_numpy()
    counts = np.ones((5, 3, 3, 2, 4))                  # +1 smoothing for rare combinations
    np.add.at(counts, (pos, g, a, cs, b), 1.0)
    rng = np.random.default_rng(0)
    res = scoring.residual(played.assign(played=1.0, played60=(played["minutes"] >= 60).astype(float)))
    residual = {int(k): rng.choice(res[pos == k], size=min(20000, int((pos == k).sum())), replace=False)
                for k in range(1, 5) if (pos == k).any()}
    return {"bonus": counts / counts.sum(axis=-1, keepdims=True), "residual": residual}


def simulate_ceiling(components: pd.DataFrame, mask: np.ndarray | None = None, sims: int = 200,
                     seed: int = 0, tables: dict | None = None) -> dict:
    """How well would a *perfect* model score, if the world worked like our component model says?

    Take the component model's probabilities as the truth, simulate every match `sims` times, and score the true expectation (the perfect prediction)
    against each simulated week. What's left is pure luck no model can remove, so the spread of
    these scores is the ceiling to compare real models against.

    `components` is `components.Predictor.components(frame)`: per-row P(play), P(60),
    expected goals/assists/saves/bonus, P(clean sheet), P(defensive contribution). Clean sheets
    and goals conceded come from one Poisson draw (a clean sheet *is* conceding none), and with
    `tables` (`match_tables(history)`) bonus follows the simulated returns and cards/penalties
    are drawn from history - without them the simulation underplays the real spread of scores.
    """
    from xpfpl import scoring

    rng = np.random.default_rng(seed)
    c = components if mask is None else components[mask]
    n = len(c)
    pos = c["position"].to_numpy().astype(int)
    p_play = np.clip(c["played"].to_numpy(), 1e-6, 1)
    p60 = np.clip(c["played60"].to_numpy(), 0, p_play)
    # The heads predict unconditional expectations; given a player plays, they scale up.
    per_app = {k: c[k].to_numpy() / p_play for k in ("goals_scored", "assists", "bonus", "saves")}
    p_cs60 = np.clip(c["clean_sheets"].to_numpy() / np.maximum(p60, 1e-6), 1e-3, 0.999)
    conceded_rate = -np.log(p_cs60)                    # so that P(concede none) = P(clean sheet)
    goal_pts = np.vectorize(scoring.GOAL_POINTS.get)(pos, 0)
    cs_pts = np.vectorize(scoring.CLEAN_SHEET_POINTS.get)(pos, 0)
    dc = np.clip(c["dc"].to_numpy() * c["dc_era"].to_numpy() / p_play, 0, 1)
    defensive = np.isin(pos, (1, 2))

    outcomes = np.empty((sims, n))
    for s in range(sims):
        played = rng.random(n) < p_play
        full = played & (rng.random(n) < p60 / p_play)
        goals = rng.poisson(per_app["goals_scored"]) * played
        assists = rng.poisson(per_app["assists"]) * played
        conceded = rng.poisson(conceded_rate)
        clean = full & (conceded == 0)
        pts = played + full.astype(float)
        pts += goal_pts * goals + scoring.ASSIST_POINTS * assists + cs_pts * clean
        pts -= defensive * full * (conceded // scoring.CONCEDED_PER_PENALTY)
        pts += (rng.poisson(per_app["saves"]) // scoring.SAVES_PER_POINT) * played * (pos == 1)
        pts += scoring.DC_POINTS * (rng.random(n) < dc) * played
        if tables is None:
            pts += np.minimum(rng.poisson(per_app["bonus"]), 3) * played
        else:
            probs = tables["bonus"][pos, np.minimum(goals, 2), np.minimum(assists, 2), clean.astype(int)]
            bonus = (rng.random((n, 1)) > np.cumsum(probs, axis=1)[:, :3]).sum(axis=1)
            pts += bonus * played
            for k, pool in tables["residual"].items():
                rows = played & (pos == k)
                pts[rows] += rng.choice(pool, size=int(rows.sum()))
        outcomes[s] = pts
    truth = outcomes.mean(axis=0)            # the perfect prediction: the expectation itself
    scores = pd.DataFrame([_scores(o, truth) for o in outcomes])
    return {"sims": sims, "rows": int(n), "outcome_variance": float(outcomes.var(axis=1).mean()),
            **{f"{m}_{q}": float(scores[m].quantile(v)) for m in ("rmse", "mae", "r2")
               for q, v in (("p5", 0.05), ("median", 0.5), ("p95", 0.95))}}


def _by_group(df: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray], key: str,
              mask: np.ndarray) -> list[dict]:
    rows = []
    for value, idx in df[mask].groupby(key, observed=True).indices.items():
        sub = np.flatnonzero(mask)[idx]
        for model, p in preds.items():
            rows.append({key: value if not isinstance(value, np.generic) else value.item(),
                         "model": model, **_scores(y[sub], p[sub])})
    return rows


def _calibration(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Average actual points inside each predicted-xP band. A well-calibrated model sits on y = x."""
    bins = pd.cut(p[mask], CALIBRATION_BINS, labels=CALIBRATION_LABELS, right=False)
    g = pd.DataFrame({"bin": bins, "xp": p[mask], "points": y[mask]}).groupby("bin", observed=True)
    return [{"bin": str(b), "predicted": float(r["xp"].mean()), "actual": float(r["points"].mean()),
             "n": int(len(r)), "se": float(r["points"].std(ddof=1) / max(len(r), 1) ** 0.5)}
            for b, r in g if len(r) >= 20]


def _deciles(y: np.ndarray, preds: dict[str, np.ndarray], mask: np.ndarray) -> list[dict]:
    """Rank rows by prediction and report the actual points in each tenth: does the order hold up?"""
    rows = []
    for model, p in preds.items():
        rank = pd.Series(p[mask]).rank(method="first", pct=True)
        decile = np.ceil(rank * DECILES).astype(int).clip(1, DECILES)
        for d, idx in pd.Series(y[mask]).groupby(decile.to_numpy()):
            rows.append({"decile": int(d), "model": model, "actual": float(idx.mean()),
                         "predicted": float(np.mean(p[mask][decile.to_numpy() == d])), "n": int(len(idx))})
    return rows


def _error_histogram(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> list[dict]:
    err = y[mask] - p[mask]  # positive = the player beat the prediction
    counts = pd.cut(err, ERROR_BINS).value_counts().sort_index()
    labels = ["worse by 10+", "-10 to -6", "-6 to -4", "-4 to -2", "-2 to -1", "within 1",
              "+1 to +2", "+2 to +4", "+4 to +6", "+6 to +10", "better by 10+"]
    total = int(counts.sum()) or 1
    return [{"band": lab, "order": i, "count": int(c), "share": float(c) / total}
            for i, (lab, c) in enumerate(zip(labels, counts))]


def _captain_test(df: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray],
                  mask: np.ndarray) -> list[dict]:
    """Each gameweek, captain the highest-xP player in the league and see what they scored.

    The comparison points are the best captain available with hindsight (perfect) and the
    average points of a player who actually started (a random pick). No ownership or price
    constraints: this is a pure "did it spot the right man" test.
    """
    rows = []
    frame = pd.DataFrame({"gw": df["gw"].to_numpy(), "points": y, "played": mask,
                          **{f"p_{m}": p for m, p in preds.items()}})
    for gw, g in frame.groupby("gw"):
        played = g[g["played"]]
        if len(played) < 20:
            continue
        row = {"gw": int(gw), "best": float(g["points"].max()), "typical": float(played["points"].mean())}
        for model in preds:
            pick = g.loc[g[f"p_{model}"].idxmax()]
            top3 = g.nlargest(3, f"p_{model}")["points"].mean()
            row[model] = float(pick["points"])
            row[f"{model} top 3"] = float(top3)
        rows.append(row)
    return rows


def build_report(val_df: pd.DataFrame, preds: dict[str, np.ndarray], *, target: str,
                 trained_on: str, best_epoch: int | None = None, primary: str | None = None,
                 horizons: dict[int, dict[str, np.ndarray]] | None = None,
                 ceiling: dict | None = None, simulation: dict | None = None) -> dict:
    """Assemble every section of the accuracy report for one held-out season.

    `preds` maps a label to that model's predictions; `primary` names the one being reported on
    (the first by default), which is the one the calibration and error charts describe.
    `horizons` maps k -> predictions made k gameweeks ahead for the same rows (k=1 is `preds`),
    and `ceiling` is `simulate_ceiling(...)` over the active rows, when a component model ran.
    `simulation` is `simulate.check(...)`: whether the Monte Carlo ranges held up on this season.
    """
    primary = primary or next(iter(preds))
    y = val_df[target].to_numpy(dtype=float)
    # "Getting minutes" = played at least once in their previous 5 matches, which is the pool a
    # manager actually chooses from. Bench-warmers are easy to predict (0) and flatter the scores.
    active = (val_df["played_r5"] > 0).to_numpy()
    df = val_df.reset_index(drop=True)

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "season": str(df["season"].iloc[0]) if len(df) else "",
        "primary": primary,
        "reference": REFERENCE if REFERENCE in preds else primary,
        "models": list(preds),
        "trained_on": trained_on,
        "best_epoch": best_epoch,
        "rows": int(len(df)),
        "active_rows": int(active.sum()),
        "headline": _headline(df, y, preds, active),
        "ranking": _ranking(df, y, preds, active),
        "return_groups": _return_groups(df, y, preds),
        "horizons": _horizons(y, {1: preds, **(horizons or {})}, active) if horizons else [],
        "ceiling": ceiling,
        "by_position": [{**r, "position": config.POSITIONS.get(r.pop("position"), "?")}
                        for r in _by_group(df, y, preds, "position", active)],
        "by_gameweek": _by_group(df, y, preds, "gw", active),
        "calibration": _calibration(y, preds[primary], active),
        "deciles": _deciles(y, preds, active),
        "errors": _error_histogram(y, preds[primary], active),
        "captain": _captain_test(df, y, preds, active),
        "simulation": simulation,
    }


def save_report(report: dict, path=None) -> None:
    path = path or config.VALIDATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")


def load_report(path=None) -> dict | None:
    path = path or config.VALIDATION_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_comparison(path=None) -> dict | None:
    """The `xpfpl compare` table: every model scored on the same held-out season."""
    path = path or config.COMPARISON_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarise(report: dict) -> str:
    """The report as a few lines of text, for the end of `xpfpl train`."""
    head = pd.DataFrame(report["headline"])
    lines = [f"Validation on {report['season']} ({report['active_rows']:,} of {report['rows']:,} rows "
             f"are players getting minutes):"]
    for subset in ("All players", "Players getting minutes"):
        part = head[head["subset"] == subset]
        lines.append(f"  {subset}")
        for r in part.itertuples():
            lines.append(f"    {r.model:22s} RMSE {r.rmse:.3f}  MAE {r.mae:.3f}  R² {r.r2:.3f}  "
                         f"rank corr {r.spearman:.3f}")
    ceiling = report.get("ceiling")
    if ceiling:
        lines.append(f"  Perfect-model ceiling    RMSE {ceiling['rmse_median']:.3f}  MAE {ceiling['mae_median']:.3f}  "
                     f"R² {ceiling['r2_median']:.3f}  (median of {ceiling['sims']} simulated seasons)")
    sim = report.get("simulation")
    if sim:
        lines.append(f"  Monte Carlo ranges       {sim['coverage_80']:.1%} of scores inside the 10th-90th percentile "
                     f"band (80% if right; club totals {sim['club_coverage_80']:.1%}); "
                     f"simulated average vs xP off by {sim['mean_gap']:.2f}")
    cap = pd.DataFrame(report["captain"])
    primary, reference = report.get("primary", "MLP"), report.get("reference", REFERENCE)
    if len(cap):
        lines.append(f"  Captain test ({len(cap)} GWs)     {primary} {cap[primary].mean():.1f} pts/GW   "
                     f"{reference} {cap[reference].mean():.1f}   "
                     f"typical starter {cap['typical'].mean():.1f}   perfect {cap['best'].mean():.1f}")
    return "\n".join(lines)
