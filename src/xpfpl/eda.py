"""What goes with points, by position: the exploratory figures for the dashboard and the website.

After Ramezani & Dinh (2026, arXiv 2505.02170, Section 7), with one change: they set each
week's stats against the same week's points, which is circular (a goal is in both). Here every
feature is as it stood *before* the match (the training frame's rows), so the figures show
what can be known in advance, the same thing the models see.

- `correlations`: Spearman rank correlation of each feature with the match's points, per
  position, for players getting minutes (played in one of their previous five).
- `spread`: how points per appearance are spread, per position (the raincloud plots): a
  histogram, the quartiles and a sample of single scores.
- `shap`: which features move LightGBM's forecast most, per position: the mean absolute
  TreeSHAP contribution (`pred_contrib=True`, no `shap` package needed) and its direction.

`build` writes all three to models/eda.json (`xpfpl eda`, and every `xpfpl train`).
"""

import json
import re

import numpy as np
import pandas as pd

from xpfpl import config
from xpfpl.features import FEATURES, TARGET

PATH = config.MODELS_DIR / "eda.json"
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SEASONS = 3                    # the most recent seasons, so the figures describe today's game
TOP = 10
SAMPLE = 400                   # single scores per position for the raincloud's dots
SHAP_ROWS = 6000               # rows per position for the SHAP averages
SKIP = {"pos_1", "pos_2", "pos_3", "pos_4", "xg_era", "dc_era", "mkt_known"}   # flags, not measures

STATS = {
    "total_points": "points", "goals_scored": "goals", "assists": "assists", "minutes": "minutes",
    "expected_goals": "xG", "expected_assists": "xA", "expected_goal_involvements": "xGI",
    "expected_goals_conceded": "xG conceded", "clean_sheets": "clean sheets",
    "goals_conceded": "goals conceded", "saves": "saves", "bonus": "bonus", "bps": "BPS",
    "bps_share": "BPS rank in match", "influence": "influence", "creativity": "creativity",
    "threat": "threat", "ict_index": "ICT index", "played": "played", "played60": "played 60+",
    "defensive_contribution": "defensive contributions", "starts": "starts",
    "yellow_cards": "yellow cards", "red_cards": "red cards", "own_goals": "own goals",
    "penalties_saved": "penalties saved", "penalties_missed": "penalties missed",
}
NAMES = {
    "price": "Price", "experience": "Matches of experience", "was_home": "At home",
    "venue_pts": "Points at this venue (10)", "price_rank_team": "Price rank in club",
    "minutes_rank_team": "Minutes rank in club", "transfer_flow": "Net transfers this week",
    "ownership": "Ownership", "fx_gf": "Fixture: goals for (ratings)", "fx_ga": "Fixture: goals against (ratings)",
    "fx_cs": "Fixture: clean-sheet chance (ratings)", "mkt_gf": "Odds: goals for", "mkt_ga": "Odds: goals against",
    "mkt_cs": "Odds: clean-sheet chance", "mkt_win": "Odds: win chance",
}


def _first_upper(text: str) -> str:
    return text if text[1:2].isupper() else text[:1].upper() + text[1:]   # "xG" and "ICT" stay as they are


def label(feature: str) -> str:
    """A feature's name in plain words, e.g. total_points_r5 -> "Points, last 5 (avg)"."""
    if feature in NAMES:
        return NAMES[feature]
    m = re.fullmatch(r"(.+)_r(\d+)", feature)
    if m and m.group(1) in STATS:
        return f"{_first_upper(STATS[m.group(1)])}, last {m.group(2)} (avg)"
    m = re.fullmatch(r"(.+)_p90_(\d+)", feature)
    if m and m.group(1) in STATS:
        return f"{_first_upper(STATS[m.group(1)])} per 90, last {m.group(2)}"
    m = re.fullmatch(r"(team|opp)_(x?g[fa])(38)?", feature)
    if m:
        side = "Club" if m.group(1) == "team" else "Opponent"
        what = {"gf": "goals for", "ga": "goals against", "xgf": "xG for", "xga": "xG against"}[m.group(2)]
        return f"{side} {what}, last {m.group(3) or 10}"
    return feature


def _rows(frame: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(frame["season"].unique())[-SEASONS:]
    return frame[frame["season"].isin(seasons) & (frame["played_r5"] > 0)]


def correlations(frame: pd.DataFrame, top: int = TOP) -> list[dict]:
    """Per position, the `top` features by absolute rank correlation with the match's points."""
    out = []
    features = [f for f in FEATURES if f not in SKIP]
    for pos, name in POSITIONS.items():
        rows = frame[frame["position"] == pos]
        varied = [f for f in features if rows[f].nunique() > 1]          # constant columns have no correlation
        ranks = rows[varied + [TARGET]].rank()
        rho = ranks[varied].corrwith(ranks[TARGET]).dropna()
        best = rho.reindex(rho.abs().sort_values(ascending=False).index).head(top)
        out += [{"position": name, "feature": f, "label": label(f), "rho": round(float(r), 4)}
                for f, r in best.items()]
    return out


def spread(frame: pd.DataFrame, sample: int = SAMPLE, seed: int = 0) -> list[dict]:
    """Per position: points per appearance (minutes > 0) as a histogram, quartiles and a sample."""
    rng = np.random.default_rng(seed)
    out = []
    for pos, name in POSITIONS.items():
        y = frame.loc[(frame["position"] == pos) & (frame["minutes"] > 0), TARGET].to_numpy(dtype="float64")
        if not len(y):
            continue
        values, counts = np.unique(np.clip(y, -3, 20), return_counts=True)
        q = np.percentile(y, [5, 25, 50, 75, 95])
        out.append({"position": name, "n": int(len(y)), "mean": round(float(y.mean()), 3),
                    **{k: float(v) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), q)},
                    "share_2_or_fewer": round(float((y <= 2).mean()), 4),
                    "share_10_plus": round(float((y >= 10).mean()), 4),
                    "histogram": [{"points": int(v), "share": round(float(c / len(y)), 5)}
                                  for v, c in zip(values, counts)],
                    "sample": [int(v) for v in rng.choice(y, size=min(sample, len(y)), replace=False)]})
    return out


def gbm_model():
    """The LightGBM model in use: the ensemble's member, else a trained gbm, else None."""
    from xpfpl import models
    for name in (config.MODEL, "gbm"):
        try:
            predictor = models.load(name)
        except (FileNotFoundError, SystemExit, ImportError, OSError):
            continue
        member = getattr(predictor, "members", {}).get("gbm") if hasattr(predictor, "members") else None
        found = member or (predictor if hasattr(predictor, "booster") else None)
        if found is not None:
            return found
    return None


def shap(frame: pd.DataFrame, gbm, top: int = TOP, rows: int = SHAP_ROWS, seed: int = 0) -> list[dict]:
    """Per position, the `top` features by mean |TreeSHAP contribution| to LightGBM's xP, with
    `direction`: the correlation between the feature's value and its contribution (+ = more of
    it, more points)."""
    out = []
    booster = gbm.booster
    for pos, name in POSITIONS.items():
        sub = frame[frame["position"] == pos]
        if len(sub) > rows:
            sub = sub.sample(rows, random_state=seed)
        x = sub[gbm.features].to_numpy(dtype="float32")
        contrib = booster.predict(x, pred_contrib=True, num_iteration=booster.best_iteration or None)[:, :-1]
        mean_abs = pd.Series(np.abs(contrib).mean(axis=0), index=gbm.features).drop(list(SKIP), errors="ignore")
        for f in mean_abs.sort_values(ascending=False).head(top).index:
            i = gbm.features.index(f)
            v, c = x[:, i], contrib[:, i]
            direction = float(np.corrcoef(v, c)[0, 1]) if v.std() > 0 and c.std() > 0 else 0.0
            out.append({"position": name, "feature": f, "label": label(f),
                        "mean_abs": round(float(mean_abs[f]), 4), "direction": round(direction, 3)})
    return out


def build(frame: pd.DataFrame, path=PATH) -> dict:
    """Every figure's data, from the training frame; written to `path` (models/eda.json)."""
    from datetime import datetime
    rows = _rows(frame)
    gbm = gbm_model()
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "seasons": sorted(rows["season"].unique().tolist()), "rows": int(len(rows)),
              "correlations": correlations(rows), "spread": spread(rows),
              "shap": shap(rows, gbm) if gbm is not None else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load(path=PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
