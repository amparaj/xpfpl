"""FPL's scoring rules, applied to match stats.

The component model (models/components.py) predicts minutes, goals, assists and the rest
separately and then combines them with these rules, so the rules live in one place and can be
tested against the points FPL actually awarded.

Only the parts worth predicting are listed. Cards, own goals, penalty misses and penalty saves
are real but rare and essentially unpredictable from form, so they are left to a residual term:
`residual(df)` is whatever the rules below fail to explain, and the component model learns its
average per player.

Every function takes one row per player-match with the columns FPL reports, plus `played` and
`played60` (0/1 for real matches, probabilities for predictions) and `position`. Feeding it
expected goals/assists/minutes instead of actual ones is exactly how expected points are built.
"""

import numpy as np
import pandas as pd

# points per goal / per clean sheet, by position id (1 GKP, 2 DEF, 3 MID, 4 FWD)
GOAL_POINTS = {1: 10, 2: 6, 3: 5, 4: 4}
CLEAN_SHEET_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_POINTS = 3
SAVES_PER_POINT = 3
CONCEDED_PER_PENALTY = 2                      # GKP/DEF lose 1 point per 2 goals conceded
DC_POINTS = 2                                 # defensive contribution, from 2025-26
DC_THRESHOLD = {1: 99, 2: 10, 3: 12, 4: 12}   # CBIT actions needed (GKPs can't score it)

COMPONENTS = ["played", "played60", "goals_scored", "assists", "clean_sheets",
              "goals_conceded", "saves", "bonus", "dc", "residual"]


def _by_position(position: pd.Series, table: dict, default: float = 0.0) -> np.ndarray:
    return position.map(table).fillna(default).to_numpy(dtype="float64")


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].fillna(0).to_numpy(dtype="float64") if name in df else np.zeros(len(df))


def _conceded_penalty(conceded: np.ndarray) -> np.ndarray:
    """Points docked for goals conceded: floor(c/2) for a real scoreline, c/2 in expectation."""
    if len(conceded) and np.allclose(conceded, np.round(conceded)):
        return np.floor(conceded / CONCEDED_PER_PENALTY)
    return conceded / CONCEDED_PER_PENALTY


def dc_awarded(df: pd.DataFrame) -> np.ndarray:
    """1 where the defensive-contribution threshold was met (0 before 2025-26, and for GKPs).

    With a predicted `dc` column (already a probability) it is passed straight through.
    """
    era = _col(df, "dc_era") if "dc_era" in df else np.ones(len(df))
    if "dc" in df:
        return _col(df, "dc") * era
    return (_col(df, "defensive_contribution") >= _by_position(df["position"], DC_THRESHOLD, 99)) * era


def points_from_stats(df: pd.DataFrame) -> np.ndarray:
    """The points FPL's rules award for the stats in `df`."""
    played, started = _col(df, "played"), _col(df, "played60")
    position = df["position"]

    pts = played + started                                      # 1 for appearing, 2 more for 60+
    pts += _by_position(position, GOAL_POINTS) * _col(df, "goals_scored")
    pts += ASSIST_POINTS * _col(df, "assists")
    pts += _by_position(position, CLEAN_SHEET_POINTS) * _col(df, "clean_sheets")
    pts += _col(df, "saves") / SAVES_PER_POINT * (position == 1).to_numpy()
    pts -= (position.isin((1, 2)).to_numpy() * started
            * _conceded_penalty(_col(df, "goals_conceded")))
    pts += _col(df, "bonus")
    pts += DC_POINTS * dc_awarded(df)
    return pts + _col(df, "residual")


def residual(df: pd.DataFrame) -> np.ndarray:
    """Points the rules above don't explain: cards, own goals, penalties, and FPL's oddities."""
    return df["total_points"].to_numpy(dtype="float64") - points_from_stats(df.drop(columns=["residual"],
                                                                                    errors="ignore"))
