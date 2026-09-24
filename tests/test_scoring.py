import numpy as np
import pandas as pd
import pytest

from xpfpl import scoring


def row(**kwargs) -> pd.DataFrame:
    base = {"position": 3, "played": 1.0, "played60": 1.0, "goals_scored": 0, "assists": 0,
            "clean_sheets": 0, "goals_conceded": 0, "saves": 0, "bonus": 0,
            "defensive_contribution": 0, "dc_era": 0.0, "total_points": 0}
    return pd.DataFrame([{**base, **kwargs}])


def points(**kwargs) -> float:
    return float(scoring.points_from_stats(row(**kwargs))[0])


def test_appearance_points():
    assert points(played=1.0, played60=0.0) == 1
    assert points(played=1.0, played60=1.0) == 2
    assert points(played=0.0, played60=0.0) == 0


def test_goals_and_clean_sheets_depend_on_position():
    assert points(position=4, goals_scored=1) == 2 + 4          # forward
    assert points(position=3, goals_scored=1) == 2 + 5          # midfielder
    assert points(position=2, goals_scored=1) == 2 + 6          # defender
    assert points(position=2, clean_sheets=1) == 2 + 4
    assert points(position=3, clean_sheets=1) == 2 + 1
    assert points(position=4, clean_sheets=1) == 2              # forwards get nothing


def test_keeper_saves_and_goals_conceded():
    assert points(position=1, saves=3) == 2 + 1                 # a point per three saves
    assert points(position=1, saves=2) == 2 + 2 / 3             # fractions are kept for expectations
    assert points(position=1, goals_conceded=2) == 2 - 1
    assert points(position=2, goals_conceded=3) == 2 - 1        # floor(3/2)
    assert points(position=3, goals_conceded=4) == 2            # midfielders aren't docked
    # A substitute who didn't reach 60 minutes isn't docked either.
    assert points(position=2, played=1.0, played60=0.0, goals_conceded=4) == 1


def test_defensive_contribution_only_counts_from_2025_26():
    assert points(position=2, defensive_contribution=10, dc_era=0.0) == 2
    assert points(position=2, defensive_contribution=10, dc_era=1.0) == 2 + 2
    assert points(position=2, defensive_contribution=9, dc_era=1.0) == 2       # under the threshold
    assert points(position=3, defensive_contribution=10, dc_era=1.0) == 2      # needs 12 for a midfielder
    assert points(position=1, defensive_contribution=20, dc_era=1.0) == 2      # keepers can't score it


def test_expected_inputs_give_expected_points():
    """Feeding probabilities instead of outcomes is how the component model builds an xP."""
    xp = points(position=3, played=0.9, played60=0.75, goals_scored=0.3, assists=0.2,
                clean_sheets=0.25, bonus=0.4)
    assert xp == pytest.approx(0.9 + 0.75 + 5 * 0.3 + 3 * 0.2 + 1 * 0.25 + 0.4)


def test_residual_is_what_the_rules_cannot_explain():
    # A midfielder who scored, played 90 minutes and got a yellow card: 2 + 5 = 7 by the rules,
    # 6 awarded, so the missing point is the card.
    r = row(position=3, goals_scored=1, total_points=6)
    assert float(scoring.residual(r)[0]) == -1.0
    assert np.isclose(float(scoring.points_from_stats(r.assign(residual=-1.0))[0]), 6.0)
