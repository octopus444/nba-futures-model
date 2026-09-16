"""
Converting team strength into game-level win probabilities.

The bridge between ratings and simulation. Ratings live in points; contracts
settle on wins, so the two have to be connected by an assumption about how
much a game's actual margin scatters around its expected margin.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def expected_margin(
    home_team: str,
    away_team: str,
    ratings: pd.Series,
    home_advantage: float,
    home_field: float = 1.0,
) -> float:
    """Expected home-minus-away point differential for a single game."""
    return (
        ratings.get(home_team, 0.0)
        - ratings.get(away_team, 0.0)
        + home_advantage * home_field
    )


def win_probability(
    margin: np.ndarray | float,
    scatter: float,
) -> np.ndarray | float:
    """
    Probability the home side wins, given the expected margin.

    A normal approximation is used rather than a logistic curve. With roughly
    two hundred possessions a game, the central limit theorem does most of the
    work, and the normal has the advantage that its scale parameter is the
    residual spread the ratings fit already produced - there is no extra
    constant to tune.

    Exact ties cannot happen: overtime resolves every NBA game.
    """
    return norm.cdf(np.asarray(margin) / scatter)


def schedule_win_probabilities(
    schedule: pd.DataFrame,
    ratings: pd.Series,
    home_advantage: float,
    scatter: float,
) -> pd.Series:
    """Home win probability for every game in a schedule, vectorised."""
    home_field = (
        (~schedule["neutral_site"].astype(bool)).astype(float)
        if "neutral_site" in schedule.columns
        else 1.0
    )

    margins = (
        ratings.reindex(schedule["home_team"]).fillna(0.0).to_numpy()
        - ratings.reindex(schedule["away_team"]).fillna(0.0).to_numpy()
        + home_advantage * np.asarray(home_field)
    )

    return pd.Series(win_probability(margins, scatter), index=schedule.index)