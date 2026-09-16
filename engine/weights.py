"""
How much each played game should count toward a team's current strength.

Two effects pull against each other, which is why the weight is not a simple
function of age.
"""

import numpy as np
import pandas as pd


def observation_weights(
    observations: pd.DataFrame,
    asof,
    half_life_days: float | None = None,
    discount_mask: np.ndarray | pd.Series | None = None,
    discount: float = 1.0,
) -> np.ndarray:
    """
    One weight per played game, normalised to average one.

    Recency: an October result was produced by a roster a February trade may
    since have dismantled, and by players in different form. Weight halves
    every half_life_days; passing None turns recency off.

    discount_mask marks games whose margin is suspect for a reason other than
    age, and discount is what those games are worth. The mask is supplied by
    the caller rather than computed here, because what makes a game unreliable
    is a property of the league, not of the arithmetic.

    Normalising to mean one keeps the effective sample size near the number of
    games, so the weight of the prior keeps its meaning across schemes.
    """
    dates = pd.to_datetime(observations["date"])

    if half_life_days is None:
        weights = np.ones(len(observations), dtype=float)
    else:
        age = (pd.Timestamp(asof) - dates).dt.days.to_numpy().astype(float)
        weights = 0.5 ** (age / half_life_days)

    if discount_mask is not None and discount != 1.0:
        weights = np.where(np.asarray(discount_mask, dtype=bool), weights * discount, weights)

    return weights / weights.mean()