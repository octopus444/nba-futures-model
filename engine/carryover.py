"""
Season-to-season carryover of team strength.

The prior for a season is last season's rating shrunk toward the league mean.
Both the shrinkage factor and the width of the prior are measured from the
data rather than assumed: regressing each team's rating on its own rating a
season earlier gives the slope, and the residual spread of that regression is
exactly the standard deviation of the prior in points.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Carryover:
    """How much of last season's strength survives the off-season."""

    beta: float       # shrinkage toward the league mean, between 0 and 1
    tau: float        # residual sd in points: the width of the prior
    n_pairs: int      # team-seasons behind the estimate


def fit_carryover(
    end_ratings: dict[int, pd.Series],
    skip_seasons: set[int] | None = None,
) -> Carryover:
    """
    Regress each team's rating on its own rating one season earlier.

    No intercept: the sum-to-zero constraint already centres both sides, so a
    fitted intercept would only absorb noise.

    Teams missing from either season are dropped. Relocations and rebrands
    change the franchise code, and treating a rebranded team as brand new is
    the conservative choice: it gets no prior rather than a wrong one.
    """
    skip_seasons = skip_seasons or set()

    xs, ys = [], []
    for season, previous in end_ratings.items():
        following = end_ratings.get(season + 1)
        if following is None:
            continue
        if season in skip_seasons or season + 1 in skip_seasons:
            continue
        common = previous.index.intersection(following.index)
        xs.append(previous.loc[common].to_numpy())
        ys.append(following.loc[common].to_numpy())

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    beta = float(x @ y / (x @ x))
    residuals = y - beta * x
    tau = float(residuals.std(ddof=1))

    return Carryover(beta=beta, tau=tau, n_pairs=len(x))


def build_prior(
    previous_ratings: pd.Series,
    carryover: Carryover,
    teams: list[str],
) -> pd.Series:
    """
    Prior ratings for the current season, one entry per team.

    A team with no previous rating gets zero, which is the league average. That
    is the honest default for an expansion or rebranded franchise, and it is
    also what the model would converge to after a handful of games anyway.
    """
    prior = pd.Series(0.0, index=teams, dtype=float)
    carried = (carryover.beta * previous_ratings).reindex(teams).dropna()
    prior.update(carried)
    return prior


def prior_weight(margin_sd: float, carryover: Carryover) -> float:
    """
    Weight of the prior pseudo-observation in the least-squares fit.

    A prior row with weight w is equivalent to a Gaussian prior on that team's
    rating with standard deviation margin_sd / w, so the weight follows from the
    width of the prior rather than being tuned. Note that w enters the
    information matrix squared: w = 3 is worth nine games, not three.
    """
    return margin_sd / carryover.tau