"""
Carrying rating uncertainty into the simulation.

Simulating with point estimates answers the wrong question. A futures price is
a distribution over a distribution: the outcome of each game is uncertain, and
so is the strength that generates those outcomes. Holding strength fixed makes
a team's games independent of one another, which understates how far a season
can drift from its central case and systematically overprices favourites.
"""

import numpy as np
import pandas as pd


def sample_ratings(
    result,
    n_sims: int,
    rng: np.random.Generator,
    drift_sd: float = 0.0,
) -> pd.DataFrame:
    """
    One rating vector per simulation, drawn from the fit's posterior.

    drift_sd widens the draw for strength changing between the as-of date and
    the day a contract resolves: trades, injuries, rest, shortened playoff
    rotations. It is added independently per team, which ignores league-wide
    shifts, but a shift common to everyone cancels out of every rating
    difference and so affects no contract in scope.
    """
    teams = list(result.rating_cov.index)
    mean = result.ratings.reindex(teams).to_numpy()
    cov = result.rating_cov.to_numpy().copy()

    if drift_sd > 0.0:
        cov = cov + np.eye(len(teams)) * drift_sd ** 2

    draws = rng.multivariate_normal(mean, cov, size=n_sims)

    # Only rating differences enter any probability, so re-centring each draw
    # keeps the scale comparable with the point estimates without changing
    # anything the model reads.
    draws -= draws.mean(axis=1, keepdims=True)

    return pd.DataFrame(draws, columns=teams)

def strength_paths(
    result,
    n_sims: int,
    n_steps: int,
    rng: np.random.Generator,
    daily_vol: float,
    step_days: float = 7.0,
) -> tuple[np.ndarray, list[str]]:
    """
    A strength trajectory per team per simulation, not a single rating.

    Two different things are uncertain and both are drawn here. Where a team
    stands today is uncertain because it was measured from a finite number of
    games, and that error is fixed for the rest of the season: it shifts the
    whole trajectory up or down. Where a team will stand in June is uncertain
    for a second reason, that strength itself moves, and that part accumulates:
    a fortnight from now it is negligible, five months from now it dominates.

    Modelling the second part as a random walk is what makes the horizon fall
    out of the dates rather than being chosen. A win total resolving in April
    reads an early point on the path and a title resolving in June reads a late
    one, from the same draw.

    Returns an array shaped (n_sims, n_steps, n_teams) and the team order.
    """
    base = sample_ratings(result, n_sims, rng, drift_sd=0.0)
    teams = list(base.columns)

    steps = rng.standard_normal((n_sims, n_steps, len(teams)))
    steps *= daily_vol * np.sqrt(step_days)
    steps[:, 0, :] = 0.0  # nothing has drifted yet at the as-of date

    paths = base.to_numpy()[:, None, :] + np.cumsum(steps, axis=1)

    # Only differences matter, so each step is re-centred across the league.
    paths -= paths.mean(axis=2, keepdims=True)

    return paths, teams