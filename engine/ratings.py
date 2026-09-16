"""
Team strength estimation from observed point differentials.

The model is deliberately the simplest thing that respects schedule strength:
each game's margin is explained by the difference between two team ratings plus
a league-wide home advantage. Averaging raw point differential would be simpler
still, but it credits a team for playing weak opponents.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatingsResult:
    """Ratings in points per game, on a scale where the league average is zero."""

    ratings: pd.Series
    home_advantage: float
    residual_sd: float
    n_games: int
    rating_cov: pd.DataFrame


def fit_ratings(
    observations: pd.DataFrame,
    prior_ratings: pd.Series | None = None,
    prior_weight: float = 0.0,
    weights: np.ndarray | pd.Series | None = None,
    fallback_sd: float = 11.0,
    teams: list[str] | None = None,
) -> RatingsResult:
    """
    Least-squares fit of team ratings to observed margins.

    prior_ratings shrinks the solution toward last season's estimates. Early in
    a season the schedule carries almost no information, and without a prior the
    fit is close to noise. A prior row with weight w is equivalent to a Gaussian
    prior on that team's rating with standard deviation sigma / w, where sigma
    is the noise sd of a single game margin. The weight enters the information
    matrix squared, so w is the square root of the number of equivalent games,
    not the number itself: w = 3 is worth nine games.

    weights lets a game count for more or less than one. An October result was
    produced by a roster a February trade may since have dismantled, and a game
    in the final week may have been played by a team resting its starters.

    Ratings are only identified up to a constant, since the model sees
    differences: the sum-to-zero constraint below pins them down.
    """
    # The team list comes from the league when it is supplied, because at the
    # start of a season there are no games to read it from and the model still
    # has to price. Falling back to the observations keeps every other call
    # unchanged.
    if teams is None:
        teams = sorted(
            set(observations["home_team"]) | set(observations["away_team"])
        )
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    n_obs = len(observations)
    # One column per team, plus one for home advantage, plus one row per team
    # for the sum-to-zero constraint and optional prior.
    design = np.zeros((n_obs + n_teams + 1, n_teams + 1))
    target = np.zeros(n_obs + n_teams + 1)

    for row, game in enumerate(observations.itertuples()):
        design[row, index[game.home_team]] = 1.0
        design[row, index[game.away_team]] = -1.0
        design[row, n_teams] = game.home_field
        target[row] = game.observed_margin

    # Weighted least squares. Scaling a row by the square root of its weight
    # makes the squared residual enter the objective multiplied by the weight
    # itself, which is what "this game counts for less" has to mean. Done once
    # the observation rows are complete, and before the prior rows are added:
    # the prior carries its own weight and must not be rescaled with them.
    if weights is not None:
        row_weights = np.asarray(weights, dtype=float)
        scale = np.sqrt(row_weights)
        design[:n_obs] *= scale[:, None]
        target[:n_obs] *= scale
    else:
        row_weights = np.ones(n_obs)

    # Prior rows: each says "this team's rating should be near its prior value",
    # weighted by prior_weight. With weight zero they contribute nothing.
    if prior_ratings is not None and prior_weight > 0:
        for team in teams:
            row = n_obs + index[team]
            design[row, index[team]] = prior_weight
            target[row] = prior_weight * prior_ratings.get(team, 0.0)

    # Sum-to-zero constraint, weighted heavily so it binds.
    design[n_obs + n_teams, :n_teams] = 100.0
    target[n_obs + n_teams] = 0.0

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)

    # Posterior covariance of the ratings.
    #
    # Stacking prior rows and the sum-to-zero row under the observations turns
    # this fit into a Gaussian posterior, whose covariance is sigma squared
    # times the inverse of X'X, with sigma the noise sd of one game. The
    # pseudo-inverse is used because the constraint row makes the overall level
    # nearly degenerate, and inverting that direction directly is unstable.
    inverse_gram = np.linalg.pinv(design.T @ design)

    ratings = pd.Series(solution[:n_teams], index=teams).sort_values(ascending=False)
    home_advantage = float(solution[n_teams])

    fitted = (
        ratings.reindex(observations["home_team"]).to_numpy()
        - ratings.reindex(observations["away_team"]).to_numpy()
        + home_advantage * observations["home_field"].to_numpy()
    )
    residuals = observations["observed_margin"].to_numpy() - fitted

    degrees_of_freedom = max(row_weights.sum() - (n_teams + 1), 1.0)
    noise_sd = float(
        np.sqrt((row_weights * residuals ** 2).sum() / degrees_of_freedom)
    )

    # With no games played, or almost none, there is nothing to estimate the
    # noise scale from. The spread of a single game's margin is a property of
    # the league rather than of one slice of one season, so a league-level
    # value stands in until the season carries evidence of its own.
    if n_obs < n_teams or not np.isfinite(noise_sd) or noise_sd <= 0:
        noise_sd = fallback_sd

    rating_cov = pd.DataFrame(
        noise_sd ** 2 * inverse_gram[:n_teams, :n_teams],
        index=teams,
        columns=teams,
    )

    return RatingsResult(
        ratings=ratings,
        home_advantage=home_advantage,
        residual_sd=noise_sd,
        n_games=n_obs,
        rating_cov=rating_cov,
    )