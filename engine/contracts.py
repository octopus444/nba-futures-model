"""
Deriving contract prices from simulated seasons.

Every contract in scope is a frequency over simulated outcomes, so this module
is thin by design: the modelling happens upstream, and what is left is counting.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine.playoffs import simulate_bracket
from engine.standings import seed_conference


def simulate_full_seasons(
    played: pd.DataFrame,
    remaining: pd.DataFrame,
    win_probabilities: pd.Series,
    ratings: pd.Series,
    home_advantage: float,
    scatter: float,
    config,
    strength_paths: np.ndarray | None = None,
    path_teams: list[str] | None = None,
    game_step: np.ndarray | None = None,
    round_step: list[int] | None = None,
    teams: list[str] | None = None,
    n_sims: int = 10_000,
    seed: int | None = 0,
) -> dict:
    """
    Simulate regular season and postseason together, returning raw outcomes.

    Regular season and playoffs are run in the same loop rather than separately
    because the bracket depends on the standings: seeding is the channel through
    which a February win affects a June contract.
    """
    # Supplied by the caller at the start of a season, when no games have been
    # played and the team list cannot be read off them.
    if teams is None:
        teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    index = {team: i for i, team in enumerate(teams)}

    base_wins = np.zeros(len(teams))
    for game in played.itertuples():
        winner = game.home_team if game.home_win else game.away_team
        base_wins[index[winner]] += 1

    home_idx = np.array([index[t] for t in remaining["home_team"]])
    away_idx = np.array([index[t] for t in remaining["away_team"]])
    probs = win_probabilities.to_numpy()
    ratings_dict = ratings.to_dict()
    home_field = (
        (~remaining["neutral_site"].astype(bool)).astype(float).to_numpy()
        if "neutral_site" in remaining.columns
        else np.ones(len(remaining))
    )
    if strength_paths is not None:
        # Reorder the paths to match this function's team ordering once, rather
        # than looking teams up by name inside the simulation loop.
        order = [path_teams.index(t) for t in teams]
        paths = strength_paths[:, :, order]
    else:
        paths = None

    round_ratings_static = [ratings_dict] * (len(config.round_names) + 1)
        # Home advantage does not apply at neutral sites, same rule as elsewhere.
    home_field = (
        (~remaining["neutral_site"].astype(bool)).astype(float).to_numpy()
        if "neutral_site" in remaining.columns
        else np.ones(len(remaining))
    )
    if strength_paths is not None:
        # Reorder the paths to match this function's team ordering once, rather
        # than looking teams up by name inside the simulation loop.
        order = [path_teams.index(t) for t in teams]
        paths = strength_paths[:, :, order]
    else:
        paths = None

    round_ratings_static = [ratings_dict] * (len(config.round_names) + 1)

    rng = np.random.default_rng(seed)

    all_wins = np.zeros((n_sims, len(teams)), dtype=np.int16)
    rounds_reached = np.full((n_sims, len(teams)), -1, dtype=np.int8)

    # The season is read off whichever frame has rows: at the opening date every
    # game is still ahead of us.
    season = (played if len(played) else remaining)["season"].iloc[0]
    use_qualification = (
        config.qualification_first_season is None
        or season >= config.qualification_first_season
    )

    for sim in range(n_sims):
        if paths is not None:
            # Each remaining game is played at the strength its date implies,
            # and each playoff round at the strength its round implies. A team
            # that drifts upward is stronger for every game after that point,
            # which is the correlation a single fixed rating throws away.
            path = paths[sim]
            at_game = path[game_step]
            rows = np.arange(len(remaining))
            margins = (
                at_game[rows, home_idx]
                - at_game[rows, away_idx]
                + home_advantage * home_field
            )
            probs = norm.cdf(margins / scatter)
            round_ratings = [
                {team: float(path[step, i]) for i, team in enumerate(teams)}
                for step in round_step
            ]
        else:
            round_ratings = round_ratings_static

        home_wins = rng.random(len(remaining)) < probs

        wins = base_wins.copy()
        np.add.at(wins, home_idx, home_wins)
        np.add.at(wins, away_idx, ~home_wins)
        all_wins[sim] = wins

        order = seed_conference(wins, teams, config.conference, rng)
        reached = simulate_bracket(
            order, round_ratings, {t: wins[i] for i, t in enumerate(teams)},
            home_advantage, scatter, config, rng, use_qualification,
        )
        for team, depth in reached.items():
            rounds_reached[sim, index[team]] = depth

    return {
        "teams": teams,
        "wins": pd.DataFrame(all_wins, columns=teams),
        "rounds": pd.DataFrame(rounds_reached, columns=teams),
        "n_sims": n_sims,
        "round_names": config.round_names,
    }


def contract_prices(results: dict) -> pd.DataFrame:
    """All four contract types, one row per team."""
    rounds = results["rounds"]
    wins = results["wins"]
    n_rounds = len(results["round_names"])

    rows = []
    for team in results["teams"]:
        depth = rounds[team]
        row = {
            "team": team,
            "make_playoffs": float((depth >= 0).mean()),
            "win_championship": float((depth == n_rounds).mean()),
            "mean_wins": float(wins[team].mean()),
            "sd_wins": float(wins[team].std()),
        }
        for i, name in enumerate(results["round_names"][1:], start=1):
            row[f"reach_{name.lower().replace(' ', '_')}"] = float((depth >= i).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("win_championship", ascending=False)