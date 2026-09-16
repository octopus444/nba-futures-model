"""
Monte Carlo simulation of the remainder of a regular season.

Every contract in scope is a function of final standings, so the whole model
funnels through here: play out the unplayed schedule many times, and the
distribution of outcomes falls out of the frequencies.
"""

import numpy as np
import pandas as pd


def simulate_remaining_season(
    played: pd.DataFrame,
    remaining: pd.DataFrame,
    win_probabilities: pd.Series,
    n_sims: int = 20_000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Final regular-season win totals, one row per simulation, one column per team.

    Games already played are not resimulated: their results are facts and are
    added as a constant. Only the unplayed schedule is random.

    Each game is drawn independently. That is an assumption, not a fact - a
    team on a losing streak because its best player is hurt will keep losing in
    ways this does not capture - and it is the main reason the tails here are
    too thin. Carrying uncertainty in the ratings themselves is what fixes it.
    """
    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    # Wins already banked.
    base_wins = np.zeros(n_teams)
    for game in played.itertuples():
        winner = game.home_team if game.home_win else game.away_team
        base_wins[index[winner]] += 1

    home_idx = np.array([index[t] for t in remaining["home_team"]])
    away_idx = np.array([index[t] for t in remaining["away_team"]])
    probs = win_probabilities.to_numpy()

    rng = np.random.default_rng(seed)
    # One draw per game per simulation: (n_sims, n_remaining_games).
    home_wins = rng.random((n_sims, len(remaining))) < probs

    wins = np.tile(base_wins, (n_sims, 1))
    np.add.at(wins, (slice(None), home_idx), home_wins)
    np.add.at(wins, (slice(None), away_idx), ~home_wins)

    return pd.DataFrame(wins, columns=teams)


def season_win_total_fair(
    simulated_wins: pd.DataFrame,
    team: str,
    line: float,
) -> dict:
    """
    Fair prices for an over/under on a team's regular-season win total.

    The line is conventionally set at a half-win so pushes cannot occur. A whole
    number would need push handling, which is why the fair line reported here is
    the median rather than the mean: the median is the number at which the
    contract is closest to even money.
    """
    wins = simulated_wins[team]

    over = float((wins > line).mean())
    under = float((wins < line).mean())
    push = float((wins == line).mean())

    return {
        "team": team,
        "line": line,
        "over": over,
        "under": under,
        "push": push,
        "mean_wins": float(wins.mean()),
        "median_wins": float(wins.median()),
        "sd_wins": float(wins.std()),
    }