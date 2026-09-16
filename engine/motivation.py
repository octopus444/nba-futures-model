"""
Whether a team still had something to play for when a game was played.

Late-season games are not alike. A team that has locked a top-six seed rests
its starters; a team that can no longer reach the play-in field plays its bench
and its rookies. Those margins measure a lineup that will not take the floor in
the postseason. A team still fighting for a seed in the final week is doing the
opposite, and its April games are the most informative it has played all year.

Discounting by the calendar treats both cases alike and cancels one effect
against the other. This module dates the discount by the standings instead.
"""

import numpy as np
import pandas as pd


def decided_flags(season_games: pd.DataFrame, config) -> pd.DataFrame:
    """
    Per game, whether each side's playoff fate was already settled.

    Settled means one of two things, both decided by arithmetic on the
    standings as they stood the morning of the game: no rival can still push
    the team out of the direct playoff seeds, or enough rivals are already
    beyond its reach that the play-in field is closed to it.

    Ties are ignored, so a team is called settled slightly later than the
    league would formally announce it. That is the right direction to err: a
    team that has only just clinched has not started resting anyone yet.
    """
    games = season_games.sort_values("date").reset_index(drop=True)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    position = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    date_code, _ = pd.factorize(games["date"], sort=True)
    n_dates = int(date_code.max()) + 1

    won = np.zeros((n_dates, n_teams))
    played = np.zeros((n_dates, n_teams))
    for row, game in enumerate(games.itertuples()):
        day = date_code[row]
        home, away = position[game.home_team], position[game.away_team]
        played[day, home] += 1
        played[day, away] += 1
        won[day, home if game.home_win else away] += 1

    # Standings as they stood before that day's games were played.
    wins_before = np.vstack([np.zeros(n_teams), np.cumsum(won, axis=0)[:-1]])
    played_before = np.vstack([np.zeros(n_teams), np.cumsum(played, axis=0)[:-1]])
    season_total = played.sum(axis=0)

    missing = [t for t in teams if t not in config.conference]
    if missing:
        raise ValueError(f"teams absent from the league config: {missing}")
    conference = np.array([config.conference[t] for t in teams])
    settled = np.zeros((n_dates, n_teams), dtype=bool)

    for day in range(n_dates):
        wins = wins_before[day]
        ceiling = wins + (season_total - played_before[day])

        for conf in np.unique(conference):
            members = conference == conf
            w, c = wins[members], ceiling[members]

            # Rivals whose best case still finishes above this team.
            can_pass = c[None, :] > w[:, None]
            np.fill_diagonal(can_pass, False)
            clinched = can_pass.sum(axis=1) <= config.direct_playoff_seeds - 1

            # Rivals already beyond this team's best case.
            out_of_reach = w[None, :] > c[:, None]
            np.fill_diagonal(out_of_reach, False)
            eliminated = out_of_reach.sum(axis=1) >= config.play_in_seeds

            settled[day, members] = clinched | eliminated

    home_pos = np.array([position[t] for t in games["home_team"]])
    away_pos = np.array([position[t] for t in games["away_team"]])

    return pd.DataFrame({
        "game_id": games["game_id"].to_numpy(),
        "home_settled": settled[date_code, home_pos],
        "away_settled": settled[date_code, away_pos],
    })