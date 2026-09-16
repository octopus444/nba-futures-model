from dataclasses import dataclass
from typing import Callable

import numpy as np

from engine.probabilities import win_probability

EASTERN = {
    "ATL": "Southeast", "BOS": "Atlantic", "BKN": "Atlantic", "CHA": "Southeast",
    "CHI": "Central", "CLE": "Central", "DET": "Central", "IND": "Central",
    "MIA": "Southeast", "MIL": "Central", "NYK": "Atlantic", "ORL": "Southeast",
    "PHI": "Atlantic", "TOR": "Atlantic", "WAS": "Southeast",
}

WESTERN = {
    "DAL": "Southwest", "DEN": "Northwest", "GSW": "Pacific", "HOU": "Southwest",
    "LAC": "Pacific", "LAL": "Pacific", "MEM": "Southwest", "MIN": "Northwest",
    "NOP": "Southwest", "OKC": "Northwest", "PHX": "Pacific", "POR": "Northwest",
    "SAC": "Pacific", "SAS": "Southwest", "UTA": "Northwest",
}

# Franchise codes used before a relocation or rebrand. The dataset spans twenty
# seasons, so these appear in it as ordinary teams. They are mapped to the
# conference of the franchise they became, which is all the engine needs: the
# ratings themselves are keyed on whatever code the feed used that season.
HISTORICAL = {
    "NJN": "East",   # New Jersey Nets, Brooklyn from 2013
    "SEA": "West",   # Seattle SuperSonics, Oklahoma City from 2009
    "NOH": "West",   # New Orleans Hornets, Pelicans from 2014
    "NOK": "West",   # New Orleans/Oklahoma City Hornets, 2006-2007
    "CHH": "East",   # Charlotte Hornets, original franchise
}

CONFERENCE = (
    {team: "East" for team in EASTERN}
    | {team: "West" for team in WESTERN}
    | HISTORICAL
)
DIVISION = EASTERN | WESTERN


def nba_play_in(
    order: list[str],
    ratings: dict,
    home_advantage: float,
    scatter: float,
    rng: np.random.Generator,
    config,
) -> list[str]:
    """
    Resolve the play-in and return the eight-team field in seed order.

    Four teams, three single games, and an asymmetry that is easy to miss: the
    seventh seed gets two chances to qualify and the tenth only one. This is a
    league rule, not machinery, which is why it lives with the league.
    """
    field = list(order[: config.direct_playoff_seeds])
    seventh, eighth, ninth, tenth = order[
        config.direct_playoff_seeds : config.play_in_seeds
    ]

    def single_game(home: str, away: str) -> tuple[str, str]:
        margin = ratings.get(home, 0.0) - ratings.get(away, 0.0) + home_advantage
        if rng.random() < win_probability(margin, scatter):
            return home, away
        return away, home

    seed_seven, loser = single_game(seventh, eighth)
    winner, _ = single_game(ninth, tenth)
    seed_eight, _ = single_game(loser, winner)

    return field + [seed_seven, seed_eight]


@dataclass(frozen=True)
class LeagueConfig:
    """
    Structural facts a simulator needs, plus a small set of calibrated numbers.

    regular_season_games is nominal: lockout and pandemic seasons ran short, and
    two teams play an extra game in years with an in-season tournament final.
    Nothing in the engine should treat it as a constant.
    """

    name: str
    conference: dict
    division: dict
    regular_season_games: int = 82

    # Postseason shape.
    playoff_teams_per_conference: int = 8
    first_round_byes: int = 0
    reseed_between_rounds: bool = False

    # One entry per round: how many wins take the series, and where the home
    # games fall for the better seed. Separate per round because leagues change
    # format as the postseason progresses.
    series_format: tuple = (
        (4, (True, True, False, False, True, False, True)),
    ) * 4

    round_names: tuple = (
        "First Round",
        "Conference Semifinals",
        "Conference Finals",
        "Finals",
    )

    # Qualification beyond finishing high enough. None means the top seeds
    # simply qualify.
    qualify: Callable | None = None
    qualification_first_season: int | None = None
    direct_playoff_seeds: int = 6
    play_in_seeds: int = 10

    # Calibrated in notebooks/calibration.ipynb over 2006-2025, excluding the
    # three seasons that did not run a full 82-game schedule. beta is how much
    # of last season's rating carries into the next one and tau is the spread
    # of what it fails to explain, both from regressing each team's rating on
    # its own rating a season earlier across 417 team-seasons.
    #
    # margin_sd is the noise scale of a single game, taken from the most recent
    # completed season rather than a long median: pace and three-point volume
    # have pushed it up by more than a point over the sample, so an average
    # across twenty years would understate today's.
    #
    # daily_strength_vol is how fast strength moves within a season, in points
    # per square root of a day. Splitting past seasons in half and removing the
    # estimation error from the spread between the halves gives a median of
    # 0.18, but the figure has roughly doubled over the sample and recent
    # seasons sit near 0.25. The higher value is used, and it is also the one
    # that makes the share of final win totals falling inside the model's
    # eighty percent interval come out at exactly eighty percent.
    carryover_beta: float = 0.619
    carryover_tau: float = 3.47
    margin_sd: float = 12.52
    daily_strength_vol: float = 0.25
    margin_cap: float = 25.0
        # Home advantage cannot be estimated before a season starts, so a league
    # value stands in until games are played. Taken from the median of recent
    # full seasons, which sit between 1.5 and 1.6 points and have been falling.
    preseason_home_advantage: float = 1.6


NBA = LeagueConfig(
    name="NBA",
    conference=CONFERENCE,
    division=DIVISION,
    qualify=nba_play_in,
    qualification_first_season=2021,
)