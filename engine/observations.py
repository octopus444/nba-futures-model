"""
Turning raw game results into the inputs a strength model can consume.

Two things happen here, and both are deliberate modelling choices rather than
plumbing: splitting a season at an as-of date, and converting a final score
into a single observation of relative strength.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservationConfig:
    """
    Parameters governing how a final score becomes an observation.

    margin_cap: absolute cap on point differential. A 40-point win does not
        mean twice what a 20-point win means; late in blowouts the starters sit
        and the closing stretch measures bench depth rather than team strength.

    overtime_is_a_draw: if a game went to overtime, regulation ended level, and
        that tie is the informative part. The overtime result turns on five
        extra minutes and is close to noise. The league itself treats overtime
        games as zero point differential in NBA Cup tiebreakers.
    """

    margin_cap: float = 25.0
    overtime_is_a_draw: bool = True


def split_at_asof(
    games: pd.DataFrame,
    season: int,
    asof,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one season's regular season into what has happened and what has not.

    Everything strictly before the as-of date is known. Games on the as-of date
    itself count as unplayed: at the moment a market is priced that morning,
    those results do not exist yet.

    Playoff and play-in games are excluded. The regular season is what gets
    simulated forward; the postseason is generated from the bracket rather than
    replayed from the schedule.
    """
    asof = pd.Timestamp(asof)

    season_games = games[
        (games["season"] == season)
        & (games["season_type"] == "Regular Season")
    ]

    played = season_games[season_games["date"] < asof].copy()
    remaining = season_games[season_games["date"] >= asof].copy()

    return played, remaining


def build_observations(
    played: pd.DataFrame,
    config: ObservationConfig = ObservationConfig(),
) -> pd.DataFrame:
    """
    One row per played game, carrying the signal the strength model reads.

    The observation is always oriented home minus away, so a positive value
    means the home side outscored the away side. Whether that margin belongs to
    strength or to home advantage is the model's problem, not this function's.
    """
    obs = played.copy()

    margin = obs["home_margin"].astype(float)

    # An overtime game is recorded as a draw: regulation ended level, and that
    # tie is what the strength model should read. The overtime period itself is
    # five minutes decided largely by which side wins a jump ball and hits one
    # more shot, so its margin is close to noise and would otherwise be credited
    # to the winner as evidence of strength.
    if config.overtime_is_a_draw and "overtime" in obs.columns:
        went_to_ot = obs["overtime"].fillna(False).astype(bool)
        margin = margin.where(~went_to_ot, 0.0)

    obs["observed_margin"] = np.clip(margin, -config.margin_cap, config.margin_cap)

    # Home advantage does not apply where neither side is at home. The flag is
    # imperfect: it catches games the feed left without a designated host, which
    # covers the Mexico City, Paris and Las Vegas games from 2024-25 onward but
    # not earlier international games, where a nominal host was assigned. Those
    # remain mislabelled as ordinary home games.
    if "neutral_site" in obs.columns:
        obs["home_field"] = (~obs["neutral_site"].astype(bool)).astype(float)
    else:
        obs["home_field"] = 1.0

    columns = [
        "season", "game_id", "date", "home_team", "away_team",
        "home_margin", "observed_margin", "home_field", "home_win",
    ]
    return obs[[c for c in columns if c in obs.columns]].reset_index(drop=True)