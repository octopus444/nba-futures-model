"""
The NFL, as far as this engine is concerned.

Nothing in engine/ changes to support it. What differs from basketball is
entirely here: a shorter season, a seven-team field with a bye for the top
seed, single-elimination rounds instead of series, and a bracket that reseeds
after every round.

Team lists are placeholders. The point of this file is to show that adding a
league is a config exercise, and the postseason logic is the part that would
otherwise force engine changes.
"""

from leagues.nba import LeagueConfig

AFC = {
    "BAL": "AFC North", "BUF": "AFC East", "CIN": "AFC North", "CLE": "AFC North",
    "DEN": "AFC West", "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South",
    "KC": "AFC West", "LAC": "AFC West", "LV": "AFC West", "MIA": "AFC East",
    "NE": "AFC East", "NYJ": "AFC East", "PIT": "AFC North", "TEN": "AFC South",
}

NFC = {
    "ARI": "NFC West", "ATL": "NFC South", "CAR": "NFC South", "CHI": "NFC North",
    "DAL": "NFC East", "DET": "NFC North", "GB": "NFC North", "LAR": "NFC West",
    "MIN": "NFC North", "NO": "NFC South", "NYG": "NFC East", "PHI": "NFC East",
    "SEA": "NFC West", "SF": "NFC West", "TB": "NFC South", "WAS": "NFC East",
}

CONFERENCE = {t: "AFC" for t in AFC} | {t: "NFC" for t in NFC}
DIVISION = AFC | NFC

NFL = LeagueConfig(
    name="NFL",
    conference=CONFERENCE,
    division=DIVISION,
    regular_season_games=17,

    playoff_teams_per_conference=7,
    first_round_byes=1,
    reseed_between_rounds=True,

    # Every round is a single game at the better seed's stadium.
    series_format=((1, (True,)),) * 4,

    round_names=(
        "Wild Card",
        "Divisional",
        "Conference Championship",
        "Super Bowl",
    ),

    # Seeding is by record alone here, which is wrong: the NFL seeds division
    # winners above wild cards regardless of record, and its tiebreakers run
    # head-to-head, then common games, then conference record. Expressing that
    # needs a seeding hook on the config alongside the qualification hook, which
    # is the one engine change NFL would actually justify.

    # Placeholders until fitted on NFL data. A seventeen-game season leaves far
    # less evidence than eighty-two, so the prior will carry most of the weight
    # for most of the season, which is exactly what the derived weight handles.
    carryover_beta=0.40,
    carryover_tau=4.0,
    margin_sd=13.5,
    daily_strength_vol=0.25,
    margin_cap=28.0,
)