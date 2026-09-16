"""
Playing out a postseason bracket from a seeded field.

Who qualifies, how the bracket is shaped, how long a series runs and where the
home games fall all arrive as configuration. What the engine contributes is the
logic of a series, the recursion through rounds, byes and reseeding, none of
which is specific to one sport.
"""

import numpy as np

from engine.probabilities import win_probability


def simulate_series(
    better_seed: str,
    worse_seed: str,
    ratings: dict,
    home_advantage: float,
    scatter: float,
    wins_needed: int,
    home_pattern: tuple,
    rng: np.random.Generator,
) -> str:
    """
    Play until one side reaches wins_needed, returning the winner.

    A single-elimination round is the same object with wins_needed of one and a
    pattern of length one, so a league that plays one game per round needs no
    separate code path.

    Home court matters here more than anywhere else in the season: the pattern
    hands the better seed the decisive game. Games are drawn independently,
    which ignores the real dynamics of a series - adjustments between games, a
    rotation shortening, an injury - and is the largest simplification in the
    postseason model.
    """
    wins = {better_seed: 0, worse_seed: 0}
    rating_gap = ratings.get(better_seed, 0.0) - ratings.get(worse_seed, 0.0)

    for game in range(len(home_pattern)):
        at_better_seed = home_pattern[game]
        margin = rating_gap + (home_advantage if at_better_seed else -home_advantage)
        if rng.random() < win_probability(margin, scatter):
            wins[better_seed] += 1
        else:
            wins[worse_seed] += 1

        if wins[better_seed] == wins_needed:
            return better_seed
        if wins[worse_seed] == wins_needed:
            return worse_seed

    raise RuntimeError("Series ended without a winner")


def simulate_bracket(
    conference_order: dict[str, list[str]],
    ratings_by_round: list[dict],
    wins_by_team: dict,
    home_advantage: float,
    scatter: float,
    config,
    rng: np.random.Generator,
    use_qualification: bool = True,
) -> dict:
    """
    Run each group's field through to a champion, then the two survivors.

    Returns, for every team that reached the postseason, the deepest round it
    got to; zero means it qualified and lost immediately, and the index matches
    config.round_names.

    Three pieces of league variation are handled generically here. Qualification
    is delegated to the league, because a play-in tournament is a rule rather
    than machinery. Byes let a top seed skip a round. Reseeding decides whether
    survivors are re-paired by seed each round or the bracket is fixed from the
    start.
    """
    if config.first_round_byes and not config.reseed_between_rounds:
        raise ValueError(
            "byes require reseeding: a fixed bracket has no defined slot for a "
            "team that did not play the first round"
        )
    if len(conference_order) != 2:
        raise ValueError("the final assumes exactly two groups feed into it")

    reached = {}
    finalists = []
    conference_rounds = len(config.round_names) - 1

    for order in conference_order.values():
        if use_qualification and config.qualify is not None:
            field = config.qualify(
                order, ratings_by_round[0], home_advantage, scatter, rng, config
            )
        else:
            field = list(order[: config.playoff_teams_per_conference])

        seed_of = {team: i for i, team in enumerate(field)}
        alive = list(field)
        for team in alive:
            reached[team] = 0

        round_index = 0
        while len(alive) > 1 and round_index < conference_rounds:
            if config.reseed_between_rounds:
                alive.sort(key=lambda team: seed_of[team])

            byes = config.first_round_byes if round_index == 0 else 0
            sitting, playing = alive[:byes], alive[byes:]

            wins_needed, pattern = config.series_format[round_index]
            ratings_now = ratings_by_round[round_index]

            winners = []
            for i in range(len(playing) // 2):
                better = playing[i]
                worse = playing[len(playing) - 1 - i]
                winners.append(
                    simulate_series(
                        better, worse, ratings_now, home_advantage, scatter,
                        wins_needed, pattern, rng,
                    )
                )

            # A bye is an advance, so the sitting teams are credited with the
            # round they skipped.
            alive = sitting + winners
            round_index += 1
            for team in alive:
                reached[team] = round_index

        finalists.append(alive[0])

    # Home court in the final goes to the better regular-season record. Ordering
    # the two survivors by wins is what makes that true; taking them in the
    # order the groups happened to be iterated would hand the advantage to
    # whichever group name sorts first.
    finalists.sort(key=lambda team: -wins_by_team.get(team, 0.0))

    wins_needed, pattern = config.series_format[-1]
    champion = simulate_series(
        finalists[0], finalists[1], ratings_by_round[-1], home_advantage,
        scatter, wins_needed, pattern, rng,
    )
    reached[champion] = len(config.round_names)

    return reached