"""
Turning simulated win totals into a seeded playoff field.

Kept generic: the engine sorts teams within whatever groups the league config
declares, and knows nothing about conferences by name.
"""

import numpy as np
import pandas as pd


def seed_conference(
    wins: np.ndarray,
    teams: list[str],
    conference_of: dict,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Order every conference by wins, breaking ties at random.

    Real NBA tiebreakers run head-to-head first, then division record, then
    conference record, and so on down a long ladder. Resolving them properly
    would mean tracking every simulated result rather than win totals alone,
    which costs far more memory than it buys accuracy: ties are common at the
    margin but the teams involved are near-identical by construction, so which
    one gets the higher seed barely moves any contract price. Random tiebreaks
    are therefore used, and the bias this introduces is discussed in the notes.
    """
    jitter = rng.random(len(teams)) * 1e-6
    ordered = {}

    for conf in sorted(set(conference_of.values())):
        members = [i for i, t in enumerate(teams) if conference_of[t] == conf]
        ranked = sorted(members, key=lambda i: -(wins[i] + jitter[i]))
        ordered[conf] = [teams[i] for i in ranked]

    return ordered