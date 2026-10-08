"""Tournament runner and empirical payoff matrix calculation."""

from __future__ import annotations

import jax
import numpy as np
from chex import PRNGKey
from kilithattu_agents.policy import Policy
from kilithattu_core.config import EnvParams

from kilithattu_eval.matches import play_full_match
from kilithattu_eval.ratings import EloTracker


def run_tournament(
    policies: dict[str, Policy],
    params: EnvParams,
    key: PRNGKey,
    matches_per_pair: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    """Execute a round-robin tournament between named policies and compute Elo ratings."""
    names = list(policies.keys())
    n = len(names)
    payoff_matrix = np.zeros((n, n), dtype=np.float32)
    tracker = EloTracker()

    for i in range(n):
        for j in range(i + 1, n):
            name_a, name_b = names[i], names[j]
            pol_a, pol_b = policies[name_a], policies[name_b]

            a_wins = 0
            b_wins = 0
            for m in range(matches_per_pair):
                k_match, key = jax.random.split(key)
                res = play_full_match(pol_a, pol_b, params, k_match)
                if res.winner == "team_a":
                    a_wins += 1
                    tracker.record_match(name_a, name_b, 1.0)
                elif res.winner == "team_b":
                    b_wins += 1
                    tracker.record_match(name_a, name_b, 0.0)
                else:
                    tracker.record_match(name_a, name_b, 0.5)

                print(
                    f"  [{m + 1}/{matches_per_pair}] {name_a} vs {name_b} -> "
                    f"Result: {res.winner.upper()} ({res.team_a_points} - {res.team_b_points})"
                )

            score_diff = (a_wins - b_wins) / float(matches_per_pair)
            payoff_matrix[i, j] = score_diff
            payoff_matrix[j, i] = -score_diff

    final_ratings = {name: tracker.get_rating(name) for name in names}
    return payoff_matrix, final_ratings
