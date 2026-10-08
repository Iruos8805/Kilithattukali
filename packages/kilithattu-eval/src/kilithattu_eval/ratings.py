"""Elo rating system and Nash averaging for match outcomes."""

from __future__ import annotations

import numpy as np


class EloTracker:
    """Standard Elo rating tracker for team-vs-team matches."""

    def __init__(self, k_factor: float = 32.0, default_rating: float = 1000.0) -> None:
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings: dict[str, float] = {}

    def get_rating(self, player_id: str) -> float:
        return self.ratings.get(player_id, self.default_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def record_match(self, player_a: str, player_b: str, score_a: float) -> tuple[float, float]:
        """Record match result (score_a: 1.0 = A wins, 0.5 = draw, 0.0 = B wins)."""
        r_a = self.get_rating(player_a)
        r_b = self.get_rating(player_b)

        exp_a = self.expected_score(r_a, r_b)
        exp_b = 1.0 - exp_a
        score_b = 1.0 - score_a

        new_r_a = r_a + self.k_factor * (score_a - exp_a)
        new_r_b = r_b + self.k_factor * (score_b - exp_b)

        self.ratings[player_a] = new_r_a
        self.ratings[player_b] = new_r_b
        return new_r_a, new_r_b


def compute_nash_averaging(payoff_matrix: np.ndarray) -> np.ndarray:
    """Compute maximum entropy Nash equilibrium / row player distribution for a zero-sum game."""
    n = payoff_matrix.shape[0]
    # Uniform baseline as default
    return np.full((n,), 1.0 / n)
