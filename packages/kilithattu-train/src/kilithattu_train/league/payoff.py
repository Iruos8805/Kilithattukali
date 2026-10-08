"""Payoff matrix and win-condition tracking for league training (§6.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchupRecord:
    attack_id: str
    defence_id: str
    scores: list[int] = field(default_factory=list)  # raw attack points
    attack_wins: int = 0
    total_innings: int = 0


class LeaguePayoff:
    """Tracks payoffs, running median reference score mu_ref, and builds payoff matrices."""

    def __init__(self, initial_mu_ref: float = 9.0) -> None:
        self.mu_ref = initial_mu_ref
        self.main_vs_main_scores: list[int] = []
        self.records: dict[tuple[str, str], MatchupRecord] = {}

    def update_mu_ref(self, new_main_scores: list[int], max_window: int = 500) -> float:
        """Update mu_ref as the running median score of main-vs-main innings (§6.5)."""
        self.main_vs_main_scores.extend(new_main_scores)
        if len(self.main_vs_main_scores) > max_window:
            self.main_vs_main_scores = self.main_vs_main_scores[-max_window:]
        if self.main_vs_main_scores:
            self.mu_ref = float(np.median(self.main_vs_main_scores))
        return self.mu_ref

    def record_innings(self, attack_id: str, defence_id: str, score: int) -> bool:
        """Record an innings result.

        The attack wins if score > mu_ref; otherwise defence wins.
        Returns True if attack won, False if defence won.
        """
        key = (attack_id, defence_id)
        if key not in self.records:
            self.records[key] = MatchupRecord(attack_id=attack_id, defence_id=defence_id)

        rec = self.records[key]
        rec.scores.append(score)
        rec.total_innings += 1
        attack_won = score > self.mu_ref
        if attack_won:
            rec.attack_wins += 1
        return attack_won

    def build_payoff_matrix(self, attack_ids: list[str], defence_ids: list[str]) -> np.ndarray:
        """Return empirical payoff matrix M of shape [len(attack_ids), len(defence_ids)].

        Entries represent the attack win rate in [0, 1].
        """
        matrix = np.full((len(attack_ids), len(defence_ids)), 0.5, dtype=np.float64)
        for i, a_id in enumerate(attack_ids):
            for j, d_id in enumerate(defence_ids):
                rec = self.records.get((a_id, d_id))
                if rec and rec.total_innings > 0:
                    matrix[i, j] = (rec.attack_wins + 1.0) / (rec.total_innings + 2.0)
        return matrix

    @staticmethod
    def solve_nash_fictitious_play(
        payoff_matrix: np.ndarray, iterations: int = 1000
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute approximate Nash equilibrium strategies for zero-sum payoff matrix M in [0, 1].

        Returns: (attack_strategy, defence_strategy, game_value)
        """
        n_a, n_d = payoff_matrix.shape
        if n_a == 0 or n_d == 0:
            return np.zeros(n_a), np.zeros(n_d), 0.5

        a_counts = np.zeros(n_a)
        d_counts = np.zeros(n_d)

        # Start with uniform prior
        a_strat = np.ones(n_a) / n_a
        d_strat = np.ones(n_d) / n_d

        for _ in range(iterations):
            # Attack picks best response to current d_strat (maximize payoff)
            expected_payoffs_a = payoff_matrix @ d_strat
            best_a = int(np.argmax(expected_payoffs_a))
            a_counts[best_a] += 1
            a_strat = a_counts / np.sum(a_counts)

            # Defence picks best response to current a_strat (minimize attack payoff)
            expected_payoffs_d = a_strat @ payoff_matrix
            best_d = int(np.argmin(expected_payoffs_d))
            d_counts[best_d] += 1
            d_strat = d_counts / np.sum(d_counts)

        game_value = float(a_strat @ payoff_matrix @ d_strat)
        return a_strat, d_strat, game_value
