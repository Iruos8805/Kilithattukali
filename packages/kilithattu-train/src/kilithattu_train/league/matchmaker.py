"""Prioritised Fictitious Self-Play (PFSP) matchmaker for league training (§6.5)."""

from __future__ import annotations

import numpy as np

from kilithattu_train.league.pool import SnapshotPool


class PFSPMatchmaker:
    """Prioritised Fictitious Self-Play matchmaker.

    Samples opponent j with probability proportional to (1 - p_hat_j)^2,
    focussing training where the learner is weakest.
    """

    def __init__(self, pool: SnapshotPool, rng: np.random.Generator | None = None) -> None:
        self.pool = pool
        self.rng = rng or np.random.default_rng()

    def sample_pfsp(self, learner_id: str, candidate_ids: list[str]) -> str:
        """Sample an opponent from candidate_ids using PFSP weighting: P(j) ~ (1 - p_j)^2."""
        if not candidate_ids:
            raise ValueError("Candidate opponent pool is empty")

        if len(candidate_ids) == 1:
            return candidate_ids[0]

        weights = []
        for opp_id in candidate_ids:
            p_win = self.pool.get_win_rate(learner_id, opp_id)
            # Weight is quadratic in loss rate (1 - p_win)^2 with small epsilon for exploration
            w = ((1.0 - p_win) ** 2) + 0.01
            weights.append(w)

        weights = np.array(weights, dtype=np.float64)
        probs = weights / np.sum(weights)
        return self.rng.choice(candidate_ids, p=probs)

    def sample_opponent_for_main(
        self,
        learner_id: str,
        opp_side: str,
        current_opp_main_id: str | None,
        main_exploiter_ids: list[str] | None = None,
    ) -> str:
        """Sample opponent for Main learner:

        - 35% current main of other side
        - 50% PFSP over all snapshots
        - 15% current main exploiters (or fallback to PFSP if none)
        """
        all_snapshots = self.pool.list_ids(side=opp_side)
        if not all_snapshots:
            if current_opp_main_id:
                return current_opp_main_id
            raise ValueError(f"No opponents found for side {opp_side}")

        roll = self.rng.random()
        has_main_exploiter = bool(main_exploiter_ids)

        if current_opp_main_id and roll < 0.35:
            return current_opp_main_id
        elif has_main_exploiter and roll < (0.35 + 0.15):
            return self.rng.choice(main_exploiter_ids)
        else:
            return self.sample_pfsp(learner_id, all_snapshots)

    def sample_opponent_for_main_exploiter(self, current_opp_main_id: str) -> str:
        """Main exploiter trains exclusively against the current main of the opposite side."""
        return current_opp_main_id

    def sample_opponent_for_league_exploiter(self, learner_id: str, opp_side: str) -> str:
        """League exploiter trains against the whole pool using PFSP."""
        all_snapshots = self.pool.list_ids(side=opp_side)
        if not all_snapshots:
            raise ValueError(f"No opponents found for side {opp_side}")
        return self.sample_pfsp(learner_id, all_snapshots)
