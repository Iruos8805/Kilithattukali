"""League manager orchestrating pool, learners, and matchmaker (§6.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kilithattu_agents.policy import Policy

from kilithattu_train.league.matchmaker import PFSPMatchmaker
from kilithattu_train.league.payoff import LeaguePayoff
from kilithattu_train.league.pool import SnapshotPool


@dataclass
class LearnerInfo:
    id: str
    side: str  # "attack" | "defence"
    role: str  # "main" | "main_exploiter" | "league_exploiter"
    current_opponent_id: str | None = None
    step: int = 0


class LeagueManager:
    """Orchestrates self-play league training according to §6.5."""

    def __init__(self) -> None:
        self.pool = SnapshotPool()
        self.matchmaker = PFSPMatchmaker(self.pool)
        self.payoff = LeaguePayoff()
        self.learners: dict[str, LearnerInfo] = {}

    def register_learner(self, learner_id: str, side: str, role: str = "main") -> LearnerInfo:
        if side not in ("attack", "defence"):
            raise ValueError(f"Invalid side: {side}")
        if role not in ("main", "main_exploiter", "league_exploiter"):
            raise ValueError(f"Invalid role: {role}")
        info = LearnerInfo(id=learner_id, side=side, role=role)
        self.learners[learner_id] = info
        return info

    def get_main_id(self, side: str) -> str | None:
        """Find the main learner ID for the given side."""
        for lid, info in self.learners.items():
            if info.side == side and info.role == "main":
                return lid
        return None

    def get_main_exploiter_ids(self, side: str) -> list[str]:
        """Find main exploiter learner IDs for the given side."""
        return [
            lid
            for lid, info in self.learners.items()
            if info.side == side and info.role == "main_exploiter"
        ]

    def sample_matchups(self) -> dict[str, str]:
        """Sample an opponent for each registered learner according to its matchmaking policy."""
        assignments: dict[str, str] = {}
        main_attack_id = self.get_main_id("attack")
        main_defence_id = self.get_main_id("defence")

        for lid, learner in self.learners.items():
            opp_side = "defence" if learner.side == "attack" else "attack"
            current_opp_main = main_defence_id if opp_side == "defence" else main_attack_id
            main_exploiters = self.get_main_exploiter_ids(opp_side)

            if learner.role == "main":
                opp_id = self.matchmaker.sample_opponent_for_main(
                    learner_id=lid,
                    opp_side=opp_side,
                    current_opp_main_id=current_opp_main,
                    main_exploiter_ids=main_exploiters,
                )
            elif learner.role == "main_exploiter":
                if not current_opp_main:
                    opp_id = self.matchmaker.sample_opponent_for_league_exploiter(lid, opp_side)
                else:
                    opp_id = self.matchmaker.sample_opponent_for_main_exploiter(current_opp_main)
            elif learner.role == "league_exploiter":
                opp_id = self.matchmaker.sample_opponent_for_league_exploiter(lid, opp_side)
            else:
                opp_id = current_opp_main or "unknown"

            learner.current_opponent_id = opp_id
            assignments[lid] = opp_id

        return assignments

    def record_matchup(self, attack_id: str, defence_id: str, scores: list[int]) -> None:
        """Record a list of scores from an attack vs defence matchup."""
        main_attack = self.get_main_id("attack")
        main_defence = self.get_main_id("defence")

        # Update mu_ref if this was main vs main
        if attack_id == main_attack and defence_id == main_defence:
            self.payoff.update_mu_ref(scores)

        for sc in scores:
            attack_won = self.payoff.record_innings(attack_id, defence_id, sc)
            self.pool.record_result(attack_id, defence_id, a_won=attack_won)

    def snapshot(
        self,
        learner_id: str,
        policy: Policy,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Snapshot a learner's current policy into the pool."""
        learner = self.learners[learner_id]
        snapshot_id = f"{learner_id}_step_{step}"
        meta = metadata or {}
        meta.update({"role": learner.role, "step": step})
        self.pool.add_snapshot(snapshot_id, policy, learner.side, step, meta)
        return snapshot_id

    def build_ladder(self) -> dict[str, str]:
        """Build difficulty ladder assignments based on available snapshots and win rates.

        Returns a dictionary mapping difficulty level ('easy', 'medium', 'hard')
        to snapshot IDs.
        """
        all_snapshots = self.pool.list_ids()
        if not all_snapshots:
            return {
                "beginner": "scripted:crosser?aggression=0.3",
                "easy": "scripted:crosser?aggression=0.5",
                "medium": "scripted:crosser?aggression=0.7",
                "hard": "scripted:crosser?aggression=0.9",
            }

        # Rank snapshots by total win count
        ranked = sorted(
            all_snapshots,
            key=lambda sid: sum(self.pool.entries[sid].wins.values()),
        )
        n = len(ranked)
        return {
            "beginner": "scripted:crosser?aggression=0.3",
            "easy": ranked[0],
            "medium": ranked[n // 2],
            "hard": ranked[-1],
            "kalari_master": ranked[-1],
        }
