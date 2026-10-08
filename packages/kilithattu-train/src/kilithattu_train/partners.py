"""Fictitious Co-Play (FCP) partner pool for human-compatible teammate training (§6.6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kilithattu_agents.human_fair import HumanFairPolicyWrapper
from kilithattu_agents.policy import Policy
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_agents.scripted.guard import ScriptedGuardPolicy
from kilithattu_agents.scripted.kili import ScriptedKiliPolicy


@dataclass
class PartnerEntry:
    id: str
    role: str  # "crosser" | "guard" | "kili"
    policy: Policy
    description: str = ""


class FCPPartnerPool:
    """Maintains a diverse pool of partner policies for training human-compatible teammates (§6.6).

    Includes:
    - Scripted bots with randomised aggression, safety margins, and reaction delays.
    - Snapshot checkpoints from training.
    - Human-fair observation delays and sticky actions.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self.partners: dict[str, PartnerEntry] = {}
        self._populate_default_scripted_partners()

    def _populate_default_scripted_partners(self) -> None:
        """Seed the pool with diverse scripted baselines (cautious, aggressive, delayed)."""
        # Crosser variants
        for agg in [0.2, 0.4, 0.6, 0.8, 1.0]:
            p = ScriptedCrosserPolicy(aggression=agg, safety_margin=0.8 + agg * 0.4)
            wrapped = HumanFairPolicyWrapper(
                p, obs_delay_ticks=int(agg * 4), sticky_action_prob=0.05
            )
            self.add_partner(
                partner_id=f"scripted_crosser_agg_{agg:.1f}",
                role="crosser",
                policy=wrapped,
                description=f"Scripted crosser with aggression={agg:.1f}",
            )

        # Guard variants
        for ant in [0.2, 0.5, 0.8]:
            p = ScriptedGuardPolicy(anticipation=ant)
            self.add_partner(
                partner_id=f"scripted_guard_ant_{ant:.1f}",
                role="guard",
                policy=p,
                description=f"Scripted guard with anticipation={ant:.1f}",
            )

        # Kili variants
        p_kili = ScriptedKiliPolicy()
        self.add_partner(
            partner_id="scripted_kili",
            role="kili",
            policy=p_kili,
            description="Scripted Kili with graph navigation and clap",
        )

    def add_partner(
        self, partner_id: str, role: str, policy: Policy, description: str = ""
    ) -> None:
        self.partners[partner_id] = PartnerEntry(
            id=partner_id, role=role, policy=policy, description=description
        )

    def sample_partner(self, role: str) -> Policy:
        """Randomly sample a partner policy for the requested role."""
        candidates = [p for p in self.partners.values() if p.role == role]
        if not candidates:
            raise ValueError(f"No partners available for role '{role}'")
        choice = self.rng.choice(candidates)
        return choice.policy

    def sample_crosser_slots(
        self, num_crossers: int, num_learner_slots: int = 2
    ) -> dict[int, str | Policy]:
        """Sample assignment of crosser slots between learner and partner pool.

        Returns a dictionary mapping slot_idx -> 'learner' or sampled Policy.
        """
        all_slots = list(range(num_crossers))
        self.rng.shuffle(all_slots)
        learner_indices = set(all_slots[:num_learner_slots])

        assignment: dict[int, str | Policy] = {}
        for idx in range(num_crossers):
            if idx in learner_indices:
                assignment[idx] = "learner"
            else:
                assignment[idx] = self.sample_partner("crosser")
        return assignment
