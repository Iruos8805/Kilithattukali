"""Scripted and AI policy controller adapter."""

from __future__ import annotations

from typing import Any
import jax
from chex import PRNGKey

from kilithattu_agents.policy import Policy
from kilithattu_core.env import Observations


class ScriptedController:
    """Adapts any Policy implementation into a slot Controller."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.carry = None
        self.key = jax.random.PRNGKey(42)

    def reset(self, slot_idx: int, seed: int) -> None:
        self.carry = self.policy.init_carry(1)
        self.key = jax.random.PRNGKey(seed + slot_idx)

    def act(self, obs: Observations, slot_idx: int) -> dict[str, int]:
        k1, self.key = jax.random.split(self.key)
        action, self.carry, _ = self.policy.act(k1, self.carry, obs, agent_idx=slot_idx)
        if isinstance(action, dict):
            return action
        return {"move": int(action)}
