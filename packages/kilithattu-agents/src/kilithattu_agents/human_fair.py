"""Human-fair wrappers: observation delay ring buffer, sticky actions, and decision intervals."""

from __future__ import annotations

from typing import Any

import jax
from chex import PRNGKey
from kilithattu_core.env import Observations

from kilithattu_agents.policy import Policy, PolicyInfo


class HumanFairPolicyWrapper:
    """Wraps any base policy to introduce human-matched latencies, decision intervals, and noise."""

    def __init__(
        self,
        base_policy: Policy,
        obs_delay_ticks: int = 3,  # ~150ms at 20 Hz
        decision_interval_ticks: int = 2,  # ~100ms
        sticky_action_prob: float = 0.05,
    ) -> None:
        self.base_policy = base_policy
        self.obs_delay_ticks = obs_delay_ticks
        self.decision_interval_ticks = decision_interval_ticks
        self.sticky_action_prob = sticky_action_prob

    def init_carry(self, num_agents: int = 16) -> dict[str, Any]:
        base_carry = self.base_policy.init_carry(num_agents)
        return {
            "base_carry": base_carry,
            "obs_buffer": [],  # Ring buffer of historical observations
            "held_actions": {},  # Dict mapping agent_idx to held action
            "tick_counter": 0,
        }

    def act(
        self,
        key: PRNGKey,
        carry: dict[str, Any],
        obs: Observations,
        agent_idx: int,
        mask: Any = None,
    ) -> tuple[Any, dict[str, Any], PolicyInfo]:
        buf = carry["obs_buffer"]
        buf.append(obs)
        if len(buf) > max(1, self.obs_delay_ticks + 1):
            buf.pop(0)

        delayed_obs = buf[0]
        counter = carry["tick_counter"]
        held = carry["held_actions"].get(agent_idx, None)

        k1, k2 = jax.random.split(key)
        # Check decision interval
        needs_decision = (held is None) or (counter % self.decision_interval_ticks == 0)

        if needs_decision:
            action, new_base_carry, info = self.base_policy.act(
                k1, carry["base_carry"], delayed_obs, agent_idx, mask
            )
            # Sticky action check
            is_sticky = jax.random.uniform(k2) < self.sticky_action_prob
            if is_sticky and held is not None:
                action = held
            carry["held_actions"][agent_idx] = action
            carry["base_carry"] = new_base_carry
        else:
            action = held
            info = PolicyInfo()

        carry["tick_counter"] = counter + 1
        return action, carry, info
