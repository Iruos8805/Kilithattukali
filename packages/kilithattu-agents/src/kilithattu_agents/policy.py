"""Policy protocol and interface definitions for all agents (scripted and neural)."""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol
from chex import Array, PRNGKey

from kilithattu_core.env import Observations


class PolicyInfo(NamedTuple):
    """Auxiliary outputs from a policy step (values, action distributions, attention)."""

    value: Array | None = None
    attention_weights: Array | None = None  # [heads, ego, target] for spectator attention overlays
    action_probs: Array | None = None
    info_dict: dict[str, Any] | None = None


class Policy(Protocol):
    """Interface implemented by scripted bots, neural networks, and wrapper policies."""

    def init_carry(self, num_agents: int) -> Any:
        """Initialize recurrent carry/memory state (e.g. GRU state or last-seen memory)."""
        ...

    def act(
        self,
        key: PRNGKey,
        carry: Any,
        obs: Observations,
        agent_idx: int,
        mask: Any = None,
    ) -> tuple[Any, Any, PolicyInfo]:
        """Compute an action for agent_idx.

        Returns:
            (action, new_carry, policy_info)
        """
        ...
