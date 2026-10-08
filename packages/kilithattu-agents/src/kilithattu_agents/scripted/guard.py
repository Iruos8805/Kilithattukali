"""Scripted Guard bot with horizontal target tracking and threat prioritization."""

from __future__ import annotations

from typing import Any

import numpy as np
from chex import PRNGKey
from kilithattu_core.env import Observations
from kilithattu_core.spaces import MAX_CROSSERS

from kilithattu_agents.policy import Policy, PolicyInfo


class ScriptedGuardPolicy(Policy):
    """Guard agent that laterally mirrors the highest-threat crosser on its line."""

    def __init__(self, anticipation: float = 0.5) -> None:
        self.anticipation = anticipation

    def init_carry(self, num_agents: int) -> Any:
        return None

    def act(
        self,
        key: PRNGKey,
        carry: Any,
        obs: Observations,
        agent_idx: int,  # MAX_CROSSERS .. MAX_CROSSERS + MAX_GUARDS - 1
        mask: Any = None,
    ) -> tuple[dict[str, int], Any, PolicyInfo]:
        # Scan active crossers and find closest in y to ego's line
        threats = []
        for i in range(MAX_CROSSERS):
            if bool(obs.player_masks[i]):
                c_tok = np.array(obs.player_tokens[agent_idx, i])
                # Check if alive (status != out, status[0] is out)
                is_alive = c_tok[6] < 0.5
                if is_alive:
                    dx = c_tok[12] * 6.0
                    dy = c_tok[13] * 15.0
                    dist_y = abs(dy)
                    threats.append((dist_y, dx, c_tok[14] * 6.0))

        move_action = 0  # 0=Stay, 1=W (-x), 2=E (+x)
        if threats:
            # Sort by proximity to line in y
            threats.sort(key=lambda t: t[0])
            primary_target = threats[0]
            target_dx = primary_target[1]

            if target_dx > 0.15:
                move_action = 2  # Move East towards target
            elif target_dx < -0.15:
                move_action = 1  # Move West towards target
            else:
                move_action = 0  # Mirrored

        action = {"move": move_action, "dash": 1, "turn": 0}
        return action, carry, PolicyInfo()
