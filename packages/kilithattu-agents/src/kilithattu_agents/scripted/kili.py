"""Scripted Kili bot with graph interception and clap triggering."""

from __future__ import annotations

from typing import Any

import numpy as np
from chex import PRNGKey
from kilithattu_core.env import Observations
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS

from kilithattu_agents.policy import Policy, PolicyInfo


class ScriptedKiliPolicy(Policy):
    """Kili agent that patrols junctions and moves along lines to intercept advancing crossers."""

    def __init__(self, patrol_speed: float = 1.0) -> None:
        self.patrol_speed = patrol_speed

    def init_carry(self, num_agents: int) -> Any:
        return None

    def act(
        self,
        key: PRNGKey,
        carry: Any,
        obs: Observations,
        agent_idx: int = MAX_CROSSERS + MAX_GUARDS,
        mask: Any = None,
    ) -> tuple[dict[str, int], Any, PolicyInfo]:
        ego_tok = np.array(obs.player_tokens[agent_idx, agent_idx])
        ego_x = ego_tok[14] * 6.0
        ego_y = ego_tok[15] * 15.0

        # Scan active crossers and target the most advanced in field (highest y for pacha, lowest y for uppu)
        crossers = []
        for i in range(MAX_CROSSERS):
            if bool(obs.player_masks[i]):
                c_tok = np.array(obs.player_tokens[agent_idx, i])
                is_alive = c_tok[6] < 0.5
                if is_alive:
                    cx = c_tok[14] * 6.0
                    cy = c_tok[15] * 15.0
                    in_field = 0.0 <= cy <= 15.0
                    if in_field:
                        dx = cx - ego_x
                        dy = cy - ego_y
                        dist = np.sqrt(dx * dx + dy * dy)
                        crossers.append((dist, dx, dy, cx, cy))

        move_action = 0  # 0=Stay, 1=N, 2=E, 3=S, 4=W
        if crossers:
            crossers.sort(key=lambda c: c[0])
            target = crossers[0]
            dx, dy = target[1], target[2]

            # Move along whichever axis has the larger gap
            if abs(dx) > abs(dy):
                move_action = 2 if dx > 0 else 4  # East or West
            else:
                move_action = 1 if dy > 0 else 3  # North or South
        else:
            # Idle patrol: sweep back and forth around centre junction
            move_action = 2 if ego_x < 3.0 else 4

        # Clap action: always clap if clap is enabled
        action = {"move": move_action, "dash": 1, "turn": 0, "clap": 1}
        return action, carry, PolicyInfo()
