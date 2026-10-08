"""Scripted Crosser bot with gap finding, defender avoidance, and foul prevention."""

from __future__ import annotations

from typing import Any

import numpy as np
from chex import PRNGKey
from kilithattu_core.env import Observations
from kilithattu_core.spaces import MAX_CROSSERS, MAX_PLAYERS

from kilithattu_agents.policy import Policy, PolicyInfo


class ScriptedCrosserPolicy(Policy):
    """Heuristic crosser agent that seeks gaps across lines and returns home safely."""

    def __init__(
        self,
        aggression: float = 0.7,
        safety_margin: float = 0.8,
    ) -> None:
        self.aggression = aggression
        self.safety_margin = safety_margin

    def init_carry(self, num_agents: int) -> Any:
        return None

    def act(
        self,
        key: PRNGKey,
        carry: Any,
        obs: Observations,
        agent_idx: int,
        mask: Any = None,
    ) -> tuple[dict[str, int], Any, PolicyInfo]:
        # Ego player token
        ego_tok = np.array(obs.player_tokens[agent_idx, agent_idx])
        # Indices:
        # 6..8: status (0=out, 1=pacha, 2=uppu)
        is_out = ego_tok[6] > 0.5
        is_pacha = ego_tok[7] > 0.5
        is_uppu = ego_tok[8] > 0.5

        if is_out:
            return {"move": 0, "sprint": 0, "call": 0}, carry, PolicyInfo()

        # Extract defender positions
        defender_positions = []
        for j in range(MAX_CROSSERS, MAX_PLAYERS):
            if bool(obs.player_masks[j]):
                tok = np.array(obs.player_tokens[agent_idx, j])
                dx = tok[12] * 6.0
                dy = tok[13] * 15.0
                dist = np.sqrt(dx * dx + dy * dy)
                defender_positions.append((dx, dy, dist, tok[14] * 6.0, tok[15] * 15.0))

        # Decision logic
        move_action = 0
        sprint_action = 1 if self.aggression > 0.6 else 0

        # Check nearest defender threat
        threat = [d for d in defender_positions if d[2] < 2.0]

        if is_pacha:
            # Outbound: goal is to move North (+y)
            if not threat:
                # Clear path -> advance North
                move_action = 1  # North
            else:
                # Threat nearby: evaluate horizontal gap
                nearest_threat = min(threat, key=lambda x: x[2])
                th_dx = nearest_threat[0]

                if abs(th_dx) < self.safety_margin:
                    # Defender directly ahead: feint or dodge sideways
                    move_action = 3 if th_dx < 0 else 7  # Move away horizontally (East or West)
                else:
                    # Defender is offset: slip past in diagonal or straight North
                    move_action = 1 if abs(th_dx) > self.safety_margin else (2 if th_dx < 0 else 8)

        elif is_uppu:
            # Inbound: goal is to return South (-y) to Home safe zone
            if not threat:
                move_action = 5  # South
            else:
                nearest_threat = min(threat, key=lambda x: x[2])
                th_dx = nearest_threat[0]
                if abs(th_dx) < self.safety_margin:
                    move_action = 3 if th_dx < 0 else 7
                else:
                    move_action = 5  # Push South

        action = {"move": move_action, "sprint": sprint_action, "call": 0}
        return action, carry, PolicyInfo()
