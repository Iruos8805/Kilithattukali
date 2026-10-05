"""Slot manager, controller assignment, and possession switching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from kilithattu_agents.policy import Policy
from kilithattu_core.config import EnvParams
from kilithattu_core.env import JointActions, Observations
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS


class Controller(Protocol):
    def reset(self, slot_idx: int, seed: int) -> None: ...
    def act(self, obs: Observations, slot_idx: int) -> dict[str, int]: ...


@dataclass
class SlotInfo:
    slot_idx: int
    role: str  # 'crosser', 'guard', 'kili'
    team: str  # 'attack', 'defence'
    is_human: bool = False


class SlotManager:
    """Manages player slots, active human possession, and controller execution."""

    def __init__(self, params: EnvParams) -> None:
        self.params = params
        self.slots: list[SlotInfo] = []
        self.controllers: dict[int, Controller] = {}
        self.human_crosser_idx: int = 0

        # Create slot definitions
        for i in range(params.num_crossers):
            self.slots.append(SlotInfo(slot_idx=i, role="crosser", team="attack", is_human=(i == 0)))
        for k in range(params.num_guards):
            idx = MAX_CROSSERS + k
            self.slots.append(SlotInfo(slot_idx=idx, role="guard", team="defence", is_human=False))
        # Kili
        kili_idx = MAX_CROSSERS + MAX_GUARDS
        self.slots.append(SlotInfo(slot_idx=kili_idx, role="kili", team="defence", is_human=False))

    def assign_controller(self, slot_idx: int, controller: Controller) -> None:
        self.controllers[slot_idx] = controller

    def switch_human_crosser(self) -> int:
        """Cycle human control to next active crosser with Tab key."""
        next_c = (self.human_crosser_idx + 1) % self.params.num_crossers
        self.slots[self.human_crosser_idx].is_human = False
        self.slots[next_c].is_human = True
        self.human_crosser_idx = next_c
        return next_c

    def get_joint_actions(self, obs: Observations) -> JointActions:
        """Query all assigned controllers to assemble the JointActions pytree."""
        import jax.numpy as jnp

        c_moves = [0] * MAX_CROSSERS
        c_sprints = [0] * MAX_CROSSERS
        c_calls = [0] * MAX_CROSSERS

        g_moves = [0] * MAX_GUARDS
        g_dashes = [0] * MAX_GUARDS
        g_turns = [0] * MAX_GUARDS

        k_move, k_dash, k_turn, k_clap = 0, 0, 0, 0

        for slot in self.slots:
            idx = slot.slot_idx
            ctrl = self.controllers.get(idx)
            act = ctrl.act(obs, idx) if ctrl else {"move": 0}

            if slot.role == "crosser":
                c_moves[idx] = act.get("move", 0)
                c_sprints[idx] = act.get("sprint", 0)
                c_calls[idx] = act.get("call", 0)
            elif slot.role == "guard":
                g_idx = idx - MAX_CROSSERS
                g_moves[g_idx] = act.get("move", 0)
                g_dashes[g_idx] = act.get("dash", 0)
                g_turns[g_idx] = act.get("turn", 0)
            elif slot.role == "kili":
                k_move = act.get("move", 0)
                k_dash = act.get("dash", 0)
                k_turn = act.get("turn", 0)
                k_clap = act.get("clap", 0)

        return JointActions(
            crosser_move=jnp.array(c_moves, dtype=jnp.int32),
            crosser_sprint=jnp.array(c_sprints, dtype=jnp.int32),
            crosser_call=jnp.array(c_calls, dtype=jnp.int32),
            guard_move=jnp.array(g_moves, dtype=jnp.int32),
            guard_dash=jnp.array(g_dashes, dtype=jnp.int32),
            guard_turn=jnp.array(g_turns, dtype=jnp.int32),
            kili_move=jnp.array(k_move, dtype=jnp.int32),
            kili_dash=jnp.array(k_dash, dtype=jnp.int32),
            kili_turn=jnp.array(k_turn, dtype=jnp.int32),
            kili_clap=jnp.array(k_clap, dtype=jnp.int32),
        )
