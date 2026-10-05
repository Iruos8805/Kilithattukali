"""Vectorised rollout generation using jax.lax.scan."""

from __future__ import annotations

from typing import Any

import jax
from chex import PRNGKey
from kilithattu_core.config import EnvParams
from kilithattu_core.env import JointActions, KilithattuEnv


def collect_rollout(
    env: KilithattuEnv,
    params: EnvParams,
    init_state: Any,
    key: PRNGKey,
    rollout_len: int = 64,
) -> tuple[Any, dict[str, Any]]:
    """Scan over rollout_len ticks collecting batched transitions."""

    def step_fn(carry, _):
        s, k = carry
        k_step, k_next = jax.random.split(k)

        # Default actions (or policy actions)
        actions = JointActions.default()
        obs, next_s, r_A, r_D, events = env.step(k_step, s, actions, params)

        transition = {
            "reward_A": r_A,
            "reward_D": r_D,
            "done": next_s.global_state.done,
            "points_attack": events.points_attack,
        }
        return (next_s, k_next), transition

    (final_state, _), traj = jax.lax.scan(step_fn, (init_state, key), None, length=rollout_len)
    return final_state, traj
