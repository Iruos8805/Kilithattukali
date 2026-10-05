"""Vectorised rollout generation and action sampling for RL training."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from chex import PRNGKey
from kilithattu_core.config import EnvParams
from kilithattu_core.env import JointActions, KilithattuEnv
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS
from kilithattu_core.state import STATUS_OUT, STATUS_UPPU


def sample_curriculum_actions(
    state: Any,
    key: PRNGKey,
    params: EnvParams,
    noise_prob: float = 0.15,
) -> tuple[JointActions, Any]:
    """Sample goal-directed exploratory actions for curriculum training.

    - Pacha/Home crossers move North (+y) toward far zone with exploration.
    - Uppu crossers move South (-y) toward home safe zone to deliver salt.
    - Guards track nearest crosser laterally.
    - Kili navigates toward nearest active crosser and claps start.
    """
    k_c, k_g, k_k = jax.random.split(key, 3)

    # 1. Crosser actions
    c_status = state.crossers.status
    c_pos = state.crossers.pos

    def sample_crosser(i, k_i):
        st = c_status[i]
        k_rnd, k_act = jax.random.split(k_i)
        is_noisy = jax.random.uniform(k_rnd) < noise_prob

        # Outbound (North) move=1 (North), with 2 (NE) or 8 (NW) noise
        noisy_north = jax.random.choice(k_act, jnp.array([1, 2, 8]))
        # Inbound (South) move=5 (South), with 4 (SE) or 6 (SW) noise
        noisy_south = jax.random.choice(k_act, jnp.array([5, 4, 6]))

        outbound_move = jnp.where(is_noisy, noisy_north, 1)
        inbound_move = jnp.where(is_noisy, noisy_south, 5)

        m = jnp.where(st == STATUS_UPPU, inbound_move, outbound_move)
        m = jnp.where(st == STATUS_OUT, 0, m)
        return m

    k_crossers = jax.random.split(k_c, MAX_CROSSERS)
    c_moves = jax.vmap(sample_crosser)(jnp.arange(MAX_CROSSERS), k_crossers)

    # 2. Guard lateral tracking
    g_x_all = state.guards.x
    target_c_x = c_pos[0, 0]  # track primary crosser

    def sample_guard(k, _):
        gx = g_x_all[k]
        # move: 0 stay, 1 West (-x), 2 East (+x)
        diff = target_c_x - gx
        return jnp.where(jnp.abs(diff) < 0.2, 0, jnp.where(diff > 0, 2, 1))

    g_moves = jax.vmap(sample_guard)(jnp.arange(MAX_GUARDS), jnp.zeros(MAX_GUARDS))

    # 3. Kili action
    kili_m = jnp.array(1, dtype=jnp.int32)  # North
    kili_clap = jnp.array(1, dtype=jnp.int32)  # clap if in pre-clap

    actions = JointActions(
        crosser_move=c_moves,
        crosser_sprint=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32),
        crosser_call=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32),
        guard_move=g_moves,
        guard_dash=jnp.zeros((MAX_GUARDS,), dtype=jnp.int32),
        guard_turn=jnp.zeros((MAX_GUARDS,), dtype=jnp.int32),
        kili_move=kili_m,
        kili_dash=jnp.array(0, dtype=jnp.int32),
        kili_turn=jnp.array(0, dtype=jnp.int32),
        kili_clap=kili_clap,
    )
    return actions, c_moves[0]


def collect_rollout(
    env: KilithattuEnv,
    params: EnvParams,
    init_state: Any,
    key: PRNGKey,
    rollout_len: int = 100,
    noise_prob: float = 0.15,
) -> tuple[Any, dict[str, Any]]:
    """Scan over rollout_len ticks collecting batched transitions and rewards."""

    def step_fn(carry, _):
        s, k = carry
        k_step, k_act, k_next = jax.random.split(k, 3)

        actions, primary_action = sample_curriculum_actions(s, k_act, params, noise_prob=noise_prob)
        obs, next_s, r_A, r_D, events = env.step(k_step, s, actions, params)

        transition = {
            "reward_A": r_A,
            "reward_D": r_D,
            "done": next_s.global_state.done,
            "points_attack": events.points_attack,
            "deliveries": events.salt_delivered[0],
            "action": primary_action,
        }
        return (next_s, k_next), transition

    (final_state, _), traj = jax.lax.scan(step_fn, (init_state, key), None, length=rollout_len)
    return final_state, traj
