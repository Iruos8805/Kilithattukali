"""Differential testing comparing vectorised JAX simulation against the Python reference engine."""

import jax
import jax.numpy as jnp
import pytest

from kilithattu_core.config import EnvParams
from kilithattu_core.env import JointActions, KilithattuEnv
from kilithattu_core.reference.reference_env import ReferenceKilithattuEnv
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS


def test_differential_step_matching():
    """Verify that pure-Python reference env and JAX env match on rollouts."""
    params = EnvParams(clap_time=0.1)
    jax_env = KilithattuEnv(params)
    ref_env = ReferenceKilithattuEnv(params)

    # Initial state
    key = jax.random.PRNGKey(777)
    obs, jax_state = jax_env.reset(key, params)

    # Align initial positions exactly for 1:1 step tracking
    # Set crosser 0 at (3.0, -1.0), Guard 0 at (3.0, 3.0), Kili at (3.0, 0.0)
    jax_state = jax_state._replace(
        crossers=jax_state.crossers._replace(pos=jax_state.crossers.pos.at[0].set(jnp.array([3.0, -1.0]))),
        guards=jax_state.guards._replace(x=jnp.full((MAX_GUARDS,), 0.5)),
        kili=jax_state.kili._replace(pos=jnp.array([0.5, 0.0])),
    )
    ref_env.crossers[0].x = 3.0
    ref_env.crossers[0].y = -1.0
    for g in ref_env.guards:
        g.x = 0.5
    ref_env.kili.x = 0.5
    ref_env.kili.y = 0.0

    # Step both environments forward
    for t in range(40):
        # Move crosser 0 North (action 1)
        c_actions = [0] * MAX_CROSSERS
        c_actions[0] = 1
        g_actions = [0] * MAX_GUARDS

        joint_actions = JointActions.default()._replace(
            crosser_move=jnp.array(c_actions, dtype=jnp.int32),
            guard_move=jnp.array(g_actions, dtype=jnp.int32),
        )

        obs, jax_state, r_A, r_D, events = jax_env.step(key, jax_state, joint_actions, params)
        ref_out = ref_env.step({"crosser_move": c_actions, "guard_move": g_actions})

        # Compare attack score
        assert int(jax_state.global_state.score_attack) == ref_out["score_attack"], (
            f"Step {t}: score mismatch JAX {int(jax_state.global_state.score_attack)} != Ref {ref_out['score_attack']}"
        )
