"""Benchmark environment simulation steps per second."""

import time
import jax
import jax.numpy as jnp

from kilithattu_core.config import EnvParams
from kilithattu_core.env import JointActions, KilithattuEnv


def run_benchmark(num_envs: int = 64, num_steps: int = 500) -> float:
    params = EnvParams()
    env = KilithattuEnv(params)

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_envs)

    # Vectorized reset
    v_reset = jax.jit(jax.vmap(lambda k: env.reset(k, params)))
    obs, state = v_reset(keys)

    # Vectorized step
    actions = JointActions.default()
    v_actions = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), num_envs, axis=0), actions)

    def step_fn(s, _):
        step_key = jax.random.PRNGKey(1)
        o, next_s, r_A, r_D, ev = jax.vmap(lambda single_s, single_a: env.step(step_key, single_s, single_a, params))(
            s, v_actions
        )
        return next_s, None

    jitted_loop = jax.jit(lambda s: jax.lax.scan(step_fn, s, None, length=num_steps))

    # Warmup JIT
    state, _ = jitted_loop(state)
    jax.block_until_ready(state.crossers.pos)

    # Timed run
    start = time.perf_counter()
    state, _ = jitted_loop(state)
    jax.block_until_ready(state.crossers.pos)
    elapsed = time.perf_counter() - start

    total_ticks = num_envs * num_steps
    steps_per_sec = total_ticks / elapsed
    print(f"Envs: {num_envs} | Steps: {num_steps} | Time: {elapsed:.3f}s | Steps/sec: {steps_per_sec:,.0f}")
    return steps_per_sec


if __name__ == "__main__":
    run_benchmark(num_envs=64, num_steps=500)
