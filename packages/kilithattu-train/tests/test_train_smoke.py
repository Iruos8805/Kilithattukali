"""Smoke tests for training components, GAE, PPO loss, and curriculum."""

import jax
import jax.numpy as jnp
from kilithattu_core.env import KilithattuEnv
from kilithattu_train.curriculum import get_stage
from kilithattu_train.mappo import compute_gae, ppo_loss
from kilithattu_train.rollout import collect_rollout


def test_compute_gae():
    T, B = 10, 4
    rewards = jnp.ones((T, B), dtype=jnp.float32)
    values = jnp.zeros((T + 1, B), dtype=jnp.float32)
    dones = jnp.zeros((T, B), dtype=jnp.float32)

    adv, ret = compute_gae(rewards, values, dones)
    assert adv.shape == (T, B)
    assert ret.shape == (T, B)
    assert bool(jnp.all(ret > 0)) is True


def test_ppo_loss():
    N, A = 16, 9
    logits = jnp.zeros((N, A), dtype=jnp.float32)
    actions = jnp.zeros((N,), dtype=jnp.int32)
    old_log_probs = jnp.full((N,), -jnp.log(A), dtype=jnp.float32)
    advantages = jnp.ones((N,), dtype=jnp.float32)

    loss, metrics = ppo_loss(logits, actions, old_log_probs, advantages)
    assert jnp.isfinite(loss)
    assert "actor_loss" in metrics


def test_curriculum_rollout_smoke():
    stage = get_stage("S0")
    env = KilithattuEnv(stage.params)
    key = jax.random.PRNGKey(123)
    obs, state = env.reset(key, stage.params)

    state, traj = collect_rollout(env, stage.params, state, key, rollout_len=16)
    assert traj["reward_A"].shape == (16,)
    assert traj["done"].shape == (16,)
