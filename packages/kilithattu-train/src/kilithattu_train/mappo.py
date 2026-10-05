"""Multi-Agent PPO (MAPPO) with centralized critic, GAE, and value clipping."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array


class TrajectoryBatch(NamedTuple):
    """Batched rollout trajectory for training updates."""

    obs_player: Array  # [T, B, MAX_PLAYERS, 27]
    obs_line: Array  # [T, B, MAX_LINES, 8]
    obs_global: Array  # [T, B, 20]
    player_mask: Array  # [T, B, MAX_PLAYERS]
    line_mask: Array  # [T, B, MAX_LINES]
    actions: Array  # [T, B, num_heads]
    log_probs: Array  # [T, B]
    rewards: Array  # [T, B]
    dones: Array  # [T, B]
    values: Array  # [T, B]


def compute_gae(
    rewards: Array,  # [T, B]
    values: Array,  # [T+1, B]
    dones: Array,  # [T, B]
    gamma: float = 0.997,
    gae_lambda: float = 0.95,
) -> tuple[Array, Array]:
    """Calculate Generalized Advantage Estimation (GAE) and discounted targets."""
    T = rewards.shape[0]

    def gae_step(carry, t):
        last_gae = carry
        next_val = values[t + 1]
        curr_val = values[t]
        done = dones[t]
        rew = rewards[t]

        delta = rew + gamma * next_val * (1.0 - done) - curr_val
        gae = delta + gamma * gae_lambda * (1.0 - done) * last_gae
        target = gae + curr_val
        return gae, (gae, target)

    _, (advantages, returns) = jax.lax.scan(
        gae_step, jnp.zeros_like(values[0]), jnp.arange(T - 1, -1, -1)
    )
    # Reverse to restore original chronological order [0..T-1]
    advantages = jnp.flip(advantages, axis=0)
    returns = jnp.flip(returns, axis=0)
    return advantages, returns


def ppo_loss(
    logits: Array,  # [N, num_actions]
    actions: Array,  # [N]
    old_log_probs: Array,  # [N]
    advantages: Array,  # [N]
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
) -> tuple[Array, dict[str, Array]]:
    """Compute clipped PPO actor loss with entropy bonus."""
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

    ratio = jnp.exp(action_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    # Entropy
    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

    total_loss = actor_loss - entropy_coeff * entropy
    metrics = {"actor_loss": actor_loss, "entropy": entropy, "ratio_mean": jnp.mean(ratio)}
    return total_loss, metrics
