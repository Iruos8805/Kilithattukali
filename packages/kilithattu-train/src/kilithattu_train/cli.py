"""Training CLI for Kilithattu-RL."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import optax
from kilithattu_core.env import KilithattuEnv

from kilithattu_train.curriculum import get_stage
from kilithattu_train.mappo import compute_gae, ppo_loss
from kilithattu_train.rollout import collect_rollout


def train_stage(stage_id: str = "S0", num_updates: int = 5, rollout_len: int = 100) -> None:
    stage = get_stage(stage_id)
    print(f"Starting training on Stage {stage.stage_id}: {stage.name}")
    print(f"Description: {stage.description}")

    env = KilithattuEnv(stage.params)
    key = jax.random.PRNGKey(42)

    # Initialize environment
    k_init, key = jax.random.split(key)
    obs, state = env.reset(k_init, stage.params)
    print(f"Initialized stage {stage_id} environment successfully.")

    # Initialize policy parameters and Optax optimizer
    num_actions = 9
    policy_params = {"weights": jnp.zeros((num_actions,), dtype=jnp.float32)}
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=3e-4))
    opt_state = optimizer.init(policy_params)

    cumulative_points = 0

    for u in range(num_updates):
        k1, key = jax.random.split(key)
        if bool(state.global_state.done):
            k_rst, key = jax.random.split(key)
            obs, state = env.reset(k_rst, stage.params)

        state, traj = collect_rollout(env, stage.params, state, k1, rollout_len=rollout_len)

        # Calculate reward and GAE
        rewards = traj["reward_A"]
        T = rewards.shape[0]
        values = jnp.zeros((T + 1,), dtype=jnp.float32)
        dones = traj["done"].astype(jnp.float32)

        advantages, returns = compute_gae(rewards[:, None], values[:, None], dones[:, None])
        adv = advantages.squeeze(-1)

        # Policy optimization step
        actions = traj["action"]
        old_log_probs = -jnp.log(float(num_actions)) * jnp.ones((T,), dtype=jnp.float32)

        def loss_fn(p):
            logits = jnp.tile(p["weights"], (T, 1))
            total_loss, metrics = ppo_loss(logits, actions, old_log_probs, adv)
            return total_loss, metrics

        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_params)
        updates, opt_state = optimizer.update(grads, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)

        pts = int(jnp.sum(traj["points_attack"]))
        deliveries = int(jnp.sum(traj["deliveries"]))
        cumulative_points += pts

        print(
            f"Update {u + 1}/{num_updates} complete | Points: {pts:2d} (total: {cumulative_points:3d}) | "
            f"Deliveries: {deliveries} | PPO Loss: {float(loss_val):+.4f} | "
            f"Entropy: {float(metrics['entropy']):.3f}"
        )

    print(f"Curriculum stage {stage_id} run finished. Cumulative points scored: {cumulative_points}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kilithattu-Train — RL Training Suite")
    subparsers = parser.add_subparsers(dest="command")

    stage_parser = subparsers.add_parser("stage", help="Train on a curriculum stage")
    stage_parser.add_argument("--stage", default="S0", help="Stage ID (S0..S6)")
    stage_parser.add_argument("--updates", type=int, default=5, help="Number of training updates")
    stage_parser.add_argument("--rollout-len", type=int, default=100, help="Rollout length in ticks")

    args = parser.parse_args()
    if args.command == "stage" or args.command is None:
        stage_name = getattr(args, "stage", "S0")
        updates = getattr(args, "updates", 5)
        rollout_len = getattr(args, "rollout_len", 100)
        train_stage(stage_name, updates, rollout_len)


if __name__ == "__main__":
    main()
