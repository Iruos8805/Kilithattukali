"""Training CLI for Kilithattu-RL."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from kilithattu_core.env import KilithattuEnv

from kilithattu_train.curriculum import get_stage
from kilithattu_train.rollout import collect_rollout


def train_stage(stage_id: str = "S0", num_updates: int = 5) -> None:
    stage = get_stage(stage_id)
    print(f"Starting training on Stage {stage.stage_id}: {stage.name}")
    print(f"Description: {stage.description}")

    env = KilithattuEnv(stage.params)
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key, stage.params)
    print(f"Initialized stage {stage_id} environment successfully.")

    for u in range(num_updates):
        k1, key = jax.random.split(key)
        state, traj = collect_rollout(env, stage.params, state, k1, rollout_len=32)
        total_pts = int(jnp.sum(traj["points_attack"]))
        print(f"Update {u + 1}/{num_updates} complete | Points scored: {total_pts}")

    print(f"Curriculum stage {stage_id} run finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kilithattu-Train — RL Training Suite")
    subparsers = parser.add_subparsers(dest="command")

    stage_parser = subparsers.add_parser("stage", help="Train on a curriculum stage")
    stage_parser.add_argument("--stage", default="S0", help="Stage ID (S0..S6)")
    stage_parser.add_argument("--updates", type=int, default=5, help="Number of training updates")

    args = parser.parse_args()
    if args.command == "stage" or args.command is None:
        stage_name = getattr(args, "stage", "S0")
        updates = getattr(args, "updates", 5)
        train_stage(stage_name, updates)


if __name__ == "__main__":
    main()
