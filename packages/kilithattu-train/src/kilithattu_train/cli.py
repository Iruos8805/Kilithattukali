"""Training CLI for Kilithattu-RL."""

from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from kilithattu_agents.checkpoint import CheckpointMetadata, save_model_checkpoint
from kilithattu_agents.networks import KilithattuActorCritic
from kilithattu_core.env import JointActions, KilithattuEnv
from kilithattu_core.spaces import MAX_CROSSERS, MAX_LINES, MAX_PLAYERS

from kilithattu_train.curriculum import get_stage
from kilithattu_train.mappo import compute_gae, ppo_loss

# Prevent double-free during ROCm runtime teardown on consumer RDNA3 GPUs
if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    atexit.register(lambda: os._exit(0))


def train_stage(
    stage_id: str = "S0",
    num_updates: int = 5,
    rollout_len: int = 50,
    output_dir: str = "checkpoints",
) -> None:
    stage = get_stage(stage_id)
    print(f"Starting Neural Network MAPPO training on Stage {stage.stage_id}: {stage.name}")
    print(f"Description: {stage.description}")

    env = KilithattuEnv(stage.params)
    key = jax.random.PRNGKey(42)

    # Initialize environment
    k_init, key = jax.random.split(key)
    obs, state = env.reset(k_init, stage.params)
    print(f"Initialized stage {stage_id} environment successfully.")

    # 1. Initialize Flax Entity-Transformer Actor-Critic
    model = KilithattuActorCritic(d_model=64, rnn_dim=128)
    p_tok = jnp.zeros((1, MAX_PLAYERS, 27))
    l_tok = jnp.zeros((1, MAX_LINES, 8))
    g_tok = jnp.zeros((1, 20))
    p_mask = jnp.ones((1, MAX_PLAYERS), dtype=jnp.bool_)
    l_mask = jnp.ones((1, MAX_LINES), dtype=jnp.bool_)

    k_nn, key = jax.random.split(key)
    nn_params = model.init(k_nn, p_tok, l_tok, g_tok, p_mask, l_mask, ego_idx=0)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=3e-4))
    opt_state = optimizer.init(nn_params)
    print("Initialized Transformer Actor-Critic network parameters.")

    cumulative_points = 0
    total_deliveries = 0

    for u in range(num_updates):
        if bool(state.global_state.done):
            k_rst, key = jax.random.split(key)
            obs, state = env.reset(k_rst, stage.params)

        p_toks, l_toks, g_toks, acts, old_logps, vals, rews, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        pts_update = 0
        deliv_update = 0

        # Collect rollout using neural network
        for _ in range(rollout_len):
            k_act, k_step, key = jax.random.split(key, 3)

            pt = obs.player_tokens[0][None, ...]
            lt = obs.line_tokens[None, ...]
            gt = obs.global_token[None, ...]
            pm = obs.player_masks[None, ...]
            lm = obs.line_mask[None, ...]

            logits, val, _ = model.apply(nn_params, pt, lt, gt, pm, lm, ego_idx=0)
            c_m = logits["c_move"][0]
            act = int(jax.random.categorical(k_act, c_m))
            logp = float(jax.nn.log_softmax(c_m)[act])

            actions = JointActions.default()
            actions = actions._replace(crosser_move=actions.crosser_move.at[0].set(act))

            obs, state, r_A, r_D, events = env.step(k_step, state, actions, stage.params)

            pts = int(events.points_attack)
            deliv = int(events.salt_delivered[0])
            pts_update += pts
            deliv_update += deliv

            p_toks.append(pt[0])
            l_toks.append(lt[0])
            g_toks.append(gt[0])
            acts.append(act)
            old_logps.append(logp)
            vals.append(float(val[0]))
            rews.append(float(r_A))
            dones.append(float(state.global_state.done))

            if bool(state.global_state.done):
                k_rst, key = jax.random.split(key)
                obs, state = env.reset(k_rst, stage.params)

        vals.append(0.0)
        advantages, returns = compute_gae(
            jnp.array(rews)[:, None], jnp.array(vals)[:, None], jnp.array(dones)[:, None]
        )
        adv = advantages.squeeze(-1)
        ret = returns.squeeze(-1)
        # Normalize advantages
        adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

        # 2. PPO Optimization Step
        b_p = jnp.stack(p_toks, axis=0)
        b_l = jnp.stack(l_toks, axis=0)
        b_g = jnp.stack(g_toks, axis=0)
        b_pm = jnp.tile(p_mask, (rollout_len, 1))
        b_lm = jnp.tile(l_mask, (rollout_len, 1))
        b_acts = jnp.array(acts, dtype=jnp.int32)
        b_old_logps = jnp.array(old_logps, dtype=jnp.float32)

        def loss_fn(p):
            b_log, b_val, _ = model.apply(p, b_p, b_l, b_g, b_pm, b_lm, ego_idx=0)
            c_log = b_log["c_move"]
            a_loss, met = ppo_loss(c_log, b_acts, b_old_logps, adv)
            v_loss = 0.5 * jnp.mean((b_val - ret) ** 2)
            tot_loss = a_loss + 0.5 * v_loss
            return tot_loss, {
                "actor_loss": a_loss,
                "val_loss": v_loss,
                "entropy": met["entropy"],
            }

        (total_loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(nn_params)
        updates, opt_state = optimizer.update(grads, opt_state, nn_params)
        nn_params = optax.apply_updates(nn_params, updates)

        cumulative_points += pts_update
        total_deliveries += deliv_update

        print(
            f"Update {u + 1}/{num_updates} complete | Points: {pts_update:2d} (total: {cumulative_points:3d}) | "
            f"Deliveries: {deliv_update} | PPO Loss: {float(metrics['actor_loss']):+.4f} | "
            f"Value Loss: {float(metrics['val_loss']):.4f} | Entropy: {float(metrics['entropy']):.3f}"
        )

    # 3. Save trained model checkpoint
    ckpt_dir = Path(output_dir) / f"stage_{stage_id}"
    latest_dir = Path(output_dir) / "latest"
    metadata = CheckpointMetadata(
        run_id=f"run_stage_{stage_id}",
        step=num_updates * rollout_len,
        side="attack",
        role_heads=["crosser_move", "crosser_sprint", "crosser_call"],
        git_commit="HEAD",
        env_version="1.0.0",
        rules_hash="classic",
        config={"d_model": 64, "rnn_dim": 128, "stage": stage_id},
        elo=1050 + cumulative_points,
    )
    save_model_checkpoint(ckpt_dir, nn_params, metadata)
    save_model_checkpoint(latest_dir, nn_params, metadata)
    print(f"Saved trained RL policy checkpoint to {ckpt_dir} and {latest_dir}")
    print(
        f"Curriculum stage {stage_id} run finished. Cumulative points: {cumulative_points} | Deliveries: {total_deliveries}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kilithattu-Train — RL Training Suite")
    subparsers = parser.add_subparsers(dest="command")

    stage_parser = subparsers.add_parser("stage", help="Train on a curriculum stage")
    stage_parser.add_argument("--stage", default="S0", help="Stage ID (S0..S6)")
    stage_parser.add_argument("--updates", type=int, default=5, help="Number of training updates")
    stage_parser.add_argument(
        "--rollout-len", type=int, default=50, help="Rollout length in ticks"
    )

    args = parser.parse_args()
    if args.command == "stage" or args.command is None:
        stage_name = getattr(args, "stage", "S0")
        updates = getattr(args, "updates", 5)
        rollout_len = getattr(args, "rollout_len", 50)
        train_stage(stage_name, updates, rollout_len)


if __name__ == "__main__":
    main()
