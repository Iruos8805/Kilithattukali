"""PPO fine-tune of an (imitation) attack against a fixed opponent. Writes runs/<run-id>/attack/{latest,best}."""
import argparse

import jax

from kilithattu_agents import load_checkpoint
from kilithattu_agents.policy import init_networks
from kilithattu_core import load_params
from kilithattu_train.config import StageCfg, load_config
from kilithattu_train.curriculum import run_curriculum
from kilithattu_train.logging_utils import RunLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--rules", default="classic"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--promote", type=float, default=24.0, help="stop early once eval points reach this")
    ap.add_argument("--opponent", default="scripted:guard?reaction=0.4")
    ap.add_argument("-o", "--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config("configs/train/curriculum.yaml", ["ppo.num_envs=64", "ppo.rollout_len=32", "ppo.minibatches=8", "ppo.epochs=2",
        f"ppo.total_updates={a.updates}", "ppo.ent_start=0.005", "ppo.ent_end=0.002", "net.gru=false", "log_every=100",
        "eval_episodes=32", f"run_id={a.run_id}", f"seed={a.seed}", f"rules={a.rules}"] + a.override)
    p = load_params(cfg.rules)
    actor, _, _ = load_checkpoint(a.init)
    critic = init_networks(jax.random.PRNGKey(a.seed + 1), p, cfg.net.build())[3]      # imitation checkpoints have no critic
    log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__)
    stage = StageCfg("ft", env={}, defence=a.opponent, updates=a.updates, promote_points=a.promote)
    run_curriculum(cfg, [stage], log, params_a={"actor": actor, "critic": critic})


if __name__ == "__main__":
    main()
