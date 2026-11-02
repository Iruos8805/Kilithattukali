"""Run the league from existing agents with a chosen update ratio, printing one line per phase.

Example:
  uv run python tools/league_run.py --run-id league_v5 --init-attack runs/league_v4/attack/latest \
      --init-defence runs/dagger/defence --seed-attack runs/finetune_v2/attack/latest runs/league_v3/attack/latest \
      --seed-defence runs/defence_v3/defence/best runs/league_v3/defence/latest --attack-updates 10 --defence-updates 25
"""
import argparse
import os

import jax
from flax import serialization

from kilithattu_agents import load_checkpoint, save_checkpoint
from kilithattu_agents.policy import init_networks
from kilithattu_core import load_params
from kilithattu_train.config import load_config
from kilithattu_train.league import League, Snapshot
from kilithattu_train.logging_utils import RunLogger

BASE = ["ppo.num_envs=64", "ppo.rollout_len=32", "ppo.minibatches=8", "ppo.epochs=2", "ppo.lr=1.0e-4", "ppo.ent_start=0.01",
        "ppo.ent_end=0.004", "net.gru=false", "win_points=30", "eval_episodes=16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rules", default="classic")
    ap.add_argument("--init-attack", required=True)
    ap.add_argument("--init-defence", required=True)
    ap.add_argument("--seed-attack", nargs="*", default=[], help="checkpoints added to the opponent pool")
    ap.add_argument("--seed-defence", nargs="*", default=[])
    ap.add_argument("--phases", type=int, default=20)
    ap.add_argument("--attack-updates", type=int, default=10)
    ap.add_argument("--defence-updates", type=int, default=25)
    ap.add_argument("-o", "--override", nargs="*", default=[], help="extra OmegaConf overrides, e.g. ppo.num_envs=32")
    a = ap.parse_args()

    cfg = load_config("configs/train/league.yaml", BASE + a.override + [f"run_id={a.run_id}", f"rules={a.rules}"])
    p = load_params(cfg.rules)
    net = cfg.net.build()
    fresh_critic = init_networks(jax.random.PRNGKey(1), p, net)[3]

    def load(d):
        actor, _, _ = load_checkpoint(d)
        cpath = os.path.join(d, "critic.msgpack")      # checkpoints from imitation have no critic: start a fresh one
        critic = serialization.from_bytes(fresh_critic, open(cpath, "rb").read()) if os.path.exists(cpath) else fresh_critic
        return {"actor": actor, "critic": critic}

    log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__)
    lg = League(cfg, p, log, init_attack=load(a.init_attack), init_defence=load(a.init_defence))
    for side, paths in (("attack", a.seed_attack), ("defence", a.seed_defence)):
        for i, d in enumerate(paths):
            lg.pool.add(Snapshot(f"seed_{side[:3]}{i}_{os.path.basename(os.path.dirname(d)) or 'x'}", side, "past", 0, load(d)))
    try:
        for phase in range(a.phases):
            ra = lg.train_agent(lg.main["attack"], a.attack_updates) if a.attack_updates else {"opp": "-", "points": float("nan")}
            rd = lg.train_agent(lg.main["defence"], a.defence_updates) if a.defence_updates else {"opp": "-", "points": float("nan")}
            print(f"phase {phase + 1:2d} | attack trained vs {ra['opp']:<24} score {ra['points']:7.1f} | defence trained vs {rd['opp']:<24} attack score {rd['points']:7.1f}", flush=True)
            log(phase + 1, {"attack_phase_points": ra["points"], "defence_phase_points": rd["points"]}, tag="league")
            if (phase + 1) % 2 == 0:
                for side in ("attack", "defence"):
                    lg._snapshot(lg.main[side], "main")
            lg.pool.save(log.dir, p, net, cfg.run_id)
            for side in ("attack", "defence"):
                ag = lg.main[side]
                save_checkpoint(log.dir / side / "latest", ag.params["actor"], p, net, side, cfg.run_id, ag.updates, critic_params=ag.params["critic"])
    finally:
        print("saved to", log.dir)


if __name__ == "__main__":
    main()
