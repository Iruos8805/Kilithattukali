"""Fictitious Co-Play (E6): does training with diverse partners give a better teammate than self-play?

Train: one learner crosser (slot 0) plays with frozen partners drawn from a pool, against a scripted defence.
Evaluate (cross-play): team = [X, Y, Y, Y] where X is the learner under test and Y a HELD-OUT partner never seen in
training. Compare X = FCP learner against X = the self-play baseline (a policy that trained with copies of itself).

  uv run python tools/fcp_experiment.py --init runs/p1_dagger_attack \
      --partners runs/p1_ft/attack/latest runs/p2_ft/attack/latest runs/p1_lg2_eq/attack/latest runs/p2_lg2_eq/attack/latest \
      --heldout runs/p3_ft/attack/latest runs/p3_lg2_eq/attack/latest --selfplay runs/p1_ft/attack/latest
"""
import argparse
import json
import os

import jax
import numpy as np
from flax import serialization

from kilithattu_agents import MixedTeam, NetworkPolicy, load_checkpoint, load_policy
from kilithattu_agents.policy import init_networks
from kilithattu_core import load_params
from kilithattu_eval.matches import play_match
from kilithattu_train.config import load_config
from kilithattu_train.logging_utils import RunLogger
from kilithattu_train.partners import train_fcp

BASE = ["ppo.num_envs=64", "ppo.rollout_len=32", "ppo.minibatches=8", "ppo.epochs=2", "ppo.lr=1.0e-4", "ppo.ent_start=0.005",
        "ppo.ent_end=0.002", "net.gru=false", "eval_episodes=16"]


def run_name(path):
    parts = path.rstrip('/').split('/')
    return parts[-3] if len(parts) >= 3 and parts[-2] in ('attack', 'defence') else parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="imitation checkpoint the FCP learner starts from")
    ap.add_argument("--partners", nargs="+", required=True); ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--selfplay", required=True, help="self-play baseline attack checkpoint")
    ap.add_argument("--learner", default=None, help="skip training and evaluate this FCP checkpoint")
    ap.add_argument("--updates", type=int, default=300); ap.add_argument("--phases", type=int, default=12)
    ap.add_argument("--defences", nargs="+", default=["scripted:guard?reaction=0.4"])
    ap.add_argument("--run-id", default="fcp_exp"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-only", action="store_true"); ap.add_argument("--eval-n", type=int, default=128); ap.add_argument("--rules", default="classic")
    ap.add_argument("--out", default=None); ap.add_argument("-o", "--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config("configs/train/fcp.yaml", BASE + a.override + [f"run_id={a.run_id}", f"seed={a.seed}", f"fcp_updates={a.updates}", f"rules={a.rules}"])
    p = load_params(cfg.rules)
    net = cfg.net.build()
    learner_dir = a.learner
    if learner_dir is None:
        fresh = init_networks(jax.random.PRNGKey(a.seed + 1), p, net)[3]
        actor, _, _ = load_checkpoint(a.init)
        partners = [load_checkpoint(d)[0] for d in a.partners]
        log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__)
        train_fcp(cfg, p, partners, defence="scripted:guard?reaction=0.4", logger=log, params_a={"actor": actor, "critic": fresh}, phases=a.phases)
        learner_dir = str(log.dir / "attack" / "fcp")
        if a.train_only:
            print("trained learner saved to", learner_dir); return
    mask = [False, True, True, True][: p.n_crossers] + [True] * max(0, p.n_crossers - 4)
    def pol(d):
        params, ncfg, _ = load_checkpoint(d)
        return NetworkPolicy("attack", p, ncfg, params)

    fcp, selfp = pol(learner_dir), pol(a.selfplay)
    rows, results = [], {}
    for h in a.heldout:
        Y = pol(h)
        for dref in a.defences:
            D = load_policy(dref, "defence", p)
            for name, team in (("FCP learner + held-out partners", MixedTeam(fcp, Y, mask)), ("self-play baseline + held-out partners", MixedTeam(selfp, Y, mask)),
                               ("FCP learner alone (x4)", fcp), ("self-play baseline alone (x4)", selfp), ("held-out partner alone (x4)", Y)):
                m = play_match(p, team, D, a.eval_n, seed=7)
                se = m["points_std"] / np.sqrt(m["innings"])
                results[f"{run_name(h)} | {dref} | {name}"] = [m["points_mean"], se]
                print(f"held-out {h[-40:]:<40} vs {dref[-28:]:<28} {name:<42} {m['points_mean']:6.1f} +- {se:4.1f}", flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
    gap = [(k, v) for k, v in results.items() if "+ held-out" in k]
    print("\nteam points when paired with a partner it has never met (higher is better):")
    for k, v in gap:
        print(f"  {k}: {v[0]:.1f} +- {v[1]:.1f}")


if __name__ == "__main__":
    main()