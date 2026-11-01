"""One command, one seed: imitation -> PPO fine-tune -> defence imitation -> two league iterations -> cross-table.

  uv run python tools/pipeline.py --seed 0            # about 1.5-2.5 h on a small GPU (see README of this script)
  uv run python tools/pipeline.py --seed 0 --dry-run  # print the commands only
Stages whose output already exists are skipped (use --force to redo). Everything lands under runs/p<seed>*.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="smoke-test sizes (not meaningful results)")
    ap.add_argument("--ratio1", default="10:25", help="attack:defence updates per phase in league iteration 1")
    ap.add_argument("--ratio2", default="8:30", help="same for league iteration 2")
    ap.add_argument("--tag", default="", help="suffix for the league/table outputs, e.g. _eq (stages 1-3 are shared)")
    ap.add_argument("--rules", default="classic", help="classic | plus"); ap.add_argument("--prefix", default=None, help="output prefix (default p, or q for plus)")
    ap.add_argument("--ft-updates", type=int, default=None, help="PPO fine-tune updates in stage 2 (default 1500)")
    ap.add_argument("--ft-promote", type=float, default=None, help="stage 2 stops early at this eval score (default 24; use 1000 to disable)")
    ap.add_argument("--ft-lr", type=float, default=None, help="stage 2 learning rate (default from the config, 3e-4)")
    ap.add_argument("--phases", type=int, default=20); ap.add_argument("--final-n", type=int, default=128)
    a = ap.parse_args()
    s = a.seed
    pre = a.prefix or ("q" if a.rules == "plus" else "p")
    rl = ["--rules", a.rules]
    R = f"runs/{pre}{s}"
    tiny = ["-o", "ppo.num_envs=4", "ppo.rollout_len=8", "ppo.minibatches=2", "ppo.epochs=1", "eval_episodes=2"] if a.tiny else []
    dag = ["--rounds", "1", "--envs", "2", "--ticks", "256", "--window", "64", "--steps", "2", "--minibatch", "2", "--eval-episodes", "0"] if a.tiny else ["--rounds", "6", "--envs", "6", "--steps", "800"]
    ft_updates = ["--updates", "2", "--promote", "1000"] if a.tiny else (
        (["--updates", str(a.ft_updates)] if a.ft_updates else []) + (["--promote", str(a.ft_promote)] if a.ft_promote is not None else []))
    ft_o = (tiny[1:] if a.tiny else []) + ([f"ppo.lr={a.ft_lr}"] if a.ft_lr else [])
    phases = "1" if a.tiny else str(a.phases)
    lg_extra = ["-o", f"seed={s}"] + (tiny[1:] if a.tiny else [])
    dagA, ft, dagD = f"{R}_dagger_attack", f"runs/{pre}{s}_ft", f"{R}_dagger_defence"
    r1, r2 = a.ratio1.split(":"), a.ratio2.split(":")
    lg1, lg2 = f"runs/{pre}{s}_lg1{a.tag}", f"runs/{pre}{s}_lg2{a.tag}"
    table = f"{R}{a.tag}_table.json"
    stages = [
        ("1 imitate the scripted attack (DAgger)", f"{dagA}/metadata.json",
         [PY, "tools/dagger.py", "--side", "attack", "--no-gru", "--seed", str(s), "--out", dagA] + rl + dag),
        ("2 PPO fine-tune the attack vs scripted guards", f"{ft}/attack/latest/metadata.json",
         [PY, "tools/finetune_attack.py", "--init", dagA, "--run-id", f"{pre}{s}_ft", "--seed", str(s)] + rl + ft_updates + (["-o"] + ft_o if ft_o else [])),
        ("3 imitate the scripted guards (DAgger)", f"{dagD}/metadata.json",
         [PY, "tools/dagger.py", "--side", "defence", "--no-gru", "--seed", str(s), "--opponents", f"scripted:crosser,path:{ft}/attack/latest", "--out", dagD] + rl + dag),
        (f"4 league iteration 1 ({a.ratio1} attack:defence updates)", f"{lg1}/attack/latest/metadata.json",
         [PY, "tools/league_run.py", "--run-id", f"{pre}{s}_lg1{a.tag}", "--init-attack", f"{ft}/attack/latest", "--init-defence", dagD,
          "--seed-attack", f"{ft}/attack/latest", "--seed-defence", dagD, "--phases", phases, "--attack-updates", r1[0], "--defence-updates", r1[1]] + rl + lg_extra),
        (f"5 league iteration 2 ({a.ratio2}), larger pool", f"{lg2}/attack/latest/metadata.json",
         [PY, "tools/league_run.py", "--run-id", f"{pre}{s}_lg2{a.tag}", "--init-attack", f"{lg1}/attack/latest", "--init-defence", f"{lg1}/defence/latest",
          "--seed-attack", f"{ft}/attack/latest", f"{lg1}/attack/latest", "--seed-defence", dagD, f"{lg1}/defence/latest",
          "--phases", phases, "--attack-updates", r2[0], "--defence-updates", r2[1]] + rl + lg_extra),
        ("6 cross-table against the scripted references", table,
         [PY, "tools/cross_table.py", "--attacks", "scripted:crosser", f"path:{ft}/attack/latest", f"path:{lg2}/attack/latest",
          "--defences", "scripted:guard?reaction=0.4", f"path:{dagD}", f"path:{lg2}/defence/latest", "-n", "4" if a.tiny else str(a.final_n), "--out", table] + rl),
    ]
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONPATH": ""}
    for name, done_file, cmd in stages:
        if Path(done_file).exists() and not a.force:
            print(f"[skip] {name}  ({done_file} exists)"); continue
        print(f"\n=== {name} ===\n$ {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t = time.time()
        subprocess.run(cmd, check=True, env=env)
        print(f"--- done in {(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
