"""Behaviour detectors (E7): pincers, decoys, guard mirroring error, fouls and tags per innings for each agent pair.

  uv run python tools/tactics.py --seeds 1 2 3 --episodes 3
"""
import argparse

from _agents import build_agents
from kilithattu_agents import load_policy
from kilithattu_core import load_params
from kilithattu_eval.behaviours import behaviour_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3]); ap.add_argument("--tag", default="_eq"); ap.add_argument("--prefix", default="p")
    ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--episodes", type=int, default=3); ap.add_argument("--rules", default="classic")
    a = ap.parse_args()
    p = load_params(a.rules)
    print(f"{'agent':<14}{'pincers':>9}{'decoys':>8}{'guard error (m)':>17}{'fouls':>7}{'tags':>7}   (per innings, {a.episodes} innings)")
    for name, (at, d) in build_agents(a.seeds, a.tag, a.prefix, a.extra).items():
        r = behaviour_report(p, load_policy(at, "attack", p), load_policy(d, "defence", p), seeds=tuple(range(a.episodes)))
        print(f"{name:<14}{r['pincers']:>9.2f}{r['decoys']:>8.2f}{r['mirroring_error']:>17.2f}{r['fouls']:>7.2f}{r['tags']:>7.2f}", flush=True)
    print("\npincer: two crossers pass the same line within 1 s at least 2.5 m apart; decoy: a sideways dash near a line followed by a teammate crossing it;")
    print("guard error: mean lateral distance between a guard and the crosser nearest its line (smaller = tighter marking).")


if __name__ == "__main__":
    main()
