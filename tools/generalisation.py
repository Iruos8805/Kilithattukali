"""Zero-shot generalisation (E3): attacks trained on the 2x5 field with 4 crossers, played on other fields / team sizes
against the scripted guards of that field. The scripted crosser is the reference.

  uv run python tools/generalisation.py --seeds 1 2 3 -n 32
"""
import argparse

import numpy as np

from _agents import build_agents
from kilithattu_agents import ScriptedDefence, load_policy
from kilithattu_core import load_params
from kilithattu_eval.generalisation import DEFAULT_VARIANTS
from kilithattu_eval.matches import play_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3]); ap.add_argument("--tag", default="_eq"); ap.add_argument("--prefix", default="p")
    ap.add_argument("-n", type=int, default=32); ap.add_argument("--rules", default="classic")
    ap.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    a = ap.parse_args()
    base = load_params(a.rules)
    agents = {n: at for n, (at, _) in build_agents(a.seeds, a.tag, a.prefix).items()}
    print(f"{'attack':<14}" + "".join(f"{v:>16}" for v in a.variants) + f"{'retention':>12}   (points per innings vs scripted guards)")
    for name, ref in agents.items():
        pts = []
        for v in a.variants:
            p = base.replace(**DEFAULT_VARIANTS[v])
            pts.append(play_match(p, load_policy(ref, "attack", p), ScriptedDefence(p), a.n, seed=3)["points_norm"])   # normalised by team size and trip value
        ret = np.mean(pts[1:]) / pts[0] if len(pts) > 1 and pts[0] > 0 else float("nan")
        print(f"{name:<14}" + "".join(f"{x:>16.3f}" for x in pts) + f"{ret:>12.2f}", flush=True)
    print("\ncells are normalised points (points / (crossers x 18)); retention = mean over the non-classic variants / classic (1.0 = no loss)")


if __name__ == "__main__":
    main()
