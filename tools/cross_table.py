"""Attack x defence table of mean attack points (and trips), with worst cases and the saddle of the zero-sum game.

  uv run python tools/cross_table.py --attacks scripted:crosser path:runs/a/attack/latest \
      --defences "scripted:guard?reaction=0.4" path:runs/d/defence/latest -n 128 --out runs/table.json
"""
import argparse
import json

import numpy as np

from kilithattu_agents import load_policy
from kilithattu_core import load_params
from kilithattu_eval.matches import play_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="+", required=True); ap.add_argument("--defences", nargs="+", required=True)
    ap.add_argument("-n", type=int, default=128); ap.add_argument("--rules", default="classic"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = load_params(a.rules)
    pts, trips = np.zeros((len(a.attacks), len(a.defences))), np.zeros((len(a.attacks), len(a.defences)))
    for j, d in enumerate(a.defences):
        for i, at in enumerate(a.attacks):
            m = play_match(p, load_policy(at, "attack", p), load_policy(d, "defence", p), a.n, seed=1000 * i + j)
            pts[i, j], trips[i, j] = m["points_mean"], m["trips_mean"]
            print(f"{at[-46:]:<46} vs {d[-46:]:<46} points {pts[i, j]:8.1f}  trips {trips[i, j]:5.2f}", flush=True)
    worst_att, worst_def = pts.min(1), pts.max(0)
    i, j = int(np.argmax(worst_att)), int(np.argmin(worst_def))
    print("\nworst case per attack (min over defences):", dict(zip(a.attacks, worst_att.round(1))))
    print("worst case per defence (max conceded):    ", dict(zip(a.defences, worst_def.round(1))))
    print(f"maximin attack: {a.attacks[i]} ({worst_att[i]:.1f});  minimax defence: {a.defences[j]} ({worst_def[j]:.1f});  "
          + ("saddle point (pure equilibrium)" if abs(worst_att[i] - worst_def[j]) < 1e-9 else "no pure saddle in this table"))
    if a.out:
        json.dump({"attacks": a.attacks, "defences": a.defences, "points": pts.tolist(), "trips": trips.tolist(), "n": a.n}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
