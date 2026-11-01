"""Round-robin tournament with Elo and Nash averaging over the final agents (E4 support).

An agent is an (attack, defence) pair; A beats B when A's attack against B's defence outscores B's attack against A's defence.
  uv run python tools/tournament.py --seeds 1 2 3 -n 32
"""
import argparse

import numpy as np

from _agents import build_agents
from kilithattu_agents import load_policy
from kilithattu_core import load_params
from kilithattu_eval.plots import heatmap, rating_bars
from kilithattu_eval.tournament import round_robin, save_tournament


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3]); ap.add_argument("--tag", default="_eq"); ap.add_argument("--prefix", default="p")
    ap.add_argument("--extra", nargs="*", default=[], help="name=attackref,defref"); ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--rules", default="classic"); ap.add_argument("--out", default="runs/tournament"); ap.add_argument("--fig", default="docs/figures")
    a = ap.parse_args()
    p = load_params(a.rules)
    spec = build_agents(a.seeds, a.tag, a.prefix, a.extra)
    agents = {n: {"attack": load_policy(at, "attack", p), "defence": load_policy(d, "defence", p)} for n, (at, d) in spec.items()}
    print(f"{len(agents)} agents, {len(agents) * (len(agents) - 1) // 2} pairings, {a.n} innings per direction", flush=True)
    res = round_robin(p, agents, a.n)
    save_tournament(res, a.out)
    names, elo = res["names"], res["elo"]
    print("\nElo (1000 = average):")
    for i in np.argsort(-elo):
        print(f"  {names[i]:<14} {elo[i]:7.1f}   nash weight {res['nash_mix'][i]:.2f}   score vs nash mix {res['vs_nash'][i]:+.3f}")
    print(f"game value of the population: {res['nash_value']:+.3f} (0 = symmetric)")
    from pathlib import Path
    Path(a.fig).mkdir(parents=True, exist_ok=True)
    heatmap(res["score"], names, names, Path(a.fig) / "tournament_scores.png", "pairwise score (row beats column > 0)")
    rating_bars(names, elo, Path(a.fig) / "tournament_elo.png")
    print("figures in", a.fig)


if __name__ == "__main__":
    main()
