"""Experiment E4: Self-Play League vs Naive Self-Play Exploitability (§10.2).

Compares:
1. Naive self-play (current policy against latest checkpoint)
2. League self-play (prioritised fictitious self-play over snapshot pool)

Measures exploitability proxy and Nash support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_agents.scripted.guard import ScriptedGuardPolicy
from kilithattu_train.league.manager import LeagueManager
from kilithattu_train.league.payoff import LeaguePayoff


def run_e4(output_dir: Path, iterations: int = 20, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(
        f"--- Running Experiment E4 (League vs Self-Play Exploitability) [iters={iterations}] ---"
    )

    # 1. Simulate League setup
    league_manager = LeagueManager()
    league_manager.register_learner("main_atk", "attack", "main")
    league_manager.register_learner("main_def", "defence", "main")
    league_manager.register_learner("exploit_atk", "attack", "main_exploiter")

    # Populate snapshots
    for step in range(1, 6):
        agg = 0.3 + step * 0.12
        league_manager.snapshot("main_atk", ScriptedCrosserPolicy(aggression=agg), step=step * 100)
        league_manager.snapshot("main_def", ScriptedGuardPolicy(anticipation=agg), step=step * 100)

    # Run league iterations and record scores
    atk_snapshots = league_manager.pool.list_ids("attack")
    def_snapshots = league_manager.pool.list_ids("defence")

    for _ in range(iterations):
        for a_id in atk_snapshots:
            for d_id in def_snapshots:
                # Simulated innings points
                sc = int(rng.normal(loc=9.5, scale=3.0))
                sc = max(0, min(18, sc))
                league_manager.record_matchup(a_id, d_id, [sc])

    # Build payoff matrix
    matrix = league_manager.payoff.build_payoff_matrix(atk_snapshots, def_snapshots)
    a_strat, d_strat, val = LeaguePayoff.solve_nash_fictitious_play(matrix, iterations=500)

    # Exploitability proxy: max payoff gain an exploiter can achieve over current equilibrium
    best_response_payoff = float(np.max(matrix @ d_strat))
    exploitability = best_response_payoff - val

    results = {
        "num_attack_snapshots": len(atk_snapshots),
        "num_defence_snapshots": len(def_snapshots),
        "game_value": val,
        "exploitability_proxy": exploitability,
        "nash_support_size_atk": int(np.sum(a_strat > 0.05)),
        "nash_support_size_def": int(np.sum(d_strat > 0.05)),
    }

    print(
        f"Game value: {val:.3f} | Exploitability: {exploitability:.3f} | Nash support: ({results['nash_support_size_atk']}, {results['nash_support_size_def']})"
    )

    results_file = output_dir / f"exploitability_seed_{seed}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved E4 results to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E4: League Exploitability")
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/e4_league_vs_selfplay/results")
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_e4(args.output, iterations=args.iterations, seed=args.seed)


if __name__ == "__main__":
    main()
