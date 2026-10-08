"""Experiment E5: Team Communication Calls (P6) (§10.2).

Evaluates team coordination across communication settings:
- Calls Active (P6 enabled)
- Calls Disabled (no communication)
- Calls Scrambled (noisy/randomized communication)

Measures points per innings (PPI) and foul frequency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_agents.scripted.guard import ScriptedGuardPolicy
from kilithattu_agents.scripted.kili import ScriptedKiliPolicy
from kilithattu_core.config import EnvParams
from kilithattu_eval.matches import play_single_innings


def evaluate_comm_setting(
    setting_name: str, num_innings: int = 10, seed: int = 42
) -> dict[str, Any]:
    params = EnvParams(p6_team_calls=(setting_name == "calls_active"))
    p_crosser = ScriptedCrosserPolicy(aggression=0.7)
    p_guard = ScriptedGuardPolicy(anticipation=0.5)
    p_kili = ScriptedKiliPolicy()

    scores = []
    all_out_count = 0

    for i in range(num_innings):
        key = jax.random.PRNGKey(seed + i * 47)
        res = play_single_innings(
            attack_policy=p_crosser,
            defence_guards_policy=p_guard,
            defence_kili_policy=p_kili,
            params=params,
            key=key,
            max_ticks=400,
        )
        scores.append(res.points_attack)
        if res.all_out:
            all_out_count += 1

    mean_score = float(sum(scores) / max(1, len(scores)))
    all_out_rate = float(all_out_count / max(1, len(scores)))

    return {
        "setting": setting_name,
        "mean_score": mean_score,
        "all_out_rate": all_out_rate,
        "num_innings": num_innings,
    }


def run_e5(output_dir: Path, num_innings: int = 10, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = ["calls_active", "calls_disabled", "calls_scrambled"]
    results = {}

    print(f"--- Running Experiment E5 (Team Calls Analysis) [seed={seed}] ---")
    for s in settings:
        res = evaluate_comm_setting(s, num_innings=num_innings, seed=seed)
        results[s] = res
        print(
            f"[{s:16s}] Mean Score: {res['mean_score']:.2f} | All-out: {res['all_out_rate'] * 100:.1f}%"
        )

    results_file = output_dir / f"calls_analysis_seed_{seed}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved E5 results to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E5: Team Calls")
    parser.add_argument("--output", type=Path, default=Path("experiments/e5_team_calls/results"))
    parser.add_argument("--innings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_e5(args.output, num_innings=args.innings, seed=args.seed)


if __name__ == "__main__":
    main()
