"""Experiment E3: Generalisation across Procedural Field Sizes (§10.2).

Evaluates policies trained on standard field sizes against unseen/held-out fields:
- In-distribution: 2x5, 2x7
- Interpolated: 2x6
- Extrapolated: 3x7, 2x9
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


def evaluate_field_size(
    lanes: int, rows: int, crossers: int, num_innings: int = 10, seed: int = 42
) -> dict[str, Any]:
    params = EnvParams(
        num_lanes=lanes,
        num_rows=rows,
        num_crossers=crossers,
        num_guards=rows - 1,
    )
    p_crosser = ScriptedCrosserPolicy(aggression=0.7)
    p_guard = ScriptedGuardPolicy(anticipation=0.5)
    p_kili = ScriptedKiliPolicy()

    scores = []
    all_out_count = 0

    for i in range(num_innings):
        key = jax.random.PRNGKey(seed + i * 23)
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
        "lanes": lanes,
        "rows": rows,
        "crossers": crossers,
        "mean_score": mean_score,
        "all_out_rate": all_out_rate,
    }


def run_e3(output_dir: Path, num_innings: int = 10, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    field_sets = {
        "in_dist_2x5": (2, 5, 4),
        "in_dist_2x7": (2, 7, 4),
        "interpolated_2x6": (2, 6, 4),
        "extrapolated_3x7": (3, 7, 4),
        "extrapolated_2x9": (2, 9, 4),
    }

    results = {}
    print(f"--- Running Experiment E3 (Field Size Generalisation) [seed={seed}] ---")
    baseline_ppi = None

    for name, (lanes, rows, crossers) in field_sets.items():
        res = evaluate_field_size(lanes, rows, crossers, num_innings=num_innings, seed=seed)
        if baseline_ppi is None:
            baseline_ppi = res["mean_score"]
        norm_ppi = res["mean_score"] / max(0.1, baseline_ppi)
        res["norm_ppi"] = norm_ppi
        results[name] = res
        print(
            f"[{name:18s}] PPI: {res['mean_score']:.2f} (norm={norm_ppi * 100:.1f}%) | All-out: {res['all_out_rate'] * 100:.1f}%"
        )

    results_file = output_dir / f"generalisation_seed_{seed}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved E3 results to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E3: Generalisation")
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/e3_generalisation/results")
    )
    parser.add_argument("--innings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_e3(args.output, num_innings=args.innings, seed=args.seed)


if __name__ == "__main__":
    main()
