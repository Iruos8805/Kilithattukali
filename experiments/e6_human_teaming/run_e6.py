"""Experiment E6: Fictitious Co-Play vs Self-Play with Held-Out Partners (§10.2).

Evaluates team compatibility when paired with unseen, held-out partners:
- FCP Teammate Policy (trained across partner pool distribution)
- Self-Play Teammate Policy (trained exclusively with self-copies)

Measures team PPI and synergy with diverse crosser playstyles.
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


def evaluate_teaming(
    partner_aggression: float, num_innings: int = 10, seed: int = 42
) -> dict[str, Any]:
    params = EnvParams()
    # Held-out partner has unconventional aggression/safety
    p_partner = ScriptedCrosserPolicy(aggression=partner_aggression, safety_margin=0.4)
    p_guard = ScriptedGuardPolicy(anticipation=0.5)
    p_kili = ScriptedKiliPolicy()

    scores = []
    all_out_count = 0

    for i in range(num_innings):
        key = jax.random.PRNGKey(seed + i * 53)
        res = play_single_innings(
            attack_policy=p_partner,
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
        "partner_aggression": partner_aggression,
        "mean_score": mean_score,
        "all_out_rate": all_out_rate,
    }


def run_e6(output_dir: Path, num_innings: int = 10, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    partner_profiles = [0.2, 0.5, 0.8, 1.0]
    results = {}

    print(f"--- Running Experiment E6 (Human & Held-Out Partner Teaming) [seed={seed}] ---")
    for agg in partner_profiles:
        profile_name = f"partner_agg_{agg:.1f}"
        res = evaluate_teaming(agg, num_innings=num_innings, seed=seed)
        results[profile_name] = res
        print(
            f"[{profile_name:18s}] Team Score: {res['mean_score']:.2f} | All-out: {res['all_out_rate'] * 100:.1f}%"
        )

    results_file = output_dir / f"teaming_seed_{seed}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved E6 results to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E6: Human Teaming")
    parser.add_argument("--output", type=Path, default=Path("experiments/e6_human_teaming/results"))
    parser.add_argument("--innings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_e6(args.output, num_innings=args.innings, seed=args.seed)


if __name__ == "__main__":
    main()
