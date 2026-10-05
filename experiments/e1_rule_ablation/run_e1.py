"""Experiment E1: Rule Depth & Ablation Study (§10.2).

Evaluates competitiveness, scoring, and tactical depth across rule variants:
- Classic
- Classic + P1 (stamina + acceleration)
- Classic + P2 (facing directions + turning)
- Classic + P3 (fog of war)
- Classic + P4 (Kili clap gating)
- Plus (all P1-P4 enabled)
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


def evaluate_variant(
    variant_name: str, params: EnvParams, num_innings: int = 10, seed: int = 42
) -> dict[str, float]:
    p_crosser = ScriptedCrosserPolicy(aggression=0.7)
    p_guard = ScriptedGuardPolicy(anticipation=0.5)
    p_kili = ScriptedKiliPolicy()

    scores = []
    all_out_count = 0
    deliveries_count = 0

    for i in range(num_innings):
        key = jax.random.PRNGKey(seed + i * 17)
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
        deliveries_count += res.salt_deliveries

    mean_score = float(sum(scores) / max(1, len(scores)))
    all_out_rate = float(all_out_count / max(1, len(scores)))
    delivery_rate = float(deliveries_count / max(1, len(scores)))

    return {
        "variant": variant_name,
        "mean_score": mean_score,
        "all_out_rate": all_out_rate,
        "delivery_rate": delivery_rate,
        "num_innings": num_innings,
    }


def run_e1(output_dir: Path, num_innings: int = 10, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base = EnvParams()
    variants = {
        "Classic": base,
        "Classic+P1": EnvParams(p1_momentum_stamina=True),
        "Classic+P2": EnvParams(p2_facing_turning=True),
        "Classic+P3": EnvParams(p3_limited_vision=True),
        "Classic+P4": EnvParams(p4_strategic_clap=True),
        "Plus": EnvParams(
            p1_momentum_stamina=True,
            p2_facing_turning=True,
            p3_limited_vision=True,
            p4_strategic_clap=True,
        ),
    }

    results = {}
    print(f"--- Running Experiment E1 (Rule Depth Ablation) [seed={seed}] ---")
    for name, params in variants.items():
        res = evaluate_variant(name, params, num_innings=num_innings, seed=seed)
        results[name] = res
        print(
            f"[{name:12s}] Mean Score: {res['mean_score']:.2f} | All-out: {res['all_out_rate'] * 100:.1f}% | Deliveries: {res['delivery_rate']:.2f}"
        )

    results_file = output_dir / f"results_seed_{seed}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E1: Rule Ablation")
    parser.add_argument("--output", type=Path, default=Path("experiments/e1_rule_ablation/results"))
    parser.add_argument("--innings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_e1(args.output, num_innings=args.innings, seed=args.seed)


if __name__ == "__main__":
    main()
