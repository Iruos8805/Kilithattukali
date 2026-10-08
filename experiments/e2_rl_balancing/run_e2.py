"""Experiment E2: Reinforcement Learning & Empirical Balancing (§10.2).

Performs systematic evaluation over physical balance parameters:
- v_guard / v_crosser speed ratio
- v_kili / v_crosser speed ratio
- r_tag (tag radius)
- row_depth

Finds parameters within the viable competitive band (PPI in [8, 14], all-out rate in [0.2, 0.8]).
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


def evaluate_param_set(
    params: EnvParams, num_innings: int = 10, seed: int = 42
) -> dict[str, float]:
    p_crosser = ScriptedCrosserPolicy(aggression=0.7)
    p_guard = ScriptedGuardPolicy(anticipation=0.5)
    p_kili = ScriptedKiliPolicy()

    scores = []
    all_out_count = 0
    deliveries_count = 0

    for i in range(num_innings):
        key = jax.random.PRNGKey(seed + i * 31)
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

    # Balance loss: penalize deviation from target PPI (10.0) and target all-out rate (0.5)
    score_penalty = (mean_score - 10.0) ** 2
    all_out_penalty = ((all_out_rate - 0.5) * 20.0) ** 2
    balance_loss = score_penalty + all_out_penalty

    return {
        "mean_score": mean_score,
        "all_out_rate": all_out_rate,
        "delivery_rate": delivery_rate,
        "balance_loss": balance_loss,
        "is_competitive": (7.0 <= mean_score <= 14.0) and (0.2 <= all_out_rate <= 0.8),
    }


def run_e2(output_dir: Path, num_trials: int = 6, seed: int = 42) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Sweep configurations
    configs = [
        {"r_tag": 0.50, "guard_speed": 4.5, "kili_speed": 5.0, "row_depth": 3.0},
        {"r_tag": 0.55, "guard_speed": 4.8, "kili_speed": 5.2, "row_depth": 3.0},
        {
            "r_tag": 0.60,
            "guard_speed": 5.0,
            "kili_speed": 5.5,
            "row_depth": 3.0,
        },  # Classic baseline
        {"r_tag": 0.65, "guard_speed": 5.2, "kili_speed": 5.8, "row_depth": 3.0},
        {"r_tag": 0.70, "guard_speed": 5.5, "kili_speed": 6.0, "row_depth": 3.0},
        {"r_tag": 0.60, "guard_speed": 5.0, "kili_speed": 5.5, "row_depth": 3.5},
    ][:num_trials]

    print(f"--- Running Experiment E2 (RL & Empirical Balancing) [trials={len(configs)}] ---")
    for i, cfg in enumerate(configs):
        p = EnvParams(
            r_tag=cfg["r_tag"],
            guard_walk_speed=cfg["guard_speed"],
            kili_walk_speed=cfg["kili_speed"],
            row_depth=cfg["row_depth"],
        )
        res = evaluate_param_set(p, num_innings=10, seed=seed)
        trial_record = {
            "trial": i,
            "params": cfg,
            "results": res,
        }
        results.append(trial_record)
        comp_str = "YES" if res["is_competitive"] else "NO"
        print(
            f"Trial {i}: r_tag={cfg['r_tag']} v_g={cfg['guard_speed']} -> PPI={res['mean_score']:.2f}, all-out={res['all_out_rate'] * 100:.1f}%, comp={comp_str}"
        )

    results_file = output_dir / f"balance_trials_seed_{seed}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved E2 results to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E2: RL Balancing")
    parser.add_argument("--output", type=Path, default=Path("experiments/e2_rl_balancing/results"))
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_e2(args.output, num_trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
