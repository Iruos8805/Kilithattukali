"""Evaluation CLI for tournaments and metrics."""

from __future__ import annotations

import argparse

import jax
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_core.config import EnvParams

from kilithattu_eval.tournament import run_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description="Kilithattu-Eval — Evaluation and Tournament CLI")
    parser.add_argument("--matches", type=int, default=2, help="Matches per pair")
    parser.add_argument(
        "--innings", type=int, default=None, help="Innings per pair (1 match = 4 innings)"
    )
    subparsers = parser.add_subparsers(dest="command")

    tourn_parser = subparsers.add_parser("tournament", help="Run a round-robin tournament")
    tourn_parser.add_argument("--matches", type=int, default=2, help="Matches per pair")
    tourn_parser.add_argument("--innings", type=int, default=None, help="Innings per pair")

    args = parser.parse_args()
    matches = getattr(args, "matches", 2)
    innings = getattr(args, "innings", None)
    if innings is not None:
        matches = max(1, innings // 4)

    policies = {
        "scripted_aggressive": ScriptedCrosserPolicy(aggression=0.9, safety_margin=0.5),
        "scripted_cautious": ScriptedCrosserPolicy(aggression=0.4, safety_margin=1.2),
        "scripted_balanced": ScriptedCrosserPolicy(aggression=0.7, safety_margin=0.8),
    }

    params = EnvParams(innings_duration=30.0)
    key = jax.random.PRNGKey(42)

    print("Running tournament among baseline policies...")
    payoff, ratings = run_tournament(policies, params, key, matches_per_pair=matches)
    print("\nTournament Final Elo Ratings:")
    for name, r in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:25s}: {r:.1f}")


if __name__ == "__main__":
    main()
