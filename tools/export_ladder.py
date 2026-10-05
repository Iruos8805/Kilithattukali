"""Export ladder manifest tool (§8.6).

Extracts or validates calibrated difficulty ladder settings into configs/game/ladder.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from kilithattu_train.league.manager import LeagueManager


def export_ladder(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manager = LeagueManager()
    ladder_tiers = manager.build_ladder()

    manifest = {
        "version": "1.0.0",
        "levels": {
            "beginner": {
                "policy": ladder_tiers.get("beginner", "scripted:crosser?aggression=0.3"),
                "observation_delay_ms": 300,
                "decision_interval_ms": 150,
                "sticky_action_prob": 0.15,
            },
            "easy": {
                "policy": ladder_tiers.get("easy", "scripted:crosser?aggression=0.5"),
                "observation_delay_ms": 250,
                "decision_interval_ms": 100,
                "sticky_action_prob": 0.10,
            },
            "medium": {
                "policy": ladder_tiers.get("medium", "scripted:crosser?aggression=0.7"),
                "observation_delay_ms": 200,
                "decision_interval_ms": 100,
                "sticky_action_prob": 0.05,
            },
            "hard": {
                "policy": ladder_tiers.get("hard", "scripted:crosser?aggression=0.9"),
                "observation_delay_ms": 150,
                "decision_interval_ms": 50,
                "sticky_action_prob": 0.0,
            },
            "kalari_master": {
                "policy": ladder_tiers.get("kalari_master", "scripted:crosser?aggression=1.0"),
                "observation_delay_ms": 50,
                "decision_interval_ms": 50,
                "sticky_action_prob": 0.0,
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Exported difficulty ladder manifest to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export game difficulty ladder manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/game/ladder.yaml"),
        help="Path to output YAML manifest",
    )
    args = parser.parse_args()
    export_ladder(args.output)


if __name__ == "__main__":
    main()
