"""Game CLI launcher and setup for Kilithattu Kali."""

from __future__ import annotations

import argparse

from kilithattu_agents.registry import load_policy
from kilithattu_core.config import load_env_params
from kilithattu_core.env import KilithattuEnv
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS

from kilithattu_game.controllers.human import HumanController
from kilithattu_game.controllers.scripted import ScriptedController
from kilithattu_game.loop import GameLoop
from kilithattu_game.slots import SlotManager


def create_game(
    mode: str = "play_crosser",
    difficulty: str = "medium",
    rules_preset: str = "classic",
    headless: bool = False,
) -> GameLoop:
    """Initialize environment, controllers, and slot manager for selected mode."""
    params = load_env_params(rules_preset)
    env = KilithattuEnv(params)
    slot_manager = SlotManager(params)

    # Resolve AI policies
    crosser_ai = load_policy("scripted:crosser")
    guard_ai = load_policy("scripted:guard")
    kili_ai = load_policy("scripted:kili")

    if mode == "play_crosser":
        # Slot 0 is human crosser; remaining slots are AI
        slot_manager.assign_controller(0, HumanController(is_crosser=True))
        for i in range(1, params.num_crossers):
            slot_manager.assign_controller(i, ScriptedController(crosser_ai))
        for k in range(params.num_guards):
            slot_manager.assign_controller(MAX_CROSSERS + k, ScriptedController(guard_ai))
        slot_manager.assign_controller(MAX_CROSSERS + MAX_GUARDS, ScriptedController(kili_ai))

    elif mode == "play_kili":
        # Human controls Kili
        for i in range(params.num_crossers):
            slot_manager.assign_controller(i, ScriptedController(crosser_ai))
        for k in range(params.num_guards):
            slot_manager.assign_controller(MAX_CROSSERS + k, ScriptedController(guard_ai))
        slot_manager.assign_controller(MAX_CROSSERS + MAX_GUARDS, HumanController(is_crosser=False))

    elif mode == "watch":
        # All AI slots
        for i in range(params.num_crossers):
            slot_manager.assign_controller(i, ScriptedController(crosser_ai))
        for k in range(params.num_guards):
            slot_manager.assign_controller(MAX_CROSSERS + k, ScriptedController(guard_ai))
        slot_manager.assign_controller(MAX_CROSSERS + MAX_GUARDS, ScriptedController(kili_ai))

    else:
        # Default fallback to play_crosser
        slot_manager.assign_controller(0, HumanController(is_crosser=True))
        for i in range(1, params.num_crossers):
            slot_manager.assign_controller(i, ScriptedController(crosser_ai))
        for k in range(params.num_guards):
            slot_manager.assign_controller(MAX_CROSSERS + k, ScriptedController(guard_ai))
        slot_manager.assign_controller(MAX_CROSSERS + MAX_GUARDS, ScriptedController(kili_ai))

    return GameLoop(env=env, slot_manager=slot_manager, params=params, headless=headless)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kilithattu Kali — Interactive Onam Game")
    parser.add_argument(
        "--mode",
        choices=["play_crosser", "play_kili", "watch", "tutorial"],
        default="play_crosser",
        help="Game mode (default: play_crosser)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["beginner", "easy", "medium", "hard", "master"],
        default="medium",
        help="AI difficulty level (default: medium)",
    )
    parser.add_argument(
        "--rules",
        choices=["classic", "plus"],
        default="classic",
        help="Rules preset (default: classic)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without graphical window")
    parser.add_argument("--ticks", type=int, default=None, help="Max simulation ticks to run")

    args = parser.parse_args()

    game = create_game(
        mode=args.mode,
        difficulty=args.difficulty,
        rules_preset=args.rules,
        headless=args.headless,
    )
    results = game.run(max_ticks=args.ticks)
    print(
        f"Match ended! Attack Score: {results['score_attack']} | Defence: {results['score_defence']}"
    )


if __name__ == "__main__":
    main()
