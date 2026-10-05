"""Application entry point and Pygame loop initializer."""

from __future__ import annotations

import argparse
import atexit
import os
import random

from kilithattu_agents.registry import load_policy
from kilithattu_core.config import load_env_params
from kilithattu_core.env import KilithattuEnv
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS

from kilithattu_game.controllers.human import HumanController
from kilithattu_game.controllers.scripted import ScriptedController
from kilithattu_game.loop import GameLoop
from kilithattu_game.slots import SlotManager

# Prevent double-free during ROCm runtime teardown on consumer RDNA3 GPUs
if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    atexit.register(lambda: os._exit(0))


def create_game(
    mode: str = "play_crosser",
    difficulty: str = "medium",
    rules_preset: str = "classic",
    headless: bool = False,
    policy_crosser: str | None = None,
    policy_guard: str | None = None,
    policy_kili: str | None = None,
) -> GameLoop:
    """Initialize environment, controllers, and slot manager for selected mode."""
    params = load_env_params(rules_preset)
    env = KilithattuEnv(params)
    slot_manager = SlotManager(params)

    # Resolve AI policies
    default_crosser_ref = "learned:crosser" if mode == "watch" else f"main@{difficulty}"
    crosser_ref = policy_crosser or default_crosser_ref
    guard_ref = policy_guard or "scripted:guard"
    kili_ref = policy_kili or "scripted:kili"

    crosser_ai = load_policy(crosser_ref)
    guard_ai = load_policy(guard_ref)
    kili_ai = load_policy(kili_ref)

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
    parser.add_argument("--policy-crosser", default=None, help="Custom policy for crossers (e.g. learned:crosser)")
    parser.add_argument("--policy-guard", default=None, help="Custom policy for guards (e.g. scripted:guard)")
    parser.add_argument("--policy-kili", default=None, help="Custom policy for kili (e.g. scripted:kili)")
    parser.add_argument("--seed", type=int, default=None, help="PRNG seed (randomized if omitted)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window")
    parser.add_argument("--ticks", type=int, default=None, help="Max simulation ticks to run")

    args = parser.parse_args()

    game = create_game(
        mode=args.mode,
        difficulty=args.difficulty,
        rules_preset=args.rules,
        headless=args.headless,
        policy_crosser=args.policy_crosser,
        policy_guard=args.policy_guard,
        policy_kili=args.policy_kili,
    )

    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    print(f"Launching Kilithattu Kali [Mode: {args.mode} | Seed: {seed}]")
    results = game.run(max_ticks=args.ticks, seed=seed)
    print(
        f"Match ended! Attack Score: {results['score_attack']} | Defence: {results['score_defence']}"
    )


if __name__ == "__main__":
    main()
