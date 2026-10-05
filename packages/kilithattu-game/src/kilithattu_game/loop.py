"""Fixed-timestep 20 Hz simulation loop with 60 fps rendering."""

from __future__ import annotations

import os
import time
import jax
import pygame
from chex import PRNGKey

from kilithattu_core.config import EnvParams
from kilithattu_core.env import KilithattuEnv
from kilithattu_game.controllers.human import HumanController
from kilithattu_game.render.renderer import GameRenderer
from kilithattu_game.slots import SlotManager


class GameLoop:
    """Fixed-timestep simulation loop running at 20 Hz with 60 fps rendering."""

    def __init__(
        self,
        env: KilithattuEnv,
        slot_manager: SlotManager,
        params: EnvParams,
        headless: bool = False,
    ) -> None:
        self.env = env
        self.slot_manager = slot_manager
        self.params = params
        self.headless = headless

        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        self.clock = pygame.time.Clock()
        self.renderer = None if self.headless else GameRenderer()
        self.running = False

    def run(self, max_ticks: int | None = None, seed: int = 42) -> dict[str, int]:
        """Execute the real-time simulation loop."""
        key = jax.random.PRNGKey(seed)
        k_reset, key = jax.random.split(key)
        obs, state = self.env.reset(k_reset, self.params)

        # Reset all controllers
        for slot in self.slot_manager.slots:
            ctrl = self.slot_manager.controllers.get(slot.slot_idx)
            if ctrl and hasattr(ctrl, "reset"):
                ctrl.reset(slot.slot_idx, seed)

        sim_dt = self.params.dt  # 0.05 s = 20 Hz
        accumulator = 0.0
        last_time = time.perf_counter()
        tick_count = 0
        self.running = True

        while self.running:
            current_time = time.perf_counter()
            frame_time = min(current_time - last_time, 0.25)
            last_time = current_time
            accumulator += frame_time

            # 1. Process Pygame Window Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_TAB:
                        # Switch human-controlled crosser
                        new_c = self.slot_manager.switch_human_crosser()
                        if self.renderer:
                            self.renderer.add_popup(f"Switched to Crosser {new_c + 1}", 3.0, 1.0)
                    else:
                        # Forward event to human controller
                        active_slot = self.slot_manager.human_crosser_idx
                        ctrl = self.slot_manager.controllers.get(active_slot)
                        if isinstance(ctrl, HumanController):
                            ctrl.handle_event(event)

            # 2. Fixed-timestep Simulation Step(s) at 20 Hz
            while accumulator >= sim_dt and self.running:
                accumulator -= sim_dt

                # Assemble joint actions from slot controllers
                joint_actions = self.slot_manager.get_joint_actions(obs)

                # Step JAX simulation
                k_step, key = jax.random.split(key)
                obs, state, r_A, r_D, events = self.env.step(k_step, state, joint_actions, self.params)

                # Trigger renderer popup notifications on events
                if self.renderer:
                    if bool(events.clapped):
                        self.renderer.add_popup("CLAP! Play Starts!", 3.0, 0.0, (255, 230, 80))
                    for i in range(self.params.num_crossers):
                        if bool(events.tagged[i]):
                            cx, cy = float(state.crossers.pos[i, 0]), float(state.crossers.pos[i, 1])
                            self.renderer.add_popup("OUT (TAGGED)!", cx, cy, (240, 60, 60))
                        elif bool(events.fouled[i]):
                            cx, cy = float(state.crossers.pos[i, 0]), float(state.crossers.pos[i, 1])
                            self.renderer.add_popup("FOUL! OUT!", cx, cy, (255, 80, 80))
                        elif bool(events.salt_delivered[i]):
                            self.renderer.add_popup("SALT DELIVERED! +6", 3.0, -1.0, (255, 215, 0))
                        elif int(events.line_crossed[i]) >= 0:
                            line_k = int(events.line_crossed[i])
                            text = "UPPU!" if line_k == self.params.num_rows else "+1"
                            cx, cy = float(state.crossers.pos[i, 0]), float(state.crossers.pos[i, 1])
                            self.renderer.add_popup(text, cx, cy, (100, 240, 140))

                tick_count += 1
                if max_ticks and tick_count >= max_ticks:
                    self.running = False
                if bool(state.global_state.done):
                    self.running = False

            # 3. Render at 60 fps
            if self.renderer and not self.headless:
                interp = accumulator / sim_dt
                self.renderer.render(
                    state, self.params, human_slot_idx=self.slot_manager.human_crosser_idx, interp_alpha=interp
                )
                self.clock.tick(60)
            elif self.headless and max_ticks:
                # In headless test mode, break if accumulated ticks reached
                if tick_count >= max_ticks:
                    self.running = False

        if not self.headless:
            pygame.quit()

        return {
            "score_attack": int(state.global_state.score_attack),
            "score_defence": int(state.global_state.score_defence),
            "ticks": tick_count,
        }
