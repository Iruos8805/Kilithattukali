"""Human controller translating keyboard and gamepad inputs to discrete factorized actions."""

from __future__ import annotations

import pygame
from kilithattu_core.env import Observations


class HumanController:
    """Translates keyboard (WASD / Arrows) and buttons into factorized discrete actions."""

    def __init__(self, is_crosser: bool = True) -> None:
        self.is_crosser = is_crosser
        self.latched_turn = False
        self.latched_clap = False

    def reset(self, slot_idx: int, seed: int) -> None:
        self.latched_turn = False
        self.latched_clap = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                self.latched_turn = True
            elif event.key == pygame.K_SPACE:
                self.latched_clap = True

    def act(self, obs: Observations, slot_idx: int) -> dict[str, int]:
        keys = pygame.key.get_pressed()

        up = keys[pygame.K_w] or keys[pygame.K_UP]
        down = keys[pygame.K_s] or keys[pygame.K_DOWN]
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        sprint = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        if self.is_crosser:
            # 8-direction quantisation: 0=Stay, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
            if up and right:
                move = 2
            elif down and right:
                move = 4
            elif down and left:
                move = 6
            elif up and left:
                move = 8
            elif up:
                move = 1
            elif right:
                move = 3
            elif down:
                move = 5
            elif left:
                move = 7
            else:
                move = 0

            return {"move": move, "sprint": 1 if sprint else 0, "call": 0}
        else:
            # Defender / Kili:
            # Guard: 0=Stay, 1=West, 2=East
            # Kili: 0=Stay, 1=N, 2=E, 3=S, 4=W
            if up:
                k_move = 1
            elif right:
                k_move = 2
            elif down:
                k_move = 3
            elif left:
                k_move = 4
            else:
                k_move = 0

            g_move = 2 if right else (1 if left else 0)

            turn_act = 1 if self.latched_turn else 0
            clap_act = 1 if self.latched_clap else 0
            self.latched_turn = False
            self.latched_clap = False

            return {
                "move": k_move if not self.is_crosser else g_move,
                "dash": 1 if sprint else 0,
                "turn": turn_act,
                "clap": clap_act,
            }
