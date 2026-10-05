"""Top-down Pygame renderer with Onam-inspired visual aesthetics and HUD."""

from __future__ import annotations

import math
from typing import Any
import pygame
from kilithattu_core.config import EnvParams
from kilithattu_core.state import EnvState, STATUS_OUT, STATUS_PACHA, STATUS_UPPU


class GameRenderer:
    """Renders the Kilithattu field, animated players, events, and HUD at 60 fps."""

    def __init__(self, screen_width: int = 960, screen_height: int = 800) -> None:
        self.width = screen_width
        self.height = screen_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Kilithattu Kali (കിളിത്തട്ടുകളി) — Onam Game")

        # Fonts
        pygame.font.init()
        self.font_large = pygame.font.SysFont("DejaVu Sans, Helvetica, Arial", 26, bold=True)
        self.font_medium = pygame.font.SysFont("DejaVu Sans, Helvetica, Arial", 18, bold=True)
        self.font_small = pygame.font.SysFont("DejaVu Sans, Helvetica, Arial", 14)

        # Field viewport transformation
        # Field is W=6m, L=15m (+ 2m safe zones on top and bottom = 19m total)
        self.margin_top = 80
        self.margin_bottom = 40
        self.margin_left = 200
        self.margin_right = 200

        self.play_h = self.height - self.margin_top - self.margin_bottom
        self.scale_y = self.play_h / 19.0  # 19m total
        self.scale_x = self.scale_y  # Square aspect ratio

        self.field_w_pixels = 6.0 * self.scale_x
        self.origin_x = (self.width - self.field_w_pixels) / 2.0
        # y=0 is at bottom of field, just above Home zone (which is y in [-2, 0))
        self.origin_y = self.margin_top + 17.0 * self.scale_y

        # Floating popup notifications
        self.popups: list[dict[str, Any]] = []

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates (meters) to screen pixel coordinates."""
        # World: x in [0, W], y in [-2, 17]
        # In screen coords: y=0 is top, so higher world y means lower screen y
        sx = int(self.origin_x + x * self.scale_x)
        sy = int(self.origin_y - y * self.scale_y)
        return sx, sy

    def add_popup(self, text: str, x: float, y: float, color: tuple[int, int, int] = (255, 230, 80)) -> None:
        self.popups.append({"text": text, "x": x, "y": y, "color": color, "ttl": 40, "max_ttl": 40})

    def render(
        self,
        state: EnvState,
        params: EnvParams,
        human_slot_idx: int = 0,
        interp_alpha: float = 1.0,
    ) -> None:
        """Render a single frame."""
        # Background: Warm Kerala sand / terracotta
        self.screen.fill((244, 233, 215))

        W, L = params.field_width, params.field_length
        z = params.safe_zone_depth
        d = params.row_depth
        w = params.lane_width

        # 1. Draw Safe Zones
        # Home safe zone (y in [-z, 0))
        hx, hy = self.world_to_screen(0.0, 0.0)
        hw = int(W * self.scale_x)
        hh = int(z * self.scale_y)
        home_rect = pygame.Rect(hx, hy, hw, hh)
        pygame.draw.rect(self.screen, (220, 245, 225), home_rect)
        pygame.draw.rect(self.screen, (160, 210, 175), home_rect, 2)

        home_label = self.font_small.render("HOME ZONE (Safe) — PACHA", True, (60, 130, 80))
        self.screen.blit(home_label, (home_rect.centerx - home_label.get_width() // 2, home_rect.centery - 8))

        # Far safe zone (y in (L, L + z])
        fx, fy = self.world_to_screen(0.0, L + z)
        far_rect = pygame.Rect(fx, fy, hw, hh)
        pygame.draw.rect(self.screen, (225, 240, 255), far_rect)
        pygame.draw.rect(self.screen, (160, 195, 240), far_rect, 2)

        far_label = self.font_small.render("FAR ZONE (Safe) → UPPU", True, (50, 100, 180))
        self.screen.blit(far_label, (far_rect.centerx - far_label.get_width() // 2, far_rect.centery - 8))

        # 2. Draw Court Boxes & Alternating Tints
        for r in range(params.num_rows):
            for c in range(params.num_lanes):
                bx, by = self.world_to_screen(c * w, (r + 1) * d)
                bw = int(w * self.scale_x)
                bh = int(d * self.scale_y)
                box_rect = pygame.Rect(bx, by, bw, bh)
                tint = (248, 238, 222) if (r + c) % 2 == 0 else (241, 230, 210)
                pygame.draw.rect(self.screen, tint, box_rect)
                pygame.draw.rect(self.screen, (215, 200, 180), box_rect, 1)

        # 3. Transverse and Longitudinal Boundary Lines
        # Transverse lines L0..L_nB
        for k in range(params.num_rows + 1):
            line_y = k * d
            p1 = self.world_to_screen(0.0, line_y)
            p2 = self.world_to_screen(W, line_y)
            is_outer = (k == 0 or k == params.num_rows)
            color = (180, 50, 30) if is_outer else (255, 255, 255)
            thickness = 4 if is_outer else 3
            pygame.draw.line(self.screen, color, p1, p2, thickness)

            # Label line
            lbl = self.font_small.render(f"L{k}", True, (130, 90, 70))
            self.screen.blit(lbl, (p1[0] - 28, p1[1] - 8))

        # Longitudinal lines
        for j in range(params.num_lanes + 1):
            line_x = j * w
            p1 = self.world_to_screen(line_x, 0.0)
            p2 = self.world_to_screen(line_x, L)
            is_side = (j == 0 or j == params.num_lanes)
            color = (180, 50, 30) if is_side else (240, 240, 240)
            thickness = 4 if is_side else 3
            pygame.draw.line(self.screen, color, p1, p2, thickness)

        # 4. Junctions (Nodes where Kili can turn)
        for k in range(params.num_rows + 1):
            for j in range(params.num_lanes + 1):
                jx, jy = self.world_to_screen(j * w, k * d)
                pygame.draw.circle(self.screen, (210, 140, 40), (jx, jy), 4)

        # 5. Render Guards
        guards = state.guards
        for g_idx in range(params.num_guards):
            if bool(guards.active_mask[g_idx]):
                gy = float(guards.line_idx[g_idx]) * d
                gx = float(guards.x[g_idx])
                sx, sy = self.world_to_screen(gx, gy)

                # Guard body: Amber circle with direction indicator
                pygame.draw.circle(self.screen, (235, 130, 20), (sx, sy), 13)
                pygame.draw.circle(self.screen, (170, 80, 0), (sx, sy), 13, 2)

                # Guard label
                lbl = self.font_small.render(f"G{g_idx + 1}", True, (255, 255, 255))
                self.screen.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))

                # Tag Reach Circle
                reach_px = int(params.r_tag * self.scale_x)
                pygame.draw.circle(self.screen, (240, 140, 40), (sx, sy), reach_px, 1)

        # 6. Render Kili
        kili = state.kili
        kx, ky = float(kili.pos[0]), float(kili.pos[1])
        ksx, ksy = self.world_to_screen(kx, ky)

        # Kili body: Crimson & Gold Leader Star
        reach_px = int(params.r_tag * self.scale_x)
        pygame.draw.circle(self.screen, (200, 30, 40), (ksx, ksy), 16)
        pygame.draw.circle(self.screen, (255, 215, 0), (ksx, ksy), 16, 3)
        pygame.draw.circle(self.screen, (255, 100, 100), (ksx, ksy), reach_px, 1)

        kili_lbl = self.font_small.render("KILI", True, (255, 255, 255))
        self.screen.blit(kili_lbl, (ksx - kili_lbl.get_width() // 2, ksy - kili_lbl.get_height() // 2))

        # 7. Render Crossers
        crossers = state.crossers
        for c_idx in range(params.num_crossers):
            if bool(crossers.active_mask[c_idx]):
                cx = float(crossers.pos[c_idx, 0])
                cy = float(crossers.pos[c_idx, 1])
                csx, csy = self.world_to_screen(cx, cy)
                status = int(crossers.status[c_idx])

                is_human = (c_idx == human_slot_idx)

                # Pulsing golden halo for human-controlled crosser
                if is_human:
                    pygame.draw.circle(self.screen, (255, 220, 0), (csx, csy), 20, 3)

                if status == STATUS_OUT:
                    # Grey cross
                    pygame.draw.circle(self.screen, (150, 150, 155), (csx, csy), 11)
                    pygame.draw.line(self.screen, (230, 70, 70), (csx - 7, csy - 7), (csx + 7, csy + 7), 2)
                    pygame.draw.line(self.screen, (230, 70, 70), (csx - 7, csy + 7), (csx + 7, csy - 7), 2)
                elif status == STATUS_UPPU:
                    # White Salt Diamond / Bright Cyan
                    pygame.draw.circle(self.screen, (245, 250, 255), (csx, csy), 13)
                    pygame.draw.circle(self.screen, (70, 160, 240), (csx, csy), 13, 2)
                    lbl = self.font_small.render("U", True, (30, 100, 200))
                    self.screen.blit(lbl, (csx - lbl.get_width() // 2, csy - lbl.get_height() // 2))
                else:
                    # STATUS_PACHA: Vibrant Leaf Green
                    pygame.draw.circle(self.screen, (40, 175, 75), (csx, csy), 13)
                    pygame.draw.circle(self.screen, (20, 110, 45), (csx, csy), 13, 2)
                    lbl = self.font_small.render(f"P{c_idx + 1}", True, (255, 255, 255))
                    self.screen.blit(lbl, (csx - lbl.get_width() // 2, csy - lbl.get_height() // 2))

        # 8. Render Floating Popups
        alive_popups = []
        for p in self.popups:
            p["ttl"] -= 1
            if p["ttl"] > 0:
                sx, sy = self.world_to_screen(p["x"], p["y"])
                sy -= int((p["max_ttl"] - p["ttl"]) * 0.8)
                surf = self.font_medium.render(p["text"], True, p["color"])
                self.screen.blit(surf, (sx - surf.get_width() // 2, sy))
                alive_popups.append(p)
        self.popups = alive_popups

        # 9. Top HUD Bar (Sleek Glassmorphic Header)
        hud_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(self.screen, (32, 38, 48), hud_rect)
        pygame.draw.line(self.screen, (218, 165, 32), (0, 60), (self.width, 60), 2)

        g = state.global_state
        score_text = f"ATTACK: {int(g.score_attack)}   |   DEFENCE: {int(g.score_defence)}"
        score_surf = self.font_large.render(score_text, True, (255, 220, 100))
        self.screen.blit(score_surf, (30, 15))

        time_left = max(0.0, params.innings_duration - float(g.time_elapsed))
        time_text = f"CLOCK: {time_left:.1f}s"
        time_surf = self.font_medium.render(time_text, True, (230, 240, 255))
        self.screen.blit(time_surf, (self.width - 200, 18))

        clap_status = "LIVE PLAY" if bool(g.clapped) else f"WAITING FOR CLAP ({float(g.pre_clap_timer):.1f}s)"
        clap_color = (80, 230, 120) if bool(g.clapped) else (255, 160, 60)
        clap_surf = self.font_small.render(clap_status, True, clap_color)
        self.screen.blit(clap_surf, (self.width // 2 - clap_surf.get_width() // 2, 22))

        # Bottom Instructions
        controls_text = "Controls: WASD=Move | Shift=Sprint | Tab=Switch Crosser | Q=Turn | Space=Clap | Esc=Menu"
        controls_surf = self.font_small.render(controls_text, True, (120, 110, 100))
        self.screen.blit(controls_surf, (self.width // 2 - controls_surf.get_width() // 2, self.height - 25))

        pygame.display.flip()
