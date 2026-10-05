"""Plain-Python reference implementation of Kilithattu Kali for differential testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kilithattu_core.config import EnvParams
from kilithattu_core.spaces import CROSSER_MOVE_VECTORS, GUARD_MOVE_VECTORS, KILI_MOVE_VECTORS


@dataclass
class ReferenceCrosser:
    x: float
    y: float
    status: str  # 'pacha', 'uppu', 'out'
    kappa: int
    trips: int
    active: bool


@dataclass
class ReferenceGuard:
    x: float
    line_idx: int
    active: bool


@dataclass
class ReferenceKili:
    x: float
    y: float
    line_type: str  # 'transverse', 'longitudinal'


class ReferenceKilithattuEnv:
    """Independent pure-Python step implementation used strictly for differential testing."""

    def __init__(self, params: EnvParams) -> None:
        self.params = params
        self.reset()

    def reset(self) -> None:
        p = self.params
        W = p.field_width
        self.time_elapsed = 0.0
        self.pre_clap_timer = 0.0
        self.clapped = False
        self.score_attack = 0
        self.score_defence = 0
        self.done = False

        self.crossers = [
            ReferenceCrosser(
                x=1.0 + i * 1.0,
                y=-1.0,
                status="pacha",
                kappa=0,
                trips=0,
                active=(i < p.num_crossers),
            )
            for i in range(p.num_crossers)
        ]
        self.guards = [
            ReferenceGuard(x=W / 2.0, line_idx=k + 1, active=(k < p.num_guards))
            for k in range(p.num_guards)
        ]
        self.kili = ReferenceKili(x=W / 2.0, y=0.0, line_type="transverse")

    def step(self, actions: dict[str, Any]) -> dict[str, Any]:
        """Perform one discrete step using pure-Python math."""
        p = self.params
        dt = p.dt
        d = p.row_depth
        W, L = p.field_width, p.field_length

        # 1. Clap timer
        if not self.clapped:
            self.pre_clap_timer += dt
            if self.pre_clap_timer >= p.clap_time:
                self.clapped = True
        else:
            self.time_elapsed += dt

        # 2. Movement
        c_actions = actions.get("crosser_move", [0] * len(self.crossers))
        g_actions = actions.get("guard_move", [0] * len(self.guards))
        k_action = actions.get("kili_move", 0)

        prev_c_pos = [(c.x, c.y) for c in self.crossers]

        # Move crossers
        for i, c in enumerate(self.crossers):
            if not c.active or c.status == "out":
                continue
            act = c_actions[i]
            dx, dy = CROSSER_MOVE_VECTORS[act]
            cand_x = c.x + dx * p.classic_crosser_speed * dt
            cand_y = c.y + dy * p.classic_crosser_speed * dt
            if not self.clapped and cand_y >= 0.0:
                cand_y = -0.05
            c.x = cand_x
            c.y = cand_y

        # Move guards
        prev_g_x = [g.x for g in self.guards]
        for j, g in enumerate(self.guards):
            if not g.active:
                continue
            act = g_actions[j]
            dx = GUARD_MOVE_VECTORS[act]
            g.x = max(0.0, min(W, g.x + dx * p.classic_guard_speed * dt))

        # Move Kili
        prev_k_pos = (self.kili.x, self.kili.y)
        k_dx, k_dy = KILI_MOVE_VECTORS[k_action]
        step_dist = p.classic_kili_speed * dt
        if self.kili.line_type == "transverse":
            self.kili.x = max(0.0, min(W, self.kili.x + k_dx * step_dist))
        else:
            self.kili.y = max(0.0, min(L, self.kili.y + k_dy * step_dist))

        # 3. Tags (Continuous swept test)
        tagged = [False] * len(self.crossers)
        if self.clapped:
            for i, c in enumerate(self.crossers):
                if not c.active or c.status == "out":
                    continue
                # Safe zone immunity
                if c.y < 0.0 or c.y > L:
                    continue

                c_start = prev_c_pos[i]
                c_end = (c.x, c.y)

                # Check guards
                for j, g in enumerate(self.guards):
                    if not g.active:
                        continue
                    gy = g.line_idx * d
                    g_start = (prev_g_x[j], gy)
                    g_end = (g.x, gy)
                    if self._swept_hit(c_start, c_end, g_start, g_end, p.r_tag):
                        tagged[i] = True

                # Check Kili
                if self._swept_hit(c_start, c_end, prev_k_pos, (self.kili.x, self.kili.y), p.r_tag):
                    tagged[i] = True

        for i, hit in enumerate(tagged):
            if hit:
                self.crossers[i].status = "out"

        # 4. Line crossings & points
        pts = 0
        if self.clapped:
            for i, c in enumerate(self.crossers):
                if not c.active or c.status == "out":
                    continue
                y0 = prev_c_pos[i][1]
                y1 = c.y

                # Check lines
                for line_k in range(p.num_rows + 1):
                    line_y = line_k * d
                    # Outbound
                    if line_k == 0 and y0 < 0.0 and y1 >= 0.0 and c.kappa == 0:
                        c.kappa = 1
                        pts += 1
                    elif line_k == p.num_rows and y0 < L and y1 >= L and c.kappa == p.num_rows:
                        c.kappa = p.num_rows + 1
                        c.status = "uppu"
                        pts += 1
                    elif (
                        1 <= line_k < p.num_rows
                        and y0 < line_y
                        and y1 >= line_y
                        and c.kappa == line_k
                    ):
                        c.kappa = line_k + 1
                        pts += 1

                    # Inbound
                    inbound_req = (p.num_rows + 1) + (p.num_rows - line_k)
                    if line_k == p.num_rows and y0 > L and y1 <= L and c.kappa == p.num_rows + 1:
                        c.kappa = p.num_rows + 2
                        pts += 1
                    elif (
                        line_k == 0
                        and y0 >= 0.0
                        and y1 < 0.0
                        and c.kappa == 2 * (p.num_rows + 1) - 1
                    ):
                        c.kappa = 0
                        c.trips += 1
                        c.status = "pacha"
                        pts += 7  # 1 crossing + 6 bonus
                    elif (
                        1 <= line_k < p.num_rows
                        and y0 > line_y
                        and y1 <= line_y
                        and c.kappa == inbound_req
                    ):
                        c.kappa = inbound_req + 1
                        pts += 1

        self.score_attack += pts

        # 5. Done check
        all_out = all(c.status == "out" for c in self.crossers if c.active)
        time_up = self.time_elapsed >= p.innings_duration
        self.done = (all_out or time_up) and self.clapped

        return {
            "score_attack": self.score_attack,
            "points_this_tick": pts,
            "done": self.done,
            "clapped": self.clapped,
        }

    def _swept_hit(self, p1_start, p1_end, p2_start, p2_end, r):
        d0_x = p1_start[0] - p2_start[0]
        d0_y = p1_start[1] - p2_start[1]
        disp_x = (p1_end[0] - p1_start[0]) - (p2_end[0] - p2_start[0])
        disp_y = (p1_end[1] - p1_start[1]) - (p2_end[1] - p2_start[1])

        denom = disp_x * disp_x + disp_y * disp_y
        numer = -(d0_x * disp_x + d0_y * disp_y)
        tau = 0.0 if denom < 1e-8 else max(0.0, min(1.0, numer / denom))

        rx = d0_x + tau * disp_x
        ry = d0_y + tau * disp_y
        return (rx * rx + ry * ry) <= (r * r)
