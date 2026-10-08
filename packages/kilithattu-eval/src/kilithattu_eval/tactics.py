"""Emergent tactical event detectors (§10.1, E7).

Detects:
- Split-guard: two crossers in the same row engage the same guard within 1 s (20 ticks).
- Decoy: a crosser's approach pulls a defender >= 2 m and a teammate crosses that line within 1 s.
- Kili bait: the Kili commits to a segment and a crosser crosses a line the Kili just left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TacticalEventCounts:
    split_guard_count: int = 0
    decoy_count: int = 0
    kili_bait_count: int = 0
    total_ticks: int = 0

    def summary(self) -> dict[str, float]:
        rate_scale = 20.0 / max(1, self.total_ticks)  # per simulated second
        return {
            "split_guard_per_sec": float(self.split_guard_count * rate_scale),
            "decoy_per_sec": float(self.decoy_count * rate_scale),
            "kili_bait_per_sec": float(self.kili_bait_count * rate_scale),
            "total_split_guards": float(self.split_guard_count),
            "total_decoys": float(self.decoy_count),
            "total_kili_baits": float(self.kili_bait_count),
            "total_ticks": float(self.total_ticks),
        }


class TacticsDetector:
    """Detects coordination and emergent tactics from trajectory histories."""

    def __init__(self, window_ticks: int = 20) -> None:
        self.window_ticks = window_ticks
        self.counts = TacticalEventCounts()
        self.recent_crosser_approaches: list[
            tuple[int, int, int]
        ] = []  # (tick, crosser_idx, line_idx)
        self.recent_kili_lines: list[tuple[int, int]] = []  # (tick, line_idx)

    def update(
        self,
        tick: int,
        crosser_positions: np.ndarray,  # [MAX_CROSSERS, 2]
        crosser_active: np.ndarray,  # [MAX_CROSSERS]
        guard_positions: np.ndarray,  # [MAX_GUARDS, 2]
        guard_active: np.ndarray,
        kili_pos: np.ndarray,  # [2]
        events: dict[str, Any] | None = None,
    ) -> None:
        self.counts.total_ticks += 1

        # 1. Split-guard detector: two crossers close in x and y to the same guard line
        for g_idx in range(len(guard_positions)):
            if not guard_active[g_idx]:
                continue
            g_pos = guard_positions[g_idx]
            close_crossers = []
            for c_idx in range(len(crosser_positions)):
                if not crosser_active[c_idx]:
                    continue
                c_pos = crosser_positions[c_idx]
                if abs(c_pos[1] - g_pos[1]) < 1.5:  # Within 1.5m of the guard line
                    close_crossers.append(c_idx)

            if len(close_crossers) >= 2:
                # Two crossers engaging the same guard in the same row
                self.counts.split_guard_count += 1

        # 2. Decoy detector & Kili bait tracking using line crossing events if provided
        if events and "lines_crossed" in events:
            lines_crossed = events["lines_crossed"]
            for c_idx, line_idx in enumerate(lines_crossed):
                if line_idx >= 0:
                    # Check if a teammate engaged recently (within window_ticks)
                    for t, other_c, l_idx in self.recent_crosser_approaches:
                        if tick - t <= self.window_ticks and other_c != c_idx and l_idx == line_idx:
                            self.counts.decoy_count += 1
                            break

                    # Check if Kili just left this line (within window_ticks)
                    for t, k_line in self.recent_kili_lines:
                        if tick - t <= self.window_ticks and k_line == line_idx:
                            self.counts.kili_bait_count += 1
                            break

        # Log current Kili approximate line
        current_kili_line = int(np.round(kili_pos[1] / 3.0))
        self.recent_kili_lines.append((tick, current_kili_line))
        if len(self.recent_kili_lines) > 50:
            self.recent_kili_lines.pop(0)
