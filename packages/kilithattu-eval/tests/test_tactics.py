"""Unit tests for tactical event detectors (E7)."""

import numpy as np
from kilithattu_eval.tactics import TacticsDetector


def test_tactics_detector():
    detector = TacticsDetector(window_ticks=20)

    # 4 crossers, 4 guards, 1 kili
    c_pos = np.array(
        [
            [3.0, 3.2],  # c0 near line 1 (y=3.0)
            [3.5, 3.3],  # c1 also near line 1 -> triggers split-guard!
            [1.0, 0.5],
            [2.0, 0.5],
        ]
    )
    c_act = np.array([True, True, True, True])

    g_pos = np.array(
        [
            [3.0, 3.0],  # guard on line 1
            [3.0, 6.0],
            [3.0, 9.0],
            [3.0, 12.0],
        ]
    )
    g_act = np.array([True, True, True, True])
    k_pos = np.array([3.0, 0.0])

    detector.update(
        tick=1,
        crosser_positions=c_pos,
        crosser_active=c_act,
        guard_positions=g_pos,
        guard_active=g_act,
        kili_pos=k_pos,
        events={"lines_crossed": [1, -1, -1, -1]},
    )

    summary = detector.counts.summary()
    assert summary["total_split_guards"] >= 1
    assert summary["total_ticks"] == 1
