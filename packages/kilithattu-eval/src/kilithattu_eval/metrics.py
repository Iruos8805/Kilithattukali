"""Evaluation metrics: Points Per Innings (PPI), salt delivery rate, and competitiveness."""

from __future__ import annotations

import numpy as np


def compute_innings_metrics(
    scores_attack: list[int] | np.ndarray,
    deliveries: list[int] | np.ndarray,
    all_out_flags: list[bool] | np.ndarray,
) -> dict[str, float]:
    """Compute summary statistics for a series of innings."""
    scores = np.array(scores_attack, dtype=np.float32)
    delivs = np.array(deliveries, dtype=np.float32)
    all_outs = np.array(all_out_flags, dtype=np.float32)

    n = len(scores)
    if n == 0:
        return {"ppi": 0.0, "deliveries_per_innings": 0.0, "all_out_rate": 0.0}

    mean_ppi = float(np.mean(scores))
    std_ppi = float(np.std(scores))
    mean_delivs = float(np.mean(delivs))
    all_out_rate = float(np.mean(all_outs))

    return {
        "ppi": mean_ppi,
        "ppi_std": std_ppi,
        "deliveries_per_innings": mean_delivs,
        "all_out_rate": all_out_rate,
    }
