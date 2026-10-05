"""Kilithattu-Train: MAPPO, curriculum, and self-play league training."""

from kilithattu_train.curriculum import CURRICULUM_STAGES, CurriculumStage, get_stage
from kilithattu_train.mappo import compute_gae, ppo_loss
from kilithattu_train.rollout import collect_rollout

__version__ = "0.1.0"

__all__ = [
    "CURRICULUM_STAGES",
    "CurriculumStage",
    "compute_gae",
    "get_stage",
    "ppo_loss",
    "collect_rollout",
]
