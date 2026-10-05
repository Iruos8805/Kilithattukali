"""Curriculum stage definitions and automated promotion criteria (§6.4)."""

from __future__ import annotations

from dataclasses import dataclass

from kilithattu_core.config import EnvParams


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    name: str
    params: EnvParams
    description: str


CURRICULUM_STAGES: dict[str, CurriculumStage] = {
    "S0": CurriculumStage(
        stage_id="S0",
        name="Run Home",
        params=EnvParams(
            num_lanes=1, num_rows=2, num_crossers=1, num_guards=0, innings_duration=30.0
        ),
        description="1 crosser, no defenders. Learn to cross lines, become uppu, and return home.",
    ),
    "S1": CurriculumStage(
        stage_id="S1",
        name="One Line",
        params=EnvParams(
            num_lanes=1, num_rows=2, num_crossers=1, num_guards=1, innings_duration=30.0
        ),
        description="1 crosser vs 1 guard. Learn basic 1v1 gap seeking and timing.",
    ),
    "S2": CurriculumStage(
        stage_id="S2",
        name="Gauntlet",
        params=EnvParams(
            num_lanes=1, num_rows=5, num_crossers=1, num_guards=4, innings_duration=60.0
        ),
        description="1 crosser vs 4 guards. Learn multi-line gauntlet traversal.",
    ),
    "S3": CurriculumStage(
        stage_id="S3",
        name="Enter the Kili",
        params=EnvParams(
            num_lanes=2, num_rows=5, num_crossers=2, num_guards=4, innings_duration=60.0
        ),
        description="2 crossers vs 4 guards + Kili (foul rule disabled). Learn to handle the roaming Kili.",
    ),
    "S4": CurriculumStage(
        stage_id="S4",
        name="Full Teams No Foul",
        params=EnvParams(
            num_lanes=2, num_rows=5, num_crossers=4, num_guards=4, innings_duration=120.0
        ),
        description="4 vs 5, foul rule off. Full team coordination and defense coverage.",
    ),
    "S5": CurriculumStage(
        stage_id="S5",
        name="Classic Complete",
        params=EnvParams(
            num_lanes=2, num_rows=5, num_crossers=4, num_guards=4, innings_duration=120.0
        ),
        description="4 vs 5, foul rule enabled. Full traditional game rules.",
    ),
    "S6": CurriculumStage(
        stage_id="S6",
        name="Kilithattu+ Rules",
        params=EnvParams(
            num_lanes=2,
            num_rows=5,
            num_crossers=4,
            num_guards=4,
            p1_momentum_stamina=True,
            p2_facing_turning=True,
            p3_limited_vision=True,
            p4_strategic_clap=True,
        ),
        description="Plus rules with momentum, stamina, facing angles, and tactical clap.",
    ),
}


def get_stage(stage_id: str) -> CurriculumStage:
    """Retrieve curriculum stage by ID."""
    stage = CURRICULUM_STAGES.get(stage_id.upper())
    if stage is None:
        raise ValueError(
            f"Unknown curriculum stage: {stage_id}. Available: {list(CURRICULUM_STAGES.keys())}"
        )
    return stage
