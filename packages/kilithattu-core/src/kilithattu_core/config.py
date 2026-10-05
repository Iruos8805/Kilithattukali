"""Configuration schema, default parameters, and rule hash computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvParams:
    """Complete physical, topological, and rule parameters for Kilithattu Kali."""

    # Geometry
    num_lanes: int = 2
    num_rows: int = 5
    lane_width: float = 3.0
    row_depth: float = 3.0
    safe_zone_depth: float = 2.0

    # Team configuration
    num_crossers: int = 4
    num_guards: int = 4  # Default: num_rows - 1

    # Time and limits
    dt: float = 0.05  # 20 Hz
    innings_duration: float = 120.0  # 2400 live ticks
    r_tag: float = 0.6  # Tag reach radius in meters
    clap_time: float = 2.0  # Automatic clap time in Classic (seconds)

    # Classic speeds (m/s)
    classic_crosser_speed: float = 5.0
    classic_guard_speed: float = 4.2
    classic_kili_speed: float = 4.8

    # Kilithattu+ P1: Momentum & Stamina
    p1_momentum_stamina: bool = False
    crosser_walk_speed: float = 3.5
    crosser_sprint_speed: float = 6.0
    guard_walk_speed: float = 3.5
    guard_dash_speed: float = 5.0
    kili_walk_speed: float = 4.0
    kili_dash_speed: float = 5.5
    a_max: float = 20.0  # Maximum acceleration (m/s^2)
    stamina_drain: float = 0.35  # Drain per second while sprinting/dashing
    stamina_regen: float = 0.20  # Regen per second while walking/standing
    sprint_stamina_threshold: float = 0.10  # Minimum stamina required to sprint

    # Kilithattu+ P2: Facing & Turning
    p2_facing_turning: bool = False
    tag_cone_deg: float = 90.0  # Tag cone half-angle (+-90 degrees = 180 deg field)
    turn_time: float = 0.30  # Duration in seconds to execute a 180 deg turn

    # Kilithattu+ P3: Limited Vision (Fog of War)
    p3_limited_vision: bool = False
    crosser_fov_deg: float = 200.0
    crosser_fov_radius: float = 9.0
    guard_fov_deg: float = 180.0
    guard_fov_radius: float = 9.0
    kili_fov_deg: float = 360.0
    kili_fov_radius: float = 12.0

    # Kilithattu+ P4: Strategic Clap
    p4_strategic_clap: bool = False
    clap_window_min: float = 1.0
    clap_window_max: float = 5.0

    # Kilithattu+ P5: Procedural Fields
    p5_procedural_fields: bool = False

    # Kilithattu+ P6: Team Calls
    p6_team_calls: bool = False
    num_calls: int = 4
    call_period_ticks: int = 10  # Every 0.5s

    # Kilithattu+ P7: Human-Fair Reflexes
    p7_human_fair: bool = False
    obs_delay_ticks: int = 0
    sticky_action_prob: float = 0.0

    # Reward shaping
    shaping_beta: float = 0.0  # Potential-based shaping coefficient (0.0 in pure zero-sum evaluation)

    @property
    def field_width(self) -> float:
        return self.num_lanes * self.lane_width

    @property
    def field_length(self) -> float:
        return self.num_rows * self.row_depth

    @property
    def max_ticks(self) -> int:
        return int(self.innings_duration / self.dt)


def load_env_params(path_or_dict: str | Path | dict[str, Any] | None = None) -> EnvParams:
    """Load EnvParams from a YAML file, dictionary, or preset name."""
    if path_or_dict is None:
        return EnvParams()

    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if not p.exists():
            # Check configs/rules/<name>.yaml
            preset_path = Path("configs/rules") / f"{p.stem}.yaml"
            if preset_path.exists():
                p = preset_path
            else:
                raise FileNotFoundError(f"Config file not found: {path_or_dict}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = dict(path_or_dict)

    # Filter out metadata keys like 'name' or 'version'
    valid_fields = {f for f in EnvParams.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return EnvParams(**filtered)


def rules_hash(params: EnvParams) -> str:
    """Compute a deterministic SHA-256 hash of the rule parameters."""
    d = asdict(params)
    serialized = json.dumps(d, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"
