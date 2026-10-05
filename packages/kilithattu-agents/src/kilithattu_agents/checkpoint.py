"""Checkpoint metadata schema and serialization utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckpointMetadata:
    run_id: str
    step: int
    side: str  # 'attack' or 'defence'
    role_heads: list[str]
    git_commit: str
    env_version: str
    rules_hash: str
    config: dict[str, Any]
    elo: int = 1000
    network: dict[str, Any] | None = None


def save_checkpoint_metadata(output_dir: str | Path, metadata: CheckpointMetadata) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
    return meta_path


def load_checkpoint_metadata(checkpoint_dir: str | Path) -> CheckpointMetadata:
    meta_path = Path(checkpoint_dir) / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CheckpointMetadata(**data)
