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


def save_model_checkpoint(
    checkpoint_dir: str | Path,
    params: Any,
    metadata: CheckpointMetadata,
) -> Path:
    """Save model weights and metadata to checkpoint directory."""
    import pickle
    import jax

    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint_metadata(out, metadata)

    weights_path = out / "model.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(jax.device_get(params), f)

    try:
        from flax import serialization
        msgpack_path = out / "model.msgpack"
        with open(msgpack_path, "wb") as f:
            f.write(serialization.to_bytes(params))
    except Exception:
        pass

    return out


def load_model_checkpoint(
    checkpoint_dir: str | Path,
) -> tuple[Any, CheckpointMetadata]:
    """Load model parameters and metadata from checkpoint directory."""
    import pickle

    out = Path(checkpoint_dir)
    metadata = load_checkpoint_metadata(out)

    weights_path = out / "model.pkl"
    if weights_path.exists():
        with open(weights_path, "rb") as f:
            params = pickle.load(f)
        return params, metadata

    msgpack_path = out / "model.msgpack"
    if msgpack_path.exists():
        from flax import serialization
        with open(msgpack_path, "rb") as f:
            raw_bytes = f.read()
        return raw_bytes, metadata

    raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
