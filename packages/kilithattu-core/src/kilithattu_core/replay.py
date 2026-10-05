"""Replay format reader, writer, and deterministic verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp

from kilithattu_core.config import EnvParams, rules_hash
from kilithattu_core.env import JointActions, KilithattuEnv
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS


def write_replay(
    output_dir: str | Path,
    header: dict[str, Any],
    actions: np.ndarray,
    events: list[dict[str, Any]] | None = None,
) -> Path:
    """Save a match replay package (.kreplay directory).

    Structure:
        output_dir/
            header.json
            actions.npz
            events.jsonl
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    header_file = out / "header.json"
    with open(header_file, "w", encoding="utf-8") as f:
        json.dump(header, f, indent=2)

    actions_file = out / "actions.npz"
    np.savez_compressed(actions_file, actions=actions)

    if events is not None:
        events_file = out / "events.jsonl"
        with open(events_file, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    return out


def read_replay(replay_dir: str | Path) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Load a replay package from disk."""
    r_dir = Path(replay_dir)
    if not r_dir.exists():
        raise FileNotFoundError(f"Replay directory not found: {replay_dir}")

    with open(r_dir / "header.json", "r", encoding="utf-8") as f:
        header = json.load(f)

    with np.load(r_dir / "actions.npz") as data:
        actions = data["actions"]

    events = []
    events_file = r_dir / "events.jsonl"
    if events_file.exists():
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

    return header, actions, events


def verify_replay(replay_dir: str | Path) -> bool:
    """Re-simulate a replay from seed and assert deterministic event reproduction."""
    header, actions, recorded_events = read_replay(replay_dir)
    params_dict = header.get("params", {})
    params = EnvParams(**params_dict)
    env = KilithattuEnv(params)

    seed = header.get("seed", 0)
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key, params)

    num_ticks = actions.shape[0]
    for t in range(num_ticks):
        # Actions shape: [num_ticks, MAX_CROSSERS + MAX_GUARDS + 1, num_heads]
        act_t = actions[t]
        joint_actions = JointActions(
            crosser_move=jnp.array(act_t[:MAX_CROSSERS, 0], dtype=jnp.int32),
            crosser_sprint=jnp.array(act_t[:MAX_CROSSERS, 1], dtype=jnp.int32),
            crosser_call=jnp.array(act_t[:MAX_CROSSERS, 2], dtype=jnp.int32),
            guard_move=jnp.array(act_t[MAX_CROSSERS : MAX_CROSSERS + MAX_GUARDS, 0], dtype=jnp.int32),
            guard_dash=jnp.array(act_t[MAX_CROSSERS : MAX_CROSSERS + MAX_GUARDS, 1], dtype=jnp.int32),
            guard_turn=jnp.array(act_t[MAX_CROSSERS : MAX_CROSSERS + MAX_GUARDS, 2], dtype=jnp.int32),
            kili_move=jnp.array(act_t[-1, 0], dtype=jnp.int32),
            kili_dash=jnp.array(act_t[-1, 1], dtype=jnp.int32),
            kili_turn=jnp.array(act_t[-1, 2], dtype=jnp.int32),
            kili_clap=jnp.array(act_t[-1, 3], dtype=jnp.int32),
        )

        obs, state, r_A, r_D, events = env.step(key, state, joint_actions, params)

        # Check recorded points if available
        if t < len(recorded_events):
            rec = recorded_events[t]
            if "points_attack" in rec:
                sim_pts = int(events.points_attack)
                rec_pts = int(rec["points_attack"])
                if sim_pts != rec_pts:
                    return False

    return True
