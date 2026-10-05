"""Structured events generated during environment steps."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from chex import Array

from kilithattu_core.spaces import MAX_CROSSERS


class StepEvents(NamedTuple):
    """Container for discrete events occurring within a single simulation tick."""

    tagged: Array  # bool [MAX_CROSSERS]
    fouled: Array  # bool [MAX_CROSSERS]
    side_exited: Array  # bool [MAX_CROSSERS]
    line_crossed: Array  # int32 [MAX_CROSSERS] (-1 if none, else line index 0..n_B)
    salt_delivered: Array  # bool [MAX_CROSSERS]
    clapped: Array  # bool scalar
    points_attack: Array  # int32 scalar points awarded this tick
    false_started: Array  # bool [MAX_CROSSERS]

    @classmethod
    def empty(cls) -> StepEvents:
        return cls(
            tagged=jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_),
            fouled=jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_),
            side_exited=jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_),
            line_crossed=jnp.full((MAX_CROSSERS,), -1, dtype=jnp.int32),
            salt_delivered=jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_),
            clapped=jnp.array(False, dtype=jnp.bool_),
            points_attack=jnp.array(0, dtype=jnp.int32),
            false_started=jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_),
        )
