"""Tests for geometry, box lookups, and swept collision detection."""

import jax.numpy as jnp
import pytest

from kilithattu_core.config import EnvParams
from kilithattu_core.geometry import continuous_swept_tag, get_box_indices, is_in_safe_zone, is_side_exit


def test_box_indices():
    params = EnvParams(num_lanes=2, num_rows=5, lane_width=3.0, row_depth=3.0)

    # In home safe zone
    box = get_box_indices(jnp.array([1.5, -0.5]), params)
    assert (box == -1).all()

    # In far safe zone
    box = get_box_indices(jnp.array([1.5, 16.0]), params)
    assert (box == -1).all()

    # In box (0, 0)
    box = get_box_indices(jnp.array([1.0, 1.0]), params)
    assert (box == jnp.array([0, 0])).all()

    # In box (1, 4)
    box = get_box_indices(jnp.array([4.5, 13.5]), params)
    assert (box == jnp.array([1, 4])).all()

    # Exact boundary line y = 3.0 (belongs to higher index row 1)
    box = get_box_indices(jnp.array([1.5, 3.0]), params)
    assert (box == jnp.array([0, 1])).all()


def test_safe_zone():
    params = EnvParams(num_rows=5, row_depth=3.0)  # L = 15.0
    assert bool(is_in_safe_zone(jnp.array([1.0, -1.0]), params)) is True
    assert bool(is_in_safe_zone(jnp.array([1.0, 16.0]), params)) is True
    assert bool(is_in_safe_zone(jnp.array([1.0, 5.0]), params)) is False


def test_side_exit():
    params = EnvParams(num_lanes=2, num_rows=5, lane_width=3.0, row_depth=3.0)  # W=6, L=15
    # Normal in-bounds
    assert bool(is_side_exit(jnp.array([2.0, 5.0]), params)) is False
    # Stepped out on left (x < 0) while between L0 and L5
    assert bool(is_side_exit(jnp.array([-0.2, 5.0]), params)) is True
    # Stepped out on right (x > W)
    assert bool(is_side_exit(jnp.array([6.5, 5.0]), params)) is True
    # Stepped out while in home zone (y < 0) -> not a side exit
    assert bool(is_side_exit(jnp.array([-0.5, -1.0]), params)) is False


def test_continuous_swept_tag_no_tunneling():
    """Verify that high-speed crossers crossing a defender line cannot tunnel through."""
    r_tag = 0.6
    # Crosser flies from y = 2.0 to y = 4.0 at x = 3.0 in 1 tick (speed = 40 m/s!)
    p_c_start = jnp.array([3.0, 2.0])
    p_c_end = jnp.array([3.0, 4.0])

    # Guard sits at (3.0, 3.0)
    p_d_start = jnp.array([3.0, 3.0])
    p_d_end = jnp.array([3.0, 3.0])

    hit, _ = continuous_swept_tag(p_c_start, p_c_end, p_d_start, p_d_end, r_tag)
    assert bool(hit) is True

    # Crosser crosses at x = 5.0, guard at x = 1.0 (far away) -> no hit
    p_c_start_far = jnp.array([5.0, 2.0])
    p_c_end_far = jnp.array([5.0, 4.0])
    hit_far, _ = continuous_swept_tag(p_c_start_far, p_c_end_far, p_d_start, p_d_end, r_tag)
    assert bool(hit_far) is False
