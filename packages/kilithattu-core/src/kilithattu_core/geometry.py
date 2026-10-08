"""Field geometry, box lookups, line topology, and continuous collision detection."""

from __future__ import annotations

import jax.numpy as jnp
from chex import Array

from kilithattu_core.config import EnvParams


def get_box_indices(pos: Array, params: EnvParams) -> Array:
    """Compute (lane_idx, row_idx) for a 2D position.

    Returns (-1, -1) if outside the playable field (e.g. in home or far safe zones).
    On interior boundaries, assigns to the higher index box.
    """
    x, y = pos[..., 0], pos[..., 1]
    w, d = params.lane_width, params.row_depth
    W, L = params.field_width, params.field_length

    in_field = (y >= 0.0) & (y <= L) & (x >= 0.0) & (x <= W)

    # Box coordinates with clamp to valid range
    j = jnp.floor(x / w).astype(jnp.int32)
    j = jnp.clip(j, 0, params.num_lanes - 1)

    k = jnp.floor(y / d).astype(jnp.int32)
    # When y == L (at line L_nB), belongs to row num_rows - 1
    k = jnp.where(y >= L, params.num_rows - 1, k)
    k = jnp.clip(k, 0, params.num_rows - 1)

    box_j = jnp.where(in_field, j, -1)
    box_k = jnp.where(in_field, k, -1)
    return jnp.stack([box_j, box_k], axis=-1)


def is_in_safe_zone(pos: Array, params: EnvParams) -> Array:
    """Check if a position is in Home zone (y < 0) or Far zone (y > L)."""
    y = pos[..., 1]
    L = params.field_length
    in_home = y < 0.0
    in_far = y > L
    return in_home | in_far


def is_side_exit(pos: Array, params: EnvParams) -> Array:
    """Check if crosser exited through field side boundaries (x < 0 or x > W) between L0 and L_nB."""
    x, y = pos[..., 0], pos[..., 1]
    W, L = params.field_width, params.field_length
    in_y_span = (y >= 0.0) & (y <= L)
    out_x = (x < 0.0) | (x > W)
    return in_y_span & out_x


def continuous_swept_tag(
    p_c_start: Array,
    p_c_end: Array,
    p_d_start: Array,
    p_d_end: Array,
    r_tag: float,
) -> tuple[Array, Array]:
    """Continuous swept-circle collision detection between crosser and defender over tick dt.

    Calculates minimum Euclidean distance between the two moving circles over tau in [0, 1].
    Returns (hit, p_c_at_tau).
    """
    delta_0 = p_c_start - p_d_start
    disp_c = p_c_end - p_c_start
    disp_d = p_d_end - p_d_start
    delta_disp = disp_c - disp_d

    denom = jnp.dot(delta_disp, delta_disp)
    numer = -jnp.dot(delta_0, delta_disp)

    # If relative motion is near zero, evaluate at tau = 0
    tau_star = jnp.where(denom > 1e-8, jnp.clip(numer / denom, 0.0, 1.0), 0.0)

    closest_rel_pos = delta_0 + tau_star * delta_disp
    min_dist_sq = jnp.dot(closest_rel_pos, closest_rel_pos)
    hit = min_dist_sq <= (r_tag * r_tag)
    p_c_at_tau = p_c_start + tau_star * disp_c
    return hit, p_c_at_tau


def point_line_distance(pos: Array, p1: Array, p2: Array) -> Array:
    """Perpendicular distance from a point to a line segment [p1, p2]."""
    line_vec = p2 - p1
    point_vec = pos - p1
    line_len_sq = jnp.dot(line_vec, line_vec)
    t = jnp.where(
        line_len_sq > 1e-8, jnp.clip(jnp.dot(point_vec, line_vec) / line_len_sq, 0.0, 1.0), 0.0
    )
    proj = p1 + t * line_vec
    diff = pos - proj
    return jnp.sqrt(jnp.dot(diff, diff))
