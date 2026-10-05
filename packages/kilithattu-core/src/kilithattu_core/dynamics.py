"""Physics dynamics, acceleration limits, stamina, line clamping, and Kili graph traversal."""

from __future__ import annotations

import jax.numpy as jnp
from chex import Array

from kilithattu_core.config import EnvParams
from kilithattu_core.spaces import CROSSER_MOVE_VECTORS, GUARD_MOVE_VECTORS, KILI_MOVE_VECTORS
from kilithattu_core.state import STATUS_OUT, CrosserState, GuardState, KiliState


def update_crosser_dynamics(
    crossers: CrosserState,
    move_actions: Array,  # int32 [MAX_CROSSERS]
    sprint_actions: Array,  # int32 [MAX_CROSSERS]
    params: EnvParams,
) -> tuple[Array, Array, Array, Array]:
    """Integrate velocities, stamina, and positions for crossers over tick dt.

    Returns:
        (new_pos, new_vel, new_stamina, new_facing)
    """
    move_vecs = jnp.array(CROSSER_MOVE_VECTORS, dtype=jnp.float32)  # [9, 2]
    dirs = move_vecs[move_actions]  # [MAX_CROSSERS, 2]

    # Speeds and stamina under Classic vs P1
    if not params.p1_momentum_stamina:
        speeds = jnp.full((crossers.pos.shape[0],), params.classic_crosser_speed, dtype=jnp.float32)
        target_vel = dirs * speeds[:, None]
        new_vel = target_vel
        new_stamina = crossers.stamina
    else:
        can_sprint = (crossers.stamina >= params.sprint_stamina_threshold) & (sprint_actions == 1)
        speeds = jnp.where(can_sprint, params.crosser_sprint_speed, params.crosser_walk_speed)
        target_vel = dirs * speeds[:, None]

        # Acceleration limiting
        max_delta_v = params.a_max * params.dt
        vel_diff = target_vel - crossers.vel
        diff_norm = jnp.linalg.norm(vel_diff, axis=-1, keepdims=True)
        scale = jnp.where(diff_norm > max_delta_v, max_delta_v / jnp.maximum(diff_norm, 1e-6), 1.0)
        new_vel = crossers.vel + vel_diff * scale

        # Stamina update
        is_moving = jnp.linalg.norm(dirs, axis=-1) > 0.0
        drain = jnp.where(is_moving & can_sprint, params.stamina_drain * params.dt, 0.0)
        regen = jnp.where(~(is_moving & can_sprint), params.stamina_regen * params.dt, 0.0)
        new_stamina = jnp.clip(crossers.stamina - drain + regen, 0.0, 1.0)

    # Inactive or out crossers do not move
    is_active_and_alive = crossers.active_mask & (crossers.status != STATUS_OUT)
    effective_vel = jnp.where(is_active_and_alive[:, None], new_vel, 0.0)

    new_pos = crossers.pos + effective_vel * params.dt

    # Update facing if moving
    facing_diff = jnp.where(
        jnp.linalg.norm(effective_vel, axis=-1, keepdims=True) > 0.1,
        effective_vel / jnp.maximum(jnp.linalg.norm(effective_vel, axis=-1, keepdims=True), 1e-6),
        crossers.facing,
    )

    return new_pos, effective_vel, new_stamina, facing_diff


def update_guard_dynamics(
    guards: GuardState,
    move_actions: Array,  # int32 [MAX_GUARDS]
    dash_actions: Array,  # int32 [MAX_GUARDS]
    turn_actions: Array,  # int32 [MAX_GUARDS]
    params: EnvParams,
) -> tuple[Array, Array, Array, Array, Array]:
    """Integrate 1D line movement along transverse lines for guards.

    Returns:
        (new_x, new_vx, new_facing, new_turn_timer, new_stamina)
    """
    move_vecs = jnp.array(GUARD_MOVE_VECTORS, dtype=jnp.float32)  # [3]
    dirs = move_vecs[move_actions]  # [MAX_GUARDS]

    if not params.p1_momentum_stamina:
        speeds = jnp.full((guards.x.shape[0],), params.classic_guard_speed, dtype=jnp.float32)
        new_vx = dirs * speeds
        new_stamina = guards.stamina
    else:
        can_dash = (guards.stamina >= params.sprint_stamina_threshold) & (dash_actions == 1)
        speeds = jnp.where(can_dash, params.guard_dash_speed, params.guard_walk_speed)
        new_vx = dirs * speeds

        is_moving = jnp.abs(dirs) > 0.0
        drain = jnp.where(is_moving & can_dash, params.stamina_drain * params.dt, 0.0)
        regen = jnp.where(~(is_moving & can_dash), params.stamina_regen * params.dt, 0.0)
        new_stamina = jnp.clip(guards.stamina - drain + regen, 0.0, 1.0)

    # Inactive guards do not move
    effective_vx = jnp.where(guards.active_mask, new_vx, 0.0)
    new_x = jnp.clip(guards.x + effective_vx * params.dt, 0.0, params.field_width)

    # P2 facing and turning
    if params.p2_facing_turning:
        is_turning = guards.turn_timer > 0.0
        new_turn_timer = jnp.maximum(0.0, guards.turn_timer - params.dt)

        # Trigger turn if requested and not already turning
        trigger_turn = (turn_actions == 1) & (~is_turning) & guards.active_mask
        new_facing = jnp.where(trigger_turn, 1 - guards.facing, guards.facing)
        new_turn_timer = jnp.where(trigger_turn, params.turn_time, new_turn_timer)
    else:
        new_facing = guards.facing
        new_turn_timer = guards.turn_timer

    return new_x, effective_vx, new_facing, new_turn_timer, new_stamina


def update_kili_dynamics(
    kili: KiliState,
    move_action: Array,  # int32 scalar (0=stay, 1=N, 2=E, 3=S, 4=W)
    dash_action: Array,  # int32 scalar
    turn_action: Array,  # int32 scalar
    params: EnvParams,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Integrate Kili graph movement along lines with junction snapping.

    Returns:
        (new_pos, new_vel, new_facing, new_turn_timer, new_stamina, new_line_type, new_line_idx)
    """
    move_vecs = jnp.array(KILI_MOVE_VECTORS, dtype=jnp.float32)  # [5, 2]
    desired_dir = move_vecs[move_action]  # [2]

    # Speeds
    if not params.p1_momentum_stamina:
        speed = params.classic_kili_speed
        new_stamina = kili.stamina
    else:
        can_dash = (kili.stamina >= params.sprint_stamina_threshold) & (dash_action == 1)
        speed = jnp.where(can_dash, params.kili_dash_speed, params.kili_walk_speed)
        is_moving = jnp.linalg.norm(desired_dir) > 0.0
        drain = jnp.where(is_moving & can_dash, params.stamina_drain * params.dt, 0.0)
        regen = jnp.where(~(is_moving & can_dash), params.stamina_regen * params.dt, 0.0)
        new_stamina = jnp.clip(kili.stamina - drain + regen, 0.0, 1.0)

    step_dist = speed * params.dt
    W, L = params.field_width, params.field_length
    w, d = params.lane_width, params.row_depth

    # Current state
    x, y = kili.pos[0], kili.pos[1]
    is_transverse = (
        kili.current_line_type == 0
    )  # 0: transverse (constant y), 1: longitudinal (constant x)

    # 1. Moving along current line
    # On transverse line, allowed along-line moves are E (move_action=2) and W (move_action=4)
    # Perpendicular moves are N (move_action=1) and S (move_action=3)
    # Nearest junction on transverse line is at nearest longitudinal line x_j = round(x / w) * w
    nearest_j = jnp.clip(jnp.round(x / w), 0, params.num_lanes).astype(jnp.int32)
    nearest_x = nearest_j.astype(jnp.float32) * w
    dist_to_junc_x = jnp.abs(x - nearest_x)

    # On longitudinal line, allowed along-line moves are N (1) and S (3)
    # Perpendicular moves are E (2) and W (4)
    # Nearest junction on longitudinal line is at nearest transverse line y_k = round(y / d) * d
    nearest_k = jnp.clip(jnp.round(y / d), 0, params.num_rows).astype(jnp.int32)
    nearest_y = nearest_k.astype(jnp.float32) * d
    dist_to_junc_y = jnp.abs(y - nearest_y)

    # Case A: On transverse line
    # Moving along-line:
    dx_along = desired_dir[0] * step_dist
    pos_along_trans = jnp.array([jnp.clip(x + dx_along, 0.0, W), y])

    # Perpendicular move on transverse line (N or S): allowed if dist_to_junc_x <= step_dist
    can_switch_from_trans = (dist_to_junc_x <= step_dist * 1.5) & (jnp.abs(desired_dir[1]) > 0.0)
    rem_dist_y = jnp.maximum(0.0, step_dist - dist_to_junc_x)
    dy_perp = desired_dir[1] * rem_dist_y
    pos_switch_from_trans = jnp.array([nearest_x, jnp.clip(y + dy_perp, 0.0, L)])

    # Case B: On longitudinal line
    # Moving along-line (N or S):
    dy_along = desired_dir[1] * step_dist
    pos_along_long = jnp.array([x, jnp.clip(y + dy_along, 0.0, L)])

    # Perpendicular move on longitudinal line (E or W): allowed if dist_to_junc_y <= step_dist
    can_switch_from_long = (dist_to_junc_y <= step_dist * 1.5) & (jnp.abs(desired_dir[0]) > 0.0)
    rem_dist_x = jnp.maximum(0.0, step_dist - dist_to_junc_y)
    dx_perp = desired_dir[0] * rem_dist_x
    pos_switch_from_long = jnp.array([jnp.clip(x + dx_perp, 0.0, W), nearest_y])

    # Select branch based on current line type and switch condition
    new_pos = jnp.where(
        is_transverse,
        jnp.where(can_switch_from_trans, pos_switch_from_trans, pos_along_trans),
        jnp.where(can_switch_from_long, pos_switch_from_long, pos_along_long),
    )

    new_line_type = jnp.where(
        is_transverse,
        jnp.where(can_switch_from_trans, 1, 0),
        jnp.where(can_switch_from_long, 0, 1),
    )

    new_line_idx = jnp.where(
        is_transverse,
        jnp.where(can_switch_from_trans, nearest_j, kili.current_line_idx),
        jnp.where(can_switch_from_long, nearest_k, kili.current_line_idx),
    )

    new_vel = (new_pos - kili.pos) / params.dt

    # P2 facing / turning
    new_facing = kili.facing
    new_turn_timer = kili.turn_timer
    if params.p2_facing_turning:
        is_turning = kili.turn_timer > 0.0
        new_turn_timer = jnp.maximum(0.0, kili.turn_timer - params.dt)
        trigger_turn = (turn_action == 1) & (~is_turning)
        new_facing = jnp.where(trigger_turn, (kili.facing + 1) % 4, kili.facing)
        new_turn_timer = jnp.where(trigger_turn, params.turn_time, new_turn_timer)

    return new_pos, new_vel, new_facing, new_turn_timer, new_stamina, new_line_type, new_line_idx
