"""Observation token builders, fog-of-war masking, action masks, and centralised critic representations."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array

from kilithattu_core.config import EnvParams
from kilithattu_core.spaces import (
    GLOBAL_TOKEN_DIM,
    LINE_TOKEN_DIM,
    MAX_CROSSERS,
    MAX_GUARDS,
    MAX_LINES,
    MAX_PLAYERS,
    MAX_TRANSVERSE_LINES,
    PLAYER_TOKEN_DIM,
)
from kilithattu_core.state import (
    STATUS_OUT,
    EnvState,
)


class AgentObservation(NamedTuple):
    """Observation container for a single agent."""

    player_tokens: Array  # float32 [MAX_PLAYERS, PLAYER_TOKEN_DIM]
    player_mask: Array  # bool [MAX_PLAYERS] (valid existence mask)
    line_tokens: Array  # float32 [MAX_LINES, LINE_TOKEN_DIM]
    line_mask: Array  # bool [MAX_LINES]
    global_token: Array  # float32 [GLOBAL_TOKEN_DIM]
    action_mask_move: Array  # bool [N_actions_move]
    action_mask_dash: Array  # bool [2]
    action_mask_turn: Array  # bool [2] (or call for crossers)


class TeamObservations(NamedTuple):
    """All agent observations for both teams."""

    crossers: list[AgentObservation]  # Up to MAX_CROSSERS
    guards: list[AgentObservation]  # Up to MAX_GUARDS
    kili: AgentObservation


def build_line_tokens(params: EnvParams) -> tuple[Array, Array]:
    """Construct static line tokens for the field geometry.

    Returns:
        (line_tokens [MAX_LINES, 8], line_mask [MAX_LINES])
    """
    W, L = params.field_width, params.field_length
    d = params.row_depth
    w = params.lane_width

    tokens = jnp.zeros((MAX_LINES, LINE_TOKEN_DIM), dtype=jnp.float32)
    mask = jnp.zeros((MAX_LINES,), dtype=jnp.bool_)

    # Transverse lines L0..L_nB
    for k in range(params.num_rows + 1):
        y_norm = (float(k) * d) / max(L, 1e-5)
        # Features: [is_trans=1, is_long=0, x1=0, y1, x2=1, y2, guard_assigned, k_norm]
        guard_assigned = 1.0 if (1 <= k < params.num_rows) else 0.0
        tok = jnp.array([1.0, 0.0, 0.0, y_norm, 1.0, y_norm, guard_assigned, float(k) / 10.0])
        tokens = tokens.at[k].set(tok)
        mask = mask.at[k].set(True)

    # Longitudinal lines x=j*w
    offset = MAX_TRANSVERSE_LINES
    for j in range(params.num_lanes + 1):
        x_norm = (float(j) * w) / max(W, 1e-5)
        # Features: [is_trans=0, is_long=1, x1, y1=0, x2, y2=1, guard_assigned=0, j_norm]
        tok = jnp.array([0.0, 1.0, x_norm, 0.0, x_norm, 1.0, 0.0, float(j) / 4.0])
        tokens = tokens.at[offset + j].set(tok)
        mask = mask.at[offset + j].set(True)

    return tokens, mask


def build_global_token(state: EnvState, params: EnvParams) -> Array:
    """Construct global token (20 features) describing game time, scores, and rules."""
    g = state.global_state
    time_rem = jnp.maximum(
        0.0, (params.innings_duration - g.time_elapsed) / params.innings_duration
    )
    pre_clap_flag = jnp.where(~g.clapped, 1.0, 0.0)
    clap_rem = jnp.maximum(0.0, (params.clap_time - g.pre_clap_timer) / params.clap_time)
    score_diff = (g.score_attack - g.score_defence).astype(jnp.float32) / 18.0

    tok = jnp.zeros((GLOBAL_TOKEN_DIM,), dtype=jnp.float32)
    tok = tok.at[0].set(time_rem)
    tok = tok.at[1].set(pre_clap_flag)
    tok = tok.at[2].set(clap_rem)
    tok = tok.at[3].set(score_diff)
    tok = tok.at[4].set(g.innings_idx.astype(jnp.float32) / 4.0)
    tok = tok.at[5].set(float(params.num_lanes) / 3.0)
    tok = tok.at[6].set(float(params.num_rows) / 10.0)
    tok = tok.at[7].set(float(params.num_crossers) / 6.0)
    # Toggles P1-P7
    tok = tok.at[8].set(1.0 if params.p1_momentum_stamina else 0.0)
    tok = tok.at[9].set(1.0 if params.p2_facing_turning else 0.0)
    tok = tok.at[10].set(1.0 if params.p3_limited_vision else 0.0)
    tok = tok.at[11].set(1.0 if params.p4_strategic_clap else 0.0)
    tok = tok.at[12].set(1.0 if params.p5_procedural_fields else 0.0)
    tok = tok.at[13].set(1.0 if params.p6_team_calls else 0.0)
    tok = tok.at[14].set(1.0 if params.p7_human_fair else 0.0)
    # Team calls
    tok = tok.at[15].set(g.last_call_attack.astype(jnp.float32) / 4.0)
    tok = tok.at[16].set(g.last_call_defence.astype(jnp.float32) / 4.0)
    return tok


def build_raw_player_tokens(state: EnvState, params: EnvParams) -> tuple[Array, Array]:
    """Build canonical list of all player positions, roles, and states.

    Order:
    0..MAX_CROSSERS-1: Crossers
    MAX_CROSSERS..MAX_CROSSERS+MAX_GUARDS-1: Guards
    MAX_CROSSERS+MAX_GUARDS: Kili

    Returns:
        (positions [MAX_PLAYERS, 2], vels [MAX_PLAYERS, 2], roles [MAX_PLAYERS, 3],
         statuses [MAX_PLAYERS, 3], zones [MAX_PLAYERS, 3], staminas [MAX_PLAYERS],
         facings [MAX_PLAYERS, 2], kappas [MAX_PLAYERS], active_mask [MAX_PLAYERS])
    """
    c = state.crossers
    guards = state.guards
    kili = state.kili
    d = params.row_depth

    # Guard positions in 2D
    guard_ys = guards.line_idx.astype(jnp.float32) * d
    guard_pos = jnp.stack([guards.x, guard_ys], axis=-1)  # [MAX_GUARDS, 2]
    guard_vel = jnp.stack([guards.vx, jnp.zeros_like(guards.vx)], axis=-1)

    # Kili position
    kili_pos = jnp.expand_dims(kili.pos, axis=0)  # [1, 2]
    kili_vel = jnp.expand_dims(kili.vel, axis=0)

    all_pos = jnp.concatenate([c.pos, guard_pos, kili_pos], axis=0)  # [MAX_PLAYERS, 2]
    all_vel = jnp.concatenate([c.vel, guard_vel, kili_vel], axis=0)  # [MAX_PLAYERS, 2]

    # Active mask
    kili_active = jnp.array([True], dtype=jnp.bool_)
    all_active = jnp.concatenate([c.active_mask, guards.active_mask, kili_active], axis=0)

    # Role one-hot: 0=crosser, 1=guard, 2=kili
    role_c = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (MAX_CROSSERS, 1))
    role_g = jnp.tile(jnp.array([0.0, 1.0, 0.0]), (MAX_GUARDS, 1))
    role_k = jnp.array([[0.0, 0.0, 1.0]])
    all_roles = jnp.concatenate([role_c, role_g, role_k], axis=0)  # [MAX_PLAYERS, 3]

    # Status: 0=out, 1=pacha, 2=uppu
    status_c = jax.nn.one_hot(c.status, 3)
    status_other = jnp.zeros((MAX_GUARDS + 1, 3), dtype=jnp.float32)
    all_status = jnp.concatenate([status_c, status_other], axis=0)

    # Zone: 0=home, 1=field, 2=far
    zone_c = jax.nn.one_hot(c.zone, 3)
    zone_other = jnp.zeros((MAX_GUARDS + 1, 3), dtype=jnp.float32)
    all_zones = jnp.concatenate([zone_c, zone_other], axis=0)

    # Stamina
    kili_stamina = jnp.expand_dims(kili.stamina, axis=0)
    all_stamina = jnp.concatenate([c.stamina, guards.stamina, kili_stamina], axis=0)

    # Facing
    guard_facings = jnp.stack(
        [jnp.zeros_like(guards.facing), jnp.where(guards.facing == 0, -1.0, 1.0)], axis=-1
    )
    kili_facing_y = jnp.where(kili.facing == 0, -1.0, jnp.where(kili.facing == 1, 1.0, 0.0))
    kili_facing_x = jnp.where(kili.facing == 2, 1.0, jnp.where(kili.facing == 3, -1.0, 0.0))
    kili_facings = jnp.array([[kili_facing_x, kili_facing_y]])
    all_facings = jnp.concatenate([c.facing, guard_facings, kili_facings], axis=0)

    # Kappa
    kappa_other = jnp.zeros((MAX_GUARDS + 1,), dtype=jnp.int32)
    all_kappas = jnp.concatenate([c.kappa, kappa_other], axis=0)

    return (
        all_pos,
        all_vel,
        all_roles,
        all_status,
        all_zones,
        all_stamina,
        all_facings,
        all_kappas,
        all_active,
    )


def observe_agent(
    ego_idx: int,  # 0..MAX_PLAYERS-1
    state: EnvState,
    params: EnvParams,
    is_critic: bool = False,
) -> tuple[Array, Array]:
    """Construct player tokens (27 features) from the perspective of agent ego_idx."""
    (
        all_pos,
        all_vel,
        all_roles,
        all_status,
        all_zones,
        all_stamina,
        all_facings,
        all_kappas,
        all_active,
    ) = build_raw_player_tokens(state, params)
    W, L = params.field_width, params.field_length
    ego_pos = all_pos[ego_idx]
    ego_is_crosser = ego_idx < MAX_CROSSERS

    tokens = jnp.zeros((MAX_PLAYERS, PLAYER_TOKEN_DIM), dtype=jnp.float32)

    def make_token(j):
        is_self = jnp.where(j == ego_idx, 1.0, 0.0)
        j_is_crosser = j < MAX_CROSSERS
        is_teammate = jnp.where((ego_is_crosser == j_is_crosser) & (j != ego_idx), 1.0, 0.0)
        is_opponent = jnp.where(ego_is_crosser != j_is_crosser, 1.0, 0.0)

        field_dims = jnp.array([jnp.maximum(W, 1e-5), jnp.maximum(L, 1e-5)])
        rel_pos = (all_pos[j] - ego_pos) / field_dims
        abs_pos = all_pos[j] / field_dims
        vel_norm = all_vel[j] / 6.0

        # Stamina: visible for own team, hidden (0) for opponent (unless critic)
        show_stamina = 1.0 if is_critic else jnp.maximum(is_self, is_teammate)
        stamina = all_stamina[j] * show_stamina

        tok = jnp.zeros((PLAYER_TOKEN_DIM,), dtype=jnp.float32)
        # 0..2: is_self, is_teammate, is_opponent
        tok = tok.at[0].set(is_self)
        tok = tok.at[1].set(is_teammate)
        tok = tok.at[2].set(is_opponent)
        # 3..5: role one-hot
        tok = tok.at[3:6].set(all_roles[j])
        # 6..8: status one-hot
        tok = tok.at[6:9].set(all_status[j])
        # 9..11: zone one-hot
        tok = tok.at[9:12].set(all_zones[j])
        # 12..13: rel_pos (dx, dy)
        tok = tok.at[12:14].set(rel_pos)
        # 14..15: abs_pos (x, y)
        tok = tok.at[14:16].set(abs_pos)
        # 16..17: vel (vx, vy)
        tok = tok.at[16:18].set(vel_norm)
        # 18..19: facing (cos, sin)
        tok = tok.at[18:20].set(all_facings[j])
        # 20: stamina
        tok = tok.at[20].set(stamina)
        # 21..22: line index / at-junction
        tok = tok.at[21].set(0.0)
        tok = tok.at[22].set(0.0)
        # 23: trip progress kappa / 12
        tok = tok.at[23].set(all_kappas[j].astype(jnp.float32) / 12.0)
        # 24..25: visible flag / time since last seen
        tok = tok.at[24].set(1.0)
        tok = tok.at[25].set(0.0)
        # 26: turn timer
        tok = tok.at[26].set(0.0)
        return tok

    tokens = jax.vmap(make_token)(jnp.arange(MAX_PLAYERS))
    return tokens, all_active


def compute_crosser_action_masks(
    c_idx: int, state: EnvState, params: EnvParams
) -> tuple[Array, Array, Array]:
    """Compute action masks for crosser c_idx: move (9), sprint (2), call (5)."""
    c = state.crossers
    is_alive = (c.status[c_idx] != STATUS_OUT) & c.active_mask[c_idx]
    has_stamina = c.stamina[c_idx] >= params.sprint_stamina_threshold

    # Move mask: if dead, only Stay (0) is allowed
    move_mask = jnp.where(
        is_alive, jnp.ones((9,), dtype=jnp.bool_), jnp.array([True] + [False] * 8, dtype=jnp.bool_)
    )
    sprint_mask = jnp.array(
        [True, has_stamina & is_alive & params.p1_momentum_stamina], dtype=jnp.bool_
    )
    call_mask = jnp.ones((5,), dtype=jnp.bool_)
    return move_mask, sprint_mask, call_mask


def compute_guard_action_masks(
    g_idx: int, state: EnvState, params: EnvParams
) -> tuple[Array, Array, Array]:
    """Compute action masks for guard g_idx: move (3), dash (2), turn (2)."""
    g = state.guards
    is_active = g.active_mask[g_idx]
    has_stamina = g.stamina[g_idx] >= params.sprint_stamina_threshold
    not_turning = g.turn_timer[g_idx] <= 0.0

    move_mask = jnp.where(
        is_active, jnp.ones((3,), dtype=jnp.bool_), jnp.array([True, False, False], dtype=jnp.bool_)
    )
    dash_mask = jnp.array(
        [True, has_stamina & is_active & params.p1_momentum_stamina], dtype=jnp.bool_
    )
    turn_mask = jnp.array(
        [True, not_turning & is_active & params.p2_facing_turning], dtype=jnp.bool_
    )
    return move_mask, dash_mask, turn_mask


def compute_kili_action_masks(
    state: EnvState, params: EnvParams
) -> tuple[Array, Array, Array, Array]:
    """Compute action masks for Kili: move (5), dash (2), turn (2), clap (2)."""
    kili = state.kili
    g = state.global_state
    has_stamina = kili.stamina >= params.sprint_stamina_threshold
    not_turning = kili.turn_timer <= 0.0

    move_mask = jnp.ones((5,), dtype=jnp.bool_)
    dash_mask = jnp.array([True, has_stamina & params.p1_momentum_stamina], dtype=jnp.bool_)
    turn_mask = jnp.array([True, not_turning & params.p2_facing_turning], dtype=jnp.bool_)

    can_clap = (
        (~g.clapped)
        & (g.pre_clap_timer >= params.clap_window_min)
        & (g.pre_clap_timer <= params.clap_window_max)
    )
    clap_mask = jnp.array([True, can_clap & params.p4_strategic_clap], dtype=jnp.bool_)
    return move_mask, dash_mask, turn_mask, clap_mask
