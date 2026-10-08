"""Resolution rules: clap gating, continuous tags, monotonic line crossings, foul detection, and termination."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chex import Array

from kilithattu_core.config import EnvParams
from kilithattu_core.events import StepEvents
from kilithattu_core.geometry import (
    continuous_swept_tag,
    get_box_indices,
    is_in_safe_zone,
    is_side_exit,
)
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS
from kilithattu_core.state import (
    STATUS_OUT,
    STATUS_PACHA,
    STATUS_UPPU,
    ZONE_FAR,
    ZONE_FIELD,
    ZONE_HOME,
    CrosserState,
    EnvState,
    GlobalState,
    GuardState,
    KiliState,
)


def resolve_tick(
    state: EnvState,
    cand_crosser_pos: Array,  # [MAX_CROSSERS, 2]
    cand_crosser_vel: Array,  # [MAX_CROSSERS, 2]
    cand_crosser_stamina: Array,  # [MAX_CROSSERS]
    cand_crosser_facing: Array,  # [MAX_CROSSERS, 2]
    cand_guard_x: Array,  # [MAX_GUARDS]
    cand_guard_vx: Array,  # [MAX_GUARDS]
    cand_guard_facing: Array,  # [MAX_GUARDS]
    cand_guard_turn_timer: Array,  # [MAX_GUARDS]
    cand_guard_stamina: Array,  # [MAX_GUARDS]
    cand_kili_pos: Array,  # [2]
    cand_kili_vel: Array,  # [2]
    cand_kili_facing: Array,  # scalar
    cand_kili_turn_timer: Array,  # scalar
    cand_kili_stamina: Array,  # scalar
    cand_kili_line_type: Array,  # scalar
    cand_kili_line_idx: Array,  # scalar
    kili_clap_action: Array,  # scalar (0 or 1)
    attack_call: Array,  # scalar
    defence_call: Array,  # scalar
    params: EnvParams,
) -> tuple[EnvState, StepEvents, Array, Array]:
    """Execute rule resolution order (§5.7) for one simulation tick.

    Returns:
        (new_state, events, reward_attack, reward_defence)
    """
    g = state.global_state
    c = state.crossers
    guards = state.guards
    kili = state.kili
    d = params.row_depth
    L = params.field_length

    # 1. Clap determination
    if not params.p4_strategic_clap:
        auto_clap = (g.pre_clap_timer + params.dt >= params.clap_time) & (~g.clapped)
        now_clapped = g.clapped | auto_clap
        clap_event = auto_clap
    else:
        window_open = (g.pre_clap_timer >= params.clap_window_min) & (
            g.pre_clap_timer <= params.clap_window_max
        )
        manual_clap = window_open & (kili_clap_action == 1) & (~g.clapped)
        force_clap = (g.pre_clap_timer + params.dt >= params.clap_window_max) & (~g.clapped)
        clap_fired = manual_clap | force_clap
        now_clapped = g.clapped | clap_fired
        clap_event = clap_fired

    new_pre_clap_timer = jnp.where(~now_clapped, g.pre_clap_timer + params.dt, g.pre_clap_timer)
    new_time_elapsed = jnp.where(now_clapped, g.time_elapsed + params.dt, g.time_elapsed)

    # 2. Constraints: Pre-clap hold & false starts
    # If not clapped, crossers cannot cross y = 0
    start_pos_c = c.pos
    end_pos_c = cand_crosser_pos

    false_started = jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_)
    if params.p4_strategic_clap:
        # P4: Crossing L0 before clap is false start -> OUT
        attempted_cross = (
            (~g.clapped) & (end_pos_c[:, 1] >= 0.0) & (c.status != STATUS_OUT) & c.active_mask
        )
        false_started = attempted_cross
    else:
        # Classic: Crossers clamped behind L0 prior to clap
        clamped_y = jnp.where((~now_clapped) & (end_pos_c[:, 1] >= -0.05), -0.05, end_pos_c[:, 1])
        end_pos_c = end_pos_c.at[:, 1].set(clamped_y)

    # 3. Side Boundary Exits (R7)
    side_exited = is_side_exit(end_pos_c, params) & (c.status != STATUS_OUT) & c.active_mask

    # 4. Continuous Swept Tag Detection (R8, D10)
    # Tags resolve before crossings (R14)
    # Safe zone immunity applies at the point of closest approach (R2)

    # Check tags vs guards
    # For each crosser i and guard j
    def check_guard_tag(i_c, j_g):
        guard_active = guards.active_mask[j_g]
        crosser_active = c.active_mask[i_c] & (c.status[i_c] != STATUS_OUT)
        guard_line_y = guards.line_idx[j_g].astype(jnp.float32) * d
        p_g_start = jnp.array([guards.x[j_g], guard_line_y])
        p_g_end = jnp.array([cand_guard_x[j_g], guard_line_y])

        hit, p_c_tau = continuous_swept_tag(
            start_pos_c[i_c], end_pos_c[i_c], p_g_start, p_g_end, params.r_tag
        )
        in_safe = is_in_safe_zone(p_c_tau, params)

        # P2: Guard tag cone check
        if params.p2_facing_turning:
            is_turning = cand_guard_turn_timer[j_g] > 0.0
            # Guard facing: 0=South (-y), 1=North (+y)
            facing_dir = jnp.where(cand_guard_facing[j_g] == 0, -1.0, 1.0)
            rel_y = start_pos_c[i_c, 1] - guard_line_y
            in_cone = (rel_y * facing_dir) >= 0.0
            hit = hit & in_cone & (~is_turning)

        return hit & guard_active & crosser_active & (~in_safe)

    # Vectorize across all crossers and guards
    tag_matrix = jax.vmap(
        lambda i_c: jax.vmap(lambda j_g: check_guard_tag(i_c, j_g))(jnp.arange(MAX_GUARDS))
    )(jnp.arange(MAX_CROSSERS))
    tagged_by_guard = jnp.any(tag_matrix, axis=-1)  # [MAX_CROSSERS]

    # Check tags vs Kili
    def check_kili_tag(i_c):
        crosser_active = c.active_mask[i_c] & (c.status[i_c] != STATUS_OUT)
        hit, p_c_tau = continuous_swept_tag(
            start_pos_c[i_c], end_pos_c[i_c], kili.pos, cand_kili_pos, params.r_tag
        )
        in_safe = is_in_safe_zone(p_c_tau, params)
        if params.p2_facing_turning:
            is_turning = cand_kili_turn_timer > 0.0
            hit = hit & (~is_turning)
        return hit & crosser_active & (~in_safe)

    tagged_by_kili = jax.vmap(check_kili_tag)(jnp.arange(MAX_CROSSERS))
    tagged = (tagged_by_guard | tagged_by_kili) & now_clapped

    # Survivors of tag, side exit, false start
    eliminated_now = tagged | side_exited | false_started

    # 5. Line Crossings & Scoring for Survivors (R9, R11, §4.4)
    # Trip progress counter kappa in 0..2*(n_B + 1)
    new_kappa = jnp.array(c.kappa)
    new_status = jnp.array(c.status)
    new_trips = jnp.array(c.trips)
    lines_crossed = jnp.full((MAX_CROSSERS,), -1, dtype=jnp.int32)
    salt_delivered = jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_)
    crosser_points = jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32)

    def evaluate_crossing(i):
        survived = (
            (~eliminated_now[i]) & (c.status[i] != STATUS_OUT) & c.active_mask[i] & now_clapped
        )
        k_curr = c.kappa[i]
        st_curr = c.status[i]
        y0 = start_pos_c[i, 1]
        y1 = end_pos_c[i, 1]

        k_next = k_curr
        st_next = st_curr
        trip_next = c.trips[i]
        pt = 0
        deliv = False
        crossed_idx = -1

        # Outbound progression
        # Line 0: y=0, kappa 0 -> 1
        cross_l0_out = (y0 < 0.0) & (y1 >= 0.0) & (k_curr == 0)
        # Lines 1..n_B-1: y=k*d, kappa k -> k+1
        # Line n_B: y=L, kappa n_B -> n_B+1, status -> UPPU
        for line_k in range(params.num_rows + 1):
            line_y = float(line_k) * d
            # Outbound check
            if line_k == 0:
                is_cross = cross_l0_out
            elif line_k == params.num_rows:
                is_cross = (y0 < L) & (y1 >= L) & (k_curr == params.num_rows)
            else:
                is_cross = (y0 < line_y) & (y1 >= line_y) & (k_curr == line_k)

            # Inbound check
            inbound_kappa_req = (params.num_rows + 1) + (params.num_rows - line_k)
            if line_k == params.num_rows:
                is_cross_in = (y0 > L) & (y1 <= L) & (k_curr == params.num_rows + 1)
            elif line_k == 0:
                is_cross_in = (y0 >= 0.0) & (y1 < 0.0) & (k_curr == 2 * (params.num_rows + 1) - 1)
            else:
                is_cross_in = (y0 > line_y) & (y1 <= line_y) & (k_curr == inbound_kappa_req)

            # Apply outbound
            if line_k == params.num_rows:
                st_next = jnp.where(survived & is_cross, STATUS_UPPU, st_next)
            k_next = jnp.where(survived & is_cross, line_k + 1, k_next)
            pt = jnp.where(survived & is_cross, pt + 1, pt)
            crossed_idx = jnp.where(survived & is_cross, line_k, crossed_idx)

            # Apply inbound
            if line_k == params.num_rows:
                k_next = jnp.where(survived & is_cross_in, params.num_rows + 2, k_next)
                pt = jnp.where(survived & is_cross_in, pt + 1, pt)
                crossed_idx = jnp.where(survived & is_cross_in, line_k, crossed_idx)
            elif line_k == 0:
                # Salt delivery: +1 crossing + 6 bonus = +7, reset kappa to 0, status to PACHA
                deliv = survived & is_cross_in
                k_next = jnp.where(deliv, 0, k_next)
                st_next = jnp.where(deliv, STATUS_PACHA, st_next)
                trip_next = jnp.where(deliv, trip_next + 1, trip_next)
                pt = jnp.where(deliv, pt + 7, pt)
                crossed_idx = jnp.where(deliv, 0, crossed_idx)
            else:
                k_next = jnp.where(survived & is_cross_in, inbound_kappa_req + 1, k_next)
                pt = jnp.where(survived & is_cross_in, pt + 1, pt)
                crossed_idx = jnp.where(survived & is_cross_in, line_k, crossed_idx)

        return k_next, st_next, trip_next, pt, deliv, crossed_idx

    res = jax.vmap(evaluate_crossing)(jnp.arange(MAX_CROSSERS))
    new_kappa, new_status, new_trips, crosser_points, salt_delivered, lines_crossed = res

    # 6. Foul Check (R10)
    # Check if an Uppu and a Pacha occupy the same box at the end of the tick
    box_end = get_box_indices(end_pos_c, params)  # [MAX_CROSSERS, 2]
    entered_this_tick = jnp.any(box_end != c.prev_box, axis=-1)

    # Check pair fouls
    def check_pair_foul(i, j):
        alive_i = (~eliminated_now[i]) & (new_status[i] != STATUS_OUT) & c.active_mask[i]
        alive_j = (~eliminated_now[j]) & (new_status[j] != STATUS_OUT) & c.active_mask[j]
        both_in_field = (box_end[i, 0] >= 0) & (box_end[j, 0] >= 0)
        same_box = (box_end[i, 0] == box_end[j, 0]) & (box_end[i, 1] == box_end[j, 1])

        is_uppu_pacha = (new_status[i] == STATUS_UPPU) & (new_status[j] == STATUS_PACHA)
        foul_occurs = alive_i & alive_j & both_in_field & same_box & is_uppu_pacha

        # Who is out? If only one entered this tick, that one is out. If both entered, Uppu (i) is out.
        both_entered = entered_this_tick[i] & entered_this_tick[j]
        only_i_entered = entered_this_tick[i] & (~entered_this_tick[j])
        only_j_entered = (~entered_this_tick[i]) & entered_this_tick[j]

        i_out = foul_occurs & (both_entered | only_i_entered | (~only_j_entered))
        j_out = foul_occurs & only_j_entered
        return i_out, j_out

    # Compute foul matrix across all pairs
    all_pairs_i, all_pairs_j = jax.vmap(
        lambda i: jax.vmap(lambda j: check_pair_foul(i, j))(jnp.arange(MAX_CROSSERS))
    )(jnp.arange(MAX_CROSSERS))

    fouled_crossers = jnp.any(all_pairs_i, axis=-1) | jnp.any(all_pairs_j, axis=0)

    # 7. Final status updates
    all_eliminated = eliminated_now | fouled_crossers
    final_status = jnp.where(all_eliminated, STATUS_OUT, new_status)
    # If eliminated this tick, zero out any points awarded this tick
    final_points_per_crosser = jnp.where(all_eliminated, 0, crosser_points)
    points_attack_tick = jnp.sum(final_points_per_crosser)

    # Zone calculation
    def get_zone(p):
        y = p[1]
        return jnp.where(y < 0.0, ZONE_HOME, jnp.where(y > L, ZONE_FAR, ZONE_FIELD))

    new_zones = jax.vmap(get_zone)(end_pos_c)

    # 8. Innings Termination Check (R12)
    # Innings ends if all active crossers are OUT, or clock >= innings_duration
    all_active_out = jnp.all(jnp.where(c.active_mask, final_status == STATUS_OUT, True))
    clock_expired = new_time_elapsed >= params.innings_duration
    is_done = (all_active_out | clock_expired) & now_clapped

    # 9. Potential-based Shaping (telescoping, zero at terminal)
    def calc_phi(pos, status, kappa, active):
        # Progress: outbound = clip(y, 0, L); once uppu = L + (L - clip(y, 0, L))
        y = jnp.clip(pos[1], 0.0, L)
        prog = jnp.where(status == STATUS_UPPU, L + (L - y), y)
        norm_prog = prog / (2.0 * L)
        return jnp.where(active & (status != STATUS_OUT), norm_prog, 0.0)

    phi_curr = jnp.mean(jax.vmap(calc_phi)(end_pos_c, final_status, new_kappa, c.active_mask))
    phi_prev = jnp.mean(jax.vmap(calc_phi)(start_pos_c, c.status, c.kappa, c.active_mask))

    # Potential is strictly 0 at terminal
    phi_next = jnp.where(is_done, 0.0, phi_curr)
    gamma = 0.997
    shaping = params.shaping_beta * (gamma * phi_next - phi_prev)

    # Primary reward: +1/18 per line point, zero-sum
    raw_reward = points_attack_tick.astype(jnp.float32) / 18.0
    reward_attack = raw_reward + shaping
    reward_defence = -reward_attack

    # Construct events
    events = StepEvents(
        tagged=tagged,
        fouled=fouled_crossers,
        side_exited=side_exited,
        line_crossed=lines_crossed,
        salt_delivered=salt_delivered,
        clapped=clap_event,
        points_attack=points_attack_tick,
        false_started=false_started,
    )

    # Construct new state
    new_crossers = CrosserState(
        pos=end_pos_c,
        vel=cand_crosser_vel,
        status=final_status,
        zone=new_zones,
        kappa=new_kappa,
        trips=new_trips,
        stamina=cand_crosser_stamina,
        facing=cand_crosser_facing,
        box=box_end,
        prev_box=c.box,
        entered_box_this_tick=entered_this_tick,
        active_mask=c.active_mask,
    )

    new_guards = GuardState(
        x=cand_guard_x,
        vx=cand_guard_vx,
        line_idx=guards.line_idx,
        facing=cand_guard_facing,
        turn_timer=cand_guard_turn_timer,
        stamina=cand_guard_stamina,
        active_mask=guards.active_mask,
    )

    new_kili = KiliState(
        pos=cand_kili_pos,
        vel=cand_kili_vel,
        facing=cand_kili_facing,
        turn_timer=cand_kili_turn_timer,
        stamina=cand_kili_stamina,
        current_line_type=cand_kili_line_type,
        current_line_idx=cand_kili_line_idx,
    )

    new_global = GlobalState(
        tick=g.tick + 1,
        time_elapsed=new_time_elapsed,
        pre_clap_timer=new_pre_clap_timer,
        clapped=now_clapped,
        score_attack=g.score_attack + points_attack_tick,
        score_defence=g.score_defence,
        innings_idx=g.innings_idx,
        done=is_done,
        last_call_attack=attack_call,
        last_call_defence=defence_call,
    )

    new_env_state = EnvState(
        crossers=new_crossers,
        guards=new_guards,
        kili=new_kili,
        global_state=new_global,
        rng_key=state.rng_key,
    )

    return new_env_state, events, reward_attack, reward_defence
