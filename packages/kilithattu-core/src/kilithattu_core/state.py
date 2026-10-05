"""State pytree definitions for Kilithattu Kali simulation."""

from __future__ import annotations

from typing import NamedTuple

from chex import Array

# Status Constants
STATUS_OUT: int = 0
STATUS_PACHA: int = 1
STATUS_UPPU: int = 2

# Zone Constants
ZONE_HOME: int = 0
ZONE_FIELD: int = 1
ZONE_FAR: int = 2


class CrosserState(NamedTuple):
    pos: Array  # float32 [MAX_CROSSERS, 2]
    vel: Array  # float32 [MAX_CROSSERS, 2]
    status: Array  # int32 [MAX_CROSSERS] (0=out, 1=pacha, 2=uppu)
    zone: Array  # int32 [MAX_CROSSERS] (0=home, 1=field, 2=far)
    kappa: Array  # int32 [MAX_CROSSERS] (trip progress counter)
    trips: Array  # int32 [MAX_CROSSERS] (salt deliveries completed)
    stamina: Array  # float32 [MAX_CROSSERS]
    facing: Array  # float32 [MAX_CROSSERS, 2] (heading vector)
    box: Array  # int32 [MAX_CROSSERS, 2] (lane_idx, row_idx)
    prev_box: Array  # int32 [MAX_CROSSERS, 2]
    entered_box_this_tick: Array  # bool [MAX_CROSSERS]
    active_mask: Array  # bool [MAX_CROSSERS]


class GuardState(NamedTuple):
    x: Array  # float32 [MAX_GUARDS]
    vx: Array  # float32 [MAX_GUARDS]
    line_idx: Array  # int32 [MAX_GUARDS] (1..num_rows - 1)
    facing: Array  # int32 [MAX_GUARDS] (0=Home/South, 1=Far/North)
    turn_timer: Array  # float32 [MAX_GUARDS] (countdown seconds)
    stamina: Array  # float32 [MAX_GUARDS]
    active_mask: Array  # bool [MAX_GUARDS]


class KiliState(NamedTuple):
    pos: Array  # float32 [2]
    vel: Array  # float32 [2]
    facing: Array  # int32 scalar (0=South, 1=North, 2=East, 3=West)
    turn_timer: Array  # float32 scalar
    stamina: Array  # float32 scalar
    current_line_type: Array  # int32 scalar (0=transverse, 1=longitudinal)
    current_line_idx: Array  # int32 scalar


class GlobalState(NamedTuple):
    tick: Array  # int32 scalar
    time_elapsed: Array  # float32 scalar
    pre_clap_timer: Array  # float32 scalar
    clapped: Array  # bool scalar
    score_attack: Array  # int32 scalar
    score_defence: Array  # int32 scalar
    innings_idx: Array  # int32 scalar
    done: Array  # bool scalar
    last_call_attack: Array  # int32 scalar
    last_call_defence: Array  # int32 scalar


class EnvState(NamedTuple):
    crossers: CrosserState
    guards: GuardState
    kili: KiliState
    global_state: GlobalState
    rng_key: Array
