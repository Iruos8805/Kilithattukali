"""Unit and invariant tests for Kilithattu Kali rules."""

import jax
import jax.numpy as jnp
from kilithattu_core.config import EnvParams
from kilithattu_core.env import JointActions, KilithattuEnv
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS
from kilithattu_core.state import STATUS_OUT, STATUS_PACHA, STATUS_UPPU


def test_safe_zone_tag_immunity():
    """Defenders cannot tag crossers in the Home or Far safe zones (R2)."""
    params = EnvParams(clap_time=0.0)
    env = KilithattuEnv(params)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, params)

    # Position crosser 0 at (3.0, -0.2) in home safe zone
    state = state._replace(
        crossers=state.crossers._replace(pos=state.crossers.pos.at[0].set(jnp.array([3.0, -0.2]))),
        kili=state.kili._replace(pos=jnp.array([3.0, 0.0])),  # Distance 0.2 < r_tag (0.6)
        global_state=state.global_state._replace(clapped=jnp.array(True)),
    )

    actions = JointActions.default()
    obs, new_state, r_A, r_D, events = env.step(key, state, actions, params)

    # Crosser 0 must NOT be tagged because it is in the safe zone
    assert bool(events.tagged[0]) is False
    assert int(new_state.crossers.status[0]) == STATUS_PACHA


def test_tag_in_field():
    """A crosser in the field within r_tag of a defender is tagged and eliminated (R8)."""
    params = EnvParams(clap_time=0.0)
    env = KilithattuEnv(params)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, params)

    # Place crosser 0 in the field at (3.0, 3.2), guard 0 on line 1 at (3.0, 3.0) -> dist 0.2 < 0.6
    state = state._replace(
        crossers=state.crossers._replace(
            pos=state.crossers.pos.at[0].set(jnp.array([3.0, 3.2])),
            status=state.crossers.status.at[0].set(STATUS_PACHA),
        ),
        guards=state.guards._replace(x=state.guards.x.at[0].set(3.0)),
        global_state=state.global_state._replace(clapped=jnp.array(True)),
    )

    actions = JointActions.default()
    obs, new_state, r_A, r_D, events = env.step(key, state, actions, params)

    assert bool(events.tagged[0]) is True
    assert int(new_state.crossers.status[0]) == STATUS_OUT


def test_round_trip_scoring_and_anti_farming():
    """Verify exact 18-point scoring for a full round trip and 0 points for re-crossing lines (R11, §4.4)."""
    params = EnvParams(clap_time=0.0)
    env = KilithattuEnv(params)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, params)

    # Crosser 0 at (3.0, -0.1), clapped
    state = state._replace(
        crossers=state.crossers._replace(
            pos=state.crossers.pos.at[0].set(jnp.array([3.0, -0.1])),
            status=state.crossers.status.at[0].set(STATUS_PACHA),
            kappa=state.crossers.kappa.at[0].set(0),
        ),
        guards=state.guards._replace(x=jnp.full((MAX_GUARDS,), 0.1)),
        kili=state.kili._replace(pos=jnp.array([0.1, 0.0])),
        global_state=state.global_state._replace(clapped=jnp.array(True)),
    )

    # Move North (action 1) to cross L0 (y=0)
    actions = JointActions.default()._replace(
        crosser_move=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32).at[0].set(1)
    )

    # Step: cross L0 -> points +1, kappa -> 1
    obs, state, r_A, r_D, events = env.step(key, state, actions, params)
    assert int(events.points_attack) == 1
    assert int(state.crossers.kappa[0]) == 1

    # Now move South (action 5) to retreat across L0 -> points must be 0!
    retreat_actions = JointActions.default()._replace(
        crosser_move=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32).at[0].set(5)
    )
    obs, state, r_A, r_D, events = env.step(key, state, retreat_actions, params)
    assert int(events.points_attack) == 0  # Anti-farming: no reward for retreating!

    # Re-cross L0 North again -> points must be 0 because kappa is already 1!
    obs, state, r_A, r_D, events = env.step(key, state, actions, params)
    assert int(events.points_attack) == 0  # Anti-farming: no reward for re-crossing!


def test_full_round_trip_18_points():
    """Test full sequential trip from Home -> Far -> Home delivers exactly 18 points."""
    params = EnvParams(clap_time=0.0)
    env = KilithattuEnv(params)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, params)

    d = params.row_depth
    L = params.field_length

    # Manually step crosser 0 through lines L0..L5 and back
    curr_state = state._replace(
        crossers=state.crossers._replace(
            pos=state.crossers.pos.at[0].set(jnp.array([3.0, -0.1])),
            status=state.crossers.status.at[0].set(STATUS_PACHA),
            kappa=state.crossers.kappa.at[0].set(0),
        ),
        # Move guards and Kili away to prevent tags
        guards=state.guards._replace(x=jnp.full((MAX_GUARDS,), 0.1)),
        kili=state.kili._replace(pos=jnp.array([0.1, 0.0])),
        global_state=state.global_state._replace(clapped=jnp.array(True)),
    )

    total_points = 0

    # 1. Outbound crossings L0..L5
    # Crossing L0
    curr_state = curr_state._replace(
        crossers=curr_state.crossers._replace(
            pos=curr_state.crossers.pos.at[0].set(jnp.array([3.0, -0.05]))
        )
    )
    actions = JointActions.default()._replace(
        crosser_move=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32).at[0].set(1)
    )
    obs, curr_state, r_A, r_D, events = env.step(key, curr_state, actions, params)
    total_points += int(events.points_attack)

    # Crossing L1..L4
    for k in range(1, params.num_rows):
        line_y = k * d
        curr_state = curr_state._replace(
            crossers=curr_state.crossers._replace(
                pos=curr_state.crossers.pos.at[0].set(jnp.array([3.0, line_y - 0.05]))
            )
        )
        obs, curr_state, r_A, r_D, events = env.step(key, curr_state, actions, params)
        total_points += int(events.points_attack)

    # Crossing L5 into Far Zone
    curr_state = curr_state._replace(
        crossers=curr_state.crossers._replace(
            pos=curr_state.crossers.pos.at[0].set(jnp.array([3.0, L - 0.05]))
        )
    )
    obs, curr_state, r_A, r_D, events = env.step(key, curr_state, actions, params)
    total_points += int(events.points_attack)
    assert int(curr_state.crossers.status[0]) == STATUS_UPPU
    assert total_points == 6  # 6 outbound points

    # 2. Inbound crossings: Re-enter across L5
    actions_south = JointActions.default()._replace(
        crosser_move=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32).at[0].set(5)
    )
    curr_state = curr_state._replace(
        crossers=curr_state.crossers._replace(
            pos=curr_state.crossers.pos.at[0].set(jnp.array([3.0, L + 0.1]))
        )
    )
    obs, curr_state, r_A, r_D, events = env.step(key, curr_state, actions_south, params)
    total_points += int(events.points_attack)

    # Crossing L4..L1 in -y direction
    for k in range(params.num_rows - 1, 0, -1):
        line_y = k * d
        curr_state = curr_state._replace(
            crossers=curr_state.crossers._replace(
                pos=curr_state.crossers.pos.at[0].set(jnp.array([3.0, line_y + 0.05]))
            )
        )
        obs, curr_state, r_A, r_D, events = env.step(key, curr_state, actions_south, params)
        total_points += int(events.points_attack)

    # Crossing L0 into Home (Salt Delivery)
    curr_state = curr_state._replace(
        crossers=curr_state.crossers._replace(
            pos=curr_state.crossers.pos.at[0].set(jnp.array([3.0, 0.05]))
        )
    )
    obs, curr_state, r_A, r_D, events = env.step(key, curr_state, actions_south, params)
    total_points += int(events.points_attack)
    assert bool(events.salt_delivered[0]) is True
    assert int(curr_state.crossers.status[0]) == STATUS_PACHA
    assert int(curr_state.crossers.trips[0]) == 1

    # Exactly 18 points for a complete round trip (6 outbound + 5 inbound lines + 1 home cross + 6 bonus = 18)
    assert total_points == 18


def test_foul_uppu_pacha_same_box():
    """An Uppu and a Pacha in the same box triggers a foul and eliminates the offender (R10)."""
    params = EnvParams(clap_time=0.0)
    env = KilithattuEnv(params)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, params)

    # Crosser 0 is Pacha already in box (0, 2) at (1.5, 7.5)
    # Crosser 1 is Uppu, enters box (0, 2) this tick from (1.5, 9.1) -> (1.5, 8.9)
    state = state._replace(
        crossers=state.crossers._replace(
            pos=state.crossers.pos.at[0]
            .set(jnp.array([1.5, 7.5]))
            .at[1]
            .set(jnp.array([1.5, 9.1])),
            status=state.crossers.status.at[0].set(STATUS_PACHA).at[1].set(STATUS_UPPU),
            box=jnp.array(
                [[0, 2], [0, 3], [-1, -1], [-1, -1], [-1, -1], [-1, -1]], dtype=jnp.int32
            ),
            prev_box=jnp.array(
                [[0, 2], [0, 3], [-1, -1], [-1, -1], [-1, -1], [-1, -1]], dtype=jnp.int32
            ),
        ),
        # Guard and Kili away
        guards=state.guards._replace(x=jnp.full((MAX_GUARDS,), 5.5)),
        kili=state.kili._replace(pos=jnp.array([5.5, 0.0])),
        global_state=state.global_state._replace(clapped=jnp.array(True)),
    )

    # Crosser 1 moves South into box (0, 2); crosser 0 stays
    actions = JointActions.default()._replace(
        crosser_move=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32).at[1].set(5)
    )

    obs, new_state, r_A, r_D, events = env.step(key, state, actions, params)

    # Crosser 1 entered the box of Crosser 0 -> Crosser 1 fouled and is OUT
    assert bool(events.fouled[1]) is True
    assert int(new_state.crossers.status[1]) == STATUS_OUT
    assert int(new_state.crossers.status[0]) == STATUS_PACHA  # Crosser 0 stays safe
