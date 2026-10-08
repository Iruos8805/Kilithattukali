"""Determinism tests and replay round-trip verification."""

import tempfile
from pathlib import Path

import jax
import numpy as np
from kilithattu_core.config import EnvParams
from kilithattu_core.env import JointActions, KilithattuEnv
from kilithattu_core.replay import verify_replay, write_replay
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS


def test_bit_for_bit_determinism():
    """Two independent simulation rollouts with the same seed and actions must match identically."""
    params = EnvParams()
    env = KilithattuEnv(params)
    seed = 12345

    def run_rollout():
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key, params)
        states = []
        events_list = []
        for t in range(50):
            # Deterministic pseudo-random actions derived from tick
            act_key = jax.random.PRNGKey(seed + t)
            k1, k2 = jax.random.split(act_key)
            c_move = jax.random.randint(k1, shape=(MAX_CROSSERS,), minval=0, maxval=9)
            g_move = jax.random.randint(k2, shape=(MAX_GUARDS,), minval=0, maxval=3)

            actions = JointActions.default()._replace(
                crosser_move=c_move,
                guard_move=g_move,
            )
            obs, state, r_A, r_D, events = env.step(key, state, actions, params)
            states.append(
                (state.crossers.pos, state.crossers.status, state.global_state.score_attack)
            )
            events_list.append(events.points_attack)
        return states, events_list

    states1, events1 = run_rollout()
    states2, events2 = run_rollout()

    for (pos1, st1, sc1), (pos2, st2, sc2) in zip(states1, states2):
        np.testing.assert_array_equal(np.array(pos1), np.array(pos2))
        np.testing.assert_array_equal(np.array(st1), np.array(st2))
        assert int(sc1) == int(sc2)

    for ev1, ev2 in zip(events1, events2):
        assert int(ev1) == int(ev2)


def test_replay_verification():
    """Verify write_replay and verify_replay functionality."""
    params = EnvParams()
    env = KilithattuEnv(params)
    seed = 999
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key, params)

    num_ticks = 30
    total_slots = MAX_CROSSERS + MAX_GUARDS + 1
    actions_arr = np.zeros((num_ticks, total_slots, 4), dtype=np.int8)
    events_log = []

    for t in range(num_ticks):
        # Step
        joint_actions = JointActions.default()
        obs, state, r_A, r_D, events = env.step(key, state, joint_actions, params)
        events_log.append({"tick": t, "points_attack": int(events.points_attack)})

    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_path = Path(tmp_dir) / "test_match.kreplay"
        header = {
            "version": "1.0.0",
            "seed": seed,
            "params": {
                "num_lanes": params.num_lanes,
                "num_rows": params.num_rows,
                "lane_width": params.lane_width,
                "row_depth": params.row_depth,
                "safe_zone_depth": params.safe_zone_depth,
            },
        }
        write_replay(replay_path, header, actions_arr, events_log)
        assert verify_replay(replay_path) is True
