"""Match and innings evaluation runner between policies."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from kilithattu_agents.policy import Policy
from kilithattu_core.config import EnvParams, rules_hash
from kilithattu_core.env import KilithattuEnv
from kilithattu_game.controllers.scripted import ScriptedController
from kilithattu_game.slots import SlotManager


class InningsResult(NamedTuple):
    points_attack: int
    points_defence: int
    salt_deliveries: int
    all_out: bool
    ticks: int


class MatchResult(NamedTuple):
    team_a_points: int
    team_b_points: int
    winner: str  # 'team_a', 'team_b', 'draw'
    innings: list[InningsResult]


_JITTED_STEP_CACHE: dict[str, Any] = {}


def _get_jitted_step(params: EnvParams) -> Any:
    h = rules_hash(params)
    if h not in _JITTED_STEP_CACHE:
        env = KilithattuEnv(params)
        _JITTED_STEP_CACHE[h] = jax.jit(lambda k, s, a: env.step(k, s, a, params))
    return _JITTED_STEP_CACHE[h]


def play_single_innings(
    attack_policy: Policy,
    defence_guards_policy: Policy,
    defence_kili_policy: Policy,
    params: EnvParams,
    key: PRNGKey,
    max_ticks: int = 400,
) -> InningsResult:
    """Run one innings with attack and defence policies."""
    env = KilithattuEnv(params)
    slot_manager = SlotManager(params)

    # Assign attack
    for i in range(params.num_crossers):
        slot_manager.assign_controller(i, ScriptedController(attack_policy))
    # Assign defence
    for k in range(params.num_guards):
        slot_manager.assign_controller(6 + k, ScriptedController(defence_guards_policy))
    slot_manager.assign_controller(6 + 8, ScriptedController(defence_kili_policy))

    k_reset, key = jax.random.split(key)
    obs, state = env.reset(k_reset, params)

    jitted_step = _get_jitted_step(params)

    ticks = 0
    for t in range(max_ticks):
        actions = slot_manager.get_joint_actions(obs)
        k_step, key = jax.random.split(key)
        obs, state, r_A, r_D, events = jitted_step(k_step, state, actions)
        ticks += 1
        if bool(state.global_state.done):
            break

    pts_a = int(state.global_state.score_attack)
    all_out = all(int(state.crossers.status[i]) == 0 for i in range(params.num_crossers))
    delivs = int(jnp.sum(state.crossers.trips[: params.num_crossers]))

    return InningsResult(
        points_attack=pts_a,
        points_defence=0,
        salt_deliveries=delivs,
        all_out=all_out,
        ticks=ticks,
    )


def play_full_match(
    team_a_policy: Policy,
    team_b_policy: Policy,
    params: EnvParams,
    key: PRNGKey,
) -> MatchResult:
    """Play a standard 4-innings match (2 attack / 2 defence per team)."""
    team_a_score = 0
    team_b_score = 0
    innings_log = []

    # Innings 1: Team A attacks, Team B defends
    k1, key = jax.random.split(key)
    inn1 = play_single_innings(team_a_policy, team_b_policy, team_b_policy, params, k1)
    team_a_score += inn1.points_attack
    innings_log.append(inn1)

    # Innings 2: Team B attacks, Team A defends
    k2, key = jax.random.split(key)
    inn2 = play_single_innings(team_b_policy, team_a_policy, team_a_policy, params, k2)
    team_b_score += inn2.points_attack
    innings_log.append(inn2)

    # Innings 3: Team A attacks, Team B defends
    k3, key = jax.random.split(key)
    inn3 = play_single_innings(team_a_policy, team_b_policy, team_b_policy, params, k3)
    team_a_score += inn3.points_attack
    innings_log.append(inn3)

    # Innings 4: Team B attacks, Team A defends
    k4, key = jax.random.split(key)
    inn4 = play_single_innings(team_b_policy, team_a_policy, team_a_policy, params, k4)
    team_b_score += inn4.points_attack
    innings_log.append(inn4)

    if team_a_score > team_b_score:
        winner = "team_a"
    elif team_b_score > team_a_score:
        winner = "team_b"
    else:
        winner = "draw"

    return MatchResult(
        team_a_points=team_a_score,
        team_b_points=team_b_score,
        winner=winner,
        innings=innings_log,
    )
