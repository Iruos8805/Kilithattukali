"""Main JaxMARL-style vectorisable environment for Kilithattu Kali."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from kilithattu_core.config import EnvParams
from kilithattu_core.dynamics import (
    update_crosser_dynamics,
    update_guard_dynamics,
    update_kili_dynamics,
)
from kilithattu_core.events import StepEvents
from kilithattu_core.observe import (
    build_global_token,
    build_line_tokens,
    observe_agent,
)
from kilithattu_core.rules import resolve_tick
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS, MAX_PLAYERS
from kilithattu_core.state import (
    STATUS_PACHA,
    ZONE_HOME,
    CrosserState,
    EnvState,
    GlobalState,
    GuardState,
    KiliState,
)


class JointActions(NamedTuple):
    """Actions for all player slots in the game."""

    # Crossers: move [MAX_CROSSERS], sprint [MAX_CROSSERS], call [MAX_CROSSERS]
    crosser_move: Array
    crosser_sprint: Array
    crosser_call: Array

    # Guards: move [MAX_GUARDS], dash [MAX_GUARDS], turn [MAX_GUARDS]
    guard_move: Array
    guard_dash: Array
    guard_turn: Array

    # Kili: move scalar, dash scalar, turn scalar, clap scalar
    kili_move: Array
    kili_dash: Array
    kili_turn: Array
    kili_clap: Array

    @classmethod
    def default(cls) -> JointActions:
        return cls(
            crosser_move=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32),
            crosser_sprint=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32),
            crosser_call=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32),
            guard_move=jnp.zeros((MAX_GUARDS,), dtype=jnp.int32),
            guard_dash=jnp.zeros((MAX_GUARDS,), dtype=jnp.int32),
            guard_turn=jnp.zeros((MAX_GUARDS,), dtype=jnp.int32),
            kili_move=jnp.array(0, dtype=jnp.int32),
            kili_dash=jnp.array(0, dtype=jnp.int32),
            kili_turn=jnp.array(0, dtype=jnp.int32),
            kili_clap=jnp.array(0, dtype=jnp.int32),
        )


class Observations(NamedTuple):
    """Vectorised token sets and masks for all agents."""

    player_tokens: Array  # [MAX_PLAYERS, MAX_PLAYERS, 27]
    player_masks: Array  # [MAX_PLAYERS]
    line_tokens: Array  # [MAX_LINES, 8]
    line_mask: Array  # [MAX_LINES]
    global_token: Array  # [20]


class KilithattuEnv:
    """Pure JAX simulation of Kilithattu Kali."""

    def __init__(self, params: EnvParams | None = None) -> None:
        self.params = params or EnvParams()

    def reset(self, key: PRNGKey, params: EnvParams | None = None) -> tuple[Observations, EnvState]:
        """Reset the environment to the initial state distribution rho_0 (§5.1)."""
        p = params or self.params
        k1, k2, k3 = jax.random.split(key, 3)

        W = p.field_width
        w = p.lane_width

        # Active slot masks
        crosser_active = jnp.arange(MAX_CROSSERS) < p.num_crossers
        guard_active = jnp.arange(MAX_GUARDS) < p.num_guards

        # 1. Crossers placed uniformly in home zone [0, W] x [-z, 0)
        c_x = jax.random.uniform(k1, shape=(MAX_CROSSERS,), minval=0.5, maxval=W - 0.5)
        c_y = jax.random.uniform(
            k2, shape=(MAX_CROSSERS,), minval=-p.safe_zone_depth + 0.2, maxval=-0.2
        )
        c_pos = jnp.stack([c_x, c_y], axis=-1)
        c_vel = jnp.zeros((MAX_CROSSERS, 2), dtype=jnp.float32)
        c_facing = jnp.tile(jnp.array([0.0, 1.0]), (MAX_CROSSERS, 1))  # Facing North toward field

        init_crossers = CrosserState(
            pos=c_pos,
            vel=c_vel,
            status=jnp.full((MAX_CROSSERS,), STATUS_PACHA, dtype=jnp.int32),
            zone=jnp.full((MAX_CROSSERS,), ZONE_HOME, dtype=jnp.int32),
            kappa=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32),
            trips=jnp.zeros((MAX_CROSSERS,), dtype=jnp.int32),
            stamina=jnp.ones((MAX_CROSSERS,), dtype=jnp.float32),
            facing=c_facing,
            box=jnp.full((MAX_CROSSERS, 2), -1, dtype=jnp.int32),
            prev_box=jnp.full((MAX_CROSSERS, 2), -1, dtype=jnp.int32),
            entered_box_this_tick=jnp.zeros((MAX_CROSSERS,), dtype=jnp.bool_),
            active_mask=crosser_active,
        )

        # 2. Guards placed on transverse lines L1..L_nB-1 at random x in [0, W]
        g_x = jax.random.uniform(k3, shape=(MAX_GUARDS,), minval=0.5, maxval=W - 0.5)
        g_lines = jnp.arange(1, MAX_GUARDS + 1, dtype=jnp.int32)
        init_guards = GuardState(
            x=g_x,
            vx=jnp.zeros((MAX_GUARDS,), dtype=jnp.float32),
            line_idx=g_lines,
            facing=jnp.zeros((MAX_GUARDS,), dtype=jnp.int32),  # Facing Home (South)
            turn_timer=jnp.zeros((MAX_GUARDS,), dtype=jnp.float32),
            stamina=jnp.ones((MAX_GUARDS,), dtype=jnp.float32),
            active_mask=guard_active,
        )

        # 3. Kili starts at centre junction of line L0 (W/2, 0)
        centre_junc_x = (float(p.num_lanes // 2)) * w
        init_kili = KiliState(
            pos=jnp.array([centre_junc_x, 0.0], dtype=jnp.float32),
            vel=jnp.zeros((2,), dtype=jnp.float32),
            facing=jnp.array(0, dtype=jnp.int32),  # South
            turn_timer=jnp.array(0.0, dtype=jnp.float32),
            stamina=jnp.array(1.0, dtype=jnp.float32),
            current_line_type=jnp.array(0, dtype=jnp.int32),  # Transverse
            current_line_idx=jnp.array(0, dtype=jnp.int32),  # Line L0
        )

        # 4. Global state
        init_global = GlobalState(
            tick=jnp.array(0, dtype=jnp.int32),
            time_elapsed=jnp.array(0.0, dtype=jnp.float32),
            pre_clap_timer=jnp.array(0.0, dtype=jnp.float32),
            clapped=jnp.array(False, dtype=jnp.bool_),
            score_attack=jnp.array(0, dtype=jnp.int32),
            score_defence=jnp.array(0, dtype=jnp.int32),
            innings_idx=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            last_call_attack=jnp.array(0, dtype=jnp.int32),
            last_call_defence=jnp.array(0, dtype=jnp.int32),
        )

        state = EnvState(
            crossers=init_crossers,
            guards=init_guards,
            kili=init_kili,
            global_state=init_global,
            rng_key=key,
        )

        obs = self.get_observations(state, p)
        return obs, state

    def step(
        self,
        key: PRNGKey,
        state: EnvState,
        actions: JointActions,
        params: EnvParams | None = None,
    ) -> tuple[Observations, EnvState, Array, Array, StepEvents]:
        """Step the environment forward by dt (20 Hz) given joint actions."""
        p = params or self.params

        # 1. Update candidate dynamics
        cand_c_pos, cand_c_vel, cand_c_stamina, cand_c_facing = update_crosser_dynamics(
            state.crossers, actions.crosser_move, actions.crosser_sprint, p
        )

        cand_g_x, cand_g_vx, cand_g_facing, cand_g_turn_timer, cand_g_stamina = (
            update_guard_dynamics(
                state.guards, actions.guard_move, actions.guard_dash, actions.guard_turn, p
            )
        )

        (
            cand_k_pos,
            cand_k_vel,
            cand_k_facing,
            cand_k_turn_timer,
            cand_k_stamina,
            cand_k_line_type,
            cand_k_line_idx,
        ) = update_kili_dynamics(
            state.kili, actions.kili_move, actions.kili_dash, actions.kili_turn, p
        )

        # 2. Resolve tick rules
        attack_call = actions.crosser_call[0]
        defence_call = jnp.array(0, dtype=jnp.int32)

        new_state, events, r_A, r_D = resolve_tick(
            state=state,
            cand_crosser_pos=cand_c_pos,
            cand_crosser_vel=cand_c_vel,
            cand_crosser_stamina=cand_c_stamina,
            cand_crosser_facing=cand_c_facing,
            cand_guard_x=cand_g_x,
            cand_guard_vx=cand_g_vx,
            cand_guard_facing=cand_g_facing,
            cand_guard_turn_timer=cand_g_turn_timer,
            cand_guard_stamina=cand_g_stamina,
            cand_kili_pos=cand_k_pos,
            cand_kili_vel=cand_k_vel,
            cand_kili_facing=cand_k_facing,
            cand_kili_turn_timer=cand_k_turn_timer,
            cand_kili_stamina=cand_k_stamina,
            cand_kili_line_type=cand_k_line_type,
            cand_kili_line_idx=cand_k_line_idx,
            kili_clap_action=actions.kili_clap,
            attack_call=attack_call,
            defence_call=defence_call,
            params=p,
        )

        obs = self.get_observations(new_state, p)
        return obs, new_state, r_A, r_D, events

    def get_observations(self, state: EnvState, params: EnvParams) -> Observations:
        """Compute all player observations and tokens."""
        # Tokens from perspective of each player slot
        all_tokens = jax.vmap(lambda i: observe_agent(i, state, params, is_critic=False)[0])(
            jnp.arange(MAX_PLAYERS)
        )
        _, player_masks = observe_agent(0, state, params)
        line_tokens, line_mask = build_line_tokens(params)
        global_tok = build_global_token(state, params)

        return Observations(
            player_tokens=all_tokens,
            player_masks=player_masks,
            line_tokens=line_tokens,
            line_mask=line_mask,
            global_token=global_tok,
        )

    def critic_state(self, state: EnvState, params: EnvParams | None = None) -> Observations:
        """Centralised critic view: full state, no fog, all stamina visible."""
        p = params or self.params
        all_tokens = jax.vmap(lambda i: observe_agent(i, state, p, is_critic=True)[0])(
            jnp.arange(MAX_PLAYERS)
        )
        _, player_masks = observe_agent(0, state, p)
        line_tokens, line_mask = build_line_tokens(p)
        global_tok = build_global_token(state, p)

        return Observations(
            player_tokens=all_tokens,
            player_masks=player_masks,
            line_tokens=line_tokens,
            line_mask=line_mask,
            global_token=global_tok,
        )


def make_env(params: EnvParams | None = None) -> KilithattuEnv:
    """Factory function for creating a KilithattuEnv instance."""
    return KilithattuEnv(params=params)
