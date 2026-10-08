"""Learned Neural Network Policy backed by Flax Entity-Transformer Actor-Critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from chex import PRNGKey
from kilithattu_core.env import Observations
from kilithattu_core.spaces import MAX_CROSSERS, MAX_GUARDS

from kilithattu_agents.checkpoint import load_model_checkpoint
from kilithattu_agents.networks import KilithattuActorCritic
from kilithattu_agents.policy import Policy, PolicyInfo


class LearnedPolicy(Policy):
    """Deep Reinforcement Learning policy using the KilithattuActorCritic Transformer."""

    def __init__(
        self,
        params: Any,
        model: KilithattuActorCritic | None = None,
        deterministic: bool = False,
    ) -> None:
        self.params = params
        self.model = model or KilithattuActorCritic(d_model=64, rnn_dim=128)
        self.deterministic = deterministic

    def init_carry(self, num_agents: int = 1) -> Any:
        return jnp.zeros((num_agents, self.model.rnn_dim), dtype=jnp.float32)

    def act(
        self,
        key: PRNGKey,
        carry: Any,
        obs: Observations,
        agent_idx: int,
        mask: Any = None,
    ) -> tuple[dict[str, int], Any, PolicyInfo]:
        if carry is None:
            carry = self.init_carry(1)

        # Ensure batched inputs of shape [1, ...]
        p_tokens = obs.player_tokens[agent_idx][None, ...]
        l_tokens = obs.line_tokens[None, ...]
        g_token = obs.global_token[None, ...]
        p_mask = obs.player_masks[None, ...]
        l_mask = obs.line_mask[None, ...]

        logits, value, new_carry = self.model.apply(
            self.params,
            p_tokens,
            l_tokens,
            g_token,
            p_mask,
            l_mask,
            ego_idx=agent_idx,
            rnn_state=carry,
        )

        k1, k2, k3, k4 = jax.random.split(key, 4)

        if agent_idx < MAX_CROSSERS:
            # Crosser action heads
            c_move = logits["c_move"][0]
            c_sprint = logits["c_sprint"][0]
            c_call = logits["c_call"][0]

            if self.deterministic:
                move = int(jnp.argmax(c_move))
                sprint = int(jnp.argmax(c_sprint))
                call = int(jnp.argmax(c_call))
            else:
                move = int(jax.random.categorical(k1, c_move))
                sprint = int(jax.random.categorical(k2, c_sprint))
                call = int(jax.random.categorical(k3, c_call))

            action = {"move": move, "sprint": sprint, "call": call}

        elif agent_idx < MAX_CROSSERS + MAX_GUARDS:
            # Guard action heads
            g_move = logits["g_move"][0]
            g_dash = logits["g_dash"][0]
            g_turn = logits["g_turn"][0]

            if self.deterministic:
                move = int(jnp.argmax(g_move))
                dash = int(jnp.argmax(g_dash))
                turn = int(jnp.argmax(g_turn))
            else:
                move = int(jax.random.categorical(k1, g_move))
                dash = int(jax.random.categorical(k2, g_dash))
                turn = int(jax.random.categorical(k3, g_turn))

            action = {"move": move, "dash": dash, "turn": turn}

        else:
            # Kili action heads
            k_move = logits["k_move"][0]
            k_dash = logits["k_dash"][0]
            k_turn = logits["k_turn"][0]
            k_clap = logits["k_clap"][0]

            if self.deterministic:
                move = int(jnp.argmax(k_move))
                dash = int(jnp.argmax(k_dash))
                turn = int(jnp.argmax(k_turn))
                clap = int(jnp.argmax(k_clap))
            else:
                move = int(jax.random.categorical(k1, k_move))
                dash = int(jax.random.categorical(k2, k_dash))
                turn = int(jax.random.categorical(k3, k_turn))
                clap = int(jax.random.categorical(k4, k_clap))

            action = {"move": move, "dash": dash, "turn": turn, "clap": clap}

        val = float(value[0])
        return action, new_carry, PolicyInfo(value=jnp.array(val))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        deterministic: bool = False,
    ) -> LearnedPolicy:
        params, metadata = load_model_checkpoint(checkpoint_dir)
        d_model = metadata.config.get("d_model", 64) if metadata.config else 64
        rnn_dim = metadata.config.get("rnn_dim", 128) if metadata.config else 128
        model = KilithattuActorCritic(d_model=d_model, rnn_dim=rnn_dim)
        return cls(params=params, model=model, deterministic=deterministic)
