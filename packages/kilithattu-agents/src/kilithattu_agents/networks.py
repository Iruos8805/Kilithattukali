"""Entity-attention Transformer actor and centralized critic networks (Flax Linen)."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from chex import Array
from kilithattu_core.spaces import (
    CROSSER_ACTION_DIMS,
    GUARD_ACTION_DIMS,
    KILI_ACTION_DIMS,
    MAX_PLAYERS,
)


class MultiHeadAttentionBlock(nn.Module):
    d_model: int = 128
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        # Self-attention with residual connection and LayerNorm
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.d_model)
        # mask shape: [batch, 1, 1, seq_len] for attention
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]
        h = attn(x, x, mask=attn_mask)
        x = nn.LayerNorm()(x + h)

        # Feed-forward block
        ff = nn.Dense(features=self.d_model * 2)(x)
        ff = nn.relu(ff)
        ff = nn.Dense(features=self.d_model)(ff)
        x = nn.LayerNorm()(x + ff)
        return x


class EntityTransformerEncoder(nn.Module):
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        player_tokens: Array,  # [B, MAX_PLAYERS, 27]
        line_tokens: Array,  # [B, MAX_LINES, 8]
        global_token: Array,  # [B, 20]
        player_mask: Array,  # [B, MAX_PLAYERS]
        line_mask: Array,  # [B, MAX_LINES]
    ) -> tuple[Array, Array, Array]:
        B = player_tokens.shape[0]

        # Embeddings
        p_emb = nn.Dense(self.d_model, name="proj_player")(player_tokens)
        l_emb = nn.Dense(self.d_model, name="proj_line")(line_tokens)
        g_emb = nn.Dense(self.d_model, name="proj_global")(global_token)[:, None, :]  # [B, 1, d]

        # Concatenate into sequence of entities: [B, 1 + MAX_PLAYERS + MAX_LINES, d]
        all_entities = jnp.concatenate([g_emb, p_emb, l_emb], axis=1)

        # Combined attention mask
        g_mask = jnp.ones((B, 1), dtype=jnp.bool_)
        combined_mask = jnp.concatenate([g_mask, player_mask, line_mask], axis=1)

        x = all_entities
        for i in range(self.num_layers):
            x = MultiHeadAttentionBlock(
                d_model=self.d_model, num_heads=self.num_heads, name=f"attn_{i}"
            )(x, mask=combined_mask)

        # Extract outputs
        global_out = x[:, 0]  # [B, d]
        players_out = x[:, 1 : 1 + MAX_PLAYERS]  # [B, MAX_PLAYERS, d]

        # Masked mean pool of all entities
        weights = combined_mask.astype(jnp.float32)[:, :, None]
        pooled_out = jnp.sum(x * weights, axis=1) / jnp.maximum(
            jnp.sum(weights, axis=1), 1.0
        )  # [B, d]

        return global_out, players_out, pooled_out


class KilithattuActorCritic(nn.Module):
    """Actor-Critic model supporting Crosser, Guard, and Kili action heads."""

    d_model: int = 128
    rnn_dim: int = 256

    @nn.compact
    def __call__(
        self,
        player_tokens: Array,  # [B, MAX_PLAYERS, 27]
        line_tokens: Array,  # [B, MAX_LINES, 8]
        global_token: Array,  # [B, 20]
        player_mask: Array,  # [B, MAX_PLAYERS]
        line_mask: Array,  # [B, MAX_LINES]
        ego_idx: int = 0,  # 0..MAX_PLAYERS-1
        rnn_state: Array | None = None,  # [B, 256]
    ) -> tuple[dict[str, Array], Array, Array]:
        B = player_tokens.shape[0]

        # 1. Entity transformer encoder
        g_out, p_out, pool_out = EntityTransformerEncoder(d_model=self.d_model)(
            player_tokens, line_tokens, global_token, player_mask, line_mask
        )

        ego_out = p_out[:, ego_idx]  # [B, d]

        # 2. Concat 384 -> MLP 256
        combined = jnp.concatenate([ego_out, pool_out, g_out], axis=-1)
        feat = nn.Dense(features=self.rnn_dim, name="mlp_dense")(combined)
        feat = nn.relu(feat)

        # 3. GRU Cell
        if rnn_state is None:
            rnn_state = jnp.zeros((B, self.rnn_dim), dtype=jnp.float32)

        rnn_cell = nn.GRUCell(features=self.rnn_dim, name="gru")
        new_rnn_state, rnn_out = rnn_cell(rnn_state, feat)

        # 4. Action Heads
        # Crosser heads
        c_move = nn.Dense(CROSSER_ACTION_DIMS[0], name="c_move")(rnn_out)
        c_sprint = nn.Dense(CROSSER_ACTION_DIMS[1], name="c_sprint")(rnn_out)
        c_call = nn.Dense(CROSSER_ACTION_DIMS[2], name="c_call")(rnn_out)

        # Guard heads
        g_move = nn.Dense(GUARD_ACTION_DIMS[0], name="g_move")(rnn_out)
        g_dash = nn.Dense(GUARD_ACTION_DIMS[1], name="g_dash")(rnn_out)
        g_turn = nn.Dense(GUARD_ACTION_DIMS[2], name="g_turn")(rnn_out)

        # Kili heads
        k_move = nn.Dense(KILI_ACTION_DIMS[0], name="k_move")(rnn_out)
        k_dash = nn.Dense(KILI_ACTION_DIMS[1], name="k_dash")(rnn_out)
        k_turn = nn.Dense(KILI_ACTION_DIMS[2], name="k_turn")(rnn_out)
        k_clap = nn.Dense(KILI_ACTION_DIMS[3], name="k_clap")(rnn_out)

        logits = {
            "c_move": c_move,
            "c_sprint": c_sprint,
            "c_call": c_call,
            "g_move": g_move,
            "g_dash": g_dash,
            "g_turn": g_turn,
            "k_move": k_move,
            "k_dash": k_dash,
            "k_turn": k_turn,
            "k_clap": k_clap,
        }

        # 5. Value Head (Centralized Critic estimate)
        critic_mlp = nn.Dense(128, name="critic_dense")(jnp.concatenate([g_out, pool_out], axis=-1))
        critic_mlp = nn.relu(critic_mlp)
        value = nn.Dense(1, name="critic_value")(critic_mlp).squeeze(-1)

        return logits, value, new_rnn_state
