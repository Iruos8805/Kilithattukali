"""Policy registry and resolution for scripted and learned agents."""

from __future__ import annotations

from kilithattu_agents.human_fair import HumanFairPolicyWrapper
from kilithattu_agents.learned import LearnedPolicy
from kilithattu_agents.policy import Policy
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_agents.scripted.guard import ScriptedGuardPolicy
from kilithattu_agents.scripted.kili import ScriptedKiliPolicy


def load_policy(ref: str) -> Policy:
    """Resolve a policy reference string to an instantiated Policy.

    Examples:
        - "scripted:crosser"
        - "scripted:guard"
        - "scripted:kili"
        - "learned:crosser"
        - "checkpoint:checkpoints/stage_S0"
        - "main@easy"
        - "main@medium"
        - "main@hard"
    """
    clean_ref = ref.strip().lower()

    if clean_ref.startswith("checkpoint:"):
        path = ref.split("checkpoint:", 1)[1]
        return LearnedPolicy.from_checkpoint(path)

    if clean_ref.startswith("learned:") or clean_ref == "learned":
        from pathlib import Path
        for candidate in ["checkpoints/stage_S0", "checkpoints/best", "checkpoints/latest"]:
            if (Path(candidate) / "model.pkl").exists() or (Path(candidate) / "model.msgpack").exists():
                return LearnedPolicy.from_checkpoint(candidate)
        # If no checkpoint exists on disk yet, initialize initialized actor-critic
        import jax
        import jax.numpy as jnp
        from kilithattu_agents.networks import KilithattuActorCritic
        from kilithattu_core.spaces import MAX_PLAYERS, MAX_LINES

        model = KilithattuActorCritic(d_model=64, rnn_dim=128)
        key = jax.random.PRNGKey(42)
        p_tok = jnp.zeros((1, MAX_PLAYERS, 27))
        l_tok = jnp.zeros((1, MAX_LINES, 8))
        g_tok = jnp.zeros((1, 20))
        p_mask = jnp.ones((1, MAX_PLAYERS), dtype=jnp.bool_)
        l_mask = jnp.ones((1, MAX_LINES), dtype=jnp.bool_)
        params = model.init(key, p_tok, l_tok, g_tok, p_mask, l_mask, ego_idx=0)
        return LearnedPolicy(params=params, model=model)

    if clean_ref.startswith("scripted:"):
        role = clean_ref.split("scripted:")[1].split("?")[0]
        if role == "crosser":
            return ScriptedCrosserPolicy()
        elif role == "guard":
            return ScriptedGuardPolicy()
        elif role == "kili":
            return ScriptedKiliPolicy()
        else:
            raise ValueError(f"Unknown scripted role: {role}")

    # Difficulty ladder levels (wrapped with human-fair latencies §7.5)
    if clean_ref in ("beginner", "easy"):
        base = ScriptedCrosserPolicy(aggression=0.4, safety_margin=1.2)
        return HumanFairPolicyWrapper(
            base, obs_delay_ticks=5, decision_interval_ticks=3, sticky_action_prob=0.10
        )
    elif clean_ref in ("medium", "main@medium"):
        base = ScriptedCrosserPolicy(aggression=0.7, safety_margin=0.8)
        return HumanFairPolicyWrapper(
            base, obs_delay_ticks=3, decision_interval_ticks=2, sticky_action_prob=0.05
        )
    elif clean_ref in ("hard", "main@hard"):
        base = ScriptedCrosserPolicy(aggression=0.9, safety_margin=0.5)
        return HumanFairPolicyWrapper(
            base, obs_delay_ticks=1, decision_interval_ticks=1, sticky_action_prob=0.0
        )
    else:
        # Default fallback to balanced scripted policy
        return ScriptedCrosserPolicy()
