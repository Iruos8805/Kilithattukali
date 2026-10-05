"""Policy registry and resolution for scripted and learned agents."""

from __future__ import annotations

from typing import Any

from kilithattu_agents.human_fair import HumanFairPolicyWrapper
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
        - "scripted:guard?aggression=0.8"
        - "main@easy"
        - "main@medium"
        - "main@hard"
    """
    clean_ref = ref.strip().lower()

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
        return HumanFairPolicyWrapper(base, obs_delay_ticks=5, decision_interval_ticks=3, sticky_action_prob=0.10)
    elif clean_ref in ("medium", "main@medium"):
        base = ScriptedCrosserPolicy(aggression=0.7, safety_margin=0.8)
        return HumanFairPolicyWrapper(base, obs_delay_ticks=3, decision_interval_ticks=2, sticky_action_prob=0.05)
    elif clean_ref in ("hard", "main@hard"):
        base = ScriptedCrosserPolicy(aggression=0.9, safety_margin=0.5)
        return HumanFairPolicyWrapper(base, obs_delay_ticks=1, decision_interval_ticks=1, sticky_action_prob=0.0)
    else:
        # Default fallback to balanced scripted policy
        return ScriptedCrosserPolicy()
