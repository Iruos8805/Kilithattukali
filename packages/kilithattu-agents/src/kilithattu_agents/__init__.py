"""Kilithattu-Agents: Policy protocols, scripted baselines, and agent registry."""

from kilithattu_agents.checkpoint import (
    CheckpointMetadata,
    load_checkpoint_metadata,
    save_checkpoint_metadata,
)
from kilithattu_agents.human_fair import HumanFairPolicyWrapper
from kilithattu_agents.policy import Policy, PolicyInfo
from kilithattu_agents.registry import load_policy
from kilithattu_agents.scripted import (
    ScriptedCrosserPolicy,
    ScriptedGuardPolicy,
    ScriptedKiliPolicy,
)

__version__ = "0.1.0"

__all__ = [
    "CheckpointMetadata",
    "HumanFairPolicyWrapper",
    "Policy",
    "PolicyInfo",
    "ScriptedCrosserPolicy",
    "ScriptedGuardPolicy",
    "ScriptedKiliPolicy",
    "load_checkpoint_metadata",
    "load_policy",
    "save_checkpoint_metadata",
]
