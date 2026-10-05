"""Tests for scripted policies and registry loading."""

import jax
import pytest

from kilithattu_agents.policy import Policy
from kilithattu_agents.registry import load_policy
from kilithattu_agents.scripted import ScriptedCrosserPolicy, ScriptedGuardPolicy, ScriptedKiliPolicy
from kilithattu_core.config import EnvParams
from kilithattu_core.env import KilithattuEnv


def test_scripted_policies_step():
    params = EnvParams()
    env = KilithattuEnv(params)
    key = jax.random.PRNGKey(101)
    obs, state = env.reset(key, params)

    crosser = ScriptedCrosserPolicy()
    guard = ScriptedGuardPolicy()
    kili = ScriptedKiliPolicy()

    # Test act
    c_act, _, _ = crosser.act(key, None, obs, agent_idx=0)
    assert "move" in c_act
    assert 0 <= c_act["move"] <= 8

    g_act, _, _ = guard.act(key, None, obs, agent_idx=params.num_crossers)
    assert "move" in g_act
    assert 0 <= g_act["move"] <= 2

    k_act, _, _ = kili.act(key, None, obs, agent_idx=params.num_crossers + params.num_guards)
    assert "move" in k_act
    assert 0 <= k_act["move"] <= 4


def test_registry_loading():
    p1 = load_policy("scripted:crosser")
    assert isinstance(p1, ScriptedCrosserPolicy)

    p2 = load_policy("scripted:guard")
    assert isinstance(p2, ScriptedGuardPolicy)

    p3 = load_policy("scripted:kili")
    assert isinstance(p3, ScriptedKiliPolicy)

    p4 = load_policy("medium")
    assert hasattr(p4, "act")
