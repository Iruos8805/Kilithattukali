"""Smoke tests for match running, metrics calculation, and Elo rating."""

import jax
import pytest
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_agents.scripted.guard import ScriptedGuardPolicy
from kilithattu_agents.scripted.kili import ScriptedKiliPolicy
from kilithattu_core.config import EnvParams
from kilithattu_eval.matches import play_single_innings
from kilithattu_eval.metrics import compute_innings_metrics
from kilithattu_eval.ratings import EloTracker


def test_metrics_calculation():
    scores = [10, 18, 0, 8]
    delivs = [1, 1, 0, 0]
    all_outs = [False, False, True, True]

    metrics = compute_innings_metrics(scores, delivs, all_outs)
    assert metrics["ppi"] == 9.0
    assert metrics["all_out_rate"] == 0.5
    assert metrics["deliveries_per_innings"] == 0.5


def test_elo_tracker():
    tracker = EloTracker(k_factor=32.0, default_rating=1000.0)
    r_a, r_b = tracker.record_match("team_a", "team_b", 1.0)
    assert r_a > 1000.0
    assert r_b < 1000.0
    assert r_a + r_b == pytest.approx(2000.0)


def test_play_single_innings_smoke():
    params = EnvParams(innings_duration=5.0)
    crosser = ScriptedCrosserPolicy()
    guard = ScriptedGuardPolicy()
    kili = ScriptedKiliPolicy()
    key = jax.random.PRNGKey(42)

    res = play_single_innings(crosser, guard, kili, params, key, max_ticks=20)
    assert res.ticks == 20
    assert res.points_attack >= 0
