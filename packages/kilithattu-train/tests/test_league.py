"""Unit tests for self-play league and partner pool."""

import numpy as np
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_agents.scripted.guard import ScriptedGuardPolicy
from kilithattu_train.league.manager import LeagueManager
from kilithattu_train.league.matchmaker import PFSPMatchmaker
from kilithattu_train.league.payoff import LeaguePayoff
from kilithattu_train.league.pool import SnapshotPool
from kilithattu_train.partners import FCPPartnerPool


def test_snapshot_pool():
    pool = SnapshotPool()
    p1 = ScriptedCrosserPolicy(aggression=0.5)
    p2 = ScriptedGuardPolicy(anticipation=0.5)

    pool.add_snapshot("atk_1", p1, "attack", step=100)
    pool.add_snapshot("def_1", p2, "defence", step=100)

    assert pool.list_ids("attack") == ["atk_1"]
    assert pool.list_ids("defence") == ["def_1"]

    # Record 3 matches where atk_1 won 2
    pool.record_result("atk_1", "def_1", a_won=True)
    pool.record_result("atk_1", "def_1", a_won=True)
    pool.record_result("atk_1", "def_1", a_won=False)

    wr = pool.get_win_rate("atk_1", "def_1")
    # Laplace smoothing: (2 + 1) / (3 + 2) = 3/5 = 0.6
    assert abs(wr - 0.6) < 1e-4


def test_pfsp_matchmaker():
    pool = SnapshotPool()
    p1 = ScriptedCrosserPolicy()
    p2_weak = ScriptedGuardPolicy(anticipation=0.1)
    p2_strong = ScriptedGuardPolicy(anticipation=0.9)

    pool.add_snapshot("atk_main", p1, "attack", step=0)
    pool.add_snapshot("def_weak", p2_weak, "defence", step=0)
    pool.add_snapshot("def_strong", p2_strong, "defence", step=0)

    # atk_main beats def_weak 10 times, loses to def_strong 10 times
    for _ in range(10):
        pool.record_result("atk_main", "def_weak", a_won=True)
        pool.record_result("atk_main", "def_strong", a_won=False)

    mm = PFSPMatchmaker(pool, rng=np.random.default_rng(42))
    # PFSP should prioritize def_strong because (1 - p_win)^2 is much higher
    samples = [mm.sample_pfsp("atk_main", ["def_weak", "def_strong"]) for _ in range(100)]
    strong_count = samples.count("def_strong")
    assert strong_count > 60


def test_league_payoff_and_nash():
    payoff = LeaguePayoff(initial_mu_ref=8.0)
    assert payoff.mu_ref == 8.0

    # Record scores: [12, 10, 4, 6]
    payoff.record_innings("atk_1", "def_1", 12)  # win
    payoff.record_innings("atk_1", "def_1", 10)  # win
    payoff.record_innings("atk_1", "def_1", 4)  # loss
    payoff.record_innings("atk_1", "def_1", 6)  # loss

    matrix = payoff.build_payoff_matrix(["atk_1"], ["def_1"])
    # 2 wins, 4 innings -> (2 + 1) / (4 + 2) = 0.5
    assert abs(matrix[0, 0] - 0.5) < 1e-4

    # Nash calculation on a 2x2 matrix
    m2 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    a_strat, d_strat, val = LeaguePayoff.solve_nash_fictitious_play(m2, iterations=500)
    assert len(a_strat) == 2
    assert len(d_strat) == 2
    assert 0.0 <= val <= 1.0


def test_league_manager():
    manager = LeagueManager()
    manager.register_learner("main_atk", "attack", "main")
    manager.register_learner("main_def", "defence", "main")

    p_atk = ScriptedCrosserPolicy()
    p_def = ScriptedGuardPolicy()

    s_atk = manager.snapshot("main_atk", p_atk, step=50)
    s_def = manager.snapshot("main_def", p_def, step=50)
    assert s_atk in manager.pool.entries
    assert s_def in manager.pool.entries

    matchups = manager.sample_matchups()
    assert "main_atk" in matchups
    assert "main_def" in matchups

    manager.record_matchup("main_atk", "main_def", [12, 14, 8])
    ladder = manager.build_ladder()
    assert "easy" in ladder
    assert "hard" in ladder


def test_fcp_partner_pool():
    fcp = FCPPartnerPool(rng=np.random.default_rng(123))
    p_crosser = fcp.sample_partner("crosser")
    assert p_crosser is not None

    p_guard = fcp.sample_partner("guard")
    assert p_guard is not None

    # Test sampling crosser slots
    assignment = fcp.sample_crosser_slots(num_crossers=4, num_learner_slots=2)
    assert len(assignment) == 4
    learner_count = sum(1 for v in assignment.values() if v == "learner")
    partner_count = sum(1 for v in assignment.values() if v != "learner")
    assert learner_count == 2
    assert partner_count == 2
