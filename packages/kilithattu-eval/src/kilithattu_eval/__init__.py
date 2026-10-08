"""Kilithattu-Eval: Tournaments, ratings, metrics, and balancing suites."""

from kilithattu_eval.matches import InningsResult, MatchResult, play_full_match, play_single_innings
from kilithattu_eval.metrics import compute_innings_metrics
from kilithattu_eval.ratings import EloTracker, compute_nash_averaging
from kilithattu_eval.tactics import TacticalEventCounts, TacticsDetector
from kilithattu_eval.tournament import run_tournament

__version__ = "0.1.0"

__all__ = [
    "EloTracker",
    "InningsResult",
    "MatchResult",
    "TacticalEventCounts",
    "TacticsDetector",
    "compute_innings_metrics",
    "compute_nash_averaging",
    "play_full_match",
    "play_single_innings",
    "run_tournament",
]
