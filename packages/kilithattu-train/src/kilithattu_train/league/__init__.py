"""Self-play league training components (§6.5)."""

from kilithattu_train.league.manager import LeagueManager, LearnerInfo
from kilithattu_train.league.matchmaker import PFSPMatchmaker
from kilithattu_train.league.payoff import LeaguePayoff, MatchupRecord
from kilithattu_train.league.pool import SnapshotEntry, SnapshotPool

__all__ = [
    "LeagueManager",
    "LearnerInfo",
    "LeaguePayoff",
    "MatchupRecord",
    "PFSPMatchmaker",
    "SnapshotEntry",
    "SnapshotPool",
]
