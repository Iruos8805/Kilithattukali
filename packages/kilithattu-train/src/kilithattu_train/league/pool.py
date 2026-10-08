"""Snapshot pool for self-play league training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kilithattu_agents.policy import Policy


@dataclass
class SnapshotEntry:
    id: str
    policy: Policy
    side: str  # "attack" | "defence"
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)  # opponent_id -> win count
    matches: dict[str, int] = field(default_factory=dict)  # opponent_id -> match count


class SnapshotPool:
    """Stores frozen policy snapshots and empirical head-to-head win records."""

    def __init__(self) -> None:
        self.entries: dict[str, SnapshotEntry] = {}

    def add_snapshot(
        self,
        snapshot_id: str,
        policy: Policy,
        side: str,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotEntry:
        if side not in ("attack", "defence"):
            raise ValueError(f"side must be 'attack' or 'defence', got '{side}'")
        entry = SnapshotEntry(
            id=snapshot_id,
            policy=policy,
            side=side,
            step=step,
            metadata=metadata or {},
        )
        self.entries[snapshot_id] = entry
        return entry

    def get(self, snapshot_id: str) -> SnapshotEntry:
        return self.entries[snapshot_id]

    def list_ids(self, side: str | None = None) -> list[str]:
        if side is None:
            return list(self.entries.keys())
        return [eid for eid, e in self.entries.items() if e.side == side]

    def record_result(self, id_a: str, id_b: str, a_won: bool) -> None:
        """Record an outcome between id_a and id_b."""
        if id_a in self.entries:
            entry_a = self.entries[id_a]
            entry_a.matches[id_b] = entry_a.matches.get(id_b, 0) + 1
            if a_won:
                entry_a.wins[id_b] = entry_a.wins.get(id_b, 0) + 1

        if id_b in self.entries:
            entry_b = self.entries[id_b]
            entry_b.matches[id_a] = entry_b.matches.get(id_a, 0) + 1
            if not a_won:
                entry_b.wins[id_a] = entry_b.wins.get(id_a, 0) + 1

    def get_win_rate(self, id_a: str, id_b: str) -> float:
        """Return empirical win rate of id_a vs id_b with Laplace smoothing (default 0.5)."""
        if id_a not in self.entries:
            return 0.5
        entry = self.entries[id_a]
        matches = entry.matches.get(id_b, 0)
        wins = entry.wins.get(id_b, 0)
        # Laplace smoothing: (wins + 1) / (matches + 2)
        return (wins + 1.0) / (matches + 2.0)
