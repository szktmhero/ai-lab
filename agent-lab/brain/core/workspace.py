"""Capacity- and lifetime-bounded global workspace."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Signal


@dataclass(frozen=True)
class WorkspaceItem:
    signal: Signal
    inserted_step: int
    expires_at: int
    priority: float


class BoundedWorkspace:
    def __init__(self, capacity: int = 4, ttl: int = 3):
        if capacity <= 0 or ttl < 0:
            raise ValueError("workspace capacity must be positive and ttl non-negative")
        self.capacity = capacity
        self.ttl = ttl
        self._items: list[WorkspaceItem] = []
        self.turnover_count = 0
        self.peak_size = 0
        self._size_samples: list[int] = []

    def _expire(self, step: int) -> None:
        retained = [item for item in self._items if item.expires_at >= step]
        self.turnover_count += len(self._items) - len(retained)
        self._items = retained

    def add(self, signal: Signal) -> None:
        self._expire(signal.step)
        self._items.append(
            WorkspaceItem(
                signal=signal,
                inserted_step=signal.step,
                expires_at=signal.step + self.ttl,
                priority=signal.salience,
            )
        )
        if len(self._items) > self.capacity:
            # Recency wins ties.  Salience can delay eviction but cannot defeat
            # expiry, which prevents the workspace becoming permanent memory.
            victim = min(
                self._items,
                key=lambda item: (item.priority, item.inserted_step, item.signal.signal_id),
            )
            self._items.remove(victim)
            self.turnover_count += 1
        self.peak_size = max(self.peak_size, len(self._items))
        self._size_samples.append(len(self._items))

    def signals(self, step: int, cue: str | None = None) -> list[Signal]:
        self._expire(step)
        result = [item.signal for item in self._items]
        if cue is not None:
            result = [signal for signal in result if signal.cue == cue]
        return sorted(result, key=lambda signal: (signal.step, signal.signal_id))

    @property
    def average_utilization(self) -> float:
        if not self._size_samples:
            return 0.0
        return sum(self._size_samples) / (len(self._size_samples) * self.capacity)

    def clear(self) -> None:
        self._items.clear()
        self.turnover_count = 0
        self.peak_size = 0
        self._size_samples.clear()
