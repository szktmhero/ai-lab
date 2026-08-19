"""Prediction-conflict monitor that never receives task correctness."""

from __future__ import annotations

from collections import Counter

from ..core.types import Signal


class ErrorMonitor:
    def __init__(self, conflict_threshold: float = 0.2):
        self.conflict_threshold = conflict_threshold
        self._signals: list[Signal] = []

    def observe(self, signal: Signal) -> None:
        if signal.kind == "evidence":
            self._signals.append(signal)

    def conflict(self, cue: str) -> float:
        relevant = [signal for signal in self._signals if signal.cue == cue]
        if len(relevant) < 2:
            return 0.0
        winners = [
            max(range(len(signal.scores)), key=lambda index: (signal.scores[index], -index))
            for signal in relevant
        ]
        largest_group = max(Counter(winners).values())
        return 1.0 - largest_group / len(winners)

    def detected(self, cue: str) -> bool:
        return self.conflict(cue) >= self.conflict_threshold
