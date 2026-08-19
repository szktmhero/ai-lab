"""Observation-to-signal conversion without evaluator information."""

from __future__ import annotations

import math

from ..core.types import EvidenceEvent, Signal


def vector_novelty(previous: tuple[float, ...] | None, current: tuple[float, ...]) -> float:
    """Normalized vector distance with explicit all-zero behavior."""

    if previous is None:
        return 0.0
    previous_norm = math.sqrt(sum(value * value for value in previous))
    current_norm = math.sqrt(sum(value * value for value in current))
    if previous_norm == 0.0 and current_norm == 0.0:
        return 0.0
    distance = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(previous, current))
    )
    return min(distance / (previous_norm + current_norm + 1e-12), 1.0)


def _score_confidence(scores: tuple[float, ...]) -> float:
    ordered = sorted(scores, reverse=True)
    if not ordered or ordered[0] <= 0.0:
        return 0.0
    second = ordered[1] if len(ordered) > 1 else 0.0
    return min(max((ordered[0] - second) / (abs(ordered[0]) + 1e-12), 0.0), 1.0)


class PerceptionGateway:
    def __init__(self) -> None:
        self._previous_by_cue: dict[str, tuple[float, ...]] = {}

    def perceive(self, episode_id: str, event: EvidenceEvent) -> tuple[Signal, float]:
        novelty = vector_novelty(self._previous_by_cue.get(event.cue), event.scores)
        self._previous_by_cue[event.cue] = event.scores
        signal = Signal(
            signal_id=f"signal:{event.event_id}",
            episode_id=episode_id,
            step=event.step,
            cue=event.cue,
            scores=event.scores,
            source=event.source,
            kind="evidence",
            salience=min(event.salience + 0.15 * novelty, 1.0),
            confidence=_score_confidence(event.scores),
            provenance=(event.event_id,),
        )
        return signal, novelty
