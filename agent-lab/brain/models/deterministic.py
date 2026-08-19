"""Weak, deterministic evidence processor used by the architecture pilot."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.budget import InferenceBudget
from ..core.types import BrainState, ModelOutput, Signal
from .base import ModelAdapter


def _confidence(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted((max(value, 0.0) for value in scores), reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    if top <= 0.0:
        return 0.0
    return min(max((top - second) / (top + second + 1e-12), 0.0), 1.0)


class DeterministicModelAdapter(ModelAdapter):
    """Aggregate structured evidence without access to evaluator truth.

    The modes are deliberately small and inspectable:

    - ``fast``: raw score sum over the two newest relevant signals;
    - ``deliberate``: raw score sum over all relevant signals;
    - ``robust``: one vote per observation, limiting a single extreme value;
    - ``integrate``: confidence-weighted hypothesis integration.

    This is a test double, not a claim that these operations are themselves a
    brain or an intelligent model.
    """

    def infer(
        self,
        signals: Sequence[Signal],
        query_cue: str,
        candidate_count: int,
        mode: str,
        state: BrainState,
        budget: InferenceBudget,
    ) -> ModelOutput:
        del state  # State affects routing; the adapter itself is condition-invariant.
        budget.consume(input_units=len(signals), output_units=1)

        if mode == "integrate":
            relevant = [signal for signal in signals if signal.kind == "hypothesis"]
        else:
            relevant = [
                signal
                for signal in signals
                if signal.kind in {"evidence", "memory"} and signal.cue == query_cue
            ]
            # A system without the required memory still has to act.  Its
            # fallback is recent unrelated evidence, never evaluator truth.
            if not relevant:
                relevant = [
                    signal
                    for signal in signals
                    if signal.kind in {"evidence", "memory"}
                ][-2:]

        if mode == "fast":
            relevant = relevant[-2:]

        totals = [0.0] * candidate_count
        if mode == "robust":
            raw = [0.0] * candidate_count
            for signal in relevant:
                winner = max(
                    range(candidate_count),
                    key=lambda index: (signal.scores[index], -index),
                )
                totals[winner] += 1.0
                for index, value in enumerate(signal.scores):
                    raw[index] += value
            # Raw evidence only breaks equal vote counts.
            scale = max((abs(value) for value in raw), default=1.0) or 1.0
            totals = [vote + 1e-4 * value / scale for vote, value in zip(totals, raw)]
        else:
            for offset, signal in enumerate(relevant):
                recency_weight = 1.0
                if mode == "fast" and len(relevant) > 1:
                    recency_weight = 0.8 + 0.2 * offset / (len(relevant) - 1)
                for index, value in enumerate(signal.scores):
                    totals[index] += value * recency_weight

        candidate = max(
            range(candidate_count),
            key=lambda index: (totals[index], -index),
        )
        return ModelOutput(
            candidate=candidate,
            confidence=_confidence(totals),
            scores=tuple(totals),
            mode=mode,
            used_signal_ids=tuple(signal.signal_id for signal in relevant),
        )
