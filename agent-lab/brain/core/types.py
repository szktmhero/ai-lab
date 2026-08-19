"""Typed public observations, internal signals, and experiment results.

The evaluator-only :class:`Task` deliberately wraps a separate :class:`TaskView`.
Only ``TaskView`` may be passed into a system under test, which makes answer
leakage both visible in code review and easy to test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """Procedural task families used by the first experiment."""

    LATENT_RULE_SWITCH = "latent_rule_switch"
    DELAYED_RECALL = "delayed_recall"
    FAULT_RECOVERY = "fault_recovery"
    PARTIAL_OBSERVATION = "partial_observation_plan"


@dataclass(frozen=True)
class EvidenceEvent:
    """One public piece of evidence.

    ``scores`` express support for public answer candidates.  They are noisy
    observations, not ground truth, and may intentionally conflict.
    """

    event_id: str
    step: int
    cue: str
    scores: tuple[float, ...]
    source: str
    salience: float = 0.5

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("scores must contain at least one candidate")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError("salience must be between 0 and 1")


@dataclass(frozen=True)
class TaskView:
    """The complete input visible to a system under test."""

    task_id: str
    task_type: TaskType
    observations: tuple[EvidenceEvent, ...]
    query_cue: str
    candidate_count: int

    def __post_init__(self) -> None:
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least two")
        if any(len(item.scores) != self.candidate_count for item in self.observations):
            raise ValueError("all evidence vectors must match candidate_count")


@dataclass(frozen=True)
class Task:
    """Evaluator-owned task including the hidden answer and annotations."""

    view: TaskView
    correct_answer: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.correct_answer < self.view.candidate_count:
            raise ValueError("correct_answer is outside the candidate range")


@dataclass(frozen=True)
class Signal:
    """A bounded internal message with explicit provenance."""

    signal_id: str
    episode_id: str
    step: int
    cue: str
    scores: tuple[float, ...]
    source: str
    kind: str
    salience: float
    confidence: float
    provenance: tuple[str, ...]


@dataclass
class BrainState:
    """Small set of inspectable dynamic variables.

    These values are not claims about human neurobiology.  They are observable
    control signals whose causal use can be disabled in an ablation.
    """

    uncertainty: float = 1.0
    novelty: float = 0.0
    prediction_error: float = 0.0
    cognitive_load: float = 0.0
    resource_pressure: float = 0.0

    def clamp(self) -> None:
        self.uncertainty = min(max(self.uncertainty, 0.0), 1.0)
        self.novelty = min(max(self.novelty, 0.0), 1.0)
        self.prediction_error = min(max(self.prediction_error, 0.0), 1.0)
        self.cognitive_load = min(max(self.cognitive_load, 0.0), 1.0)
        self.resource_pressure = min(max(self.resource_pressure, 0.0), 1.0)


@dataclass(frozen=True)
class RouterDecision:
    """Auditable routing result."""

    activated_modules: tuple[str, ...]
    retrieve_memory: bool
    use_robust_cortex: bool
    reason: str


@dataclass(frozen=True)
class ModelOutput:
    """One candidate hypothesis produced through a ModelAdapter."""

    candidate: int
    confidence: float
    scores: tuple[float, ...]
    mode: str
    used_signal_ids: tuple[str, ...]


@dataclass(frozen=True)
class Usage:
    """Model-budget accounting for one episode."""

    model_calls: int
    input_units: int
    output_units: int


@dataclass(frozen=True)
class SystemResult:
    """Final system decision plus a fully inspectable trace summary."""

    candidate: int
    confidence: float
    provenance: tuple[str, ...]
    usage: Usage
    trace: Mapping[str, Any]


def dominant_candidate(signals: Sequence[Signal], candidate_count: int) -> int | None:
    """Return the score-sum winner, or ``None`` when no signal is available."""

    if not signals:
        return None
    totals = [0.0] * candidate_count
    for signal in signals:
        for index, value in enumerate(signal.scores):
            totals[index] += value
    return max(range(candidate_count), key=lambda index: (totals[index], -index))
