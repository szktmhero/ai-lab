"""Backend-neutral model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core.budget import InferenceBudget
from ..core.types import BrainState, ModelOutput, Signal


class ModelAdapter(ABC):
    """A model call that can be audited and budgeted.

    The deterministic pilot adapter and the Phase 3 Responses API adapter share
    this condition-invariant contract. Architecture code chooses the signals and
    mode; adapters account for every successful inference through the budget.
    """

    @abstractmethod
    def infer(
        self,
        signals: Sequence[Signal],
        query_cue: str,
        candidate_count: int,
        mode: str,
        state: BrainState,
        budget: InferenceBudget,
    ) -> ModelOutput:
        raise NotImplementedError
