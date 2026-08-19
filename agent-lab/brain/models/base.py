"""Backend-neutral model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core.budget import InferenceBudget
from ..core.types import BrainState, ModelOutput, Signal


class ModelAdapter(ABC):
    """A model call that can be audited and budgeted.

    Real LLM adapters can implement the same method later.  The first pilot
    intentionally uses a weak deterministic adapter so it validates the harness
    without API cost or hidden model variance.
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
