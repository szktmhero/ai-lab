"""Budget enforcement shared by every architecture condition."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Usage


class BudgetExceeded(RuntimeError):
    """Raised before a model call would exceed its declared episode budget."""


@dataclass(frozen=True)
class BudgetLimits:
    """Immutable per-episode limits shared by every comparison condition."""

    max_calls: int = 4
    max_input_units: int = 48
    max_output_units: int = 8
    unit_name: str = "signals"

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if self.max_input_units < 0 or self.max_output_units < 0:
            raise ValueError("budget limits must be non-negative")
        if not self.unit_name:
            raise ValueError("unit_name must not be empty")

    def create_budget(self) -> InferenceBudget:
        return InferenceBudget(
            max_calls=self.max_calls,
            max_input_units=self.max_input_units,
            max_output_units=self.max_output_units,
            unit_name=self.unit_name,
        )


@dataclass
class InferenceBudget:
    max_calls: int = 4
    max_input_units: int = 48
    max_output_units: int = 8
    unit_name: str = "signals"
    model_calls: int = 0
    input_units: int = 0
    output_units: int = 0

    def ensure_can_consume(
        self,
        input_units: int,
        output_units: int = 1,
    ) -> None:
        """Reject a call before execution without mutating the ledger.

        Real adapters use the provider's token-count endpoint here and reserve
        their configured maximum output.  The response's actual usage is only
        committed by :meth:`consume` after the inference succeeds.
        """

        if input_units < 0 or output_units < 0:
            raise ValueError("budget units must be non-negative")
        if self.model_calls + 1 > self.max_calls:
            raise BudgetExceeded("model call budget exceeded")
        if self.input_units + input_units > self.max_input_units:
            raise BudgetExceeded("model input budget exceeded")
        if self.output_units + output_units > self.max_output_units:
            raise BudgetExceeded("model output budget exceeded")

    def consume(self, input_units: int, output_units: int = 1) -> None:
        self.ensure_can_consume(input_units, output_units)
        self.model_calls += 1
        self.input_units += input_units
        self.output_units += output_units

    def snapshot(self) -> Usage:
        return Usage(
            model_calls=self.model_calls,
            input_units=self.input_units,
            output_units=self.output_units,
            unit_name=self.unit_name,
        )
