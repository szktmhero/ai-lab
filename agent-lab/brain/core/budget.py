"""Budget enforcement shared by every architecture condition."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Usage


class BudgetExceeded(RuntimeError):
    """Raised before a model call would exceed its declared episode budget."""


@dataclass
class InferenceBudget:
    max_calls: int = 4
    max_input_units: int = 48
    max_output_units: int = 8
    model_calls: int = 0
    input_units: int = 0
    output_units: int = 0

    def consume(self, input_units: int, output_units: int = 1) -> None:
        if input_units < 0 or output_units < 0:
            raise ValueError("budget units must be non-negative")
        if self.model_calls + 1 > self.max_calls:
            raise BudgetExceeded("model call budget exceeded")
        if self.input_units + input_units > self.max_input_units:
            raise BudgetExceeded("model input budget exceeded")
        if self.output_units + output_units > self.max_output_units:
            raise BudgetExceeded("model output budget exceeded")
        self.model_calls += 1
        self.input_units += input_units
        self.output_units += output_units

    def snapshot(self) -> Usage:
        return Usage(
            model_calls=self.model_calls,
            input_units=self.input_units,
            output_units=self.output_units,
        )
