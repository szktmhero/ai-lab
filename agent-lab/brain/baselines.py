"""Non-brain comparison systems using the identical model adapter."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from .core.budget import InferenceBudget
from .core.system import _hypothesis_signal
from .core.types import (
    BrainState,
    ModelOutput,
    Signal,
    SystemResult,
    TaskView,
    dominant_candidate,
)
from .models.base import ModelAdapter
from .models.deterministic import DeterministicModelAdapter
from .modules.perception import PerceptionGateway
from .observer.observer import BrainObserver

if TYPE_CHECKING:
    from .core.system import BrainSystem

BASELINE_CONDITIONS = (
    "single_long_context",
    "sequential_reflection",
    "flat_ensemble",
)


class BaselineSystem:
    def __init__(self, condition: str, model: ModelAdapter | None = None):
        if condition not in BASELINE_CONDITIONS:
            raise ValueError(f"unknown baseline condition: {condition}")
        self.condition = condition
        self.model = model or DeterministicModelAdapter()

    def run(
        self,
        task: TaskView,
        observer: BrainObserver | None = None,
    ) -> SystemResult:
        observer = observer or BrainObserver()
        budget = InferenceBudget(max_calls=4, max_input_units=48, max_output_units=8)
        state = BrainState()
        perception = PerceptionGateway()
        signals: list[Signal] = []
        online_predictions: list[int | None] = []
        for event in task.observations:
            signal, novelty = perception.perceive(task.task_id, event)
            signals.append(signal)
            recent = signals[-4:]
            relevant = [item for item in recent if item.cue == task.query_cue]
            online_predictions.append(
                dominant_candidate(relevant or recent, task.candidate_count)
            )
            observer.record(
                task.task_id,
                event.step,
                "perception",
                signal_id=signal.signal_id,
                cue=signal.cue,
                novelty=novelty,
            )

        final_step = max((event.step for event in task.observations), default=-1) + 1
        outputs: list[ModelOutput] = []
        if self.condition == "single_long_context":
            final = self._infer(task, signals, "deliberate", state, budget)
            outputs.append(final)
        elif self.condition == "sequential_reflection":
            outputs.extend(
                [
                    self._infer(task, signals[-2:], "fast", state, budget),
                    self._infer(task, signals, "deliberate", state, budget),
                    self._infer(task, signals, "robust", state, budget),
                ]
            )
            final = self._integrate(task, outputs, state, budget, final_step)
        else:
            partitions = [signals[index::3] for index in range(3)]
            outputs.extend(
                self._infer(task, partition, "deliberate", state, budget)
                for partition in partitions
            )
            final = self._integrate(task, outputs, state, budget, final_step)

        provenance = tuple(
            dict.fromkeys(
                signal_id
                for output in outputs
                for signal_id in output.used_signal_ids
            )
        )
        observer.record(
            task.task_id,
            final_step,
            "final_action",
            candidate=final.candidate,
            confidence=final.confidence,
            provenance=list(provenance),
            usage=asdict(budget.snapshot()),
        )
        return SystemResult(
            candidate=final.candidate,
            confidence=final.confidence,
            provenance=provenance,
            usage=budget.snapshot(),
            trace={
                "condition": self.condition,
                "activated_modules": [self.condition],
                "router_reason": "baseline has no salience router",
                "robust_activated": self.condition == "sequential_reflection",
                "conflict_detected": False,
                "state": asdict(state),
                "fast_candidate": outputs[0].candidate if outputs else None,
                "deliberate_candidate": outputs[-1].candidate if outputs else None,
                "robust_candidate": outputs[-1].candidate
                if outputs and outputs[-1].mode == "robust"
                else None,
                "online_predictions": online_predictions,
                "workspace_turnover": 0,
                "workspace_peak": len(signals),
                "workspace_utilization": 1.0 if signals else 0.0,
                "memory_writes": 0,
                "memory_retrievals": 0,
                "memory_hits": 0,
                "retrieved_records": 0,
                "model_modes": [output.mode for output in outputs]
                + ([] if final in outputs else ["integrate"]),
                "invalid_actions": 0,
                "no_ops": 0,
            },
        )

    def _infer(
        self,
        task: TaskView,
        signals: list[Signal],
        mode: str,
        state: BrainState,
        budget: InferenceBudget,
    ) -> ModelOutput:
        return self.model.infer(
            signals,
            query_cue=task.query_cue,
            candidate_count=task.candidate_count,
            mode=mode,
            state=state,
            budget=budget,
        )

    def _integrate(
        self,
        task: TaskView,
        outputs: list[ModelOutput],
        state: BrainState,
        budget: InferenceBudget,
        step: int,
    ) -> ModelOutput:
        hypotheses = [
            _hypothesis_signal(
                task.task_id,
                step,
                output,
                task.candidate_count,
            )
            for output in outputs
        ]
        return self._infer(task, hypotheses, "integrate", state, budget)


def make_system(condition: str) -> BrainSystem | BaselineSystem:
    if condition in BASELINE_CONDITIONS:
        return BaselineSystem(condition)
    from .core.system import BRAIN_CONDITIONS, BrainSystem

    if condition in BRAIN_CONDITIONS:
        return BrainSystem(BRAIN_CONDITIONS[condition])
    raise ValueError(f"unknown condition: {condition}")
