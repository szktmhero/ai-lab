"""Recurrent Brain Kernel with independently ablatable causal paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from ..models.base import ModelAdapter
from ..models.deterministic import DeterministicModelAdapter
from ..modules.error_monitor import ErrorMonitor
from ..modules.perception import PerceptionGateway
from ..modules.router import SalienceRouter
from ..observer.observer import BrainObserver
from .budget import InferenceBudget
from .memory import EpisodicMemory
from .types import (
    BrainState,
    ModelOutput,
    Signal,
    SystemResult,
    TaskView,
    dominant_candidate,
)
from .workspace import BoundedWorkspace


@dataclass(frozen=True)
class BrainConfig:
    condition: str = "brain_full"
    enable_memory: bool = True
    enable_error_monitor: bool = True
    adaptive_routing: bool = True
    internal_state_coupling: bool = True
    workspace_capacity: int = 4
    workspace_ttl: int = 3
    memory_retrieval_limit: int = 6
    max_model_calls: int = 4
    max_input_units: int = 48
    max_output_units: int = 8


BRAIN_CONDITIONS: dict[str, BrainConfig] = {
    "brain_full": BrainConfig(condition="brain_full"),
    "brain_no_memory": BrainConfig(
        condition="brain_no_memory",
        enable_memory=False,
    ),
    "brain_no_error_monitor": BrainConfig(
        condition="brain_no_error_monitor",
        enable_error_monitor=False,
    ),
    "brain_static_routing": BrainConfig(
        condition="brain_static_routing",
        adaptive_routing=False,
    ),
    "brain_no_internal_state_coupling": BrainConfig(
        condition="brain_no_internal_state_coupling",
        internal_state_coupling=False,
    ),
}


def _deduplicate(signals: Iterable[Signal]) -> list[Signal]:
    by_origin: dict[str, Signal] = {}
    for signal in signals:
        origin = signal.provenance[0] if signal.provenance else signal.signal_id
        by_origin[origin] = signal
    return sorted(by_origin.values(), key=lambda item: (item.step, item.signal_id))


def _hypothesis_signal(
    episode_id: str,
    step: int,
    output: ModelOutput,
    candidate_count: int,
    *,
    weight: float = 1.0,
) -> Signal:
    scores = [0.0] * candidate_count
    # The floor prevents an uncertain hypothesis from disappearing entirely;
    # confidence still controls its influence.
    scores[output.candidate] = weight * (0.25 + output.confidence)
    return Signal(
        signal_id=f"hypothesis:{output.mode}:{step}:{len(output.used_signal_ids)}",
        episode_id=episode_id,
        step=step,
        cue="final_hypothesis",
        scores=tuple(scores),
        source=output.mode,
        kind="hypothesis",
        salience=min(weight / 2.0, 1.0),
        confidence=output.confidence,
        provenance=output.used_signal_ids,
    )


def _evidence_uncertainty(signals: list[Signal], candidate_count: int) -> float:
    if not signals:
        return 1.0
    totals = [0.0] * candidate_count
    for signal in signals:
        for index, value in enumerate(signal.scores):
            totals[index] += value
    ordered = sorted((max(value, 0.0) for value in totals), reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    if top <= 0.0:
        return 1.0
    confidence = (top - second) / (top + second + 1e-12)
    return 1.0 - min(max(confidence, 0.0), 1.0)


class BrainSystem:
    """One integrated individual with persistent episodic memory."""

    def __init__(
        self,
        config: BrainConfig | None = None,
        *,
        model: ModelAdapter | None = None,
        memory: EpisodicMemory | None = None,
    ) -> None:
        self.config = config or BRAIN_CONDITIONS["brain_full"]
        self.model = model or DeterministicModelAdapter()
        self.memory = memory or EpisodicMemory()

    def run(
        self,
        task: TaskView,
        observer: BrainObserver | None = None,
    ) -> SystemResult:
        observer = observer or BrainObserver()
        state = BrainState()
        workspace = BoundedWorkspace(
            capacity=self.config.workspace_capacity,
            ttl=self.config.workspace_ttl,
        )
        perception = PerceptionGateway()
        error_monitor = ErrorMonitor()
        router = SalienceRouter()
        budget = InferenceBudget(
            max_calls=self.config.max_model_calls,
            max_input_units=self.config.max_input_units,
            max_output_units=self.config.max_output_units,
        )
        episode_signals: list[Signal] = []
        online_predictions: list[int | None] = []
        memory_writes_before = self.memory.write_count
        memory_retrievals_before = self.memory.retrieval_count
        memory_hits_before = self.memory.hit_count

        for event in task.observations:
            signal, novelty = perception.perceive(task.task_id, event)
            episode_signals.append(signal)
            workspace.add(signal)
            state.novelty = 0.7 * state.novelty + 0.3 * novelty
            current = workspace.signals(event.step)
            state.cognitive_load = len(current) / workspace.capacity
            state.clamp()
            if self.config.enable_memory:
                # Novel observations can cross the ordinary salience write gate.
                # This makes novelty a real memory-control signal rather than a
                # display-only metric.
                self.memory.write(signal, force=state.novelty >= 0.45)
            if self.config.enable_error_monitor:
                error_monitor.observe(signal)
            query_signals = [item for item in current if item.cue == task.query_cue]
            online_predictions.append(
                dominant_candidate(query_signals or current, task.candidate_count)
            )
            observer.record(
                task.task_id,
                event.step,
                "perception",
                signal_id=signal.signal_id,
                cue=signal.cue,
                novelty=novelty,
                workspace_size=len(current),
            )

        final_step = max((event.step for event in task.observations), default=-1) + 1
        workspace_signals = workspace.signals(final_step)
        query_workspace = [
            signal for signal in workspace_signals if signal.cue == task.query_cue
        ]
        state.cognitive_load = len(workspace_signals) / workspace.capacity
        state.uncertainty = _evidence_uncertainty(
            query_workspace,
            task.candidate_count,
        )
        # Before inference, pressure is the fraction of the input-unit cap
        # already represented by active workspace items.
        state.resource_pressure = len(workspace_signals) / self.config.max_input_units
        if self.config.enable_error_monitor:
            state.prediction_error = error_monitor.conflict(task.query_cue)
        state.clamp()
        routing_state = asdict(state)

        route = router.decide(
            state,
            enable_memory=self.config.enable_memory,
            enable_error_monitor=self.config.enable_error_monitor,
            adaptive=self.config.adaptive_routing,
            state_coupling=self.config.internal_state_coupling,
        )
        observer.record(
            task.task_id,
            final_step,
            "routing",
            modules=list(route.activated_modules),
            retrieve_memory=route.retrieve_memory,
            robust=route.use_robust_cortex,
            reason=route.reason,
            state=asdict(state),
        )

        retrieved: list[Signal] = []
        if route.retrieve_memory:
            records = self.memory.retrieve(
                task.query_cue,
                limit=self.config.memory_retrieval_limit,
            )
            retrieved = self.memory.as_signals(records)
        context = _deduplicate([*retrieved, *workspace_signals])
        fast_context = [
            signal for signal in workspace_signals if signal.cue == task.query_cue
        ][-2:] or workspace_signals[-2:]

        outputs: list[ModelOutput] = []
        fast = self._infer(
            task,
            fast_context,
            "fast",
            state,
            budget,
            observer,
            final_step,
        )
        outputs.append(fast)
        deliberate = self._infer(
            task,
            context,
            "deliberate",
            state,
            budget,
            observer,
            final_step,
        )
        outputs.append(deliberate)
        if route.use_robust_cortex:
            robust = self._infer(
                task,
                context,
                "robust",
                state,
                budget,
                observer,
                final_step,
            )
            outputs.append(robust)

        hypotheses = [
            _hypothesis_signal(
                task.task_id,
                final_step,
                output,
                task.candidate_count,
                # Robust evidence is promoted only after the independent error
                # monitor reports conflict.  The ablation removes this path.
                weight=2.0 if output.mode == "robust" else 1.0,
            )
            for output in outputs
        ]
        final = self._infer(
            task,
            hypotheses,
            "integrate",
            state,
            budget,
            observer,
            final_step,
        )

        state.uncertainty = 1.0 - final.confidence
        state.resource_pressure = budget.model_calls / budget.max_calls
        state.clamp()
        provenance = tuple(
            dict.fromkeys(
                origin
                for output in outputs
                for origin in output.used_signal_ids
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
                "condition": self.config.condition,
                "activated_modules": list(route.activated_modules),
                "router_reason": route.reason,
                "robust_activated": route.use_robust_cortex,
                "conflict_detected": error_monitor.detected(task.query_cue)
                if self.config.enable_error_monitor
                else False,
                "state": asdict(state),
                "routing_state": routing_state,
                "fast_candidate": fast.candidate,
                "deliberate_candidate": deliberate.candidate,
                "robust_candidate": outputs[-1].candidate
                if outputs[-1].mode == "robust"
                else None,
                "online_predictions": online_predictions,
                "workspace_turnover": workspace.turnover_count,
                "workspace_peak": workspace.peak_size,
                "workspace_utilization": workspace.average_utilization,
                "memory_writes": self.memory.write_count - memory_writes_before,
                "memory_retrievals": (
                    self.memory.retrieval_count - memory_retrievals_before
                ),
                "memory_hits": self.memory.hit_count - memory_hits_before,
                "retrieved_records": len(retrieved),
                "model_modes": [output.mode for output in outputs] + ["integrate"],
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
        observer: BrainObserver,
        step: int,
    ) -> ModelOutput:
        output = self.model.infer(
            signals,
            query_cue=task.query_cue,
            candidate_count=task.candidate_count,
            mode=mode,
            state=state,
            budget=budget,
        )
        observer.record(
            task.task_id,
            step,
            "model_output",
            mode=mode,
            candidate=output.candidate,
            confidence=output.confidence,
            used_signal_ids=list(output.used_signal_ids),
        )
        return output


def brain_config(condition: str) -> BrainConfig:
    try:
        return BRAIN_CONDITIONS[condition]
    except KeyError as error:
        raise ValueError(
            f"unknown brain condition {condition!r}; expected one of "
            f"{sorted(BRAIN_CONDITIONS)}"
        ) from error
