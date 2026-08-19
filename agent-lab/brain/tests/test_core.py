from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from brain.baselines import make_system
from brain.core.budget import BudgetExceeded, InferenceBudget
from brain.core.memory import EpisodicMemory
from brain.core.system import BRAIN_CONDITIONS, BrainSystem
from brain.core.types import BrainState, Signal, TaskType
from brain.core.workspace import BoundedWorkspace
from brain.modules.error_monitor import ErrorMonitor
from brain.modules.perception import vector_novelty
from brain.modules.router import SalienceRouter
from brain.observer.observer import BrainObserver
from brain.tasks.generator import generate_task, generate_task_suite


def _signal(identifier: str, step: int, scores=(1.0, 0.0), salience=0.6) -> Signal:
    return Signal(
        signal_id=identifier,
        episode_id="episode",
        step=step,
        cue="target",
        scores=tuple(scores),
        source="test",
        kind="evidence",
        salience=salience,
        confidence=1.0,
        provenance=(identifier,),
    )


def test_task_view_contains_no_answer_or_evaluator_metadata():
    task = generate_task(42, TaskType.FAULT_RECOVERY, 0)
    serialized = json.dumps(asdict(task.view), default=str)
    assert "correct_answer" not in serialized
    assert "fault_event_id" not in serialized
    assert "correct_answer" not in task.view.__dataclass_fields__


def test_task_generation_is_deterministic_and_condition_independent():
    left = generate_task_suite(7, n_per_type=2)
    right = generate_task_suite(7, n_per_type=2)
    assert left == right
    assert len({task.view.task_id for task in left}) == len(left)


def test_all_zero_novelty_is_zero():
    assert vector_novelty((0.0, 0.0), (0.0, 0.0)) == 0.0
    assert vector_novelty((0.0, 0.0), (1.0, 0.0)) > 0.9


def test_workspace_is_capacity_and_ttl_bounded():
    workspace = BoundedWorkspace(capacity=2, ttl=1)
    workspace.add(_signal("a", 0, salience=0.9))
    workspace.add(_signal("b", 1, salience=0.6))
    workspace.add(_signal("c", 2, salience=0.7))
    identifiers = [signal.signal_id for signal in workspace.signals(2)]
    assert "a" not in identifiers
    assert len(identifiers) <= 2
    assert workspace.turnover_count >= 1


def test_memory_preserves_provenance_and_supports_reset_and_load():
    memory = EpisodicMemory()
    signal = _signal("origin", 0)
    assert memory.write(signal)
    record = memory.retrieve("target")[0]
    restored = memory.as_signals([record])[0]
    assert restored.provenance[0] == "origin"

    exported = memory.export_records()
    memory.reset()
    assert memory.retrieve("target") == []
    memory.load_records(exported)
    assert len(memory.retrieve("target")) == 1


def test_novelty_can_force_a_memory_write():
    memory = EpisodicMemory(write_threshold=0.9)
    low_salience = _signal("novel", 0, salience=0.2)
    assert not memory.write(low_salience)
    assert memory.write(low_salience, force=True)


def test_error_monitor_detects_conflict_without_truth():
    monitor = ErrorMonitor(conflict_threshold=0.2)
    for index, scores in enumerate(((1.0, 0.0), (1.0, 0.0), (0.0, 2.0))):
        monitor.observe(_signal(str(index), index, scores=scores))
    assert monitor.conflict("target") == pytest.approx(1 / 3)
    assert monitor.detected("target")


def test_router_ablation_disconnects_internal_state():
    state = BrainState(prediction_error=0.5)
    router = SalienceRouter()
    coupled = router.decide(
        state,
        enable_memory=True,
        enable_error_monitor=True,
        adaptive=True,
        state_coupling=True,
    )
    disconnected = router.decide(
        state,
        enable_memory=True,
        enable_error_monitor=True,
        adaptive=True,
        state_coupling=False,
    )
    assert coupled.use_robust_cortex
    assert not disconnected.use_robust_cortex


def test_router_uses_uncertainty_load_and_resource_pressure():
    router = SalienceRouter()
    memory_route = router.decide(
        BrainState(uncertainty=0.8, cognitive_load=0.1),
        enable_memory=True,
        enable_error_monitor=True,
        adaptive=True,
        state_coupling=True,
    )
    constrained = router.decide(
        BrainState(prediction_error=0.8, resource_pressure=0.95),
        enable_memory=True,
        enable_error_monitor=True,
        adaptive=True,
        state_coupling=True,
    )
    assert memory_route.retrieve_memory
    assert not constrained.use_robust_cortex


def test_budget_rejects_calls_before_overrun():
    budget = InferenceBudget(max_calls=1, max_input_units=2, max_output_units=1)
    budget.consume(2)
    with pytest.raises(BudgetExceeded):
        budget.consume(0)
    assert budget.snapshot().model_calls == 1


def test_full_system_is_deterministic_with_clean_state():
    task = generate_task(0, TaskType.LATENT_RULE_SWITCH, 0)
    left = BrainSystem(BRAIN_CONDITIONS["brain_full"]).run(task.view)
    right = BrainSystem(BRAIN_CONDITIONS["brain_full"]).run(task.view)
    assert left == right


def test_memory_ablation_selectively_breaks_delayed_recall():
    task = generate_task(0, TaskType.DELAYED_RECALL, 0)
    full = make_system("brain_full").run(task.view)
    no_memory = make_system("brain_no_memory").run(task.view)
    assert full.candidate == task.correct_answer
    assert no_memory.candidate != task.correct_answer
    assert full.trace["memory_hits"] == 1
    assert no_memory.trace["memory_retrievals"] == 0


def test_error_and_state_paths_are_causal_on_fault_recovery():
    task = generate_task(0, TaskType.FAULT_RECOVERY, 0)
    full = make_system("brain_full").run(task.view)
    no_error = make_system("brain_no_error_monitor").run(task.view)
    disconnected = make_system("brain_no_internal_state_coupling").run(task.view)
    assert full.candidate == task.correct_answer
    assert full.trace["robust_activated"]
    assert no_error.candidate != task.correct_answer
    assert not no_error.trace["robust_activated"]
    assert disconnected.trace["conflict_detected"]
    assert not disconnected.trace["robust_activated"]
    assert disconnected.candidate != task.correct_answer


def test_adaptive_routing_uses_fewer_calls_on_low_conflict_task():
    task = generate_task(0, TaskType.DELAYED_RECALL, 0)
    adaptive = make_system("brain_full").run(task.view)
    static = make_system("brain_static_routing").run(task.view)
    assert adaptive.candidate == static.candidate == task.correct_answer
    assert adaptive.usage.model_calls == 3
    assert static.usage.model_calls == 4


def test_every_condition_respects_declared_budget():
    task = generate_task(0, TaskType.PARTIAL_OBSERVATION, 0)
    conditions = [
        "single_long_context",
        "sequential_reflection",
        "flat_ensemble",
        *BRAIN_CONDITIONS,
    ]
    for condition in conditions:
        result = make_system(condition).run(task.view)
        assert result.usage.model_calls <= 4
        assert result.usage.input_units <= 48
        assert result.usage.output_units <= 8


def test_observer_can_reconstruct_an_episode():
    task = generate_task(0, TaskType.FAULT_RECOVERY, 0)
    observer = BrainObserver()
    make_system("brain_full").run(task.view, observer=observer)
    events = observer.episode_events(task.view.task_id)
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert events[-1]["event_type"] == "final_action"
    assert events[-1]["provenance"]
