"""Evaluator-side metrics; no value here is visible to the system under test."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from ..core.types import SystemResult, Task, TaskType


def _stable_recovery_steps(task: Task, result: SystemResult) -> int | None:
    if task.view.task_type != TaskType.LATENT_RULE_SWITCH:
        return None
    switch_index = int(task.metadata["switch_index"])
    predictions = list(result.trace.get("online_predictions", []))
    for index in range(switch_index, len(predictions)):
        if predictions[index] == task.correct_answer and all(
            later == task.correct_answer for later in predictions[index:]
        ):
            return index - switch_index
    return None


def evaluate_result(
    seed: int,
    condition: str,
    task: Task,
    result: SystemResult,
) -> dict[str, Any]:
    correct = int(result.candidate == task.correct_answer)
    recovery_steps = _stable_recovery_steps(task, result)
    trace = result.trace
    fast_candidate = trace.get("fast_candidate")
    return {
        "seed": seed,
        "condition": condition,
        "task_id": task.view.task_id,
        "task_type": task.view.task_type.value,
        "correct": correct,
        "candidate": result.candidate,
        "confidence": result.confidence,
        "brier": (result.confidence - correct) ** 2,
        "regime_switch_success": correct
        if task.view.task_type == TaskType.LATENT_RULE_SWITCH
        else None,
        "recovery_steps": recovery_steps,
        "delayed_recall_success": correct
        if task.view.task_type == TaskType.DELAYED_RECALL
        else None,
        "fault_detection": int(bool(trace.get("conflict_detected")))
        if task.view.task_type == TaskType.FAULT_RECOVERY
        else None,
        "fault_recovery_success": correct
        if task.view.task_type == TaskType.FAULT_RECOVERY
        else None,
        "self_correction": int(
            fast_candidate is not None
            and fast_candidate != result.candidate
            and correct == 1
        ),
        "model_calls": result.usage.model_calls,
        "input_units": result.usage.input_units,
        "output_units": result.usage.output_units,
        "workspace_turnover": int(trace.get("workspace_turnover", 0)),
        "workspace_utilization": float(trace.get("workspace_utilization", 0.0)),
        "module_activations": len(trace.get("activated_modules", [])),
        "robust_activated": int(bool(trace.get("robust_activated"))),
        "memory_writes": int(trace.get("memory_writes", 0)),
        "memory_retrievals": int(trace.get("memory_retrievals", 0)),
        "memory_hits": int(trace.get("memory_hits", 0)),
        "retrieved_records": int(trace.get("retrieved_records", 0)),
        "prediction_error": float(trace.get("state", {}).get("prediction_error", 0.0)),
        "provenance_count": len(result.provenance),
        "invalid_actions": int(trace.get("invalid_actions", 0)),
        "no_ops": int(trace.get("no_ops", 0)),
    }


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return fmean(values) if values else None


def _binary_entropy(values: Iterable[int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    probability = sum(values) / len(values)
    if probability in {0.0, 1.0}:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


def aggregate_task_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["task_type"]].append(row)
    memory_retrievals = sum(row["memory_retrievals"] for row in rows)
    return {
        "tasks": len(rows),
        "accuracy": _mean_present(rows, "correct"),
        "brier": _mean_present(rows, "brier"),
        "regime_switch_accuracy": _mean_present(rows, "regime_switch_success"),
        "mean_recovery_steps": _mean_present(rows, "recovery_steps"),
        "delayed_recall_accuracy": _mean_present(rows, "delayed_recall_success"),
        "fault_detection_rate": _mean_present(rows, "fault_detection"),
        "fault_recovery_accuracy": _mean_present(rows, "fault_recovery_success"),
        "self_correction_rate": _mean_present(rows, "self_correction"),
        "avg_model_calls": _mean_present(rows, "model_calls"),
        "avg_input_units": _mean_present(rows, "input_units"),
        "avg_output_units": _mean_present(rows, "output_units"),
        "avg_workspace_turnover": _mean_present(rows, "workspace_turnover"),
        "avg_workspace_utilization": _mean_present(rows, "workspace_utilization"),
        "avg_module_activations": _mean_present(rows, "module_activations"),
        "routing_entropy": _binary_entropy(row["robust_activated"] for row in rows),
        "memory_hit_rate": (
            sum(row["memory_hits"] for row in rows) / memory_retrievals
            if memory_retrievals
            else 0.0
        ),
        "avg_retrieved_records": _mean_present(rows, "retrieved_records"),
        "avg_prediction_error": _mean_present(rows, "prediction_error"),
        "avg_provenance_count": _mean_present(rows, "provenance_count"),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "no_ops": sum(row["no_ops"] for row in rows),
        "accuracy_by_task": {
            task_type: _mean_present(task_rows, "correct")
            for task_type, task_rows in sorted(by_type.items())
        },
    }
