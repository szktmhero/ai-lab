from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from brain.experiments.formal import (
    CONDITIONS,
    FORMAL_MODEL_CONFIG,
    ProtocolLockError,
    _condition_order,
    evaluate_primary_thresholds,
    formal_preflight,
    implementation_sha256,
    load_and_verify_protocol,
    run_formal,
)
from brain.models.openai_responses import OpenAIResponsesAdapter
from brain.tasks.blind import (
    BLIND_SEEDS,
    EXPECTED_BLIND_SUITE_SHA256,
    generate_blind_task_suite,
    validate_blind_suite,
)


def test_blind_suite_matches_committed_evaluator_inclusive_hash():
    assert validate_blind_suite() == EXPECTED_BLIND_SUITE_SHA256


def test_blind_task_view_does_not_expose_answer_or_fault_annotation():
    tasks = generate_blind_task_suite(BLIND_SEEDS[0])
    serialized = json.dumps(asdict(tasks[0].view), default=str)
    assert "correct_answer" not in serialized
    assert "fault_event_id" not in serialized
    assert all("latent_rule" not in task.view.task_id for task in tasks)


def test_protocol_lock_matches_current_code_and_preflight_is_offline():
    lock = load_and_verify_protocol()
    summary = formal_preflight()
    assert summary["lock_sha256"] == lock["lock_sha256"]
    assert summary["task_count"] == 800
    assert summary["episode_count"] == 6400
    assert summary["maximum_inference_calls"] == 25600
    assert summary["maximum_provider_requests"] == 51200
    assert summary["live_api_called"] is False


def test_implementation_hash_covers_runtime_sources_not_tests():
    digest, files = implementation_sha256()
    assert len(digest) == 64
    int(digest, 16)
    assert "pyproject.toml" in files
    assert "experiments/formal.py" in files
    assert not any(path.startswith("tests/") for path in files)


def test_condition_order_is_deterministic_complete_permutation():
    first = _condition_order("opaque-task-a")
    second = _condition_order("opaque-task-a")
    assert first == second
    assert set(first) == set(CONDITIONS)


def test_formal_run_requires_lock_acknowledgement_before_creating_output(tmp_path):
    output_dir = tmp_path / "formal"
    with pytest.raises(PermissionError):
        run_formal(acknowledgement="wrong", output_dir=output_dir)
    assert not output_dir.exists()


def test_formal_run_rejects_unlocked_sdk_before_creating_output(tmp_path):
    class WrongSdkAdapter(OpenAIResponsesAdapter):
        @staticmethod
        def sdk_version() -> str:
            return "0.0.0"

    summary = formal_preflight()
    output_dir = tmp_path / "formal"
    adapter = WrongSdkAdapter(FORMAL_MODEL_CONFIG, client=object())
    with pytest.raises(ProtocolLockError, match="SDK version"):
        run_formal(
            acknowledgement=summary["lock_sha256"][:16],
            output_dir=output_dir,
            adapter=adapter,
        )
    assert not output_dir.exists()


def test_primary_effect_thresholds_are_applied_mechanically():
    comparison = {
        "brain_full": {
            "accuracy": 0.90,
            "regime_switch_accuracy": 0.90,
            "delayed_recall_accuracy": 0.90,
            "fault_recovery_accuracy": 0.90,
            "partial_observation_accuracy": 0.90,
            "avg_total_units": 90.0,
        },
        "sequential_reflection": {
            "regime_switch_accuracy": 0.80,
            "fault_recovery_accuracy": 0.80,
        },
        "flat_ensemble": {
            "regime_switch_accuracy": 0.80,
            "fault_recovery_accuracy": 0.80,
        },
        "brain_no_memory": {
            "delayed_recall_accuracy": 0.75,
            "partial_observation_accuracy": 0.75,
        },
        "brain_no_error_monitor": {"fault_recovery_accuracy": 0.75},
        "brain_static_routing": {
            "accuracy": 0.91,
            "avg_total_units": 110.0,
        },
    }
    decisions = evaluate_primary_thresholds(comparison)
    assert all(decisions[key]["status"] == "PASS" for key in ("H1", "H2", "H3", "H4"))
    assert decisions["H5"]["status"] == "NOT_EVALUATED"

    comparison["flat_ensemble"]["regime_switch_accuracy"] = 0.87
    assert evaluate_primary_thresholds(comparison)["H1"]["status"] == "FAIL"
