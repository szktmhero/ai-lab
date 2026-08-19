from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from brain.baselines import make_system
from brain.core.budget import BudgetExceeded, BudgetLimits, InferenceBudget
from brain.core.types import BrainState, Signal, TaskType
from brain.experiments.run import CONDITIONS
from brain.models.openai_responses import (
    InvalidModelResponse,
    OpenAIResponsesAdapter,
    OpenAIResponsesConfig,
    SnapshotRequired,
    TokenAccountingMismatch,
    prompt_spec_sha256,
)
from brain.tasks.generator import generate_task


class FakeInputTokens:
    def __init__(self, input_tokens: int):
        self.input_tokens = input_tokens
        self.payloads: list[dict[str, object]] = []

    def count(self, **payload):
        self.payloads.append(payload)
        return SimpleNamespace(input_tokens=self.input_tokens)


class FakeResponses:
    def __init__(
        self,
        *,
        counted_input: int = 37,
        actual_input: int = 37,
        actual_output: int = 19,
    ):
        self.input_tokens = FakeInputTokens(counted_input)
        self.actual_input = actual_input
        self.actual_output = actual_output
        self.payloads: list[dict[str, object]] = []

    def create(self, **payload):
        self.payloads.append(payload)
        return SimpleNamespace(
            id="resp_test",
            model="gpt-5.4-mini-2026-03-17",
            status="completed",
            usage=SimpleNamespace(
                input_tokens=self.actual_input,
                output_tokens=self.actual_output,
            ),
            output_text=json.dumps(
                {
                    "candidate": 1,
                    "confidence": 0.75,
                    "scores": [0.1, 0.8, 0.1, 0.0],
                    "used_signal_ids": ["s0"],
                }
            ),
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.responses = FakeResponses(**kwargs)


def _signal() -> Signal:
    return Signal(
        signal_id="private-long-task-identifier",
        episode_id="hidden-episode",
        step=0,
        cue="public-cue",
        scores=(0.1, 0.8, 0.1, 0.0),
        source="channel_0",
        kind="evidence",
        salience=0.7,
        confidence=0.8,
        provenance=("private-long-task-identifier",),
    )


def _config() -> OpenAIResponsesConfig:
    return OpenAIResponsesConfig(
        model="gpt-5.4-mini-2026-03-17",
        min_request_interval_seconds=0.0,
    )


def test_formal_config_rejects_mutable_model_alias():
    with pytest.raises(SnapshotRequired):
        OpenAIResponsesConfig(model="gpt-5.4-mini")


def test_adapter_fingerprint_freezes_client_retry_timeout_and_pacing():
    fingerprint = _config().public_fingerprint()
    assert fingerprint["client_max_retries"] == 0
    assert fingerprint["client_timeout_seconds"] == 120.0
    assert fingerprint["min_request_interval_seconds"] == 0.0


def test_adapter_rejects_invalid_candidate_count_before_provider_request():
    client = FakeClient()
    adapter = OpenAIResponsesAdapter(_config(), client=client)
    budget = InferenceBudget(unit_name="tokens")
    with pytest.raises(ValueError):
        adapter.infer([_signal()], "public-cue", 0, "fast", BrainState(), budget)
    assert adapter.token_count_requests == adapter.response_requests == 0


def test_adapter_rejects_mismatched_signal_shape_before_provider_request():
    client = FakeClient()
    adapter = OpenAIResponsesAdapter(_config(), client=client)
    budget = InferenceBudget(unit_name="tokens")
    malformed = replace(_signal(), scores=(0.2, 0.8))
    with pytest.raises(ValueError, match="score vector"):
        adapter.infer([malformed], "public-cue", 4, "fast", BrainState(), budget)
    assert adapter.token_count_requests == adapter.response_requests == 0


def test_local_parser_rejects_fields_outside_frozen_schema():
    payload = json.dumps(
        {
            "candidate": 1,
            "confidence": 0.75,
            "scores": [0.1, 0.8, 0.1, 0.0],
            "used_signal_ids": ["s0"],
            "unexpected": True,
        }
    )
    with pytest.raises(InvalidModelResponse):
        OpenAIResponsesAdapter._parse_output(
            payload,
            candidate_count=4,
            alias_to_id={"s0": "original"},
        )


def test_openai_adapter_preflights_and_commits_provider_usage():
    client = FakeClient()
    adapter = OpenAIResponsesAdapter(_config(), client=client)
    budget = InferenceBudget(
        max_calls=4,
        max_input_units=200,
        max_output_units=256,
        unit_name="tokens",
    )
    output = adapter.infer(
        [_signal()],
        query_cue="public-cue",
        candidate_count=4,
        mode="deliberate",
        state=BrainState(prediction_error=1.0),
        budget=budget,
    )

    assert output.candidate == 1
    assert output.used_signal_ids == ("private-long-task-identifier",)
    assert output.input_units == 37
    assert output.output_units == 19
    assert budget.snapshot().input_units == 37
    assert budget.snapshot().output_units == 19
    assert budget.snapshot().unit_name == "tokens"
    assert adapter.token_count_requests == adapter.response_requests == 1

    count_payload = client.responses.input_tokens.payloads[0]
    response_payload = client.responses.payloads[0]
    assert count_payload["model"] == response_payload["model"]
    assert count_payload["input"] == response_payload["input"]
    assert count_payload["text"] == response_payload["text"]
    assert response_payload["store"] is False
    rendered = str(count_payload["input"])
    assert "private-long-task-identifier" not in rendered
    assert "hidden-episode" not in rendered
    assert "prediction_error" not in rendered


def test_output_allowance_is_checked_before_paid_inference():
    client = FakeClient()
    adapter = OpenAIResponsesAdapter(_config(), client=client)
    budget = InferenceBudget(
        max_calls=4,
        max_input_units=200,
        max_output_units=64,
        unit_name="tokens",
    )
    with pytest.raises(BudgetExceeded):
        adapter.infer(
            [_signal()],
            "public-cue",
            4,
            "fast",
            BrainState(),
            budget,
        )
    assert adapter.token_count_requests == 1
    assert adapter.response_requests == 0
    assert budget.snapshot().model_calls == 0


def test_token_count_mismatch_aborts_without_committing_ledger():
    client = FakeClient(counted_input=37, actual_input=38)
    adapter = OpenAIResponsesAdapter(_config(), client=client)
    budget = InferenceBudget(
        max_calls=4,
        max_input_units=200,
        max_output_units=256,
        unit_name="tokens",
    )
    with pytest.raises(TokenAccountingMismatch):
        adapter.infer(
            [_signal()],
            "public-cue",
            4,
            "robust",
            BrainState(),
            budget,
        )
    assert adapter.response_requests == 1
    assert budget.snapshot().model_calls == 0


def test_prompt_spec_has_stable_sha256_shape():
    fingerprint = prompt_spec_sha256()
    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_real_adapter_wiring_respects_token_budget_in_every_condition():
    adapter = OpenAIResponsesAdapter(_config(), client=FakeClient())
    limits = BudgetLimits(
        max_calls=4,
        max_input_units=1000,
        max_output_units=512,
        unit_name="tokens",
    )
    task = generate_task(0, TaskType.PARTIAL_OBSERVATION, 0)
    for condition in CONDITIONS:
        result = make_system(
            condition,
            model=adapter,
            budget_limits=limits,
        ).run(task.view)
        assert result.usage.unit_name == "tokens"
        assert result.usage.model_calls <= limits.max_calls
        assert result.usage.input_units <= limits.max_input_units
        assert result.usage.output_units <= limits.max_output_units
        assert all("candidate" in record for record in result.trace["model_call_records"])
