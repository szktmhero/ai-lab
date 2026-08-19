"""OpenAI Responses API adapter with provider-counted token accounting.

The adapter is intentionally condition-blind.  It receives only the signals,
query cue, candidate count, inference mode, and shared budget from the existing
``ModelAdapter`` contract.  It never receives evaluator answers, condition names,
task annotations, or prior Responses API conversation state.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from ..core.budget import InferenceBudget
from ..core.types import BrainState, ModelOutput, Signal
from .base import ModelAdapter


class OpenAIAdapterError(RuntimeError):
    """Base class for adapter and protocol failures."""


class SnapshotRequired(OpenAIAdapterError):
    """Raised when a mutable model alias is used for a frozen experiment."""


class TokenAccountingMismatch(OpenAIAdapterError):
    """Raised when preflight and response token counts disagree."""


class InvalidModelResponse(OpenAIAdapterError):
    """Raised when a response violates the frozen structured-output contract."""


SYSTEM_INSTRUCTIONS = (
    "You are a condition-invariant decision kernel in a controlled experiment. "
    "Use only the public signals supplied in the request. Never assume a hidden "
    "answer, task family, experiment condition, or evaluator annotation. Return "
    "only the structured decision requested by the response schema."
)

MODE_INSTRUCTIONS: Mapping[str, str] = {
    "fast": (
        "Make an economical first-pass decision. Prioritize the newest directly "
        "relevant evidence and do not invent missing observations."
    ),
    "deliberate": (
        "Integrate all supplied relevant evidence. Preserve temporal order, cue "
        "identity, confidence, and provenance when resolving conflicts."
    ),
    "robust": (
        "Make a fault-tolerant decision. Compare independent provenance roots, "
        "discount a single extreme observation, and prefer corroborated evidence."
    ),
    "integrate": (
        "Integrate the supplied hypothesis signals. Use their scores, confidence, "
        "and provenance without adding external evidence."
    ),
}

PROMPT_SPEC_VERSION = "brain-openai-responses-v1"
OUTPUT_SCHEMA_VERSION = "brain-decision-schema-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    encoded = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prompt_spec_sha256() -> str:
    """Fingerprint every static prompt choice used by the adapter."""

    return _sha256(
        {
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "system_instructions": SYSTEM_INSTRUCTIONS,
            "mode_instructions": dict(MODE_INSTRUCTIONS),
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "state_visible_to_model": False,
            "condition_visible_to_model": False,
        }
    )


@dataclass(frozen=True)
class OpenAIResponsesConfig:
    """Frozen provider configuration for one experimental run."""

    model: str
    reasoning_effort: str = "none"
    max_output_tokens: int = 128
    store: bool = False
    require_snapshot: bool = True
    client_max_retries: int = 0
    client_timeout_seconds: float = 120.0
    min_request_interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.client_max_retries < 0:
            raise ValueError("client_max_retries must be non-negative")
        if self.client_timeout_seconds <= 0.0:
            raise ValueError("client_timeout_seconds must be positive")
        if self.min_request_interval_seconds < 0.0:
            raise ValueError("min_request_interval_seconds must be non-negative")
        if self.require_snapshot and not re.search(
            r"-\d{4}-\d{2}-\d{2}$",
            self.model,
        ):
            raise SnapshotRequired(
                "formal runs require a dated model snapshot, not a mutable alias"
            )

    def public_fingerprint(self) -> dict[str, object]:
        return {
            **asdict(self),
            "provider": "openai",
            "endpoint": "responses",
            "prompt_spec_sha256": prompt_spec_sha256(),
            "temperature": "provider default (parameter omitted)",
            "tools": [],
            "conversation_state": "disabled",
        }


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _short_provenance(identifier: str) -> str:
    return f"p_{hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:12]}"


def _signal_payloads(
    signals: Sequence[Signal],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    alias_to_id: dict[str, str] = {}
    payloads: list[dict[str, object]] = []
    for index, signal in enumerate(signals):
        alias = f"s{index}"
        alias_to_id[alias] = signal.signal_id
        payloads.append(
            {
                "id": alias,
                "step": signal.step,
                "cue": signal.cue,
                "scores": list(signal.scores),
                "source": signal.source,
                "kind": signal.kind,
                "salience": signal.salience,
                "confidence": signal.confidence,
                "provenance_roots": [
                    _short_provenance(item) for item in signal.provenance
                ],
            }
        )
    return payloads, alias_to_id


def _decision_schema(
    candidate_count: int,
    signal_aliases: Sequence[str],
) -> dict[str, object]:
    used_items: dict[str, object] = {"type": "string"}
    if signal_aliases:
        used_items["enum"] = list(signal_aliases)
    return {
        "type": "json_schema",
        "name": "brain_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidate": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": candidate_count - 1,
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "scores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": candidate_count,
                    "maxItems": candidate_count,
                },
                "used_signal_ids": {
                    "type": "array",
                    "items": used_items,
                    "minItems": 1 if signal_aliases else 0,
                    "maxItems": len(signal_aliases),
                },
            },
            "required": [
                "candidate",
                "confidence",
                "scores",
                "used_signal_ids",
            ],
            "additionalProperties": False,
        },
    }


class OpenAIResponsesAdapter(ModelAdapter):
    """Real-model adapter using exact provider-side token preflight."""

    def __init__(
        self,
        config: OpenAIResponsesConfig,
        *,
        client: object | None = None,
    ) -> None:
        self.config = config
        self.client = client if client is not None else self._create_client()
        self.token_count_requests = 0
        self.response_requests = 0
        self.call_records: list[dict[str, object]] = []
        self._last_request_started_at: float | None = None

    def _create_client(self) -> object:
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - environment dependent
            raise OpenAIAdapterError(
                "install the optional 'openai' dependency before a live run"
            ) from error
        return OpenAI(
            max_retries=self.config.client_max_retries,
            timeout=self.config.client_timeout_seconds,
        )

    def _pace_provider_request(self) -> None:
        interval = self.config.min_request_interval_seconds
        now = time.monotonic()
        if self._last_request_started_at is not None:
            remaining = self._last_request_started_at + interval - now
            if remaining > 0.0:
                time.sleep(remaining)
        self._last_request_started_at = time.monotonic()

    @staticmethod
    def sdk_version() -> str:
        try:
            return importlib.metadata.version("openai")
        except importlib.metadata.PackageNotFoundError:
            return "client-injected-or-not-installed"

    def infer(
        self,
        signals: Sequence[Signal],
        query_cue: str,
        candidate_count: int,
        mode: str,
        state: BrainState,
        budget: InferenceBudget,
    ) -> ModelOutput:
        del state  # Internal state changes routing, never the invariant model prompt.
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if any(len(signal.scores) != candidate_count for signal in signals):
            raise ValueError("every signal score vector must match candidate_count")
        if budget.unit_name != "tokens":
            raise OpenAIAdapterError(
                "OpenAIResponsesAdapter requires a token-denominated budget"
            )
        try:
            mode_instruction = MODE_INSTRUCTIONS[mode]
        except KeyError as error:
            raise ValueError(f"unsupported inference mode: {mode}") from error

        signal_payloads, alias_to_id = _signal_payloads(signals)
        input_payload = {
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "mode": mode,
            "mode_instruction": mode_instruction,
            "query_cue": query_cue,
            "candidate_ids": list(range(candidate_count)),
            "signals": signal_payloads,
        }
        text_config = {
            "format": _decision_schema(candidate_count, tuple(alias_to_id))
        }
        count_payload = {
            "model": self.config.model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": _canonical_json(input_payload),
            "text": text_config,
            "reasoning": {"effort": self.config.reasoning_effort},
        }
        request_payload = {
            **count_payload,
            "max_output_tokens": self.config.max_output_tokens,
            "store": self.config.store,
        }
        request_hash = _sha256(request_payload)

        self._pace_provider_request()
        counter = self.client.responses.input_tokens.count(**count_payload)
        self.token_count_requests += 1
        counted_input = int(_field(counter, "input_tokens", -1))
        if counted_input < 0:
            raise TokenAccountingMismatch(
                "token-count response did not contain a valid input_tokens value"
            )
        budget.ensure_can_consume(
            counted_input,
            self.config.max_output_tokens,
        )

        self._pace_provider_request()
        response = self.client.responses.create(**request_payload)
        self.response_requests += 1
        status = _field(response, "status")
        if status != "completed":
            raise InvalidModelResponse(f"response status was {status!r}")

        usage = _field(response, "usage")
        actual_input = int(_field(usage, "input_tokens", -1))
        actual_output = int(_field(usage, "output_tokens", -1))
        if actual_input != counted_input:
            raise TokenAccountingMismatch(
                "provider preflight input count did not match response usage: "
                f"{counted_input} != {actual_input}"
            )
        if actual_output < 0 or actual_output > self.config.max_output_tokens:
            raise TokenAccountingMismatch(
                "response output token count was missing or exceeded max_output_tokens"
            )

        model_id = str(_field(response, "model", ""))
        if model_id != self.config.model:
            raise InvalidModelResponse(
                "provider response model differed from the frozen snapshot: "
                f"{model_id!r} != {self.config.model!r}"
            )

        raw_output = _field(response, "output_text", "")
        if callable(raw_output):
            raw_output = raw_output()
        if not isinstance(raw_output, str) or not raw_output:
            raise InvalidModelResponse("response did not contain output_text")
        parsed = self._parse_output(
            raw_output,
            candidate_count=candidate_count,
            alias_to_id=alias_to_id,
        )
        budget.consume(actual_input, actual_output)

        response_id = str(_field(response, "id", "")) or None
        output_hash = _sha256(raw_output)
        record = {
            "request_hash": request_hash,
            "output_hash": output_hash,
            "response_id": response_id,
            "model_id": model_id,
            "mode": mode,
            "counted_input_tokens": counted_input,
            "actual_input_tokens": actual_input,
            "actual_output_tokens": actual_output,
        }
        self.call_records.append(record)
        return ModelOutput(
            candidate=parsed["candidate"],
            confidence=parsed["confidence"],
            scores=parsed["scores"],
            mode=mode,
            used_signal_ids=parsed["used_signal_ids"],
            input_units=actual_input,
            output_units=actual_output,
            response_id=response_id,
            model_id=model_id,
            request_hash=request_hash,
            output_hash=output_hash,
        )

    @staticmethod
    def _parse_output(
        raw_output: str,
        *,
        candidate_count: int,
        alias_to_id: Mapping[str, str],
    ) -> dict[str, object]:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise InvalidModelResponse("output_text was not valid JSON") from error
        if not isinstance(payload, dict):
            raise InvalidModelResponse("structured output must be an object")

        expected_fields = {
            "candidate",
            "confidence",
            "scores",
            "used_signal_ids",
        }
        if set(payload) != expected_fields:
            raise InvalidModelResponse(
                "structured output fields differed from the frozen schema"
            )

        candidate = payload.get("candidate")
        confidence = payload.get("confidence")
        scores = payload.get("scores")
        used_aliases = payload.get("used_signal_ids")
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise InvalidModelResponse("candidate must be an integer")
        if not 0 <= candidate < candidate_count:
            raise InvalidModelResponse("candidate is outside the public range")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise InvalidModelResponse("confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise InvalidModelResponse("confidence must be finite and between 0 and 1")
        if not isinstance(scores, list) or len(scores) != candidate_count:
            raise InvalidModelResponse("scores must match candidate_count")
        normalized_scores: list[float] = []
        for value in scores:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidModelResponse("every score must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise InvalidModelResponse("every score must be finite")
            normalized_scores.append(value)
        if not isinstance(used_aliases, list) or any(
            not isinstance(item, str) for item in used_aliases
        ):
            raise InvalidModelResponse("used_signal_ids must be a string array")
        if len(used_aliases) != len(set(used_aliases)):
            raise InvalidModelResponse("used_signal_ids must not contain duplicates")
        if alias_to_id and not used_aliases:
            raise InvalidModelResponse("at least one supplied signal must be cited")
        if any(item not in alias_to_id for item in used_aliases):
            raise InvalidModelResponse("used_signal_ids cited an unknown signal")
        return {
            "candidate": candidate,
            "confidence": confidence,
            "scores": tuple(normalized_scores),
            "used_signal_ids": tuple(alias_to_id[item] for item in used_aliases),
        }
