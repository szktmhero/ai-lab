"""Phase 3 real-model protocol, preflight, and explicitly gated formal runner.

Running this module without ``--execute`` performs only a local protocol
preflight.  The paid blind run additionally requires the committed lock hash as
an acknowledgement, so tests and casual commands cannot unblind the task set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..baselines import make_system
from ..core.budget import BudgetLimits
from ..metrics.collector import aggregate_task_metrics, evaluate_result
from ..models.openai_responses import (
    OpenAIResponsesAdapter,
    OpenAIResponsesConfig,
)
from ..observer.observer import BrainObserver
from ..tasks.blind import (
    BLIND_INSTANCES_PER_TYPE,
    BLIND_SEEDS,
    BLIND_SUITE_VERSION,
    generate_blind_task_suite,
    validate_blind_suite,
)
from .analysis import paired_contrasts
from .run import ANALYSIS_METRICS, CONDITIONS, CONTRASTS

PROTOCOL_ID = "brain-real-llm-v1"
FORMAL_MODEL = "gpt-5.4-mini-2026-03-17"
FORMAL_OPENAI_SDK_VERSION = "2.51.0"
FORMAL_MODEL_CONFIG = OpenAIResponsesConfig(
    model=FORMAL_MODEL,
    reasoning_effort="none",
    max_output_tokens=128,
    store=False,
    require_snapshot=True,
    client_max_retries=0,
    client_timeout_seconds=120.0,
    min_request_interval_seconds=0.15,
)
FORMAL_BUDGET = BudgetLimits(
    max_calls=4,
    max_input_units=8192,
    max_output_units=512,
    unit_name="tokens",
)
FORMAL_ANALYSIS_METRICS = [
    *ANALYSIS_METRICS,
    "partial_observation_accuracy",
    "avg_output_units",
    "avg_total_units",
]
PRIMARY_EFFECT_THRESHOLDS: dict[str, float | str] = {
    "H1_min_regime_switch_gain_over_each_strong_baseline": 0.05,
    "H1_min_fault_recovery_gain_over_each_strong_baseline": 0.05,
    "H2_min_delayed_recall_memory_ablation_effect": 0.10,
    "H2_min_partial_integration_memory_ablation_effect": 0.10,
    "H3_min_selective_fault_ablation_effect": 0.10,
    "H4_max_accuracy_loss": 0.02,
    "H4_min_compute_reduction_fraction": 0.10,
    "H4_compute_metric": "avg_total_units",
}
LOCK_PATH = Path(__file__).with_name("formal_protocol.lock.json")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / PROTOCOL_ID
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ProtocolLockError(RuntimeError):
    """Raised before inference when code and committed preregistration differ."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def protocol_sha256(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(protocol).encode("utf-8")).hexdigest()


def implementation_sha256() -> tuple[str, list[str]]:
    """Hash executable Brain sources and the dependency declaration.

    Tests, generated results, documentation, and bytecode do not affect the
    decision path and are intentionally outside this formal lock.
    """

    source_paths = [
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(PACKAGE_ROOT).parts
        and "__pycache__" not in path.relative_to(PACKAGE_ROOT).parts
    ]
    source_paths.append(PACKAGE_ROOT / "pyproject.toml")
    relative_paths = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix() for path in source_paths
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((PACKAGE_ROOT / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def build_protocol() -> dict[str, Any]:
    """Return the complete preregistration payload compared to the lock file."""

    source_hash, source_files = implementation_sha256()
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "LOCKED_NOT_EXECUTED",
        "implementation": {
            "source_sha256": source_hash,
            "files": source_files,
            "exclusions": ["tests", "generated results", "documentation", "bytecode"],
        },
        "model_adapter": {
            **FORMAL_MODEL_CONFIG.public_fingerprint(),
            "openai_sdk_version": FORMAL_OPENAI_SDK_VERSION,
            "provider_token_counting": "responses.input_tokens.count",
            "response_usage_authority": "response.usage input_tokens/output_tokens",
            "input_count_mismatch_policy": "abort the entire run",
        },
        "episode_budget": {
            "max_calls": FORMAL_BUDGET.max_calls,
            "max_input_tokens": FORMAL_BUDGET.max_input_units,
            "max_output_tokens": FORMAL_BUDGET.max_output_units,
            "max_output_tokens_per_call": FORMAL_MODEL_CONFIG.max_output_tokens,
            "matching_rule": (
                "Every condition receives the identical hard cap. Each call is "
                "preflighted with the provider count and its full output allowance; "
                "actual response usage is committed. No padding or dummy calls."
            ),
        },
        "blind_tasks": {
            "suite_version": BLIND_SUITE_VERSION,
            "suite_sha256": validate_blind_suite(),
            "seeds": list(BLIND_SEEDS),
            "instances_per_task_type_per_seed": BLIND_INSTANCES_PER_TYPE,
            "task_types": [
                "latent_rule_switch",
                "delayed_recall",
                "fault_recovery",
                "partial_observation_plan",
            ],
            "task_count": (
                len(BLIND_SEEDS) * BLIND_INSTANCES_PER_TYPE * 4
            ),
            "unblinding_rule": (
                "No live model call or outcome inspection before the acknowledged "
                "all-condition formal run. Development smoke tests use pilot tasks."
            ),
        },
        "conditions": list(CONDITIONS),
        "condition_order": (
            "Deterministic task-block shuffle derived from protocol_id and opaque task_id"
        ),
        "model_invariance": {
            "one_adapter_instance": True,
            "condition_visible_to_adapter": False,
            "task_answer_visible_to_system": False,
            "conversation_state_between_calls": False,
            "internal_state_visible_to_model": False,
        },
        "primary_hypotheses": {
            "H1": (
                "brain_full improves regime-switch and fault recovery over both "
                "sequential_reflection and flat_ensemble"
            ),
            "H2": (
                "brain_full selectively improves delayed recall and partial "
                "integration over brain_no_memory"
            ),
            "H3": (
                "brain_full selectively improves fault recovery over "
                "brain_no_error_monitor"
            ),
            "H4": (
                "brain_full reduces token/call compute versus brain_static_routing "
                "without a material accuracy loss"
            ),
            "H5": (
                "history dependence remains a separate causal manipulation and is "
                "not a primary formal emergence outcome"
            ),
        },
        "primary_effect_thresholds": dict(PRIMARY_EFFECT_THRESHOLDS),
        "analysis": {
            "inference_unit": "per-seed mean over matched tasks",
            "matched_seed_count": len(BLIND_SEEDS),
            "metrics": list(FORMAL_ANALYSIS_METRICS),
            "threshold_decision_rule": (
                "PASS iff every preregistered point-effect threshold for that "
                "hypothesis passes. Exact sign-flip tests and paired bootstrap "
                "intervals are reported as uncertainty, not substituted post hoc "
                "as extra pass gates."
            ),
            "contrasts": [
                {"left": left, "right": right, "description": description}
                for left, right, description in CONTRASTS
            ],
            "retain_task_level_failures": True,
            "claim_labels": [
                "DESIGNED",
                "OBSERVED",
                "INFERRED",
                "SPECULATIVE",
            ],
        },
        "failure_and_amendment_policy": {
            "automatic_inference_retries": 0,
            "protocol_violation": "abort and preserve partial append-only artifacts",
            "nonempty_output_directory": "refuse to start",
            "post_outcome_threshold_changes": "forbidden",
            "blind_task_tuning": "forbidden",
            "protocol_amendments": (
                "must create a new protocol id and lock before any new outcomes"
            ),
        },
        "interpretation_limits": [
            "A model-dependent effect is not a model-independent law of intelligence.",
            "Designed modules and prompts are not emergent roles.",
            "Accuracy gains do not establish personality, subjectivity, or consciousness.",
            "Failure to find an effect does not establish impossibility for stronger models.",
        ],
    }


def load_and_verify_protocol() -> dict[str, Any]:
    if not LOCK_PATH.exists():
        raise ProtocolLockError(f"missing protocol lock: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked_protocol = lock.get("protocol")
    locked_hash = lock.get("lock_sha256")
    if not isinstance(locked_protocol, dict) or not isinstance(locked_hash, str):
        raise ProtocolLockError("protocol lock has an invalid structure")
    actual_locked_hash = protocol_sha256(locked_protocol)
    if actual_locked_hash != locked_hash:
        raise ProtocolLockError("protocol lock hash does not match its payload")
    current = build_protocol()
    if current != locked_protocol:
        raise ProtocolLockError(
            "current Phase 3 code/config differs from the committed protocol lock"
        )
    return lock


def formal_preflight() -> dict[str, Any]:
    lock = load_and_verify_protocol()
    task_count = len(BLIND_SEEDS) * BLIND_INSTANCES_PER_TYPE * 4
    episode_count = task_count * len(CONDITIONS)
    maximum_inference_calls = episode_count * FORMAL_BUDGET.max_calls
    maximum_provider_requests = maximum_inference_calls * 2
    installed_sdk_version = OpenAIResponsesAdapter.sdk_version()
    api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "protocol_id": PROTOCOL_ID,
        "lock_sha256": lock["lock_sha256"],
        "model": FORMAL_MODEL,
        "task_count": task_count,
        "condition_count": len(CONDITIONS),
        "episode_count": episode_count,
        "maximum_inference_calls": maximum_inference_calls,
        "maximum_token_count_requests": maximum_inference_calls,
        "maximum_provider_requests": maximum_provider_requests,
        "minimum_request_start_span_seconds": (
            max(maximum_provider_requests - 1, 0)
            * FORMAL_MODEL_CONFIG.min_request_interval_seconds
        ),
        "live_api_called": False,
        "openai_sdk_version": installed_sdk_version,
        "sdk_version_matches": installed_sdk_version == FORMAL_OPENAI_SDK_VERSION,
        "api_key_present": api_key_present,
        "ready_for_paid_run": (
            api_key_present and installed_sdk_version == FORMAL_OPENAI_SDK_VERSION
        ),
    }


def _condition_order(task_id: str) -> list[str]:
    digest = hashlib.sha256(f"{PROTOCOL_ID}:{task_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    ordered = list(CONDITIONS)
    rng.shuffle(ordered)
    return ordered


def _assert_output_directory_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"formal output directory is not empty; refusing overwrite: {output_dir}"
        )


def _prepare_output_directory(output_dir: Path) -> None:
    _assert_output_directory_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _save_adapter_call_records(
    output_dir: Path,
    adapter: OpenAIResponsesAdapter,
) -> None:
    path = output_dir / "adapter_call_records.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in adapter.call_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _formal_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_task_metrics(rows)
    if not rows:
        return metrics
    metrics["partial_observation_accuracy"] = metrics["accuracy_by_task"].get(
        "partial_observation_plan"
    )
    metrics["avg_total_units"] = sum(
        row["input_units"] + row["output_units"] for row in rows
    ) / len(rows)
    return metrics


def _minimum_check(observed: float, threshold: float) -> dict[str, Any]:
    return {
        "observed": observed,
        "criterion": ">=",
        "threshold": threshold,
        "passed": observed >= threshold,
    }


def _maximum_check(observed: float, threshold: float) -> dict[str, Any]:
    return {
        "observed": observed,
        "criterion": "<=",
        "threshold": threshold,
        "passed": observed <= threshold,
    }


def evaluate_primary_thresholds(
    comparison: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Mechanically apply the frozen point-effect criteria without discretion."""

    full = comparison["brain_full"]
    sequential = comparison["sequential_reflection"]
    flat = comparison["flat_ensemble"]
    no_memory = comparison["brain_no_memory"]
    no_error_monitor = comparison["brain_no_error_monitor"]
    static = comparison["brain_static_routing"]
    h1_switch = float(
        PRIMARY_EFFECT_THRESHOLDS[
            "H1_min_regime_switch_gain_over_each_strong_baseline"
        ]
    )
    h1_fault = float(
        PRIMARY_EFFECT_THRESHOLDS[
            "H1_min_fault_recovery_gain_over_each_strong_baseline"
        ]
    )
    h2_recall = float(
        PRIMARY_EFFECT_THRESHOLDS[
            "H2_min_delayed_recall_memory_ablation_effect"
        ]
    )
    h2_partial = float(
        PRIMARY_EFFECT_THRESHOLDS[
            "H2_min_partial_integration_memory_ablation_effect"
        ]
    )
    h3_fault = float(
        PRIMARY_EFFECT_THRESHOLDS["H3_min_selective_fault_ablation_effect"]
    )
    h4_accuracy_loss = float(
        PRIMARY_EFFECT_THRESHOLDS["H4_max_accuracy_loss"]
    )
    h4_compute = float(
        PRIMARY_EFFECT_THRESHOLDS["H4_min_compute_reduction_fraction"]
    )
    h4_compute_metric = str(PRIMARY_EFFECT_THRESHOLDS["H4_compute_metric"])

    decisions: dict[str, dict[str, Any]] = {
        "H1": {
            "checks": {
                "regime_switch_gain_vs_sequential_reflection": _minimum_check(
                    full["regime_switch_accuracy"]
                    - sequential["regime_switch_accuracy"],
                    h1_switch,
                ),
                "regime_switch_gain_vs_flat_ensemble": _minimum_check(
                    full["regime_switch_accuracy"]
                    - flat["regime_switch_accuracy"],
                    h1_switch,
                ),
                "fault_recovery_gain_vs_sequential_reflection": _minimum_check(
                    full["fault_recovery_accuracy"]
                    - sequential["fault_recovery_accuracy"],
                    h1_fault,
                ),
                "fault_recovery_gain_vs_flat_ensemble": _minimum_check(
                    full["fault_recovery_accuracy"]
                    - flat["fault_recovery_accuracy"],
                    h1_fault,
                ),
            }
        },
        "H2": {
            "checks": {
                "delayed_recall_gain_vs_no_memory": _minimum_check(
                    full["delayed_recall_accuracy"]
                    - no_memory["delayed_recall_accuracy"],
                    h2_recall,
                ),
                "partial_integration_gain_vs_no_memory": _minimum_check(
                    full["partial_observation_accuracy"]
                    - no_memory["partial_observation_accuracy"],
                    h2_partial,
                ),
            }
        },
        "H3": {
            "checks": {
                "fault_recovery_gain_vs_no_error_monitor": _minimum_check(
                    full["fault_recovery_accuracy"]
                    - no_error_monitor["fault_recovery_accuracy"],
                    h3_fault,
                )
            }
        },
    }

    static_total = float(static[h4_compute_metric])
    if static_total > 0.0:
        compute_check = _minimum_check(
            (static_total - float(full[h4_compute_metric])) / static_total,
            h4_compute,
        )
    else:
        compute_check = {
            "observed": None,
            "criterion": ">=",
            "threshold": h4_compute,
            "passed": False,
            "reason": "static-routing total token use was not positive",
        }
    compute_check["metric"] = h4_compute_metric
    decisions["H4"] = {
        "checks": {
            "accuracy_loss_vs_static_routing": _maximum_check(
                float(static["accuracy"]) - float(full["accuracy"]),
                h4_accuracy_loss,
            ),
            "total_token_reduction_fraction_vs_static_routing": compute_check,
        }
    }
    for decision in decisions.values():
        decision["status"] = (
            "PASS"
            if all(check["passed"] for check in decision["checks"].values())
            else "FAIL"
        )
    decisions["H5"] = {
        "status": "NOT_EVALUATED",
        "reason": "separate causal manipulation; not a primary formal emergence outcome",
    }
    return decisions


def run_formal(
    *,
    acknowledgement: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    adapter: OpenAIResponsesAdapter | None = None,
) -> str:
    """Execute the complete blind matrix after an explicit paid-run gate."""

    lock = load_and_verify_protocol()
    lock_hash = str(lock["lock_sha256"])
    if acknowledgement != lock_hash[:16]:
        raise PermissionError(
            "formal run acknowledgement must equal the first 16 lock-hash characters"
        )
    _assert_output_directory_available(output_dir)
    adapter = adapter or OpenAIResponsesAdapter(FORMAL_MODEL_CONFIG)
    if adapter.config.public_fingerprint() != FORMAL_MODEL_CONFIG.public_fingerprint():
        raise ProtocolLockError("adapter configuration differs from the protocol lock")
    actual_sdk_version = adapter.sdk_version()
    if actual_sdk_version != FORMAL_OPENAI_SDK_VERSION:
        raise ProtocolLockError(
            "OpenAI SDK version differs from the protocol lock: "
            f"{actual_sdk_version!r} != {FORMAL_OPENAI_SDK_VERSION!r}"
        )
    _prepare_output_directory(output_dir)

    started_at = datetime.now(timezone.utc).isoformat()
    (output_dir / "protocol.lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "lock_sha256": lock_hash,
        "started_at": started_at,
        "model": FORMAL_MODEL,
        "openai_sdk_version": actual_sdk_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "implementation_source_sha256": lock["protocol"]["implementation"][
            "source_sha256"
        ],
        "status": "RUNNING",
    }
    provenance_path = output_dir / "run_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    seed_metrics: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    task_metrics_path = output_dir / "task_metrics.jsonl"

    try:
        for seed in BLIND_SEEDS:
            systems = {
                condition: make_system(
                    condition,
                    model=adapter,
                    budget_limits=FORMAL_BUDGET,
                )
                for condition in CONDITIONS
            }
            observers = {condition: BrainObserver() for condition in CONDITIONS}
            seed_rows: dict[str, list[dict[str, Any]]] = {
                condition: [] for condition in CONDITIONS
            }
            with task_metrics_path.open("a", encoding="utf-8") as task_file:
                for task in generate_blind_task_suite(seed):
                    for condition in _condition_order(task.view.task_id):
                        result = systems[condition].run(
                            task.view,
                            observer=observers[condition],
                        )
                        row = evaluate_result(seed, condition, task, result)
                        row["protocol_lock_sha256"] = lock_hash
                        row["model_call_records"] = result.trace.get(
                            "model_call_records",
                            [],
                        )
                        row["usage_unit"] = result.usage.unit_name
                        seed_rows[condition].append(row)
                        by_condition[condition].append(row)
                        task_file.write(json.dumps(row, sort_keys=True) + "\n")
                        task_file.flush()

            for condition in CONDITIONS:
                seed_metrics[condition].append(
                    {"seed": seed, **_formal_aggregate(seed_rows[condition])}
                )
                observers[condition].save(
                    output_dir
                    / "events"
                    / condition
                    / f"seed{seed:04d}.jsonl"
                )

        comparison = {
            condition: _formal_aggregate(rows)
            for condition, rows in by_condition.items()
        }
        threshold_decisions = evaluate_primary_thresholds(comparison)
        statistics = paired_contrasts(
            seed_metrics,
            CONTRASTS,
            FORMAL_ANALYSIS_METRICS,
        )
        report = _generate_formal_report(
            comparison,
            statistics,
            threshold_decisions,
            lock_hash,
        )
        for filename, payload in {
            "comparison.json": comparison,
            "seed_metrics.json": seed_metrics,
            "paired_statistics.json": statistics,
            "threshold_decisions.json": threshold_decisions,
        }.items():
            (output_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        (output_dir / "COMPARISON.md").write_text(report, encoding="utf-8")
        _save_adapter_call_records(output_dir, adapter)
        provenance.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "COMPLETED",
                "response_requests": adapter.response_requests,
                "token_count_requests": adapter.token_count_requests,
            }
        )
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report
    except Exception as error:
        try:
            _save_adapter_call_records(output_dir, adapter)
        except OSError as record_error:
            provenance["adapter_call_record_save_error"] = str(record_error)
        provenance.update(
            {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "status": "ABORTED",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "response_requests": adapter.response_requests,
                "token_count_requests": adapter.token_count_requests,
            }
        )
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise


def _generate_formal_report(
    comparison: dict[str, dict[str, Any]],
    statistics: list[dict[str, Any]],
    threshold_decisions: dict[str, dict[str, Any]],
    lock_hash: str,
) -> str:
    lines = [
        "# Brain Architecture: Phase 3 Real-Model Run",
        "",
        f"Protocol lock: `{lock_hash}`",
        "",
        "## OBSERVED",
        "",
        "| Condition | Accuracy | Fault recovery | Calls | Input tokens | Output tokens | Total tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = comparison[condition]
        lines.append(
            f"| {condition} | {metrics['accuracy']:.3f} | "
            f"{metrics['fault_recovery_accuracy']:.3f} | "
            f"{metrics['avg_model_calls']:.2f} | "
            f"{metrics['avg_input_units']:.2f} | "
            f"{metrics['avg_output_units']:.2f} | "
            f"{metrics['avg_total_units']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## PREREGISTERED EFFECT-SIZE GATES",
            "",
            *[
                f"- {hypothesis}: {decision['status']}"
                for hypothesis, decision in threshold_decisions.items()
            ],
            "",
            "## INFERRED",
            "",
            (
                "- Threshold decisions must be evaluated against the committed protocol; "
                "the report generator does not rewrite them after observing results."
            ),
            (
                f"- {len(statistics)} preregistered paired contrasts are retained in "
                "`paired_statistics.json` without combining unlike effects."
            ),
            "",
            "## SPECULATIVE / NOT ESTABLISHED",
            "",
            "- Generalization to other model families or capability levels.",
            "- Emergent roles, personality, subjectivity, consciousness, or biological realism.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-lock", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(formal_preflight(), indent=2, sort_keys=True))
        return
    print(
        run_formal(
            acknowledgement=args.acknowledge_lock,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
