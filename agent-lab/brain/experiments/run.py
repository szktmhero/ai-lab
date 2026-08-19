"""Run the matched deterministic Brain Architecture pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..baselines import BASELINE_CONDITIONS, make_system
from ..metrics.collector import aggregate_task_metrics, evaluate_result
from ..observer.observer import BrainObserver
from ..tasks.generator import generate_task_suite
from .analysis import paired_contrasts
from .individuality import run_individuality_smoke_test

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONDITIONS = (
    *BASELINE_CONDITIONS,
    "brain_full",
    "brain_no_memory",
    "brain_no_error_monitor",
    "brain_static_routing",
    "brain_no_internal_state_coupling",
)

CONTRASTS = [
    (
        "brain_full",
        "single_long_context",
        "Full recurrent architecture versus a single full-context inference",
    ),
    (
        "brain_full",
        "sequential_reflection",
        "Full recurrent architecture versus a strong sequential reflection baseline",
    ),
    (
        "brain_full",
        "flat_ensemble",
        "Full recurrent architecture versus flat partitioned aggregation",
    ),
    (
        "brain_full",
        "brain_no_memory",
        "Causal effect of bounded episodic retrieval",
    ),
    (
        "brain_full",
        "brain_no_error_monitor",
        "Causal effect of conflict-triggered robust processing",
    ),
    (
        "brain_full",
        "brain_static_routing",
        "Adaptive versus fixed robust routing",
    ),
    (
        "brain_full",
        "brain_no_internal_state_coupling",
        "Effect of connecting observed prediction error to routing",
    ),
]

ANALYSIS_METRICS = [
    "accuracy",
    "regime_switch_accuracy",
    "delayed_recall_accuracy",
    "fault_recovery_accuracy",
    "avg_model_calls",
    "avg_input_units",
]


def _design(n_seeds: int, n_per_type: int) -> dict[str, Any]:
    return {
        "status": "pre-registered deterministic pilot; not a formal LLM result",
        "n_seeds": n_seeds,
        "n_instances_per_task_type_per_seed": n_per_type,
        "task_types": [
            "latent_rule_switch",
            "delayed_recall",
            "fault_recovery",
            "partial_observation_plan",
        ],
        "conditions": list(CONDITIONS),
        "model": "DeterministicModelAdapter shared unchanged by every condition",
        "episode_budget_cap": {
            "model_calls": 4,
            "input_units": 48,
            "output_units": 8,
        },
        "matched_factors": [
            "task instance",
            "correct candidate",
            "public evidence sequence",
            "evidence noise",
        ],
        "independent_random_streams": (
            "task seeds derive from seed, task type, and instance only; condition "
            "is excluded from generation"
        ),
        "primary_hypotheses": {
            "H1": "brain_full improves regime-switch and fault recovery over equal-cap baselines",
            "H2": "memory selectively improves delayed recall and distributed evidence integration",
            "H3": "error monitoring selectively improves injected-fault recovery",
            "H4": "adaptive routing avoids robust calls on low-conflict tasks",
            "H5": "different histories can produce persistent choices that follow memory reset/swap",
        },
        "inference_unit": "per-seed mean over the matched task suite",
        "formal_run_gate": (
            "freeze a real ModelAdapter and exact token-matched condition matrix after "
            "reviewing this harness pilot"
        ),
        "known_limitations": [
            "The deterministic adapter and evidence task grammar are designed test doubles.",
            "A performance difference validates a causal path, not biological realism or emergence.",
            "single_long_context intentionally uses fewer calls; compute is reported rather than padded.",
            "Three pilot seeds cannot support precise inferential claims.",
        ],
    }


def run_pilot(
    *,
    n_seeds: int = 3,
    n_per_type: int = 20,
    output_dir: Path = RESULTS_DIR,
    save_event_logs: bool = True,
) -> str:
    if n_seeds <= 0 or n_per_type <= 0:
        raise ValueError("n_seeds and n_per_type must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    design = _design(n_seeds, n_per_type)
    tasks_by_seed = {
        seed: generate_task_suite(seed, n_per_type=n_per_type)
        for seed in range(n_seeds)
    }
    task_metrics: list[dict[str, Any]] = []
    seed_metrics: dict[str, list[dict[str, Any]]] = {}
    comparison: dict[str, dict[str, Any]] = {}

    for condition in CONDITIONS:
        condition_rows: list[dict[str, Any]] = []
        condition_seed_metrics: list[dict[str, Any]] = []
        for seed in range(n_seeds):
            system = make_system(condition)
            observer = BrainObserver()
            seed_rows: list[dict[str, Any]] = []
            for task in tasks_by_seed[seed]:
                result = system.run(task.view, observer=observer)
                row = evaluate_result(seed, condition, task, result)
                seed_rows.append(row)
                condition_rows.append(row)
                task_metrics.append(row)
            summary = {"seed": seed, **aggregate_task_metrics(seed_rows)}
            condition_seed_metrics.append(summary)
            if save_event_logs:
                observer.save(
                    output_dir
                    / "pilot"
                    / condition
                    / f"seed{seed:03d}"
                    / "events.jsonl"
                )
        seed_metrics[condition] = condition_seed_metrics
        comparison[condition] = aggregate_task_metrics(condition_rows)

    statistics = paired_contrasts(
        seed_metrics,
        CONTRASTS,
        ANALYSIS_METRICS,
    )
    individuality = run_individuality_smoke_test()
    report = _generate_report(design, comparison, statistics, individuality)

    files = {
        "design.json": design,
        "comparison.json": comparison,
        "seed_metrics.json": seed_metrics,
        "paired_statistics.json": statistics,
        "individuality.json": individuality,
    }
    for filename, payload in files.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    pilot_dir = output_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / "task_metrics.json").write_text(
        json.dumps(task_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "COMPARISON.md").write_text(report, encoding="utf-8")
    return report


def _generate_report(
    design: dict[str, Any],
    comparison: dict[str, dict[str, Any]],
    statistics: list[dict[str, Any]],
    individuality: dict[str, Any],
) -> str:
    lines = [
        "# Brain Architecture: Deterministic Pilot",
        "",
        "## Status",
        "",
        (
            "This is a harness and causal-path pilot using a deterministic test double. "
            "It is **not** evidence that an LLM brain, personality, subjectivity, or "
            "emergent intelligence has been created."
        ),
        "",
        "## Design",
        "",
        (
            f"{design['n_seeds']} matched seeds × "
            f"{design['n_instances_per_task_type_per_seed']} instances × 4 task types. "
            "All conditions received the exact same immutable TaskView objects."
        ),
        "",
        "| Condition | Accuracy | Switch | Recall | Fault recovery | Calls | Input units |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = comparison[condition]
        lines.append(
            f"| {condition} | {metrics['accuracy']:.3f} | "
            f"{metrics['regime_switch_accuracy']:.3f} | "
            f"{metrics['delayed_recall_accuracy']:.3f} | "
            f"{metrics['fault_recovery_accuracy']:.3f} | "
            f"{metrics['avg_model_calls']:.2f} | "
            f"{metrics['avg_input_units']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Seed Contrasts",
            "",
            (
                "Differences are left minus right. With only three pilot seeds, exact "
                "p-values are coarse and are included to validate the analysis path, "
                "not to support formal claims."
            ),
            "",
            "| Contrast | Accuracy Δ | 95% bootstrap CI | exact p | Calls Δ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for contrast in statistics:
        accuracy = contrast["metrics"]["accuracy"]
        calls = contrast["metrics"]["avg_model_calls"]
        low, high = accuracy["bootstrap_95_ci"]
        lines.append(
            f"| {contrast['left']} − {contrast['right']} | "
            f"{accuracy['mean_difference']:+.3f} | "
            f"[{low:+.3f}, {high:+.3f}] | "
            f"{accuracy['exact_sign_flip_p']:.3f} | "
            f"{calls['mean_difference']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## History-dependence manipulation",
            "",
            f"- Different-history divergence: {individuality['between_history_divergence']}",
            f"- Divergence after memory reset: {individuality['reset_divergence']}",
            f"- Choice followed swapped memory: {individuality['swap_followed_memory']}",
            (
                "- Interpretation: this only verifies that memory is a causal path for "
                "history-dependent choices. The histories and probe were deliberately "
                "constructed, so this is not emergent individuality."
            ),
            "",
            "## Interpretation discipline",
            "",
            "### DESIGNED",
            "",
            (
                "- Bounded workspace, six-record retrieval, conflict-triggered robust "
                "processing, and robust-output gating are explicit architecture choices."
            ),
            "- Task families were constructed to exercise those paths.",
            "",
            "### OBSERVED",
            "",
            "- The table above records deterministic harness behavior and compute use.",
            "- Reset and swap interventions change the history-probe result through memory.",
            "",
            "### INFERRED",
            "",
            (
                "- A selective ablation difference indicates that the implemented path is "
                "active and measurable in this harness."
            ),
            "",
            "### SPECULATIVE / NOT YET TESTED",
            "",
            (
                "- Whether the architecture helps a real language model under a strict "
                "token-matched budget."
            ),
            (
                "- Whether stable behavioral differentiation appears without an explicitly "
                "constructed history cue."
            ),
            (
                "- Any claim of personality, subjectivity, consciousness, biological "
                "similarity, or emergent intelligence."
            ),
            "",
            "## Next gate",
            "",
            (
                "Freeze one real ModelAdapter, a token-matched matrix, and a blind procedural "
                "evaluation set. Do not reuse this pilot's tuned task instances as formal evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--instances-per-type", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-event-logs", action="store_true")
    args = parser.parse_args()
    report = run_pilot(
        n_seeds=args.seeds,
        n_per_type=args.instances_per_type,
        output_dir=args.output_dir,
        save_event_logs=not args.no_event_logs,
    )
    print(report)


if __name__ == "__main__":
    main()
