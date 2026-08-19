"""Experiment runner for society simulation."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..core.world import Society
from ..policies.random import RandomPolicy
from ..policies.simple import SimplePolicy
from ..observer.observer import Observer
from ..tasks import generate_task_sequence
from ..simulation import run_multi_task
from ..visualization.renderer import Renderer
from .analysis import ANALYSIS_METRICS, paired_contrasts


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

# This factorial slice separates network reach from the explicit support-count
# term in SimplePolicy.  The latter can create conformity without any emergent
# social structure, so it must not be bundled into the topology comparison.
CONDITIONS = {
    "independent": {
        "network_mode": "independent",
        "social_evidence_weight": 0.0,
    },
    "full_evidence": {
        "network_mode": "full",
        "social_evidence_weight": 0.0,
    },
    "local_evidence": {
        "network_mode": "local",
        "social_evidence_weight": 0.0,
    },
    "full_social": {
        "network_mode": "full",
        "social_evidence_weight": 0.05,
    },
    "local_social": {
        "network_mode": "local",
        "social_evidence_weight": 0.05,
    },
}

CONTRASTS = [
    (
        "full_evidence",
        "independent",
        "Full-network exchange versus fragmented independent proposals, with the support-count term disabled",
    ),
    (
        "local_evidence",
        "independent",
        "Local-network exchange versus fragmented independent proposals, with the support-count term disabled",
    ),
    (
        "full_social",
        "full_evidence",
        "Effect of the support-count term in the full network",
    ),
    (
        "local_social",
        "local_evidence",
        "Effect of the support-count term in the local network",
    ),
    (
        "local_evidence",
        "full_evidence",
        "Effect of sparse versus full topology without the support-count term",
    ),
]


def _resolve_condition(condition: str) -> dict:
    """Resolve canonical conditions while retaining the old labels as aliases."""
    aliases = {"full": "full_social", "local": "local_social"}
    canonical = aliases.get(condition, condition)
    if canonical not in CONDITIONS:
        raise ValueError(
            f"Unknown condition: {condition}. Expected one of {sorted(CONDITIONS)}"
        )
    return {"name": canonical, **CONDITIONS[canonical]}


def run_experiment(
    policy_name: str = "simple",
    condition: str = "local_social",
    n_agents: int = 128,
    n_tasks: int = 50,
    n_seeds: int = 3,
    max_rounds: int = 20,
    output_dir: Optional[Path] = None,
    save_artifacts: bool = True,
    include_run_details: bool = True,
) -> Dict:
    """Run a single policy experiment across multiple seeds and tasks."""
    condition_config = _resolve_condition(condition)
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR / condition_config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    seed_metrics = []

    for seed in range(n_seeds):
        seed_dir = output_dir / f"seed{seed:03d}"
        if save_artifacts:
            seed_dir.mkdir(exist_ok=True)

        # Create society
        society = Society(
            agent_count=n_agents,
            seed=seed,
            network_mode=condition_config["network_mode"],
        )

        # Create policy
        if policy_name == "random":
            policy = RandomPolicy(seed=seed)
        elif policy_name == "simple":
            policy = SimplePolicy(
                seed=seed,
                social_evidence_weight=condition_config["social_evidence_weight"],
            )
            policy.set_agents(society.agents)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # Generate tasks
        tasks = generate_task_sequence(seed, n_tasks, n_agents)

        # Create observer
        observer = Observer()

        # Run multi-task simulation
        results = run_multi_task(
            society, policy, tasks,
            max_rounds_per_task=max_rounds,
            observer=observer,
        )

        if save_artifacts:
            observer.save_events(seed_dir / "events.jsonl")
            observer.save_rounds(seed_dir / "rounds.json")
            observer.save_tasks(seed_dir / "tasks.json")
            (seed_dir / "agent_metrics.json").write_text(
                json.dumps(observer.agent_metrics(society), indent=2)
            )

            config = {
                "policy": policy_name,
                "condition": condition_config["name"],
                "network_mode": condition_config["network_mode"],
                "social_evidence_weight": condition_config["social_evidence_weight"],
                "seed": seed,
                "n_agents": n_agents,
                "n_tasks": n_tasks,
                "max_rounds": max_rounds,
                "action_costs": {
                    key.name: value for key, value in society.action_costs.items()
                },
                "decision_quality_definition": (
                    "normalized latent quality of the selected synthetic option"
                ),
            }
            (seed_dir / "config.json").write_text(json.dumps(config, indent=2))

            society_snapshot = society.snapshot()
            (seed_dir / "society_state.json").write_text(
                json.dumps(society_snapshot, default=str, indent=2)
            )

            renderer = Renderer(seed_dir)
            renderer.render_network(society_snapshot, observer.event_log)
            renderer.render_timeline(observer.event_log)
            renderer.render_metrics(observer.round_summaries)
            renderer.render_proposals(
                society_snapshot, observer.event_log, observer.task_summaries
            )

            chronicle = observer.generate_chronicle()
            (seed_dir / "chronicle.md").write_text(chronicle)

        all_results.append({
            "seed": seed,
            "results": results,
            "observer": observer,
        })
        seed_metrics.append(_summarize_results(seed, results, observer.task_summaries))

        print(f"  Seed {seed}: {sum(r['rounds'] for r in results)} rounds, "
              f"{sum(r['proposals'] for r in results)} proposals")

    # Compute aggregate metrics
    agg = _compute_aggregates(all_results, n_tasks)
    (output_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (output_dir / "seed_metrics.json").write_text(json.dumps(seed_metrics, indent=2))

    return {
        "policy": policy_name,
        "seeds": all_results if include_run_details else [],
        "seed_metrics": seed_metrics,
        "condition": condition_config,
        "aggregate": agg,
    }


def _summarize_results(seed: int, results: List[dict], task_metrics: List[dict]) -> Dict:
    """Summarize one seed so tasks are not treated as independent replicates."""
    winners = [result["winner"] for result in results]
    return {
        "seed": seed,
        "avg_rounds": float(np.mean([result["rounds"] for result in results]))
        if results else 0.0,
        "avg_proposals": float(np.mean([result["proposals"] for result in results]))
        if results else 0.0,
        "winner_rate": sum(winner is not None for winner in winners) /
        max(len(winners), 1),
        "avg_consensus": float(np.mean([
            metric["consensus_level"] for metric in task_metrics
        ])) if task_metrics else 0.0,
        "avg_communication_cost": float(np.mean([
            metric["communication_cost"] for metric in task_metrics
        ])) if task_metrics else 0.0,
        "avg_information_spread": float(np.mean([
            metric["information_spread"] for metric in task_metrics
        ])) if task_metrics else 0.0,
        "avg_trust": float(np.mean([
            metric["average_trust"] for metric in task_metrics
        ])) if task_metrics else 0.0,
        "avg_network_density": float(np.mean([
            metric["network_density"] for metric in task_metrics
        ])) if task_metrics else 0.0,
        "avg_decision_quality": float(np.mean([
            metric["decision_quality"] for metric in task_metrics
        ])) if task_metrics else 0.0,
        "round_limit_rate": float(np.mean([
            metric["reached_round_limit"] for metric in task_metrics
        ])) if task_metrics else 0.0,
        "majority_agreement": float(np.mean([
            result["winner"] == result["majority_winner"] for result in results
        ])) if results else 0.0,
    }


def _compute_aggregates(all_results: List[Dict], n_tasks: int) -> Dict:
    """Compute descriptive metrics across every generated task."""
    winners = []
    total_rounds = []
    total_proposals = []
    task_metrics = []
    majority_agreement = []

    for res in all_results:
        for r in res["results"]:
            winners.append(r["winner"])
            total_rounds.append(r["rounds"])
            total_proposals.append(r["proposals"])
            task_metrics.append(r["task_summary"])
            majority_agreement.append(r["winner"] == r["majority_winner"])

    return {
        "avg_rounds": float(np.mean(total_rounds)) if total_rounds else 0,
        "avg_proposals": float(np.mean(total_proposals)) if total_proposals else 0,
        "total_tasks_with_winner": sum(1 for w in winners if w is not None),
        "total_tasks": len(winners),
        "winner_rate": sum(1 for w in winners if w is not None) / max(len(winners), 1),
        "avg_consensus": float(np.mean([m["consensus_level"] for m in task_metrics])) if task_metrics else 0,
        "avg_communication_cost": float(np.mean([m["communication_cost"] for m in task_metrics])) if task_metrics else 0,
        "avg_information_spread": float(np.mean([m["information_spread"] for m in task_metrics])) if task_metrics else 0,
        "avg_trust": float(np.mean([m["average_trust"] for m in task_metrics])) if task_metrics else 0,
        "avg_network_density": float(np.mean([m["network_density"] for m in task_metrics])) if task_metrics else 0,
        "avg_decision_quality": float(np.mean([m["decision_quality"] for m in task_metrics])) if task_metrics else 0,
        "round_limit_rate": float(np.mean([
            m["reached_round_limit"] for m in task_metrics
        ])) if task_metrics else 0,
        "majority_agreement": float(np.mean(majority_agreement)) if majority_agreement else 0,
    }


def run_comparison(
    n_agents: int = 128,
    n_tasks: int = 50,
    n_seeds: int = 10,
    max_rounds: int = 20,
    output_dir: Optional[Path] = None,
    save_artifacts: bool = True,
    parallel: bool = False,
    max_workers: int = 4,
) -> str:
    """Run the network × support-count ablation with matched random seeds."""
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {}
    seed_metrics = {}
    condition_configs = {}

    def record_result(condition: str, result: dict):
        comparison[condition] = result["aggregate"]
        seed_metrics[condition] = result["seed_metrics"]
        condition_configs[condition] = result["condition"]

    experiment_arguments = {
        condition: {
            "policy_name": "simple",
            "condition": condition,
            "n_agents": n_agents,
            "n_tasks": n_tasks,
            "n_seeds": n_seeds,
            "max_rounds": max_rounds,
            "output_dir": output_dir / condition,
            "save_artifacts": save_artifacts,
            "include_run_details": not parallel,
        }
        for condition in CONDITIONS
    }

    if parallel:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(CONDITIONS))) as executor:
            futures = {
                executor.submit(run_experiment, **arguments): condition
                for condition, arguments in experiment_arguments.items()
            }
            for future in as_completed(futures):
                condition = futures[future]
                record_result(condition, future.result())
    else:
        for condition, arguments in experiment_arguments.items():
            print(f"\nRunning {condition} condition...")
            record_result(condition, run_experiment(**arguments))

    # Preserve the declared condition order regardless of parallel completion.
    comparison = {condition: comparison[condition] for condition in CONDITIONS}
    seed_metrics = {condition: seed_metrics[condition] for condition in CONDITIONS}
    condition_configs = {
        condition: condition_configs[condition] for condition in CONDITIONS
    }

    statistics = paired_contrasts(seed_metrics, CONTRASTS)
    design = {
        "n_agents": n_agents,
        "n_tasks_per_seed": n_tasks,
        "n_seeds": n_seeds,
        "max_rounds": max_rounds,
        "per_seed_artifacts_saved": save_artifacts,
        "matched_factors": [
            "agent attributes",
            "generated tasks",
            "initial information correctness and recipients",
        ],
        "conditions": condition_configs,
        "analysis_unit": "per-seed mean across the persistent 50-task sequence",
    }

    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (output_dir / "seed_metrics.json").write_text(json.dumps(seed_metrics, indent=2))
    (output_dir / "paired_statistics.json").write_text(
        json.dumps(statistics, indent=2)
    )
    (output_dir / "design.json").write_text(json.dumps(design, indent=2))

    report = _generate_comparison_report(comparison, statistics, design)
    (output_dir / "COMPARISON.md").write_text(report)

    return report


def _generate_comparison_report(
    comparison: Dict,
    statistics: List[dict],
    design: Dict,
) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# Society Experiment: Comparison Report\n",
        "## Design\n",
        f"{design['n_seeds']} matched seeds × {design['n_tasks_per_seed']} persistent tasks "
        f"× {design['n_agents']} agents; up to {design['max_rounds']} rounds per task.\n",
        "The comparison crosses network reach (`independent`, `full`, `local`) with "
        "the SimplePolicy support-count term (`0.00` or `0.05`). Agent attributes, "
        "tasks, and initial information assignments are matched within each seed.\n",
        "## Metrics Summary\n",
        "| Condition | Rounds | Limit rate | Proposals | Quality | Consensus | Comm. cost | Info spread |",
        "|-----------|--------|------------|-----------|---------|-----------|------------|-------------|",
    ]

    for condition, agg in comparison.items():
        lines.append(
            f"| {condition} | {agg['avg_rounds']:.1f} | "
            f"{agg['round_limit_rate']:.3f} | {agg['avg_proposals']:.1f} | "
            f"{agg['avg_decision_quality']:.3f} | "
            f"{agg['avg_consensus']:.3f} | {agg['avg_communication_cost']:.1f} | "
            f"{agg['avg_information_spread']:.3f} |"
        )

    lines.extend([
        "\n## Paired Seed Contrasts\n",
        "Differences are `left - right`. Intervals bootstrap the paired seed means; "
        "p-values are two-sided exact sign-flip tests over seed blocks.\n",
        "| Contrast | Metric | Difference | 95% CI | p |",
        "|----------|--------|------------|--------|---|",
    ])
    report_metrics = [
        "avg_decision_quality",
        "avg_consensus",
        "avg_information_spread",
        "round_limit_rate",
    ]
    for contrast in statistics:
        label = f"{contrast['left']} − {contrast['right']}"
        for metric in report_metrics:
            result = contrast["metrics"][metric]
            low, high = result["bootstrap_95_ci"]
            lines.append(
                f"| {label} | {ANALYSIS_METRICS[metric]} | "
                f"{result['mean_difference']:+.3f} | [{low:+.3f}, {high:+.3f}] | "
                f"{result['exact_sign_flip_p']:.4f} |"
            )

    lines.extend([
        "\n## Interpretation Boundaries\n",
        "**DESIGNED:** communication limits, action costs, topology, the optional "
        "support-count term, trust updates, and the final decision score.",
        "",
        "**OBSERVED:** the first table contains task-level descriptive means. The "
        "contrast table treats a complete persistent task sequence as one replicate.",
        "",
        "**INFERRED:** a topology effect is separated from the explicit conformity "
        "term only by the named paired contrasts; p-values describe this simulator, "
        "not a population of real societies.",
        "",
        "**NOT TESTED:** trust values persist and update, but SimplePolicy does not "
        "yet use trust to value evidence or choose communication partners. Therefore "
        "cross-task trust learning has no policy-level causal path in this experiment.",
        "",
        "**BASELINE LIMITATION:** independent agents create separate proposals that "
        "remain tied at one supporter each. The final proposal-level tie break does "
        "not aggregate their private choices by option, so network-versus-independent "
        "quality differences are not a pure estimate of information sharing.",
        "",
        "**SPECULATIVE:** leadership, coalitions, specialization, authority, and "
        "institutions are not established by this run.",
        "",
        "`decision_quality` is normalized latent option quality in this synthetic "
        "task model; it is not a general measure of judgment or intelligence.",
    ])

    return "\n".join(lines)
