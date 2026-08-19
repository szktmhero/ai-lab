"""Experiment runner for society simulation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from ..core.world import Society
from ..policies.random import RandomPolicy
from ..policies.simple import SimplePolicy
from ..observer.observer import Observer
from ..tasks import generate_task, generate_task_sequence
from ..simulation import run_simulation, run_multi_task
from ..visualization.renderer import Renderer


def run_experiment(
    policy_name: str = "simple",
    condition: str = "local",
    n_agents: int = 128,
    n_tasks: int = 50,
    n_seeds: int = 3,
    max_rounds: int = 20,
    output_dir: Path = Path("results"),
) -> Dict:
    """Run a single policy experiment across multiple seeds and tasks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in range(n_seeds):
        seed_dir = output_dir / f"seed{seed:03d}"
        seed_dir.mkdir(exist_ok=True)

        # Create society
        society = Society(agent_count=n_agents, seed=seed, network_mode=condition)

        # Create policy
        if policy_name == "random":
            policy = RandomPolicy(seed=seed)
        elif policy_name == "simple":
            policy = SimplePolicy(seed=seed)
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

        # Save results
        observer.save_events(seed_dir / "events.jsonl")
        observer.save_rounds(seed_dir / "rounds.json")
        observer.save_tasks(seed_dir / "tasks.json")
        (seed_dir / "agent_metrics.json").write_text(
            json.dumps(observer.agent_metrics(society), indent=2)
        )

        # Save config
        config = {
            "policy": policy_name,
            "condition": condition,
            "seed": seed,
            "n_agents": n_agents,
            "n_tasks": n_tasks,
            "max_rounds": max_rounds,
            "action_costs": {key.name: value for key, value in society.action_costs.items()},
            "decision_quality_definition": "normalized latent quality of the selected synthetic option",
        }
        (seed_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Save society state
        society_snapshot = society.snapshot()
        (seed_dir / "society_state.json").write_text(
            json.dumps(society_snapshot, default=str, indent=2)
        )

        renderer = Renderer(seed_dir)
        renderer.render_network(society_snapshot, observer.event_log)
        renderer.render_timeline(observer.event_log)
        renderer.render_metrics(observer.round_summaries)
        renderer.render_proposals(society_snapshot, observer.event_log, observer.task_summaries)

        # Generate chronicle
        chronicle = observer.generate_chronicle()
        (seed_dir / "chronicle.md").write_text(chronicle)

        all_results.append({
            "seed": seed,
            "results": results,
            "observer": observer,
        })

        print(f"  Seed {seed}: {sum(r['rounds'] for r in results)} rounds, "
              f"{sum(r['proposals'] for r in results)} proposals")

    # Compute aggregate metrics
    agg = _compute_aggregates(all_results, n_tasks)
    (output_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))

    return {
        "policy": policy_name,
        "seeds": all_results,
        "aggregate": agg,
    }


def _compute_aggregates(all_results: List[Dict], n_tasks: int) -> Dict:
    """Compute aggregate metrics across seeds."""
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
        "majority_agreement": float(np.mean(majority_agreement)) if majority_agreement else 0,
    }


def run_comparison(
    n_agents: int = 128,
    n_tasks: int = 50,
    n_seeds: int = 3,
    max_rounds: int = 20,
    output_dir: Path = Path("results"),
) -> str:
    """Run comparison between all policies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {}

    for condition in ["independent", "full", "local"]:
        print(f"\nRunning {condition} condition...")
        policy_dir = output_dir / condition
        result = run_experiment(
            policy_name="simple",
            condition=condition,
            n_agents=n_agents,
            n_tasks=n_tasks,
            n_seeds=n_seeds,
            max_rounds=max_rounds,
            output_dir=policy_dir,
        )
        comparison[condition] = result["aggregate"]

    # Save comparison
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    # Generate report
    report = _generate_comparison_report(comparison)
    (output_dir / "COMPARISON.md").write_text(report)

    return report


def _generate_comparison_report(comparison: Dict) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# Society Experiment: Comparison Report\n",
        "## Metrics Summary\n",
        "| Condition | Rounds | Proposals | Quality | Consensus | Comm. cost | Info spread |",
        "|-----------|--------|-----------|---------|-----------|------------|-------------|",
    ]

    for policy, agg in comparison.items():
        lines.append(
            f"| {policy} | {agg['avg_rounds']:.1f} | {agg['avg_proposals']:.1f} | "
            f"{agg['avg_decision_quality']:.3f} | "
            f"{agg['avg_consensus']:.3f} | {agg['avg_communication_cost']:.1f} | "
            f"{agg['avg_information_spread']:.3f} |"
        )

    lines.extend([
        "\n## Interpretation Boundaries\n",
        "**DESIGNED:** communication limits, action costs, sparse/full/absent networks, trust updates, and the decision score.",
        "",
        "**OBSERVED:** the table reports direct measurements only.",
        "",
        "**INFERRED:** none automatically. Differences between conditions require seed-level statistical analysis.",
        "",
        "**SPECULATIVE:** leadership, coalitions, specialization, authority, and institutions are not established by this run.",
        "",
        "`decision_quality` is normalized latent option quality in this synthetic task model; `evidence_coverage` is reported separately.",
    ])

    return "\n".join(lines)
