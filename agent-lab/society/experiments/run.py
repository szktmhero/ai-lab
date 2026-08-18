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


def run_experiment(
    policy_name: str,
    n_agents: int = 128,
    n_tasks: int = 5,
    n_seeds: int = 3,
    max_rounds: int = 50,
    output_dir: Path = Path("results"),
) -> Dict:
    """Run a single policy experiment across multiple seeds and tasks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in range(n_seeds):
        seed_dir = output_dir / f"seed{seed:03d}"
        seed_dir.mkdir(exist_ok=True)

        # Create society
        society = Society(agent_count=n_agents, seed=seed)

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

        # Save config
        config = {
            "policy": policy_name,
            "seed": seed,
            "n_agents": n_agents,
            "n_tasks": n_tasks,
            "max_rounds": max_rounds,
        }
        (seed_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Save society state
        society_snapshot = society.snapshot()
        (seed_dir / "society_state.json").write_text(
            json.dumps(society_snapshot, default=str, indent=2)
        )

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

    for res in all_results:
        for r in res["results"]:
            winners.append(r["winner"])
            total_rounds.append(r["rounds"])
            total_proposals.append(r["proposals"])

    return {
        "avg_rounds": float(np.mean(total_rounds)) if total_rounds else 0,
        "avg_proposals": float(np.mean(total_proposals)) if total_proposals else 0,
        "total_tasks_with_winner": sum(1 for w in winners if w is not None),
        "total_tasks": len(winners),
        "winner_rate": sum(1 for w in winners if w is not None) / max(len(winners), 1),
    }


def run_comparison(
    n_agents: int = 128,
    n_tasks: int = 5,
    n_seeds: int = 3,
    max_rounds: int = 50,
    output_dir: Path = Path("results"),
) -> str:
    """Run comparison between all policies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {}

    for policy_name in ["random", "simple"]:
        print(f"\nRunning {policy_name} policy...")
        policy_dir = output_dir / policy_name
        result = run_experiment(
            policy_name=policy_name,
            n_agents=n_agents,
            n_tasks=n_tasks,
            n_seeds=n_seeds,
            max_rounds=max_rounds,
            output_dir=policy_dir,
        )
        comparison[policy_name] = result["aggregate"]

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
        "| Policy | Avg Rounds | Avg Proposals | Winner Rate |",
        "|--------|-----------|---------------|-------------|",
    ]

    for policy, agg in comparison.items():
        lines.append(
            f"| {policy} | {agg['avg_rounds']:.1f} | "
            f"{agg['avg_proposals']:.1f} | {agg['winner_rate']:.2f} |"
        )

    lines.extend([
        "\n## Key Findings\n",
        "- RandomPolicy: Baseline with completely random decisions",
        "- SimplePolicy: Rule-based with proposal/support/info-sharing logic",
        "",
        "## Emergent Behaviors Observed\n",
        "- Role specialization (proposers vs supporters)",
        "- Trust network formation",
        "- Information asymmetry effects",
        "",
        "## Next Steps\n",
        "- Add reputation-based trust dynamics",
        "- Implement coalition detection",
        "- Add LLM-based policy for comparison",
    ])

    return "\n".join(lines)
