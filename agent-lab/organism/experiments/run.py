"""Run a single organism experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from organism.core.world import World
from organism.core.cell import Cell
from organism.policies.random import RandomPolicy
from organism.policies.rule_based import RuleBasedPolicy
from organism.metrics.collector import MetricsCollector
from organism.visualization.renderer import render_html


def run_experiment(
    policy_name: str = "random",
    seed: int = 42,
    steps: int = 1000,
    cell_count: int = 128,
    world_w: int = 32,
    world_h: int = 32,
    snapshot_interval: int = 10,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run one experiment and save results."""

    # Create policy
    if policy_name == "random":
        policy = RandomPolicy(seed=seed)
    elif policy_name == "rule_based":
        policy = RuleBasedPolicy(seed=seed)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Create world
    world = World(
        width=world_w,
        height=world_h,
        cell_count=cell_count,
        seed=seed,
    )

    # Metrics
    collector = MetricsCollector()
    collector.collect(world)

    # Snapshots for visualization
    snapshots = [world.snapshot()]

    # Policy function wrapper
    def policy_fn(cell: Cell, obs: dict):
        return policy.decide(cell, obs)

    # Run simulation
    start_time = time.time()
    for step in range(steps):
        world.run_step(policy_fn)
        record = collector.collect(world)

        if world.step_count % snapshot_interval == 0:
            snapshots.append(world.snapshot())

    if snapshots[-1]["step"] != world.step_count:
        snapshots.append(world.snapshot())

    elapsed = time.time() - start_time

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Config
        config = {
            "policy": policy_name,
            "random_seed": seed,
            "steps": steps,
            "cell_count": cell_count,
            "world_size": [world_w, world_h],
            "snapshot_interval": snapshot_interval,
            "parameters": {
                "signal_dim": world.signal_dim,
                "state_dim": world.state_dim,
                "base_metabolism": 0.5,
                "move_cost": 0.3,
                "signal_cost": 0.2,
                "resource_gain_limit": 5.0,
            },
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Metrics CSV
        collector.save_csv(output_dir / "metrics.csv")

        # Metrics JSON
        collector.save_json(output_dir / "metrics.json")

        # Final state
        final_state = world.snapshot()
        (output_dir / "final_state.json").write_text(json.dumps(final_state, indent=2))

        with open(output_dir / "events.jsonl", "w") as f:
            for event in world.events:
                f.write(json.dumps(event) + "\n")

        # Visualization
        render_html(
            snapshots=snapshots,
            metrics=collector.records,
            resource_field=world.resource_field.tolist(),
            output_path=output_dir / "visualization.html",
            world_w=world_w,
            world_h=world_h,
        )

    # Summary
    summary = collector.get_summary()
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["final_alive"] = world.alive_count()
    summary["policy"] = policy_name
    summary["seed"] = seed

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run organism experiment")
    parser.add_argument("--policy", default="random", choices=["random", "rule_based"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--output", type=str, default="experiments/output")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output
    result = run_experiment(
        policy_name=args.policy,
        seed=args.seed,
        steps=args.steps,
        cell_count=args.cells,
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2))
