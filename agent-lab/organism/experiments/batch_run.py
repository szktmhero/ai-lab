"""Batch run all experiments: 10 seeds × 2 policies."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from organism.experiments.run import run_experiment
from organism.experiments.compare import compare_policies


def main():
    seeds = list(range(10))
    policies = ["random", "rule_based"]
    steps = 1000
    output_base = Path(__file__).parent / "results"

    all_results = []

    for policy in policies:
        for seed in seeds:
            exp_name = f"{policy}_seed{seed:02d}"
            output_dir = output_base / exp_name
            print(f"Running {exp_name}...")

            result = run_experiment(
                policy_name=policy,
                seed=seed,
                steps=steps,
                cell_count=128,
                world_w=32,
                world_h=32,
                snapshot_interval=10,
                output_dir=output_dir,
            )
            result["exp_name"] = exp_name
            all_results.append(result)
            print(f"  Done: alive={result['final_alive']}, elapsed={result['elapsed_seconds']}s")

    # Save summary
    summary_path = output_base / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    (output_base / "COMPARISON.md").write_text(compare_policies(output_base))
    print(f"\nAll experiments complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
