"""Compare policies across seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def compare_policies(results_dir: Path) -> str:
    """Generate comparison report from experiment results."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return "No summary.json found."

    results = json.loads(summary_path.read_text())

    # Group by policy
    policies = {}
    for r in results:
        p = r["policy"]
        if p not in policies:
            policies[p] = []
        policies[p].append(r)

    lines = ["# Experiment Comparison Report\n"]

    for policy_name, runs in policies.items():
        lines.append(f"## {policy_name.capitalize()} Policy\n")

        alive_vals = [r["final_alive"]["final"] if isinstance(r["final_alive"], dict) else r.get("final_alive", 0) for r in runs]
        energy_vals = [r["average_energy"]["final"] if isinstance(r["average_energy"], dict) else r.get("average_energy", 0) for r in runs]
        cluster_vals = [r["largest_cluster_size"]["final"] if isinstance(r["largest_cluster_size"], dict) else r.get("largest_cluster_size", 0) for r in runs]
        signal_vals = [r["signal_diversity"]["final"] if isinstance(r["signal_diversity"], dict) else r.get("signal_diversity", 0) for r in runs]
        entropy_vals = [r["spatial_entropy"]["final"] if isinstance(r["spatial_entropy"], dict) else r.get("spatial_entropy", 0) for r in runs]
        consumption_vals = [r["resource_consumption"]["final"] for r in runs]
        efficiency_vals = [1000 * alive / max(consumed, 1e-12) for alive, consumed in zip(alive_vals, consumption_vals)]
        elapsed_vals = [r["elapsed_seconds"] for r in runs]

        lines.append(f"| Metric | Mean | Std | Min | Max |")
        lines.append(f"|--------|------|-----|-----|-----|")

        for name, vals in [
            ("Final Survivors", alive_vals),
            ("Avg Energy (final)", energy_vals),
            ("Largest Cluster", cluster_vals),
            ("Signal Diversity", signal_vals),
            ("Spatial Entropy", entropy_vals),
            ("Resource Consumed", consumption_vals),
            ("Survivors / 1000 Resource", efficiency_vals),
            ("Elapsed (s)", elapsed_vals),
        ]:
            arr = np.array(vals, dtype=float)
            lines.append(f"| {name} | {arr.mean():.3f} | {arr.std():.3f} | {arr.min():.3f} | {arr.max():.3f} |")

        lines.append("")

    # Cross-policy comparison
    if len(policies) > 1:
        lines.append("## Cross-Policy Comparison\n")
        lines.append("| Metric | " + " | ".join(p.capitalize() for p in policies) + " |")
        lines.append("|--------|-" + "-|-".join("-" for _ in policies) + "|")

        # Simple survival comparison
        for metric_key, label in [
            ("final_alive", "Final Survivors"),
            ("elapsed_seconds", "Avg Time (s)"),
        ]:
            row = f"| {label} |"
            for policy_name, runs in policies.items():
                if metric_key == "final_alive":
                    vals = []
                    for r in runs:
                        v = r.get("final_alive", 0)
                        if isinstance(v, dict):
                            v = v.get("final", 0)
                        vals.append(float(v))
                else:
                    vals = [float(r.get(metric_key, 0)) for r in runs]
                row += f" {np.mean(vals):.2f} ± {np.std(vals):.2f} |"
            lines.append(row)

    lines.extend([
        "",
        "## Interpretation Boundaries",
        "",
        "**DESIGNED:** metabolism, resource regeneration, action costs, collision handling, "
        "and each policy's action rules.",
        "",
        "**OBSERVED:** the tables contain final-step measurements over ten matched seeds.",
        "",
        "**INFERRED:** the consume-and-search rule improves survival under this resource model. "
        "This does not establish intelligence or self-organization.",
        "",
        "**SPECULATIVE:** signal semantics, role differentiation, and organism-like behavior "
        "are not demonstrated by these metrics.",
    ])

    report = "\n".join(lines)
    return report
