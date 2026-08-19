"""Seed-blocked analysis for matched society experiment conditions."""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ANALYSIS_METRICS = {
    "avg_decision_quality": "Decision quality",
    "avg_consensus": "Consensus",
    "avg_information_spread": "Information spread",
    "avg_communication_cost": "Communication cost",
    "round_limit_rate": "Round-limit rate",
}


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Two-sided paired randomization p-value using seed blocks.

    With ten or fewer seeds this enumerates every possible sign assignment.
    For larger experiments it uses a deterministic Monte Carlo approximation.
    """
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    if np.allclose(values, 0.0):
        return 1.0

    observed = abs(float(np.mean(values)))
    tolerance = 1e-12
    if values.size <= 20:
        permuted = (
            abs(float(np.mean(values * np.asarray(signs, dtype=float))))
            for signs in product((-1.0, 1.0), repeat=values.size)
        )
        total = 2 ** values.size
        extreme = sum(value >= observed - tolerance for value in permuted)
        return extreme / total

    rng = np.random.default_rng(0)
    sample_count = 100_000
    signs = rng.choice((-1.0, 1.0), size=(sample_count, values.size))
    permuted = np.abs(np.mean(signs * values, axis=1))
    return float((np.count_nonzero(permuted >= observed - tolerance) + 1) /
                 (sample_count + 1))


def paired_bootstrap_interval(
    differences: Sequence[float],
    confidence: float = 0.95,
    sample_count: int = 20_000,
) -> Tuple[float, float]:
    """Percentile interval for the mean paired seed-level difference."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(0)
    indices = rng.integers(0, values.size, size=(sample_count, values.size))
    means = np.mean(values[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_contrasts(
    seed_metrics: Dict[str, List[dict]],
    contrasts: Iterable[Tuple[str, str, str]],
) -> List[dict]:
    """Compare condition means after matching observations by random seed."""
    analyses = []
    for left, right, question in contrasts:
        left_by_seed = {entry["seed"]: entry for entry in seed_metrics[left]}
        right_by_seed = {entry["seed"]: entry for entry in seed_metrics[right]}
        matched_seeds = sorted(set(left_by_seed) & set(right_by_seed))
        metric_results = {}

        for metric in ANALYSIS_METRICS:
            left_values = np.asarray(
                [left_by_seed[seed][metric] for seed in matched_seeds], dtype=float
            )
            right_values = np.asarray(
                [right_by_seed[seed][metric] for seed in matched_seeds], dtype=float
            )
            differences = left_values - right_values
            interval = paired_bootstrap_interval(differences)
            standard_deviation = float(np.std(differences, ddof=1)) \
                if len(differences) > 1 else 0.0
            standardized = (
                float(np.mean(differences) / standard_deviation)
                if standard_deviation > 1e-12 else 0.0
            )
            metric_results[metric] = {
                "left_mean": float(np.mean(left_values)),
                "right_mean": float(np.mean(right_values)),
                "mean_difference": float(np.mean(differences)),
                "bootstrap_95_ci": list(interval),
                "paired_effect_dz": standardized,
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
            }

        analyses.append({
            "left": left,
            "right": right,
            "difference_definition": "left - right",
            "question": question,
            "matched_seeds": matched_seeds,
            "metrics": metric_results,
        })

    return analyses
