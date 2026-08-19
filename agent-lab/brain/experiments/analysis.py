"""Seed-blocked paired analysis for the Brain Architecture pilot."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable
from statistics import fmean


def exact_sign_flip_pvalue(differences: Iterable[float]) -> float:
    differences = [float(value) for value in differences if value != 0.0]
    if not differences:
        return 1.0
    observed = abs(fmean(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(fmean(sign * value for sign, value in zip(signs, differences)))
        extreme += int(permuted >= observed - 1e-12)
        total += 1
    return extreme / total


def bootstrap_interval(
    differences: list[float],
    *,
    iterations: int = 5000,
    seed: int = 20260819,
) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    rng = random.Random(seed)
    samples = sorted(
        fmean(rng.choice(differences) for _ in differences)
        for _ in range(iterations)
    )
    low = samples[int(0.025 * (iterations - 1))]
    high = samples[int(0.975 * (iterations - 1))]
    return [low, high]


def paired_contrasts(
    seed_metrics: dict[str, list[dict]],
    contrasts: list[tuple[str, str, str]],
    metrics: list[str],
) -> list[dict]:
    analyses: list[dict] = []
    for left, right, description in contrasts:
        left_by_seed = {entry["seed"]: entry for entry in seed_metrics[left]}
        right_by_seed = {entry["seed"]: entry for entry in seed_metrics[right]}
        seeds = sorted(left_by_seed.keys() & right_by_seed.keys())
        metric_results = {}
        for metric in metrics:
            differences = [
                float(left_by_seed[seed][metric]) - float(right_by_seed[seed][metric])
                for seed in seeds
            ]
            metric_results[metric] = {
                "mean_difference": fmean(differences) if differences else 0.0,
                "bootstrap_95_ci": bootstrap_interval(differences),
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "paired_differences": differences,
            }
        analyses.append(
            {
                "left": left,
                "right": right,
                "description": description,
                "matched_seeds": seeds,
                "metrics": metric_results,
            }
        )
    return analyses
