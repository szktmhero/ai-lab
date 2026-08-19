"""Tests for seed-blocked experiment analysis."""

from society.experiments.analysis import exact_sign_flip_pvalue, paired_contrasts


def _seed_entry(seed: int, value: float) -> dict:
    return {
        "seed": seed,
        "avg_decision_quality": value,
        "avg_consensus": value,
        "avg_information_spread": value,
        "avg_communication_cost": value,
        "round_limit_rate": value,
    }


def test_exact_sign_flip_uses_seed_blocks():
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == 0.25
    assert exact_sign_flip_pvalue([0.0, 0.0]) == 1.0


def test_paired_contrasts_matches_by_seed_not_list_order():
    seed_metrics = {
        "left": [_seed_entry(1, 3.0), _seed_entry(0, 2.0)],
        "right": [_seed_entry(0, 1.0), _seed_entry(1, 2.0)],
    }
    result = paired_contrasts(
        seed_metrics,
        [("left", "right", "test contrast")],
    )[0]

    assert result["matched_seeds"] == [0, 1]
    assert result["metrics"]["avg_decision_quality"]["mean_difference"] == 1.0
