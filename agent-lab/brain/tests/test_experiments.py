from __future__ import annotations

import json

from brain.experiments.analysis import exact_sign_flip_pvalue, paired_contrasts
from brain.experiments.individuality import run_individuality_smoke_test
from brain.experiments.run import CONDITIONS, run_pilot


def test_exact_sign_flip_uses_seed_blocks():
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == 0.25
    assert exact_sign_flip_pvalue([0.0, 0.0]) == 1.0


def test_paired_analysis_matches_seed_not_input_order():
    seed_metrics = {
        "left": [
            {"seed": 1, "accuracy": 3.0},
            {"seed": 0, "accuracy": 2.0},
        ],
        "right": [
            {"seed": 0, "accuracy": 1.0},
            {"seed": 1, "accuracy": 2.0},
        ],
    }
    result = paired_contrasts(
        seed_metrics,
        [("left", "right", "test")],
        ["accuracy"],
    )[0]
    assert result["matched_seeds"] == [0, 1]
    assert result["metrics"]["accuracy"]["mean_difference"] == 1.0


def test_history_smoke_test_follows_reset_and_swap():
    result = run_individuality_smoke_test()
    assert result["between_history_divergence"] == 1
    assert result["reset_divergence"] == 0
    assert result["swap_followed_memory"] == 1
    assert "not evidence of personality" in result["status"]


def test_small_pilot_writes_complete_aggregate_artifacts(tmp_path):
    report = run_pilot(
        n_seeds=1,
        n_per_type=1,
        output_dir=tmp_path,
        save_event_logs=True,
    )
    expected = {
        "design.json",
        "comparison.json",
        "seed_metrics.json",
        "paired_statistics.json",
        "individuality.json",
        "COMPARISON.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert set(comparison) == set(CONDITIONS)
    assert all(metrics["tasks"] == 4 for metrics in comparison.values())
    assert "not a formal LLM result" in (tmp_path / "design.json").read_text()
    assert "not evidence that an LLM brain" in report.replace("**", "")
    task_metrics = json.loads(
        (tmp_path / "pilot" / "task_metrics.json").read_text()
    )
    assert len(task_metrics) == len(CONDITIONS) * 4

    event_path = (
        tmp_path / "pilot" / "brain_full" / "seed000" / "events.jsonl"
    )
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert events
    assert events[-1]["event_type"] == "final_action"
