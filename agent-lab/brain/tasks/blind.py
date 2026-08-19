"""Frozen holdout task generator for the real-model Phase 3 protocol.

No system or model receives the hidden answer or evaluator metadata.  The
generator uses a new namespace and structural distributions rather than the
deterministic pilot's tuned instances.  Formal execution verifies the complete
suite hash before making any inference call.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

from ..core.types import EvidenceEvent, Task, TaskType, TaskView

BLIND_SUITE_VERSION = "brain-blind-v1"
BLIND_SEEDS = tuple(range(1000, 1010))
BLIND_INSTANCES_PER_TYPE = 20
BLIND_CANDIDATE_COUNT = 4

# Filled from the canonical, evaluator-inclusive suite payload.  Any task
# generator or answer change must be treated as a protocol amendment.
EXPECTED_BLIND_SUITE_SHA256 = (
    "3ddb5460674ce6b85ce4bde0744a430f83f0b59e0abfc8d9c1b99b33d3d8fc99"
)


def _derived_seed(seed: int, task_type: TaskType, instance: int) -> int:
    digest = hashlib.sha256(
        f"{BLIND_SUITE_VERSION}:{seed}:{task_type.value}:{instance}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _opaque(prefix: str, *parts: object) -> str:
    payload = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(f"{BLIND_SUITE_VERSION}:{payload}".encode()).hexdigest()
    return f"{prefix}_{digest[:16]}"


def _different_candidate(rng: random.Random, candidate: int) -> int:
    return rng.choice(
        [index for index in range(BLIND_CANDIDATE_COUNT) if index != candidate]
    )


def _scores(
    rng: random.Random,
    candidate: int,
    strength: float,
    *,
    noise: float = 0.04,
) -> tuple[float, ...]:
    values = [rng.uniform(0.0, noise) for _ in range(BLIND_CANDIDATE_COUNT)]
    values[candidate] += strength
    return tuple(round(value, 6) for value in values)


def _event(
    task_id: str,
    step: int,
    cue: str,
    scores: tuple[float, ...],
    *,
    salience: float,
    channel: int,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=f"{task_id}:e{step:02d}",
        step=step,
        cue=cue,
        scores=scores,
        source=f"channel_{channel}",
        salience=salience,
    )


def generate_blind_task(seed: int, task_type: TaskType, instance: int) -> Task:
    if seed not in BLIND_SEEDS:
        raise ValueError("seed is outside the frozen blind split")
    if not 0 <= instance < BLIND_INSTANCES_PER_TYPE:
        raise ValueError("instance is outside the frozen blind split")

    rng = random.Random(_derived_seed(seed, task_type, instance))
    correct = rng.randrange(BLIND_CANDIDATE_COUNT)
    wrong = _different_candidate(rng, correct)
    task_id = _opaque("blind", seed, task_type.value, instance)
    cue = _opaque("cue", seed, task_type.value, instance)
    observations: list[EvidenceEvent] = []
    metadata: dict[str, int | str | float] = {}

    if task_type == TaskType.LATENT_RULE_SWITCH:
        old_count = rng.randint(3, 5)
        new_count = rng.randint(2, 4)
        step = 0
        for _ in range(old_count):
            observations.append(
                _event(
                    task_id,
                    step,
                    cue,
                    _scores(rng, wrong, rng.uniform(0.9, 1.2)),
                    salience=rng.uniform(0.55, 0.7),
                    channel=rng.randrange(6),
                )
            )
            step += 1
        switch_index = step
        for _ in range(new_count):
            observations.append(
                _event(
                    task_id,
                    step,
                    cue,
                    _scores(rng, correct, rng.uniform(1.1, 1.5)),
                    salience=rng.uniform(0.65, 0.8),
                    channel=rng.randrange(6),
                )
            )
            step += 1
        metadata = {
            "switch_index": switch_index,
            "old_answer": wrong,
            "old_observations": old_count,
            "new_observations": new_count,
        }

    elif task_type == TaskType.DELAYED_RECALL:
        observations.append(
            _event(
                task_id,
                0,
                cue,
                _scores(rng, correct, rng.uniform(1.8, 2.4)),
                salience=0.95,
                channel=rng.randrange(6),
            )
        )
        delay = rng.randint(7, 9)
        for step in range(1, delay + 1):
            distractor_cue = _opaque("noise", seed, instance, step)
            distractor = rng.randrange(BLIND_CANDIDATE_COUNT)
            observations.append(
                _event(
                    task_id,
                    step,
                    distractor_cue,
                    _scores(rng, distractor, rng.uniform(0.9, 1.4)),
                    salience=rng.uniform(0.2, 0.4),
                    channel=rng.randrange(6),
                )
            )
        metadata = {"delay": delay}

    elif task_type == TaskType.FAULT_RECOVERY:
        fault_step = rng.randint(1, 3)
        for step in range(5):
            is_fault = step == fault_step
            candidate = wrong if is_fault else correct
            strength = (
                rng.uniform(2.7, 3.4) if is_fault else rng.uniform(0.85, 1.15)
            )
            observations.append(
                _event(
                    task_id,
                    step,
                    cue,
                    _scores(rng, candidate, strength),
                    salience=rng.uniform(0.65, 0.8),
                    channel=step,
                )
            )
        metadata = {"fault_event_id": observations[fault_step].event_id}

    elif task_type == TaskType.PARTIAL_OBSERVATION:
        target_steps = (0, 3, 6, 8)
        wrong_step = rng.choice(target_steps[1:])
        for step in range(9):
            if step in target_steps:
                is_decoy = step == wrong_step
                candidate = wrong if is_decoy else correct
                strength = (
                    rng.uniform(1.2, 1.45)
                    if is_decoy
                    else rng.uniform(0.7, 0.9)
                )
                event_cue = cue
                salience = rng.uniform(0.6, 0.75)
            else:
                candidate = rng.randrange(BLIND_CANDIDATE_COUNT)
                strength = rng.uniform(0.8, 1.1)
                event_cue = _opaque("partial_noise", seed, instance, step)
                salience = rng.uniform(0.2, 0.4)
            observations.append(
                _event(
                    task_id,
                    step,
                    event_cue,
                    _scores(rng, candidate, strength),
                    salience=salience,
                    channel=rng.randrange(6),
                )
            )
        metadata = {"fragments": len(target_steps), "decoy_step": wrong_step}
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported task type: {task_type}")

    return Task(
        view=TaskView(
            task_id=task_id,
            task_type=task_type,
            observations=tuple(observations),
            query_cue=cue,
            candidate_count=BLIND_CANDIDATE_COUNT,
        ),
        correct_answer=correct,
        metadata=metadata,
    )


def generate_blind_task_suite(seed: int) -> list[Task]:
    tasks: list[Task] = []
    for instance in range(BLIND_INSTANCES_PER_TYPE):
        for task_type in TaskType:
            tasks.append(generate_blind_task(seed, task_type, instance))
    return tasks


def _task_payload(task: Task) -> dict[str, object]:
    return {
        "view": {
            "task_id": task.view.task_id,
            "task_type": task.view.task_type.value,
            "query_cue": task.view.query_cue,
            "candidate_count": task.view.candidate_count,
            "observations": [
                {
                    "event_id": event.event_id,
                    "step": event.step,
                    "cue": event.cue,
                    "scores": list(event.scores),
                    "source": event.source,
                    "salience": event.salience,
                }
                for event in task.view.observations
            ],
        },
        "correct_answer": task.correct_answer,
        "metadata": dict(sorted(task.metadata.items())),
    }


def blind_suite_sha256(seeds: Sequence[int] = BLIND_SEEDS) -> str:
    payload = {
        "suite_version": BLIND_SUITE_VERSION,
        "seeds": list(seeds),
        "instances_per_type": BLIND_INSTANCES_PER_TYPE,
        "tasks": [
            _task_payload(task)
            for seed in seeds
            for task in generate_blind_task_suite(seed)
        ],
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_blind_suite() -> str:
    actual = blind_suite_sha256()
    if actual != EXPECTED_BLIND_SUITE_SHA256:
        raise RuntimeError(
            "blind task suite differs from the frozen Phase 3 protocol: "
            f"{actual} != {EXPECTED_BLIND_SUITE_SHA256}"
        )
    return actual
