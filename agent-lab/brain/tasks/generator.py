"""Matched, procedurally generated tasks with evaluator-only answers."""

from __future__ import annotations

import hashlib
import random

from ..core.types import EvidenceEvent, Task, TaskType, TaskView

CANDIDATE_COUNT = 4


def _derived_seed(seed: int, task_type: TaskType, instance: int) -> int:
    digest = hashlib.sha256(
        f"brain-task-v1:{seed}:{task_type.value}:{instance}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _different_candidate(rng: random.Random, candidate: int) -> int:
    choices = [index for index in range(CANDIDATE_COUNT) if index != candidate]
    return rng.choice(choices)


def _scores(
    rng: random.Random,
    candidate: int,
    strength: float,
    *,
    noise: float = 0.025,
) -> tuple[float, ...]:
    values = [rng.uniform(0.0, noise) for _ in range(CANDIDATE_COUNT)]
    values[candidate] += strength
    return tuple(round(value, 6) for value in values)


def _event(
    task_id: str,
    step: int,
    cue: str,
    scores: tuple[float, ...],
    *,
    salience: float,
    source: str | None = None,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=f"{task_id}:event:{step}",
        step=step,
        cue=cue,
        scores=scores,
        source=source or f"sensor_{step}",
        salience=salience,
    )


def generate_task(seed: int, task_type: TaskType, instance: int) -> Task:
    """Generate a task independently of any architecture condition."""

    rng = random.Random(_derived_seed(seed, task_type, instance))
    correct = rng.randrange(CANDIDATE_COUNT)
    wrong = _different_candidate(rng, correct)
    task_id = f"seed{seed:03d}:{task_type.value}:{instance:03d}"
    cue = f"target:{seed}:{task_type.value}:{instance}"
    observations: list[EvidenceEvent] = []
    metadata: dict[str, int | str] = {}

    if task_type == TaskType.LATENT_RULE_SWITCH:
        old_answer = wrong
        for step in range(4):
            observations.append(
                _event(
                    task_id,
                    step,
                    cue,
                    _scores(rng, old_answer, 1.05),
                    salience=0.65,
                    source=f"recent_example_{step}",
                )
            )
        for step in range(4, 7):
            observations.append(
                _event(
                    task_id,
                    step,
                    cue,
                    _scores(rng, correct, 1.35),
                    salience=0.7,
                    source=f"recent_example_{step}",
                )
            )
        metadata = {"switch_index": 4, "old_answer": old_answer}

    elif task_type == TaskType.DELAYED_RECALL:
        observations.append(
            _event(
                task_id,
                0,
                cue,
                _scores(rng, correct, 2.5),
                salience=0.95,
                source="initial_instruction",
            )
        )
        for step in range(1, 7):
            distractor = rng.randrange(CANDIDATE_COUNT)
            observations.append(
                _event(
                    task_id,
                    step,
                    f"distractor:{seed}:{instance}:{step}",
                    _scores(rng, distractor, 1.4),
                    salience=0.35,
                    source=f"distractor_{step}",
                )
            )
        metadata = {"delay": 6}

    elif task_type == TaskType.FAULT_RECOVERY:
        fault_step = 2
        for step in range(4):
            candidate = wrong if step == fault_step else correct
            strength = 3.4 if step == fault_step else 1.05
            observations.append(
                _event(
                    task_id,
                    step,
                    cue,
                    _scores(rng, candidate, strength),
                    salience=0.75,
                    source=f"redundant_sensor_{step}",
                )
            )
        # ``fault_event_id`` remains evaluator-only in Task.metadata.
        metadata = {"fault_event_id": observations[fault_step].event_id}

    elif task_type == TaskType.PARTIAL_OBSERVATION:
        target_events = {
            0: (correct, 0.85),
            2: (correct, 0.85),
            4: (wrong, 1.4),
            6: (correct, 0.85),
        }
        for step in range(7):
            if step in target_events:
                candidate, strength = target_events[step]
                observations.append(
                    _event(
                        task_id,
                        step,
                        cue,
                        _scores(rng, candidate, strength),
                        salience=0.7,
                        source=f"partial_sensor_{step}",
                    )
                )
            else:
                distractor = rng.randrange(CANDIDATE_COUNT)
                observations.append(
                    _event(
                        task_id,
                        step,
                        f"irrelevant:{seed}:{instance}:{step}",
                        _scores(rng, distractor, 1.0),
                        salience=0.3,
                        source=f"irrelevant_sensor_{step}",
                    )
                )
        metadata = {"fragments": 4}
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported task type: {task_type}")

    view = TaskView(
        task_id=task_id,
        task_type=task_type,
        observations=tuple(observations),
        query_cue=cue,
        candidate_count=CANDIDATE_COUNT,
    )
    return Task(view=view, correct_answer=correct, metadata=metadata)


def generate_task_suite(seed: int, n_per_type: int = 20) -> list[Task]:
    if n_per_type <= 0:
        raise ValueError("n_per_type must be positive")
    tasks: list[Task] = []
    for instance in range(n_per_type):
        for task_type in TaskType:
            tasks.append(generate_task(seed, task_type, instance))
    return tasks


def generate_history_episode(
    history_id: str,
    preferred_candidate: int,
    index: int,
    *,
    cue: str = "shared_identity_probe",
) -> Task:
    """One experience for the history-dependence causal-path smoke test."""

    if not 0 <= preferred_candidate < CANDIDATE_COUNT:
        raise ValueError("preferred_candidate is outside the candidate range")
    task_id = f"history:{history_id}:{index}"
    rng = random.Random(_derived_seed(index, TaskType.DELAYED_RECALL, preferred_candidate))
    observation = _event(
        task_id,
        0,
        cue,
        _scores(rng, preferred_candidate, 1.0, noise=0.0),
        salience=0.8,
        source=f"history_{history_id}",
    )
    return Task(
        view=TaskView(
            task_id=task_id,
            task_type=TaskType.DELAYED_RECALL,
            observations=(observation,),
            query_cue=cue,
            candidate_count=CANDIDATE_COUNT,
        ),
        correct_answer=preferred_candidate,
        metadata={"history_group": history_id},
    )


def generate_history_probe(
    probe_id: str,
    *,
    cue: str = "shared_identity_probe",
) -> Task:
    """Ambiguous common probe; its nominal answer is not used as evidence."""

    task_id = f"history_probe:{probe_id}"
    rng = random.Random(_derived_seed(999, TaskType.PARTIAL_OBSERVATION, 0))
    observations = (
        _event(task_id, 0, cue, _scores(rng, 0, 0.8, noise=0.0), salience=0.7),
        _event(task_id, 1, cue, _scores(rng, 1, 0.8, noise=0.0), salience=0.7),
    )
    return Task(
        view=TaskView(
            task_id=task_id,
            task_type=TaskType.PARTIAL_OBSERVATION,
            observations=observations,
            query_cue=cue,
            candidate_count=CANDIDATE_COUNT,
        ),
        correct_answer=0,
        metadata={"ambiguous_probe": 1},
    )
