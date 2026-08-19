"""History-dependence smoke test; explicitly not a personality claim."""

from __future__ import annotations

from collections import Counter

from ..core.memory import EpisodicMemory, MemoryRecord
from ..core.system import BRAIN_CONDITIONS, BrainSystem
from ..tasks.generator import generate_history_episode, generate_history_probe


def _train(history_id: str, preferred_candidate: int) -> tuple[MemoryRecord, ...]:
    system = BrainSystem(BRAIN_CONDITIONS["brain_full"])
    for index in range(8):
        system.run(generate_history_episode(history_id, preferred_candidate, index).view)
    return system.memory.export_records()


def _probe(records: tuple[MemoryRecord, ...], repeats: int = 3) -> list[int]:
    choices: list[int] = []
    for repeat in range(repeats):
        memory = EpisodicMemory()
        memory.load_records(records)
        system = BrainSystem(BRAIN_CONDITIONS["brain_full"], memory=memory)
        choices.append(system.run(generate_history_probe(str(repeat)).view).candidate)
    return choices


def run_individuality_smoke_test() -> dict:
    history_a = _train("a", 0)
    history_b = _train("b", 1)
    choices_a = _probe(history_a)
    choices_b = _probe(history_b)
    reset_a = _probe(())
    reset_b = _probe(())
    swapped_a = _probe(history_b)
    swapped_b = _probe(history_a)

    modal_a = Counter(choices_a).most_common(1)[0][0]
    modal_b = Counter(choices_b).most_common(1)[0][0]
    return {
        "status": "causal-path smoke test only; not evidence of personality",
        "history_a_choices": choices_a,
        "history_b_choices": choices_b,
        "within_history_consistency": {
            "a": choices_a.count(modal_a) / len(choices_a),
            "b": choices_b.count(modal_b) / len(choices_b),
        },
        "between_history_divergence": int(modal_a != modal_b),
        "after_memory_reset": {"a": reset_a, "b": reset_b},
        "reset_divergence": int(reset_a[0] != reset_b[0]),
        "after_memory_swap": {"a": swapped_a, "b": swapped_b},
        "swap_followed_memory": int(swapped_a[0] == modal_b and swapped_b[0] == modal_a),
        "interpretation": (
            "The harness can create and causally remove history-dependent choices. "
            "Because the history cue and deterministic memory rule were designed, "
            "this validates the manipulation rather than demonstrating emergent individuality."
        ),
    }
