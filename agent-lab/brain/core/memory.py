"""Episodic memory with provenance, bounded capacity, reset, and swap support."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .types import Signal


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    episode_id: str
    step: int
    cue: str
    scores: tuple[float, ...]
    source: str
    salience: float
    confidence: float
    provenance: tuple[str, ...]


class EpisodicMemory:
    def __init__(self, max_records: int = 256, write_threshold: float = 0.55):
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.max_records = max_records
        self.write_threshold = write_threshold
        self._records: list[MemoryRecord] = []
        self._record_ids: set[str] = set()
        self.write_count = 0
        self.retrieval_count = 0
        self.hit_count = 0

    def write(self, signal: Signal, *, force: bool = False) -> bool:
        if signal.kind != "evidence" or (
            signal.salience < self.write_threshold and not force
        ):
            return False
        record_id = f"memory:{signal.signal_id}"
        if record_id in self._record_ids:
            return False
        record = MemoryRecord(
            record_id=record_id,
            episode_id=signal.episode_id,
            step=signal.step,
            cue=signal.cue,
            scores=signal.scores,
            source=signal.source,
            salience=signal.salience,
            confidence=signal.confidence,
            provenance=signal.provenance,
        )
        self._records.append(record)
        self._record_ids.add(record_id)
        self.write_count += 1
        if len(self._records) > self.max_records:
            removed = self._records.pop(0)
            self._record_ids.remove(removed.record_id)
        return True

    def retrieve(self, cue: str, limit: int = 6) -> list[MemoryRecord]:
        self.retrieval_count += 1
        matches = [record for record in self._records if record.cue == cue]
        if matches:
            self.hit_count += 1
        # Newer evidence is favored.  The fixed retrieval capacity is part of
        # the declared architecture and is ablated by disabling memory.
        return matches[-limit:]

    def reset(self) -> None:
        self._records.clear()
        self._record_ids.clear()
        self.write_count = 0
        self.retrieval_count = 0
        self.hit_count = 0

    def export_records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records)

    def load_records(self, records: Iterable[MemoryRecord]) -> None:
        self.reset()
        for index, record in enumerate(records):
            copied = replace(record, record_id=f"loaded:{index}:{record.record_id}")
            self._records.append(copied)
            self._record_ids.add(copied.record_id)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
            self._record_ids = {record.record_id for record in self._records}

    @property
    def hit_rate(self) -> float:
        if self.retrieval_count == 0:
            return 0.0
        return self.hit_count / self.retrieval_count

    def as_signals(self, records: Iterable[MemoryRecord]) -> list[Signal]:
        return [
            Signal(
                signal_id=record.record_id,
                episode_id=record.episode_id,
                step=record.step,
                cue=record.cue,
                scores=record.scores,
                source="episodic_memory",
                kind="memory",
                salience=record.salience,
                confidence=record.confidence,
                provenance=record.provenance + (record.record_id,),
            )
            for record in records
        ]
