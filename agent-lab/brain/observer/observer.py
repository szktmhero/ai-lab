"""Append-only event log that is outside the decision path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BrainObserver:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, episode_id: str, step: int, event_type: str, **payload: Any) -> None:
        self.events.append(
            {
                "event_index": len(self.events),
                "episode_id": episode_id,
                "step": step,
                "event_type": event_type,
                **payload,
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def episode_events(self, episode_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["episode_id"] == episode_id]
