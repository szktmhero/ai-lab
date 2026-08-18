"""Observer: collects events and computes metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np


class Observer:
    """Observes the simulation without participating."""

    def __init__(self):
        self.event_log: List[dict] = []
        self.round_summaries: List[dict] = []
        self.task_summaries: List[dict] = []

    def record_event(self, event: dict):
        """Record a single event."""
        self.event_log.append(event)

    def record_round(self, society, round_num: int):
        """Record round summary."""
        agents = list(society.agents.values())
        proposals = list(society.proposals.values())
        alive_proposals = [p for p in proposals if p.alive]

        # Proposal stats
        total_support = sum(p.support_count for p in alive_proposals)
        total_oppose = sum(p.oppose_count for p in alive_proposals)

        # Trust stats
        all_trust = []
        for a in agents:
            all_trust.extend(a.trust.values())

        # Network stats
        degrees = [len(a.known_agents) for a in agents]

        summary = {
            "round": round_num,
            "alive_proposals": len(alive_proposals),
            "total_support": total_support,
            "total_oppose": total_oppose,
            "avg_trust": float(np.mean(all_trust)) if all_trust else 0.0,
            "trust_variance": float(np.var(all_trust)) if all_trust else 0.0,
            "avg_degree": float(np.mean(degrees)),
            "total_energy": sum(a.energy for a in agents),
        }
        self.round_summaries.append(summary)
        return summary

    def record_task_end(self, society, task_num: int, winner: Optional[int]):
        """Record task completion summary."""
        agents = list(society.agents.values())
        proposals = list(society.proposals.values())

        # Agent role metrics
        proposal_counts = Counter()
        support_counts = Counter()
        info_shares = Counter()

        for event in self.event_log:
            if event.get("round", -1) >= 0:  # Current task events
                if event.get("type") == "propose":
                    proposal_counts[event["agent"]] += 1
                elif event.get("type") == "support":
                    support_counts[event["agent"]] += 1

        # Centrality (degree)
        degrees = {a.id: len(a.known_agents) for a in agents}
        max_degree = max(degrees.values()) if degrees else 1

        summary = {
            "task": task_num,
            "winner": winner,
            "proposal_count": len(proposals),
            "events_this_task": len([e for e in self.event_log if e.get("task", task_num) == task_num]),
            "agents_proposed": len(proposal_counts),
            "agents_supported": len(support_counts),
            "max_degree": max_degree,
            "avg_degree": float(np.mean(list(degrees.values()))) if degrees else 0,
        }
        self.task_summaries.append(summary)
        return summary

    def save_events(self, path: Path):
        """Save all events to JSONL."""
        with open(path, 'w') as f:
            for event in self.event_log:
                f.write(json.dumps(event, default=str) + "\n")

    def save_rounds(self, path: Path):
        """Save round summaries."""
        path.write_text(json.dumps(self.round_summaries, indent=2))

    def save_tasks(self, path: Path):
        """Save task summaries."""
        path.write_text(json.dumps(self.task_summaries, indent=2))

    def generate_chronicle(self) -> str:
        """Generate human-readable chronicle."""
        lines = ["# Society Chronicle\n"]

        for ts in self.task_summaries:
            task_num = ts["task"]
            lines.append(f"\n## Task {task_num}\n")
            lines.append(f"Proposals created: {ts['proposal_count']}")
            lines.append(f"Agents who proposed: {ts['agents_proposed']}")
            lines.append(f"Agents who supported: {ts['agents_supported']}")
            if ts["winner"] is not None:
                lines.append(f"Winner: Proposal #{ts['winner']}")

        return "\n".join(lines)
