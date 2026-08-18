"""MetricsCollector: accumulates per-round and per-task metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


class MetricsCollector:
    """Collects detailed metrics during simulation."""

    def __init__(self):
        self.round_metrics: List[dict] = []
        self.task_metrics: List[dict] = []
        self.agent_snapshots: List[dict] = []

    def record_round(self, round_num: int, society_snapshot: dict, event_count: int):
        agents = society_snapshot.get("agents", {})
        proposals = society_snapshot.get("proposals", {})

        degrees = [len(a.get("known_agents", [])) for a in agents.values()]
        energies = [a.get("energy", 0) for a in agents.values()]
        all_trust = []
        for a in agents.values():
            all_trust.extend(a.get("trust", {}).values())

        alive_props = [p for p in proposals.values() if p.get("alive", True)]

        self.round_metrics.append({
            "round": round_num,
            "event_count": event_count,
            "alive_proposals": len(alive_props),
            "total_support": sum(p.get("support_count", 0) for p in alive_props),
            "total_oppose": sum(p.get("oppose_count", 0) for p in alive_props),
            "avg_degree": float(np.mean(degrees)) if degrees else 0,
            "avg_energy": float(np.mean(energies)) if energies else 0,
            "avg_trust": float(np.mean(all_trust)) if all_trust else 0,
            "trust_variance": float(np.var(all_trust)) if all_trust else 0,
        })

    def record_task(self, task_num: int, winner: Optional[int], society_snapshot: dict):
        agents = society_snapshot.get("agents", {})
        proposals = society_snapshot.get("proposals", {})

        # Role metrics
        proposal_creators = set()
        supporters = set()
        for p in proposals.values():
            proposal_creators.add(p.get("creator", -1))
            supporters.update(p.get("supporters", []))
        supporters -= proposal_creators

        degrees = {aid: len(a.get("known_agents", [])) for aid, a in agents.items()}
        max_degree = max(degrees.values()) if degrees else 0
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10] if degrees else []

        self.task_metrics.append({
            "task": task_num,
            "winner": winner,
            "total_proposals": len(proposals),
            "proposal_creators": len(proposal_creators),
            "exclusive_supporters": len(supporters - proposal_creators),
            "max_degree": max_degree,
            "top_nodes": top_nodes,
        })

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "round_metrics.json").write_text(
            json.dumps(self.round_metrics, indent=2))
        (output_dir / "task_metrics.json").write_text(
            json.dumps(self.task_metrics, indent=2))

    def summary(self) -> dict:
        if not self.round_metrics:
            return {}
        return {
            "total_rounds": len(self.round_metrics),
            "total_tasks": len(self.task_metrics),
            "avg_events_per_round": float(np.mean([r["event_count"] for r in self.round_metrics])),
            "final_avg_trust": self.round_metrics[-1]["avg_trust"],
            "final_avg_degree": self.round_metrics[-1]["avg_degree"],
        }
