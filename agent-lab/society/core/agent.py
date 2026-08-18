"""Agent dataclass for society simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
import numpy as np

from .types import PREFERENCE_DIM, BELIEF_DIM, INITIAL_BUDGET


@dataclass
class Agent:
    id: int

    # Internal state
    preferences: np.ndarray = field(default_factory=lambda: np.random.randn(PREFERENCE_DIM).astype(np.float32) * 0.3)
    beliefs: np.ndarray = field(default_factory=lambda: np.random.randn(BELIEF_DIM).astype(np.float32) * 0.3)
    confidence: float = 0.5
    risk_tolerance: float = 0.5
    novelty_preference: float = 0.5
    communication_tendency: float = 0.5

    # Social state
    trust: Dict[int, float] = field(default_factory=dict)
    known_agents: Set[int] = field(default_factory=set)

    # Task state
    energy: float = INITIAL_BUDGET
    current_support: Optional[int] = None
    current_proposal: Optional[int] = None
    has_proposed: bool = False

    # History (persists across tasks)
    history: List[dict] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)

    # Task-specific info
    known_information: List[str] = field(default_factory=list)

    def initialize_social(self, all_agent_ids: List[int], rng: np.random.Generator, avg_degree: int = 6):
        """Initialize random social connections."""
        n_agents = len(all_agent_ids)
        n_connections = min(avg_degree, n_agents - 1)

        # Random connections
        others = [aid for aid in all_agent_ids if aid != self.id]
        connections = rng.choice(others, size=min(n_connections, len(others)), replace=False)
        for aid in connections:
            self.known_agents.add(int(aid))
            self.trust[int(aid)] = float(rng.uniform(0.3, 0.7))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "preferences": self.preferences.tolist(),
            "beliefs": self.beliefs.tolist(),
            "confidence": self.confidence,
            "risk_tolerance": self.risk_tolerance,
            "novelty_preference": self.novelty_preference,
            "communication_tendency": self.communication_tendency,
            "trust": self.trust,
            "known_agents": list(self.known_agents),
            "energy": self.energy,
            "current_support": self.current_support,
            "current_proposal": self.current_proposal,
            "has_proposed": self.has_proposed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Agent:
        agent = cls(id=d["id"])
        agent.preferences = np.array(d["preferences"], dtype=np.float32)
        agent.beliefs = np.array(d["beliefs"], dtype=np.float32)
        agent.confidence = d["confidence"]
        agent.risk_tolerance = d["risk_tolerance"]
        agent.novelty_preference = d["novelty_preference"]
        agent.communication_tendency = d["communication_tendency"]
        agent.trust = d["trust"]
        agent.known_agents = set(d["known_agents"])
        agent.energy = d["energy"]
        agent.current_support = d.get("current_support")
        agent.current_proposal = d.get("current_proposal")
        agent.has_proposed = d.get("has_proposed", False)
        return agent
