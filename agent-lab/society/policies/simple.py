"""Simple rule-based policy for agents."""

from __future__ import annotations

from typing import Optional
import numpy as np

from .base import AgentPolicy
from ..core.agent import Agent
from ..core.types import Action, ActionType, ACTION_COSTS


class SimplePolicy(AgentPolicy):
    """
    Simple rule-based policy:
    1. If no proposal exists and energy > 20: propose
    2. If supporting a proposal and energy > 10: share information
    3. If energy < 30: request information from trusted agents
    4. If high energy and known agents: follow trusted
    5. Otherwise: support best matching proposal
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def decide(self, agent: Agent, observation: dict) -> Optional[Action]:
        known = list(agent.known_agents)
        alive_proposals = [
            p for p in observation.get("proposals", {}).values()
            if p.get("alive", True)
        ]
        info_ids = observation.get("information_ids", [])
        current_round = observation.get("round", 0)

        action = Action(
            agent_id=agent.id,
            action_type=ActionType.WAIT,
            round=current_round,
        )

        # Rule 1: If no proposal exists and energy > 20: propose
        if not alive_proposals and agent.energy > 20 and not agent.has_proposed:
            action.action_type = ActionType.PROPOSE
            action.content = f"Proposal_{agent.id}_{current_round}"
            return action

        # Rule 2: If supporting a proposal and energy > 10: share info
        if agent.current_support is not None and agent.energy > 10:
            known_with_info = [
                aid for aid in known
                if aid in self.agents and self.agents[aid].known_information
            ] if hasattr(self, 'agents') else known

            if known and self.rng.random() < 0.3:
                action.action_type = ActionType.SHARE_INFORMATION
                action.target_id = int(self.rng.choice(known))
                if info_ids:
                    action.information_id = str(self.rng.choice(info_ids))
                return action

        # Rule 3: If energy < 30: request information
        if agent.energy < 30 and known:
            action.action_type = ActionType.REQUEST_INFORMATION
            action.target_id = int(self.rng.choice(known))
            return action

        # Rule 4: If high energy: follow trusted
        if agent.energy > 60 and known:
            if self.rng.random() < 0.2:
                action.action_type = ActionType.FOLLOW
                action.target_id = int(self.rng.choice(known))
                return action

        # Rule 5: Support best matching proposal
        if alive_proposals and agent.current_support is None:
            # Find proposal with most supporters (simple heuristic)
            best = max(alive_proposals, key=lambda p: p.get("support_count", 0))
            action.action_type = ActionType.SUPPORT
            action.proposal_id = best["id"]
            action.confidence = float(self.rng.uniform(0.5, 0.9))
            return action

        return action

    def set_agents(self, agents):
        """Store reference to agents for info lookup."""
        self.agents = agents

    def reset(self):
        pass
