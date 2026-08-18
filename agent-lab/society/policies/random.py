"""Random policy: completely random actions."""

from __future__ import annotations

from typing import Optional
import numpy as np

from .base import AgentPolicy
from ..core.agent import Agent
from ..core.types import Action, ActionType, ACTION_COSTS


class RandomPolicy(AgentPolicy):
    """Baseline: random actions weighted by available budget."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def decide(self, agent: Agent, observation: dict) -> Optional[Action]:
        # Filter affordable actions
        affordable = [
            at for at, cost in ACTION_COSTS.items()
            if agent.energy >= cost and at != ActionType.WAIT
        ]

        if not affordable:
            return None  # WAIT

        # Weight by inverse cost (cheaper actions more likely)
        weights = np.array([1.0 / max(ACTION_COSTS[at], 1) for at in affordable])
        weights = weights / weights.sum()

        action_type = ActionType(int(self.rng.choice(affordable, p=weights)))

        # Build action
        known = list(agent.known_agents)
        alive_proposals = [
            p for p in observation.get("proposals", {}).values()
            if p.get("alive", True)
        ]

        action = Action(
            agent_id=agent.id,
            action_type=action_type,
            round=observation.get("round", 0),
        )

        # Fill required fields based on action type
        if action_type in (ActionType.SUPPORT, ActionType.OPPOSE, ActionType.MODIFY, ActionType.MERGE):
            if alive_proposals:
                p = self.rng.choice(alive_proposals)
                action.proposal_id = p["id"]
                action.confidence = float(self.rng.uniform(0.3, 0.9))

        if action_type in (ActionType.SHARE_INFORMATION, ActionType.REQUEST_INFORMATION, ActionType.CONTACT, ActionType.FOLLOW):
            if known:
                action.target_id = int(self.rng.choice(known))

        if action_type == ActionType.PROPOSE:
            action.content = f"Proposal_{agent.id}_{observation.get('round', 0)}"

        if action_type == ActionType.MODIFY:
            action.change_description = f"Modified by agent {agent.id}"

        if action_type == ActionType.MERGE and alive_proposals and len(alive_proposals) >= 2:
            pair = self.rng.choice(alive_proposals, size=2, replace=False)
            action.proposal_id = int(pair[0]["id"])
            action.merge_target = int(pair[1]["id"])

        return action

    def reset(self):
        pass
