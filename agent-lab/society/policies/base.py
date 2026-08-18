"""Abstract base class for agent decision policies."""

from __future__ import annotations

import abc
from typing import Tuple, Optional

from ..core.agent import Agent
from ..core.types import Action, ActionType


class AgentPolicy(abc.ABC):
    """Abstract base class for agent decision-making policies."""

    @abc.abstractmethod
    def decide(self, agent: Agent, observation: dict) -> Optional[Action]:
        """
        Given an agent and its observation, return an Action or None (WAIT).

        Args:
            agent: The agent making the decision
            observation: Dict with 'known_agents', 'proposals', 'information', 'round', 'budget_remaining'

        Returns:
            Action or None
        """
        ...

    def reset(self):
        """Reset any internal state between tasks."""
        pass
