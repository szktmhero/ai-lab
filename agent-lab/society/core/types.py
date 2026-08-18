"""Core types for society simulation."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


# Dimensions
PREFERENCE_DIM = 8
BELIEF_DIM = 8
SIGNAL_DIM = 16


class ActionType(enum.IntEnum):
    PROPOSE = 0
    SUPPORT = 1
    OPPOSE = 2
    MODIFY = 3
    MERGE = 4
    SHARE_INFORMATION = 5
    REQUEST_INFORMATION = 6
    CONTACT = 7
    FOLLOW = 8
    UNFOLLOW = 9
    WAIT = 10


# Action costs
ACTION_COSTS = {
    ActionType.PROPOSE: 5,
    ActionType.SUPPORT: 1,
    ActionType.OPPOSE: 1,
    ActionType.MODIFY: 3,
    ActionType.MERGE: 4,
    ActionType.SHARE_INFORMATION: 3,
    ActionType.REQUEST_INFORMATION: 2,
    ActionType.CONTACT: 5,
    ActionType.FOLLOW: 2,
    ActionType.UNFOLLOW: 1,
    ActionType.WAIT: 0,
}

INITIAL_BUDGET = 100.0


@dataclass
class Action:
    agent_id: int
    action_type: ActionType
    round: int
    target_id: Optional[int] = None
    proposal_id: Optional[int] = None
    information_id: Optional[str] = None
    confidence: float = 0.5
    content: Optional[str] = None
    change_description: Optional[str] = None
    merge_target: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type.name,
            "round": self.round,
            "target_id": self.target_id,
            "proposal_id": self.proposal_id,
            "information_id": self.information_id,
            "confidence": self.confidence,
            "content": self.content,
            "change_description": self.change_description,
            "merge_target": self.merge_target,
        }


@dataclass
class Information:
    id: str
    content: str
    category: str
    accuracy: float  # 0.0 - 1.0
    source_agent: int
    round_created: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "accuracy": self.accuracy,
            "source_agent": self.source_agent,
            "round_created": self.round_created,
        }


@dataclass
class ProposalState:
    id: int
    content: str
    creator: int
    parent: Optional[int] = None
    merged_from: List[int] = field(default_factory=list)
    support_count: int = 0
    oppose_count: int = 0
    supporters: List[int] = field(default_factory=list)
    opposers: List[int] = field(default_factory=list)
    alive: bool = True
    created_round: int = 0
    support_history: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "creator": self.creator,
            "parent": self.parent,
            "merged_from": self.merged_from,
            "support_count": self.support_count,
            "oppose_count": self.oppose_count,
            "supporters": self.supporters,
            "opposers": self.opposers,
            "alive": self.alive,
            "created_round": self.created_round,
        }
