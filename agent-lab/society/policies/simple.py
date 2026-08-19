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
    2. Explore or leave local links at low probability
    3. Revise a supported proposal when local evidence favors another option
    4. Share or request information within the social graph
    5. Support a materially better known proposal

    The support-count term is configurable so that information exchange can be
    compared with and without an explicit conformity incentive.
    """

    def __init__(
        self,
        seed: int = 42,
        social_evidence_weight: float = 0.05,
        support_switch_threshold: float = 0.1,
    ):
        if social_evidence_weight < 0:
            raise ValueError("social_evidence_weight must be non-negative")
        if support_switch_threshold < 0:
            raise ValueError("support_switch_threshold must be non-negative")
        self.rng = np.random.default_rng(seed)
        self.social_evidence_weight = social_evidence_weight
        self.support_switch_threshold = support_switch_threshold

    def decide(self, agent: Agent, observation: dict) -> Optional[Action]:
        known = list(agent.known_agents)
        proposal_map = observation.get("proposals", {})
        alive_proposals = list(proposal_map.values())
        info_ids = observation.get("information_ids", [])
        current_round = observation.get("round", 0)
        options = observation.get("options", [])
        option_scores = agent.preferences.copy()
        for info in observation.get("information", []):
            option_id = info.get("option_id")
            if option_id is not None and 0 <= option_id < len(option_scores):
                option_scores[option_id] += info.get("value", 0.0)

        action = Action(
            agent_id=agent.id,
            action_type=ActionType.WAIT,
            round=current_round,
        )

        if observation.get("network_mode") == "local":
            unknown = [
                aid for aid in observation.get("all_agent_ids", [])
                if aid != agent.id and aid not in agent.known_agents
            ]
            low_trust = [
                aid for aid in known if agent.trust.get(aid, 0.0) < 0.1
            ]
            if low_trust and self.rng.random() < 0.1:
                action.action_type = ActionType.UNFOLLOW
                action.target_id = int(self.rng.choice(low_trust))
                return action
            if unknown and self.rng.random() < 0.03:
                action.action_type = ActionType.CONTACT
                action.target_id = int(self.rng.choice(unknown))
                return action

        preferred_option = int(np.argmax(option_scores))
        current_proposal = proposal_map.get(agent.current_support)
        if (
            current_proposal is not None
            and current_proposal.get("option_id") != preferred_option
            and agent.energy >= ACTION_COSTS[ActionType.MODIFY]
            and self.rng.random() < 0.05
        ):
            action.action_type = ActionType.MODIFY
            action.proposal_id = current_proposal["id"]
            action.option_id = preferred_option
            action.change_description = (
                options[preferred_option] if options else f"option_{preferred_option}"
            )
            return action

        # Rule 1: If no proposal exists and energy > 20: propose
        if not alive_proposals and agent.energy > 20 and not agent.has_proposed:
            action.action_type = ActionType.PROPOSE
            action.option_id = int(np.argmax(option_scores))
            action.content = options[action.option_id] if options else f"option_{action.option_id}"
            return action

        # Rule 2: If supporting a proposal and energy > 10: share info
        if agent.current_support is not None and agent.energy > 10:
            if known and info_ids and self.rng.random() < 0.3:
                action.action_type = ActionType.SHARE_INFORMATION
                action.target_id = int(self.rng.choice(known))
                action.information_id = str(self.rng.choice(info_ids))
                return action

        # Rule 3: If energy < 30: request information
        if agent.energy < 30 and known:
            action.action_type = ActionType.REQUEST_INFORMATION
            action.target_id = int(self.rng.choice(known))
            return action

        # Rule 4: Support a materially better known proposal. This permits
        # support changes without prescribing consensus or a leader.
        if alive_proposals:
            def proposal_score(proposal):
                option_id = proposal.get("option_id")
                preference = option_scores[option_id] if option_id is not None else 0.0
                social_evidence = self.social_evidence_weight * (
                    proposal.get("support_count", 0) - proposal.get("oppose_count", 0)
                )
                return preference + social_evidence, -proposal["id"]

            # Proposals for the same option have the same evidence score for an
            # agent.  Retain only the strongest representative per option so a
            # 128-proposal round requires at most eight final comparisons.
            prepared = observation.get("proposal_representatives")
            if prepared is not None:
                representative_proposals = prepared[
                    "social" if self.social_evidence_weight > 0 else "evidence"
                ]
            else:
                representatives = {}
                for proposal in alive_proposals:
                    option_id = proposal.get("option_id")
                    existing = representatives.get(option_id)
                    social_rank = (
                        self.social_evidence_weight * (
                            proposal.get("support_count", 0)
                            - proposal.get("oppose_count", 0)
                        ),
                        -proposal["id"],
                    )
                    if existing is None or social_rank > existing[0]:
                        representatives[option_id] = (social_rank, proposal)
                representative_proposals = [
                    entry[1] for entry in representatives.values()
                ]

            best = max(representative_proposals, key=proposal_score)
            current = proposal_map.get(agent.current_support)
            current_score = proposal_score(current)[0] if current else -np.inf
            if (
                best["id"] != agent.current_support
                and proposal_score(best)[0]
                > current_score + self.support_switch_threshold
            ):
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
