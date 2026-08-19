"""Society world state and simulation engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

from .types import (
    Action, ActionType, ACTION_COSTS, ProposalState, Information,
    INITIAL_BUDGET, PREFERENCE_DIM, BELIEF_DIM,
)
from .agent import Agent


class Society:
    """Main simulation world for the society experiment."""

    def __init__(
        self,
        agent_count: int = 128,
        seed: int = 42,
        avg_degree: int = 6,
        budget: float = INITIAL_BUDGET,
        network_mode: str = "local",
        action_costs: Optional[Dict[ActionType, float]] = None,
    ):
        self.agent_count = agent_count
        self.seed = seed
        self.avg_degree = avg_degree
        self.budget = budget
        if network_mode not in {"independent", "full", "local"}:
            raise ValueError(f"Unknown network mode: {network_mode}")
        self.network_mode = network_mode
        self.action_costs = dict(ACTION_COSTS if action_costs is None else action_costs)

        # Keep condition-specific randomness (the network) isolated from the
        # agent population and task information.  Without independent streams,
        # using the same seed under ``independent``, ``full``, and ``local``
        # produced different preferences and evidence assignments because each
        # topology consumed a different number of random draws during setup.
        seed_sequence = np.random.SeedSequence(seed)
        agent_seed, network_seed, information_seed, dynamics_seed = seed_sequence.spawn(4)
        self.agent_rng = np.random.default_rng(agent_seed)
        self.network_rng = np.random.default_rng(network_seed)
        self.information_rng = np.random.default_rng(information_seed)
        self.rng = np.random.default_rng(dynamics_seed)
        self.round_count = 0
        self.task_count = 0

        # Agents
        self.agents: Dict[int, Agent] = {}
        self._init_agents()

        # Proposals
        self.proposals: Dict[int, ProposalState] = {}
        self.next_proposal_id = 0

        # Information pool (per-task)
        self.information_pool: Dict[str, Information] = {}
        self.next_info_id = 0

        # Event log
        self.events: List[dict] = []

        # Current task
        self.current_task: Optional[dict] = None

    def _init_agents(self):
        """Create and initialize agents."""
        for i in range(self.agent_count):
            agent = Agent(
                id=i,
                preferences=(self.agent_rng.standard_normal(PREFERENCE_DIM) * 0.3).astype(np.float32),
                beliefs=(self.agent_rng.standard_normal(BELIEF_DIM) * 0.3).astype(np.float32),
                confidence=float(self.agent_rng.uniform(0.2, 0.8)),
                risk_tolerance=float(self.agent_rng.random()),
                novelty_preference=float(self.agent_rng.random()),
                communication_tendency=float(self.agent_rng.random()),
            )
            if self.network_mode == "local":
                agent.initialize_social(
                    list(range(self.agent_count)), self.network_rng, self.avg_degree
                )
            elif self.network_mode == "full":
                agent.known_agents = set(range(self.agent_count)) - {i}
                agent.trust = {
                    aid: float(self.network_rng.uniform(0.1, 0.4))
                    for aid in agent.known_agents
                }
            # Random initial energy variance
            agent.energy = self.budget
            self.agents[i] = agent

    def get_known_agents(self, agent: Agent) -> List[Agent]:
        """Get agents that this agent knows."""
        return [self.agents[aid] for aid in agent.known_agents if aid in self.agents]

    def get_all_alive_agents(self) -> List[Agent]:
        """Get all agents (all are alive in this experiment)."""
        return list(self.agents.values())

    def compute_preference_match(self, agent1: Agent, agent2: Agent) -> float:
        """Compute preference similarity between two agents."""
        cos_sim = np.dot(agent1.preferences, agent2.preferences) / (
            np.linalg.norm(agent1.preferences) * np.linalg.norm(agent2.preferences) + 1e-8
        )
        return float((cos_sim + 1) / 2)  # Normalize to 0-1

    def apply_action(self, agent: Agent, action: Action) -> Optional[dict]:
        """Apply an agent's action and return event dict."""
        if action.agent_id != agent.id:
            return None
        cost = self.action_costs.get(action.action_type, 0)

        # Check budget
        if agent.energy < cost:
            return None

        event = None

        if action.action_type == ActionType.PROPOSE:
            event = self._handle_propose(agent, action)
        elif action.action_type == ActionType.SUPPORT:
            event = self._handle_support(agent, action)
        elif action.action_type == ActionType.OPPOSE:
            event = self._handle_oppose(agent, action)
        elif action.action_type == ActionType.MODIFY:
            event = self._handle_modify(agent, action)
        elif action.action_type == ActionType.MERGE:
            event = self._handle_merge(agent, action)
        elif action.action_type == ActionType.SHARE_INFORMATION:
            event = self._handle_share_info(agent, action)
        elif action.action_type == ActionType.REQUEST_INFORMATION:
            event = self._handle_request_info(agent, action)
        elif action.action_type == ActionType.CONTACT:
            event = self._handle_contact(agent, action)
        elif action.action_type == ActionType.FOLLOW:
            event = self._handle_follow(agent, action)
        elif action.action_type == ActionType.UNFOLLOW:
            event = self._handle_unfollow(agent, action)
        elif action.action_type == ActionType.WAIT:
            event = {"type": "wait", "agent": agent.id, "round": self.round_count}

        if event and not event["type"].endswith("_failed"):
            agent.energy -= cost
            agent.history.append({**event, "round": self.round_count})
            event["round"] = self.round_count
            event["agent"] = agent.id
            self.events.append(event)

        return event

    def _handle_propose(self, agent: Agent, action: Action) -> dict:
        """Handle proposal creation."""
        proposal = ProposalState(
            id=self.next_proposal_id,
            content=action.content or f"Proposal from agent {agent.id}",
            creator=agent.id,
            created_round=self.round_count,
            option_id=action.option_id,
        )
        proposal.supporters.append(agent.id)
        proposal.support_count = 1
        proposal.support_confidence[agent.id] = action.confidence
        self.proposals[self.next_proposal_id] = proposal
        agent.current_proposal = self.next_proposal_id
        agent.current_support = self.next_proposal_id
        agent.has_proposed = True
        self.next_proposal_id += 1
        return {"type": "propose", "proposal_id": proposal.id, "content": proposal.content}

    def _handle_support(self, agent: Agent, action: Action) -> dict:
        """Handle support action."""
        pid = action.proposal_id
        if pid is None or pid not in self.proposals:
            return {"type": "support_failed", "reason": "invalid_proposal"}

        proposal = self.proposals[pid]
        if not proposal.alive:
            return {"type": "support_failed", "reason": "inactive_proposal"}
        if agent.current_support is not None and agent.current_support != pid:
            previous = self.proposals.get(agent.current_support)
            if previous and agent.id in previous.supporters:
                previous.supporters.remove(agent.id)
                previous.support_count -= 1
                previous.support_confidence.pop(agent.id, None)
        # Remove from opposing if was opposing
        if agent.id in proposal.opposers:
            proposal.opposers.remove(agent.id)
            proposal.oppose_count -= 1

        if agent.id not in proposal.supporters:
            proposal.supporters.append(agent.id)
            proposal.support_count += 1
        proposal.support_confidence[agent.id] = action.confidence

        agent.current_support = pid
        return {"type": "support", "proposal_id": pid, "confidence": action.confidence}

    def _handle_oppose(self, agent: Agent, action: Action) -> dict:
        """Handle oppose action."""
        pid = action.proposal_id
        if pid is None or pid not in self.proposals:
            return {"type": "oppose_failed", "reason": "invalid_proposal"}

        proposal = self.proposals[pid]
        # Remove from supporting if was supporting
        if agent.id in proposal.supporters:
            proposal.supporters.remove(agent.id)
            proposal.support_count -= 1
            proposal.support_confidence.pop(agent.id, None)
            if agent.current_support == pid:
                agent.current_support = None

        if agent.id not in proposal.opposers:
            proposal.opposers.append(agent.id)
            proposal.oppose_count += 1

        return {"type": "oppose", "proposal_id": pid, "confidence": action.confidence}

    def _handle_modify(self, agent: Agent, action: Action) -> dict:
        """Handle proposal modification."""
        pid = action.proposal_id
        if pid is None or pid not in self.proposals:
            return {"type": "modify_failed", "reason": "invalid_proposal"}

        parent = self.proposals[pid]
        if not parent.alive:
            return {"type": "modify_failed", "reason": "inactive_proposal"}
        child_id = self.next_proposal_id
        content = action.change_description or parent.content + " (modified)"
        child = ProposalState(
            id=child_id,
            content=content,
            creator=parent.creator,
            parent=pid,
            created_round=self.round_count,
            supporters=[agent.id],
            support_count=1,
            modifiers=parent.modifiers + [agent.id],
            mutation_history=parent.mutation_history + [{
                "round": self.round_count, "modifier": agent.id, "from": pid,
            }],
            support_confidence={agent.id: action.confidence},
            option_id=action.option_id if action.option_id is not None else parent.option_id,
        )
        if agent.id in parent.supporters:
            parent.supporters.remove(agent.id)
            parent.support_count -= 1
            parent.support_confidence.pop(agent.id, None)
        self.proposals[child_id] = child
        self.next_proposal_id += 1
        agent.current_support = child_id
        return {"type": "modify", "proposal_id": pid, "new_proposal_id": child_id,
                "old": parent.content, "new": content}

    def _handle_merge(self, agent: Agent, action: Action) -> dict:
        """Handle proposal merge."""
        pid = action.proposal_id
        merge_target = action.merge_target
        if pid is None or merge_target is None:
            return {"type": "merge_failed", "reason": "missing_ids"}
        if pid not in self.proposals or merge_target not in self.proposals:
            return {"type": "merge_failed", "reason": "invalid_proposal"}

        if pid == merge_target:
            return {"type": "merge_failed", "reason": "same_proposal"}
        p1 = self.proposals[pid]
        p2 = self.proposals[merge_target]
        if not p1.alive or not p2.alive:
            return {"type": "merge_failed", "reason": "inactive_proposal"}
        merged_id = self.next_proposal_id
        supporters = sorted(set(p1.supporters) | set(p2.supporters) | {agent.id})
        merged = ProposalState(
            id=merged_id,
            content=f"[Merged] {p1.content} + {p2.content}",
            creator=agent.id,
            merged_from=[pid, merge_target],
            supporters=supporters,
            support_count=len(supporters),
            created_round=self.round_count,
            modifiers=[agent.id],
            support_confidence={sid: max(
                p1.support_confidence.get(sid, 0.5), p2.support_confidence.get(sid, 0.5)
            ) for sid in supporters},
            option_id=p1.option_id if p1.option_id == p2.option_id else None,
        )
        self.proposals[merged_id] = merged
        self.next_proposal_id += 1
        p1.alive = False
        p2.alive = False
        for supporter in supporters:
            self.agents[supporter].current_support = merged_id
        return {"type": "merge", "proposal_a": pid, "proposal_b": merge_target,
                "new_proposal_id": merged_id}

    def _handle_share_info(self, agent: Agent, action: Action) -> dict:
        """Handle information sharing."""
        target_id = action.target_id
        if target_id is None or target_id not in agent.known_agents:
            return {"type": "share_failed", "reason": "invalid_target"}

        target = self.agents[target_id]
        info_id = action.information_id

        if info_id and info_id in self.information_pool:
            if info_id not in agent.known_information:
                return {"type": "share_failed", "reason": "sender_lacks_info"}
            if info_id not in target.known_information:
                target.known_information.append(info_id)
                target.information_sources[info_id] = agent.id
            return {
                "type": "share_information", "target": target_id, "info_id": info_id,
                "information_correct": self.information_pool[info_id].correct,
            }

        return {"type": "share_failed", "reason": "invalid_info"}

    def _handle_request_info(self, agent: Agent, action: Action) -> dict:
        """Handle information request."""
        target_id = action.target_id
        if target_id is None or target_id not in agent.known_agents:
            return {"type": "request_failed", "reason": "invalid_target"}

        target = self.agents[target_id]

        # Transfer one piece of info from target to agent
        if target.known_information:
            info_id = self.rng.choice(target.known_information)
            if info_id not in agent.known_information:
                agent.known_information.append(info_id)
                agent.information_sources[str(info_id)] = target_id
            return {
                "type": "request_information", "target": target_id, "info_id": str(info_id),
                "information_correct": self.information_pool[str(info_id)].correct,
            }

        return {"type": "request_failed", "reason": "target_has_no_info"}

    def _handle_contact(self, agent: Agent, action: Action) -> dict:
        """Handle new social contact."""
        target_id = action.target_id
        if target_id is None or target_id not in self.agents or target_id == agent.id:
            return {"type": "contact_failed", "reason": "invalid_target"}
        if self.network_mode != "local":
            return {"type": "contact_failed", "reason": "network_fixed"}

        if target_id not in agent.known_agents:
            agent.known_agents.add(target_id)
            agent.trust[target_id] = 0.25

        return {"type": "contact", "target": target_id}

    def _handle_follow(self, agent: Agent, action: Action) -> dict:
        """Handle follow action."""
        if self.network_mode != "local":
            return {"type": "follow_failed", "reason": "network_fixed"}
        target_id = action.target_id
        if target_id is None or target_id not in agent.known_agents:
            return {"type": "follow_failed", "reason": "invalid_target"}

        if target_id not in agent.known_agents:
            agent.known_agents.add(target_id)
        agent.trust[target_id] = min(1.0, agent.trust.get(target_id, 0.5) + 0.1)

        return {"type": "follow", "target": target_id, "trust": agent.trust[target_id]}

    def _handle_unfollow(self, agent: Agent, action: Action) -> dict:
        """Handle unfollow action."""
        if self.network_mode != "local":
            return {"type": "unfollow_failed", "reason": "network_fixed"}
        target_id = action.target_id
        if target_id is None or target_id not in agent.known_agents:
            return {"type": "unfollow_failed", "reason": "no_target"}

        if target_id in agent.known_agents:
            agent.known_agents.discard(target_id)
            agent.trust.pop(target_id, None)

        return {"type": "unfollow", "target": target_id}

    def update_trust(self, agent: Agent, source_id: int, info_correct: bool):
        """Update trust based on information accuracy."""
        if source_id in agent.trust:
            delta = 0.05 if info_correct else -0.1
            agent.trust[source_id] = max(0.0, min(1.0, agent.trust[source_id] + delta))

    def settle_task(self, winner: Optional[int]):
        """Resolve information outcomes and retain learning across tasks."""
        for agent in self.agents.values():
            for info_id, source_id in agent.information_sources.items():
                info = self.information_pool.get(info_id)
                if info is not None:
                    self.update_trust(agent, source_id, info.correct)
            agent.memory.append(f"task={self.current_task['id']} winner={winner}")

    def distribute_information(self, task_info: dict):
        """Distribute information asymmetrically to agents."""
        self.information_pool.clear()
        self.next_info_id = 0

        infos = task_info.get("information_fragments", [])
        agent_ids = list(self.agents.keys())

        for info in infos:
            info_id = f"info_{self.task_count}_{self.next_info_id}"
            info_obj = Information(
                id=info_id,
                content=info.get("content", ""),
                category=info.get("category", "general"),
                accuracy=info.get("accuracy", 0.8),
                source_agent=-1,
                round_created=self.round_count,
                correct=bool(self.information_rng.random() < info.get("accuracy", 0.8)),
                option_id=int(info.get("option_id", 0)),
                value=float(info.get("value", 0.0)),
            )
            self.information_pool[info_id] = info_obj
            self.next_info_id += 1

        # Distribute to random subset of agents
        for info_id, info_obj in self.information_pool.items():
            if not agent_ids:
                continue
            minimum = min(5, self.agent_count)
            maximum = min(40, self.agent_count)
            n_recipients = int(self.information_rng.integers(minimum, maximum + 1))
            recipients = self.information_rng.choice(
                agent_ids, size=n_recipients, replace=False
            )
            for rid in recipients:
                self.agents[int(rid)].known_information.append(info_id)

    def start_task(self, task: dict):
        """Start a new task."""
        self.current_task = task
        self.task_count += 1
        self.round_count = 0

        # Reset agent task state
        for agent in self.agents.values():
            agent.energy = self.budget
            agent.current_support = None
            agent.current_proposal = None
            agent.has_proposed = False
            agent.known_information.clear()
            agent.information_sources.clear()

        # Distribute information
        self.distribute_information(task)

        # Clear old proposals
        self.proposals.clear()
        self.next_proposal_id = 0

    def decide(self) -> Optional[int]:
        """Decide which proposal wins (emergent mechanism)."""
        alive_proposals = [p for p in self.proposals.values() if p.alive and p.support_count > 0]
        if not alive_proposals:
            return None

        # Weighted score: support * (1 + avg_trust_of_supporters)
        scores = {}
        for p in alive_proposals:
            trust_sum = 0
            for sid in p.supporters:
                if sid in self.agents:
                    # Average trust from all agents toward this supporter
                    trust_vals = [a.trust.get(sid, 0.5) for a in self.agents.values() if sid in a.known_agents]
                    avg_trust = np.mean(trust_vals) if trust_vals else 0.0
                    trust_sum += avg_trust
            confidence = np.mean(list(p.support_confidence.values())) if p.support_confidence else 0.5
            opposition = p.oppose_count / max(self.agent_count, 1)
            diffusion = len({neighbor for sid in p.supporters
                             for neighbor in self.agents[sid].known_agents}) / max(self.agent_count, 1)
            scores[p.id] = p.support_count * (1 + trust_sum / max(p.support_count, 1))
            scores[p.id] *= confidence * (1 - opposition) * (1 + diffusion)

        # Add stability bonus (older proposals are more stable)
        for p in alive_proposals:
            age = self.round_count - p.created_round
            scores[p.id] *= (1 + age * 0.01)

        return max(scores, key=scores.get)

    def majority_vote(self) -> Optional[int]:
        """Simple majority vote baseline."""
        alive_proposals = [p for p in self.proposals.values() if p.alive and p.support_count > 0]
        if not alive_proposals:
            return None
        return max(alive_proposals, key=lambda p: p.support_count).id

    def snapshot(self) -> dict:
        """Capture full world state."""
        return {
            "round": self.round_count,
            "task": self.task_count,
            "network_mode": self.network_mode,
            "agents": {id: a.to_dict() for id, a in self.agents.items()},
            "proposals": {id: p.to_dict() for id, p in self.proposals.items()},
            "information": {id: i.to_dict() for id, i in self.information_pool.items()},
        }
