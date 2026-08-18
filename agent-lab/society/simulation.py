"""Society simulation loop."""

from __future__ import annotations

from typing import List, Optional, Callable
from pathlib import Path

import numpy as np

from .core.world import Society
from .core.types import Action, ActionType
from .policies.base import AgentPolicy
from .observer.observer import Observer


def run_simulation(
    society: Society,
    policy: AgentPolicy,
    task: dict,
    max_rounds: int = 50,
    observer: Optional[Observer] = None,
    on_round_end: Optional[Callable] = None,
    decision_mode: str = "emergent",
) -> dict:
    """
    Run a single society simulation.

    Args:
        society: Society world state
        policy: Agent decision policy
        task: Task definition
        max_rounds: Maximum rounds per task
        observer: Optional observer for metrics
        on_round_end: Optional callback after each round
        decision_mode: "emergent" or "majority"

    Returns:
        Simulation results dict
    """
    if observer is None:
        observer = Observer()

    society.start_task(task)

    for round_num in range(max_rounds):
        society.round_count = round_num
        round_events = []

        # Build observation for each agent
        for agent in society.get_all_alive_agents():
            observation = _build_observation(agent, society, round_num)

            # Get action from policy
            action = policy.decide(agent, observation)

            if action is not None:
                action.round = round_num
                event = society.apply_action(agent, action)
                if event:
                    event["task"] = task["id"]
                    observer.record_event(event)
                    round_events.append(event)

        # Record round summary
        round_summary = observer.record_round(society, round_num)

        # Check termination: enough proposals with sufficient support?
        alive_proposals = [p for p in society.proposals.values() if p.alive and p.support_count > 5]
        if len(alive_proposals) >= 3 and round_num > 10:
            # Check convergence: support not changing much
            if round_num > 15:
                break

        if on_round_end:
            on_round_end(round_num, society, round_summary)

    # Decide winner
    if decision_mode == "emergent":
        winner = society.decide()
    else:
        winner = society.majority_vote()

    # Record task end
    task_summary = observer.record_task_end(society, task["id"], winner)

    return {
        "winner": winner,
        "rounds": society.round_count + 1,
        "proposals": len(society.proposals),
        "task_summary": task_summary,
    }


def _build_observation(agent, society: Society, round_num: int) -> dict:
    """Build observation dict for an agent."""
    # Known proposals (only alive ones the agent knows about)
    known_proposals = {}
    for pid, proposal in society.proposals.items():
        if proposal.alive:
            # Agent knows about proposals they support/oppose or from known agents
            if (agent.id in proposal.supporters or
                agent.id in proposal.opposers or
                proposal.creator in agent.known_agents):
                known_proposals[pid] = proposal.to_dict()

    # Known information
    known_info = [
        society.information_pool[info_id].to_dict()
        for info_id in agent.known_information
        if info_id in society.information_pool
    ]

    return {
        "round": round_num,
        "proposals": known_proposals,
        "information": known_info,
        "information_ids": agent.known_information,
        "known_agents": list(agent.known_agents),
        "trust": agent.trust,
        "budget_remaining": agent.energy,
    }


def run_multi_task(
    society: Society,
    policy: AgentPolicy,
    tasks: List[dict],
    max_rounds_per_task: int = 50,
    observer: Optional[Observer] = None,
    decision_mode: str = "emergent",
) -> List[dict]:
    """Run multiple tasks with persistent society state."""
    if observer is None:
        observer = Observer()

    results = []
    policy.reset()

    for task in tasks:
        result = run_simulation(
            society, policy, task,
            max_rounds=max_rounds_per_task,
            observer=observer,
            decision_mode=decision_mode,
        )
        results.append(result)

    return results
