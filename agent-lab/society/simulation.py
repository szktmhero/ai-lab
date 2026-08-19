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
    previous_support = None
    stable_rounds = 0
    stop_reason = "max_rounds"

    for round_num in range(max_rounds):
        society.round_count = round_num
        round_events = []
        decisions = []
        proposal_snapshot = {
            proposal_id: proposal.to_dict()
            for proposal_id, proposal in society.proposals.items()
            if proposal.alive
        }
        proposal_representatives = (
            _group_proposal_representatives(proposal_snapshot)
            if society.network_mode == "full" else None
        )

        # All decisions use the same start-of-round state, avoiding ID-order bias.
        for agent in society.get_all_alive_agents():
            observation = _build_observation(
                agent,
                society,
                round_num,
                proposal_snapshot=proposal_snapshot,
                proposal_representatives=proposal_representatives,
            )
            action = policy.decide(agent, observation)
            if action is not None:
                action.round = round_num
                decisions.append((agent, action))

        society.rng.shuffle(decisions)
        for agent, action in decisions:
            event = society.apply_action(agent, action)
            if event:
                event["task"] = task["id"]
                observer.record_event(event)
                round_events.append(event)

        # Record round summary
        round_summary = observer.record_round(society, round_num)

        if on_round_end:
            on_round_end(round_num, society, round_summary)

        support = tuple(
            sorted((p.id, p.support_count, p.oppose_count)
                   for p in society.proposals.values() if p.alive)
        )
        stable_rounds = stable_rounds + 1 if support == previous_support else 0
        previous_support = support
        if round_num >= 4 and stable_rounds >= 3 and any(count > 0 for _, count, _ in support):
            stop_reason = "stable_support"
            break

    # Decide winner
    if decision_mode == "emergent":
        winner = society.decide()
    elif decision_mode == "majority":
        winner = society.majority_vote()
    else:
        raise ValueError(f"Unknown decision mode: {decision_mode}")

    society.settle_task(winner)

    # Record task end
    task_summary = observer.record_task_end(
        society, task["id"], winner, stop_reason=stop_reason
    )

    return {
        "winner": winner,
        "rounds": society.round_count + 1,
        "proposals": len(society.proposals),
        "majority_winner": society.majority_vote(),
        "stop_reason": stop_reason,
        "task_summary": task_summary,
    }


def _build_observation(
    agent,
    society: Society,
    round_num: int,
    proposal_snapshot: Optional[dict] = None,
    proposal_representatives: Optional[dict] = None,
) -> dict:
    """Build observation dict for an agent."""
    if proposal_snapshot is None:
        proposal_snapshot = {
            proposal_id: proposal.to_dict()
            for proposal_id, proposal in society.proposals.items()
            if proposal.alive
        }

    # Known proposals (only alive ones the agent knows about)
    if society.network_mode == "full":
        known_proposals = proposal_snapshot
    else:
        known_proposals = {}
        for proposal_id, proposal in proposal_snapshot.items():
            if (
                agent.id in proposal["supporters"]
                or agent.id in proposal["opposers"]
                or proposal["creator"] in agent.known_agents
                or not agent.known_agents.isdisjoint(proposal["supporters"])
            ):
                known_proposals[proposal_id] = proposal

    # Known information
    known_info = [
        {
            **society.information_pool[info_id].public_dict(),
            "source_agent": agent.information_sources.get(info_id, -1),
        }
        for info_id in agent.known_information
        if info_id in society.information_pool
    ]

    return {
        "round": round_num,
        "proposals": known_proposals,
        "proposal_representatives": proposal_representatives,
        "information": known_info,
        "information_ids": agent.known_information,
        "known_agents": list(agent.known_agents),
        "all_agent_ids": list(society.agents),
        "network_mode": society.network_mode,
        "trust": agent.trust,
        "budget_remaining": agent.energy,
        "options": society.current_task.get("options", []),
    }


def _group_proposal_representatives(proposals: dict) -> dict:
    """Pre-compute full-network candidates once for every agent in a round."""
    evidence = {}
    social = {}
    for proposal in proposals.values():
        option_id = proposal.get("option_id")
        if option_id not in evidence or proposal["id"] < evidence[option_id]["id"]:
            evidence[option_id] = proposal

        rank = (
            proposal.get("support_count", 0) - proposal.get("oppose_count", 0),
            -proposal["id"],
        )
        existing = social.get(option_id)
        if existing is None:
            social[option_id] = proposal
        else:
            existing_rank = (
                existing.get("support_count", 0) - existing.get("oppose_count", 0),
                -existing["id"],
            )
            if rank > existing_rank:
                social[option_id] = proposal

    return {
        "evidence": list(evidence.values()),
        "social": list(social.values()),
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
