"""Non-participating event collector and social metrics calculator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np


class Observer:
    def __init__(self):
        self.event_log: List[dict] = []
        self.round_summaries: List[dict] = []
        self.task_summaries: List[dict] = []

    def record_event(self, event: dict):
        self.event_log.append(event)

    @staticmethod
    def _entropy(values) -> float:
        counts = np.array(list(Counter(values).values()), dtype=float)
        if counts.size == 0:
            return 0.0
        probabilities = counts / counts.sum()
        return float(-np.sum(probabilities * np.log2(probabilities)))

    @staticmethod
    def _network_metrics(agents) -> dict:
        ids = set(agents)
        adjacency = {aid: set(agents[aid].known_agents) & ids - {aid} for aid in ids}
        degrees = [len(neighbors) for neighbors in adjacency.values()]
        edge_count = sum(degrees)
        possible_edges = len(ids) * max(len(ids) - 1, 1)

        if not ids:
            return {
                "average_degree": 0.0,
                "network_density": 0.0,
                "clustering_coefficient": 0.0,
                "connected_components": 0,
                "centralization": 0.0,
                "max_degree": 0,
            }
        if edge_count == 0:
            return {
                "average_degree": 0.0,
                "network_density": 0.0,
                "clustering_coefficient": 0.0,
                "connected_components": len(ids),
                "centralization": 0.0,
                "max_degree": 0,
            }
        if len(ids) > 1 and all(degree == len(ids) - 1 for degree in degrees):
            return {
                "average_degree": float(len(ids) - 1),
                "network_density": 1.0,
                "clustering_coefficient": 1.0,
                "connected_components": 1,
                "centralization": 0.0,
                "max_degree": len(ids) - 1,
            }

        undirected = {aid: set() for aid in ids}
        for source, targets in adjacency.items():
            for target in targets:
                undirected[source].add(target)
                undirected[target].add(source)

        remaining = set(ids)
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                node = stack.pop()
                discovered = undirected[node] & remaining
                remaining -= discovered
                stack.extend(discovered)

        coefficients = []
        for node, neighbors in undirected.items():
            degree = len(neighbors)
            if degree < 2:
                coefficients.append(0.0)
                continue
            links = sum(
                len(undirected[neighbor] & neighbors) for neighbor in neighbors
            ) / 2
            coefficients.append(2 * links / (degree * (degree - 1)))

        max_degree = max(degrees, default=0)
        centralization = 0.0
        if len(ids) > 2:
            centralization = sum(max_degree - degree for degree in degrees) / ((len(ids) - 1) ** 2)

        return {
            "average_degree": float(np.mean(degrees)) if degrees else 0.0,
            "network_density": edge_count / possible_edges,
            "clustering_coefficient": float(np.mean(coefficients)) if coefficients else 0.0,
            "connected_components": components,
            "centralization": centralization,
            "max_degree": max_degree,
        }

    def record_round(self, society, round_num: int):
        agents = society.agents
        alive_proposals = [p for p in society.proposals.values() if p.alive]
        all_trust = [value for agent in agents.values() for value in agent.trust.values()]
        support_counts = [p.support_count for p in alive_proposals]
        total_support = sum(support_counts)
        consensus = max(support_counts, default=0) / max(society.agent_count, 1)
        minority = total_support - max(support_counts, default=0)
        current_events = [
            event["type"] for event in self.event_log
            if event.get("task") == society.current_task["id"] and event.get("round") == round_num
        ]

        for proposal in alive_proposals:
            proposal.support_history.append(list(proposal.supporters))

        summary = {
            "task": society.current_task["id"],
            "round": round_num,
            "alive_proposals": len(alive_proposals),
            "total_support": total_support,
            "total_oppose": sum(p.oppose_count for p in alive_proposals),
            "consensus_level": consensus,
            "minority_size": minority,
            "average_trust": float(np.mean(all_trust)) if all_trust else 0.0,
            "trust_variance": float(np.var(all_trust)) if all_trust else 0.0,
            "trust_concentration": float(max(all_trust) / sum(all_trust)) if all_trust and sum(all_trust) else 0.0,
            "opinion_entropy": self._entropy(
                agent.current_support for agent in agents.values() if agent.current_support is not None
            ),
            "preference_diversity": float(np.mean(np.var(
                np.array([agent.preferences for agent in agents.values()]), axis=0
            ))),
            "behavioral_diversity": self._entropy(current_events),
            "total_energy": sum(a.energy for a in agents.values()),
            **self._network_metrics(agents),
        }
        self.round_summaries.append(summary)
        return summary

    def record_task_end(
        self,
        society,
        task_num: int,
        winner: Optional[int],
        stop_reason: str = "unknown",
    ):
        events = [event for event in self.event_log if event.get("task") == task_num]
        event_types = Counter(event.get("type") for event in events)
        proposal_counts = Counter(event["agent"] for event in events if event.get("type") == "propose")
        support_counts = Counter(event["agent"] for event in events if event.get("type") == "support")
        shares = [event for event in events if event.get("type") == "share_information"]
        reached = {event.get("target") for event in shares}
        known_pairs = sum(len(agent.known_information) for agent in society.agents.values())
        possible_pairs = len(society.information_pool) * max(society.agent_count, 1)
        accurate_pairs = sum(
            1 for agent in society.agents.values() for info_id in agent.known_information
            if society.information_pool[info_id].correct
        )

        winner_proposal = society.proposals.get(winner) if winner is not None else None
        winner_support = winner_proposal.support_count if winner_proposal else 0
        supporter_coverage = []
        if winner_proposal and society.information_pool:
            for agent_id in winner_proposal.supporters:
                known_accuracy = sum(
                    society.information_pool[info_id].accuracy
                    for info_id in society.agents[agent_id].known_information
                )
                supporter_coverage.append(known_accuracy / len(society.information_pool))

        final_round = next(
            (record for record in reversed(self.round_summaries) if record["task"] == task_num),
            {},
        )
        all_trust = [value for agent in society.agents.values() for value in agent.trust.values()]
        qualities = society.current_task.get("option_quality", [])
        selected_quality = 0.0
        if winner_proposal and winner_proposal.option_id is not None and qualities:
            raw_quality = qualities[winner_proposal.option_id]
            low, high = min(qualities), max(qualities)
            selected_quality = (raw_quality - low) / max(high - low, 1e-12)

        summary = {
            "task": task_num,
            "winner": winner,
            "decision_quality": selected_quality,
            "evidence_coverage": float(np.mean(supporter_coverage)) if supporter_coverage else 0.0,
            "decision_time": final_round.get("round", -1) + 1,
            "stop_reason": stop_reason,
            "reached_round_limit": stop_reason == "max_rounds",
            "consensus_level": winner_support / max(society.agent_count, 1),
            "minority_size": sum(
                p.support_count for p in society.proposals.values() if p.alive and p.id != winner
            ),
            "proposal_count": len(society.proposals),
            "proposal_mutation_count": event_types["modify"],
            "proposal_merge_count": event_types["merge"],
            "communication_cost": society.agent_count * society.budget - sum(
                agent.energy for agent in society.agents.values()
            ),
            "information_spread": known_pairs / max(possible_pairs, 1),
            "information_accuracy": accurate_pairs / max(known_pairs, 1),
            "information_reach": len(reached) / max(society.agent_count, 1),
            "information_loss": 1 - known_pairs / max(possible_pairs, 1),
            "average_trust": float(np.mean(all_trust)) if all_trust else 0.0,
            "trust_variance": float(np.var(all_trust)) if all_trust else 0.0,
            "trust_concentration": float(max(all_trust) / sum(all_trust)) if all_trust and sum(all_trust) else 0.0,
            "events_this_task": len(events),
            "agents_proposed": len(proposal_counts),
            "agents_supported": len(support_counts),
            **self._network_metrics(society.agents),
        }
        self.task_summaries.append(summary)
        return summary

    def save_events(self, path: Path):
        with open(path, "w") as f:
            for event in self.event_log:
                f.write(json.dumps(event, default=str) + "\n")

    def save_rounds(self, path: Path):
        path.write_text(json.dumps(self.round_summaries, indent=2))

    def save_tasks(self, path: Path):
        path.write_text(json.dumps(self.task_summaries, indent=2))

    def agent_metrics(self, society) -> dict:
        """Return post-hoc measurements without assigning role labels."""
        metrics = {}
        for agent_id, agent in society.agents.items():
            events = [event for event in self.event_log if event.get("agent") == agent_id]
            created = [
                (event.get("task"), event["proposal_id"])
                for event in events if event.get("type") == "propose"
            ]
            winners = {summary["task"]: summary["winner"] for summary in self.task_summaries}
            adopted = sum(
                1 for task_id, proposal_id in created if winners.get(task_id) == proposal_id
            )
            shared = [event for event in events if event.get("type") == "share_information"]
            accurate = sum(bool(event.get("information_correct")) for event in shared)
            incoming = sum(
                1 for event in self.event_log
                if event.get("target") == agent_id and event.get("type") in {"contact", "follow"}
            )
            trust_received = [
                other.trust[agent_id] for other in society.agents.values() if agent_id in other.trust
            ]
            metrics[str(agent_id)] = {
                "proposal_creation_count": len(created),
                "proposal_adoption_rate": adopted / max(len(created), 1),
                "information_accuracy": accurate / max(len(shared), 1),
                "information_propagation": len(shared),
                "incoming_contacts": incoming,
                "outgoing_contacts": sum(
                    event.get("type") in {"contact", "follow"} for event in events
                ),
                "out_degree_centrality": len(agent.known_agents) / max(society.agent_count - 1, 1),
                "support_influence": sum(
                    event.get("type") == "support" and
                    winners.get(event.get("task")) == event.get("proposal_id")
                    for event in events
                ),
                "trust_received": float(np.mean(trust_received)) if trust_received else 0.0,
                "trust_given": float(np.mean(list(agent.trust.values()))) if agent.trust else 0.0,
                "successful_modifications": sum(event.get("type") == "modify" for event in events),
                "bridge_behavior_proxy": len(agent.known_agents),
            }
        return metrics

    def generate_chronicle(self) -> str:
        lines = [
            "# Society Chronicle",
            "",
            "The entries below are measured events. Role labels and social authority are not inferred automatically.",
        ]
        for summary in self.task_summaries:
            lines.extend([
                "",
                f"## Task {summary['task']}",
                "",
                f"Observed proposals: {summary['proposal_count']}",
                f"Observed mutations / merges: {summary['proposal_mutation_count']} / {summary['proposal_merge_count']}",
                f"Agents proposing / supporting: {summary['agents_proposed']} / {summary['agents_supported']}",
                f"Final consensus: {summary['consensus_level']:.3f}",
                f"Information spread: {summary['information_spread']:.3f}",
                f"Communication cost: {summary['communication_cost']:.1f}",
                f"Stop reason: {summary['stop_reason']}",
                f"Selected proposal: {summary['winner']}",
            ])
        return "\n".join(lines) + "\n"
