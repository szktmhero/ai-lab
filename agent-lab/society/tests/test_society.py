"""Tests for society experiment."""

import pytest
import numpy as np

from society.core.types import Action, ActionType, ACTION_COSTS, ProposalState, Information
from society.core.agent import Agent
from society.core.world import Society
from society.policies.random import RandomPolicy
from society.policies.simple import SimplePolicy
from society.tasks import generate_task, generate_task_sequence
from society.simulation import run_simulation, run_multi_task
from society.observer.observer import Observer


class TestAgent:
    def test_create(self):
        agent = Agent(id=0)
        assert agent.id == 0
        assert agent.energy > 0
        assert len(agent.preferences) == 8

    def test_initialize_social(self):
        agent = Agent(id=0)
        rng = np.random.default_rng(42)
        agent.initialize_social(list(range(10)), rng, avg_degree=4)
        assert len(agent.known_agents) > 0
        assert all(0 <= t <= 1 for t in agent.trust.values())

    def test_to_dict_roundtrip(self):
        agent = Agent(id=42)
        d = agent.to_dict()
        agent2 = Agent.from_dict(d)
        assert agent2.id == 42
        assert np.allclose(agent2.preferences, agent.preferences)


class TestSociety:
    def test_create(self):
        soc = Society(agent_count=16, seed=42)
        assert len(soc.agents) == 16
        assert soc.round_count == 0

    def test_known_agents(self):
        soc = Society(agent_count=16, seed=42)
        agent = soc.agents[0]
        known = soc.get_known_agents(agent)
        assert len(known) > 0

    def test_propose(self):
        soc = Society(agent_count=16, seed=42)
        agent = soc.agents[0]
        action = Action(agent_id=0, action_type=ActionType.PROPOSE, round=0, content="Test proposal")
        event = soc.apply_action(agent, action)
        assert event is not None
        assert event["type"] == "propose"
        assert len(soc.proposals) == 1
        assert agent.has_proposed

    def test_support(self):
        soc = Society(agent_count=16, seed=42)
        # First propose
        a0 = soc.agents[0]
        soc.apply_action(a0, Action(0, ActionType.PROPOSE, 0, content="P0"))

        # Support
        a1 = soc.agents[1]
        soc.apply_action(a1, Action(1, ActionType.SUPPORT, 0, proposal_id=0, confidence=0.8))
        assert soc.proposals[0].support_count == 2  # proposer + supporter

    def test_oppose(self):
        soc = Society(agent_count=16, seed=42)
        soc.apply_action(soc.agents[0], Action(0, ActionType.PROPOSE, 0, content="P0"))
        soc.apply_action(soc.agents[1], Action(1, ActionType.OPPOSE, 0, proposal_id=0, confidence=0.7))
        assert soc.proposals[0].oppose_count == 1

    def test_budget_enforcement(self):
        soc = Society(agent_count=16, seed=42)
        agent = soc.agents[0]
        agent.energy = 0.1  # Very low energy
        event = soc.apply_action(agent, Action(0, ActionType.PROPOSE, 0))
        assert event is None  # Can't afford

    def test_contact(self):
        soc = Society(agent_count=16, seed=42)
        agent = soc.agents[0]
        soc.apply_action(agent, Action(0, ActionType.CONTACT, 0, target_id=5))
        assert 5 in agent.known_agents
        assert 5 in agent.trust

    def test_decide(self):
        soc = Society(agent_count=16, seed=42)
        # Create some proposals with support
        soc.apply_action(soc.agents[0], Action(0, ActionType.PROPOSE, 0, content="P0"))
        soc.apply_action(soc.agents[1], Action(1, ActionType.PROPOSE, 0, content="P1"))
        soc.apply_action(soc.agents[2], Action(2, ActionType.SUPPORT, 0, proposal_id=1))
        winner = soc.decide()
        assert winner is not None

    def test_seed_controls_agent_state(self):
        first = Society(agent_count=16, seed=7)
        second = Society(agent_count=16, seed=7)
        assert np.array_equal(first.agents[0].preferences, second.agents[0].preferences)
        assert first.agents[0].trust == second.agents[0].trust

    def test_support_switch_is_exclusive(self):
        soc = Society(agent_count=16, seed=42)
        soc.apply_action(soc.agents[0], Action(0, ActionType.PROPOSE, 0, content="P0"))
        soc.apply_action(soc.agents[1], Action(1, ActionType.PROPOSE, 0, content="P1"))
        agent = soc.agents[2]
        soc.apply_action(agent, Action(2, ActionType.SUPPORT, 0, proposal_id=0))
        soc.apply_action(agent, Action(2, ActionType.SUPPORT, 0, proposal_id=1))
        assert agent.id not in soc.proposals[0].supporters
        assert agent.id in soc.proposals[1].supporters

    def test_invalid_action_does_not_cost_budget(self):
        soc = Society(agent_count=16, seed=42)
        agent = soc.agents[0]
        initial = agent.energy
        event = soc.apply_action(agent, Action(0, ActionType.SUPPORT, 0, proposal_id=999))
        assert event["type"] == "support_failed"
        assert agent.energy == initial

    def test_modify_creates_lineage(self):
        soc = Society(agent_count=16, seed=42)
        soc.apply_action(soc.agents[0], Action(0, ActionType.PROPOSE, 0, content="P0"))
        event = soc.apply_action(
            soc.agents[1], Action(1, ActionType.MODIFY, 1, proposal_id=0, change_description="P1")
        )
        child = soc.proposals[event["new_proposal_id"]]
        assert child.parent == 0
        assert child.content == "P1"
        assert soc.agents[1].id not in soc.proposals[0].supporters

    def test_network_conditions(self):
        independent = Society(agent_count=8, seed=1, network_mode="independent")
        full = Society(agent_count=8, seed=1, network_mode="full")
        assert all(not agent.known_agents for agent in independent.agents.values())
        assert all(len(agent.known_agents) == 7 for agent in full.agents.values())

    def test_matched_seed_isolates_population_and_information_from_topology(self):
        societies = [
            Society(agent_count=16, seed=7, network_mode=mode)
            for mode in ("independent", "full", "local")
        ]
        for agent_id in range(16):
            reference = societies[0].agents[agent_id]
            for society in societies[1:]:
                candidate = society.agents[agent_id]
                assert np.array_equal(reference.preferences, candidate.preferences)
                assert np.array_equal(reference.beliefs, candidate.beliefs)
                assert reference.confidence == candidate.confidence

        for society in societies:
            society.start_task(generate_task(seed=7, task_num=0))

        reference_information = {
            info_id: info.to_dict()
            for info_id, info in societies[0].information_pool.items()
        }
        reference_recipients = {
            agent_id: list(agent.known_information)
            for agent_id, agent in societies[0].agents.items()
        }
        for society in societies[1:]:
            assert {
                info_id: info.to_dict()
                for info_id, info in society.information_pool.items()
            } == reference_information
            assert {
                agent_id: list(agent.known_information)
                for agent_id, agent in society.agents.items()
            } == reference_recipients


class TestPolicies:
    def test_random_policy(self):
        soc = Society(agent_count=16, seed=42)
        policy = RandomPolicy(seed=42)
        agent = soc.agents[0]
        obs = {"proposals": {}, "information_ids": [], "round": 0}
        action = policy.decide(agent, obs)
        assert action is not None
        assert isinstance(action.action_type, ActionType)

    def test_simple_policy(self):
        soc = Society(agent_count=16, seed=42)
        policy = SimplePolicy(seed=42)
        agent = soc.agents[0]
        obs = {"proposals": {}, "information_ids": [], "round": 0}
        action = policy.decide(agent, obs)
        assert action is not None

    def test_social_evidence_weight_is_an_explicit_ablation(self):
        agent = Agent(id=0, preferences=np.zeros(8, dtype=np.float32))
        agent.current_support = 0
        observation = {
            "round": 2,
            "proposals": {
                0: {"id": 0, "option_id": 0, "support_count": 1, "oppose_count": 0},
                1: {"id": 1, "option_id": 1, "support_count": 10, "oppose_count": 0},
            },
            "information": [],
            "information_ids": [],
            "known_agents": [],
            "all_agent_ids": [0],
            "network_mode": "independent",
            "options": [f"option_{index}" for index in range(8)],
        }

        evidence_only = SimplePolicy(seed=1, social_evidence_weight=0.0)
        social = SimplePolicy(seed=1, social_evidence_weight=0.05)

        assert evidence_only.decide(agent, observation).action_type == ActionType.WAIT
        social_action = social.decide(agent, observation)
        assert social_action.action_type == ActionType.SUPPORT
        assert social_action.proposal_id == 1


class TestTasks:
    def test_generate_task(self):
        task = generate_task(seed=42, task_num=0)
        assert "goal" in task
        assert len(task["information_fragments"]) > 0

    def test_generate_sequence(self):
        tasks = generate_task_sequence(seed=42, n_tasks=3)
        assert len(tasks) == 3

    def test_reliability_is_hidden_from_agents(self):
        soc = Society(agent_count=16, seed=42)
        soc.start_task(generate_task(seed=42, task_num=0))
        agent = next(agent for agent in soc.agents.values() if agent.known_information)
        from society.simulation import _build_observation
        information = _build_observation(agent, soc, 0)["information"][0]
        assert "accuracy" not in information
        assert "correct" not in information


class TestSimulation:
    def test_run_simulation(self):
        soc = Society(agent_count=16, seed=42)
        policy = RandomPolicy(seed=42)
        task = generate_task(seed=42, task_num=0)
        result = run_simulation(soc, policy, task, max_rounds=5)
        assert "winner" in result
        assert result["rounds"] > 0

    def test_observer(self):
        soc = Society(agent_count=16, seed=42)
        policy = RandomPolicy(seed=42)
        task = generate_task(seed=42, task_num=0)
        observer = Observer()
        result = run_simulation(soc, policy, task, max_rounds=5, observer=observer)
        assert len(observer.round_summaries) > 0

    def test_task_metrics_do_not_mix_tasks(self):
        soc = Society(agent_count=16, seed=42)
        policy = SimplePolicy(seed=42)
        observer = Observer()
        run_multi_task(
            soc, policy, generate_task_sequence(seed=42, n_tasks=2),
            max_rounds_per_task=3, observer=observer,
        )
        task_zero_events = sum(event.get("task") == 0 for event in observer.event_log)
        assert observer.task_summaries[0]["events_this_task"] == task_zero_events
        assert observer.task_summaries[1]["agents_proposed"] <= 16

    def test_stop_reason_distinguishes_stability_from_round_limit(self):
        task = generate_task(seed=42, task_num=0)
        stable_result = run_simulation(
            Society(agent_count=16, seed=42, network_mode="independent"),
            SimplePolicy(seed=42, social_evidence_weight=0.0),
            task,
            max_rounds=10,
        )
        limited_result = run_simulation(
            Society(agent_count=16, seed=42, network_mode="independent"),
            SimplePolicy(seed=42, social_evidence_weight=0.0),
            task,
            max_rounds=2,
        )

        assert stable_result["stop_reason"] == "stable_support"
        assert not stable_result["task_summary"]["reached_round_limit"]
        assert limited_result["stop_reason"] == "max_rounds"
        assert limited_result["task_summary"]["reached_round_limit"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
