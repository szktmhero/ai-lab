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


class TestTasks:
    def test_generate_task(self):
        task = generate_task(seed=42, task_num=0)
        assert "goal" in task
        assert len(task["information_fragments"]) > 0

    def test_generate_sequence(self):
        tasks = generate_task_sequence(seed=42, n_tasks=3)
        assert len(tasks) == 3


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
