"""Tests for organism simulation core."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from organism.core.types import Vec2, Action, CELL_STATE_DIM, SIGNAL_DIM
from organism.core.cell import Cell
from organism.core.world import World
from organism.policies.random import RandomPolicy
from organism.policies.rule_based import RuleBasedPolicy
from organism.metrics.collector import MetricsCollector


def test_vec2():
    v1 = Vec2(1, 2)
    v2 = Vec2(3, 4)
    v3 = v1 + v2
    assert v3.x == 4 and v3.y == 6
    assert v1 == Vec2(1, 2)
    assert v1 != v2
    assert hash(v1) == hash(Vec2(1, 2))
    print("PASS: test_vec2")


def test_cell_creation():
    cell = Cell(id=0, pos=Vec2(5, 10))
    assert cell.id == 0
    assert cell.pos.x == 5 and cell.pos.y == 10
    assert cell.energy == 50.0
    assert cell.alive is True
    assert cell.internal_state.shape == (CELL_STATE_DIM,)
    assert cell.signal_output.shape == (SIGNAL_DIM,)
    print("PASS: test_cell_creation")


def test_cell_serialization():
    cell = Cell(id=42, pos=Vec2(3, 7), energy=75.5, age=100)
    d = cell.to_dict()
    cell2 = Cell.from_dict(d)
    assert cell2.id == 42
    assert cell2.pos.x == 3 and cell2.pos.y == 7
    assert abs(cell2.energy - 75.5) < 0.01
    assert cell2.age == 100
    print("PASS: test_cell_serialization")


def test_world_creation():
    world = World(width=16, height=16, cell_count=10, seed=42)
    assert world.width == 16
    assert world.height == 16
    assert world.alive_count() == 10
    assert world.resource_field.shape == (16, 16)
    # Check no overlapping positions
    positions = [(c.pos.x, c.pos.y) for c in world.cells.values()]
    assert len(positions) == len(set(positions))
    print("PASS: test_world_creation")


def test_world_wrap():
    world = World(width=16, height=16, cell_count=1, seed=42)
    assert world.wrap_pos(Vec2(-1, 0)) == Vec2(15, 0)
    assert world.wrap_pos(Vec2(16, 0)) == Vec2(0, 0)
    assert world.wrap_pos(Vec2(0, -1)) == Vec2(0, 15)
    print("PASS: test_world_wrap")


def test_world_neighbors():
    world = World(width=16, height=16, cell_count=2, seed=42)
    # Place cells adjacent
    world.cells[0].pos = Vec2(5, 5)
    world.cells[1].pos = Vec2(6, 5)
    neighbors = world.get_neighbors(world.cells[0])
    assert len(neighbors) == 1
    assert neighbors[0].id == 1
    print("PASS: test_world_neighbors")


def test_toroidal_relative_position_is_local():
    world = World(width=16, height=16, cell_count=2, seed=42)
    world.cells[0].pos = Vec2(0, 0)
    world.cells[1].pos = Vec2(15, 0)
    observation = world.observe(world.cells[0])
    assert observation["neighbors"][0]["relative_pos"] == [-1, 0]


def test_world_observe():
    world = World(width=16, height=16, cell_count=5, seed=42)
    cell = world.cells[0]
    obs = world.observe(cell)
    assert "self_state" in obs
    assert "neighbors" in obs
    assert "local_resource" in obs
    assert obs["self_state"].shape[0] == CELL_STATE_DIM + 2  # energy + age + state
    assert 0 <= obs["local_resource"] <= 1.0
    print("PASS: test_world_observe")


def test_apply_action_consume():
    world = World(width=16, height=16, cell_count=1, seed=42)
    cell = world.cells[0]
    cell.pos = Vec2(5, 5)
    initial_energy = cell.energy
    world.resource_field[5, 5] = 5.0

    world.apply_action(cell, 5)  # CONSUME
    assert cell.energy > initial_energy
    assert world.resource_field[5, 5] < 5.0
    print("PASS: test_apply_action_consume")


def test_apply_action_move():
    world = World(width=16, height=16, cell_count=1, seed=42)
    cell = world.cells[0]
    cell.pos = Vec2(5, 5)
    initial_energy = cell.energy

    world.apply_action(cell, 1)  # MOVE_N
    assert cell.pos.y == 4
    assert cell.energy < initial_energy
    print("PASS: test_apply_action_move")


def test_cell_death():
    world = World(width=16, height=16, cell_count=1, seed=42)
    cell = world.cells[0]
    cell.energy = 0.1
    world.apply_action(cell, 0)  # STAY (metabolism cost)
    assert not cell.alive
    print("PASS: test_cell_death")


def test_random_policy():
    policy = RandomPolicy(seed=42)
    cell = Cell(id=0, pos=Vec2(5, 5))
    obs = {"self_state": np.zeros(14), "neighbors": [], "local_resource": 0.5}
    action, signal = policy.decide(cell, obs)
    assert 0 <= action <= 6
    if action == 6:
        assert signal is not None
        assert signal.shape == (SIGNAL_DIM,)
    print("PASS: test_random_policy")


def test_rule_based_policy():
    policy = RuleBasedPolicy(seed=42)
    cell = Cell(id=0, pos=Vec2(5, 5), energy=15.0)
    obs = {"self_state": np.zeros(14), "neighbors": [], "local_resource": 0.8}
    action, signal = policy.decide(cell, obs)
    assert action == 5  # Should consume (low energy, high resource)
    print("PASS: test_rule_based_policy")


def test_rule_based_low_energy():
    policy = RuleBasedPolicy(seed=42)
    cell = Cell(id=0, pos=Vec2(5, 5), energy=5.0)
    obs = {
        "self_state": np.zeros(14),
        "neighbors": [{"signal": np.zeros(SIGNAL_DIM)}],
        "local_resource": 0.1,
        "directional_resource": {1: 0.1, 2: 0.2, 3: 0.9, 4: 0.3},
    }
    action, signal = policy.decide(cell, obs)
    assert action == 3
    print("PASS: test_rule_based_low_energy")


def test_metrics_collector():
    world = World(width=16, height=16, cell_count=10, seed=42)
    collector = MetricsCollector()
    record = collector.collect(world)
    assert record["alive_cells"] == 10
    assert record["step"] == 0
    assert record["average_energy"] > 0
    assert record["largest_cluster_size"] >= 1
    assert record["number_of_clusters"] >= 1
    print("PASS: test_metrics_collector")


def test_movement_does_not_overlap():
    world = World(width=16, height=16, cell_count=2, seed=42)
    world.cells[0].pos = Vec2(5, 5)
    world.cells[1].pos = Vec2(6, 5)
    world.apply_action(world.cells[0], Action.MOVE_E)
    assert world.cells[0].pos == Vec2(5, 5)


def test_resource_consumption_is_cumulative():
    world = World(width=16, height=16, cell_count=1, seed=42)
    cell = world.cells[0]
    world.resource_field[cell.pos.y, cell.pos.x] = 5.0
    world.apply_action(cell, Action.CONSUME)
    record = MetricsCollector().collect(world)
    assert record["resource_consumption"] == 5.0
    assert record["resource_remaining"] > record["resource_consumption"]


def test_simulation_loop():
    world = World(width=16, height=16, cell_count=10, seed=42)
    policy = RandomPolicy(seed=42)

    def policy_fn(cell, obs):
        return policy.decide(cell, obs)

    for _ in range(100):
        world.run_step(policy_fn)

    assert world.step_count == 100
    assert world.alive_count() <= 10
    print("PASS: test_simulation_loop")


def test_reproducibility():
    world1 = World(width=16, height=16, cell_count=10, seed=42)
    world2 = World(width=16, height=16, cell_count=10, seed=42)

    policy = RandomPolicy(seed=42)

    def policy_fn(cell, obs):
        return policy.decide(cell, obs)

    for _ in range(50):
        world1.run_step(policy_fn)

    policy2 = RandomPolicy(seed=42)
    def policy_fn2(cell, obs):
        return policy2.decide(cell, obs)

    for _ in range(50):
        world2.run_step(policy_fn2)

    assert world1.alive_count() == world2.alive_count()
    # Energy should match
    e1 = sum(c.energy for c in world1.cells.values() if c.alive)
    e2 = sum(c.energy for c in world2.cells.values() if c.alive)
    assert abs(e1 - e2) < 0.01
    print("PASS: test_reproducibility")


if __name__ == "__main__":
    test_vec2()
    test_cell_creation()
    test_cell_serialization()
    test_world_creation()
    test_world_wrap()
    test_world_neighbors()
    test_toroidal_relative_position_is_local()
    test_world_observe()
    test_apply_action_consume()
    test_apply_action_move()
    test_cell_death()
    test_random_policy()
    test_rule_based_policy()
    test_rule_based_low_energy()
    test_metrics_collector()
    test_movement_does_not_overlap()
    test_resource_consumption_is_cumulative()
    test_simulation_loop()
    test_reproducibility()
    print("\nAll tests passed!")
