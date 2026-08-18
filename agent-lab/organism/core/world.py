from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .types import Vec2, CellState, CELL_STATE_DIM, SIGNAL_DIM
from .cell import Cell


WORLD_W = 32
WORLD_H = 32
RESOURCE_REGEN_RATE = 0.05
RESOURCE_MAX = 10.0
RESOURCE_INITIAL = 3.0


def _perlin_like_field(w: int, h: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth spatial field via low-frequency noise summation."""
    field = np.zeros((h, w), dtype=np.float64)
    for octave in range(3):
        freq = scale * (2 ** octave)
        amplitude = 1.0 / (2 ** octave)
        noise = rng.random((h, w))
        # Smooth via box blur
        kernel_size = max(1, int(3 / freq))
        from scipy.ndimage import uniform_filter
        try:
            smoothed = uniform_filter(noise, size=kernel_size)
        except ImportError:
            # Fallback: simple averaging
            smoothed = noise
        field += smoothed * amplitude
    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field


class World:
    def __init__(
        self,
        width: int = WORLD_W,
        height: int = WORLD_H,
        cell_count: int = 128,
        seed: int = 42,
        signal_dim: int = SIGNAL_DIM,
        state_dim: int = CELL_STATE_DIM,
    ):
        self.width = width
        self.height = height
        self.cell_count = cell_count
        self.seed = seed
        self.signal_dim = signal_dim
        self.state_dim = state_dim

        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        # Resource grid
        self.resource_field = self._init_resources()

        # Cells
        self.cells: Dict[int, Cell] = {}
        self._init_cells()

    def _init_resources(self) -> np.ndarray:
        """Initialize resource field with spatial heterogeneity."""
        try:
            from scipy.ndimage import gaussian_filter
            base = self.rng.random((self.height, self.width))
            field = gaussian_filter(base, sigma=4.0)
        except ImportError:
            # Fallback: use cumulative sum for smoothness
            base = self.rng.random((self.height, self.width))
            field = np.cumsum(np.cumsum(base, axis=0), axis=1)
            # Normalize
            field = (field - field.min()) / (field.max() - field.min() + 1e-8)

        field = field * RESOURCE_MAX * 0.8 + RESOURCE_MAX * 0.1
        return field.astype(np.float64)

    def _init_cells(self):
        """Place cells at random positions (no overlaps)."""
        positions = set()
        for i in range(self.cell_count):
            while True:
                x = self.rng.integers(0, self.width)
                y = self.rng.integers(0, self.height)
                if (x, y) not in positions:
                    positions.add((x, y))
                    break
            self.cells[i] = Cell(id=i, pos=Vec2(int(x), int(y)))

    def wrap_pos(self, pos: Vec2) -> Vec2:
        """Toroidal wrapping."""
        return Vec2(pos.x % self.width, pos.y % self.height)

    def get_neighbors(self, cell: Cell) -> List[Cell]:
        """Get cells in Moore neighborhood (8 directions)."""
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (cell.pos.x + dx) % self.width
                ny = (cell.pos.y + dy) % self.height
                for c in self.cells.values():
                    if c.alive and c.pos.x == nx and c.pos.y == ny:
                        neighbors.append(c)
        return neighbors

    def get_cell_at(self, x: int, y: int) -> Optional[Cell]:
        """Get cell at specific position."""
        for c in self.cells.values():
            if c.alive and c.pos.x == x and c.pos.y == y:
                return c
        return None

    def observe(self, cell: Cell) -> dict:
        """
        Build observation for a cell.
        Returns dict with:
          - self_state: cell's own state vector
          - neighbors: list of neighbor observations
          - local_resource: resource level at cell position
        """
        # Self state
        self_state = np.concatenate([
            [cell.energy / 100.0],  # normalized energy
            [cell.age / 1000.0],    # normalized age
            cell.internal_state,
        ])

        # Neighbors
        neighbors = self.get_neighbors(cell)
        neighbor_obs = []
        for n in neighbors:
            dist = abs(n.pos.x - cell.pos.x) + abs(n.pos.y - cell.pos.y)
            # Handle toroidal distance
            if dist > self.width // 2:
                dist = self.width - dist
            neighbor_obs.append({
                "relative_pos": [n.pos.x - cell.pos.x, n.pos.y - cell.pos.y],
                "energy": n.energy / 100.0,
                "signal": n.signal_output.copy(),
                "distance": dist,
            })

        # Local resource
        local_resource = self.resource_field[cell.pos.y, cell.pos.x] / RESOURCE_MAX

        return {
            "self_state": self_state,
            "neighbors": neighbor_obs,
            "local_resource": local_resource,
        }

    def apply_action(self, cell: Cell, action: int, signal: Optional[np.ndarray] = None):
        """Apply a cell's action to the world."""
        if not cell.alive:
            return

        cell.age += 1

        # Base metabolism cost
        cell.energy -= 0.5

        # Movement actions
        if action == 1:  # MOVE_N
            cell.pos = self.wrap_pos(Vec2(cell.pos.x, cell.pos.y - 1))
            cell.energy -= 0.3
        elif action == 2:  # MOVE_S
            cell.pos = self.wrap_pos(Vec2(cell.pos.x, cell.pos.y + 1))
            cell.energy -= 0.3
        elif action == 3:  # MOVE_E
            cell.pos = self.wrap_pos(Vec2(cell.pos.x + 1, cell.pos.y))
            cell.energy -= 0.3
        elif action == 4:  # MOVE_W
            cell.pos = self.wrap_pos(Vec2(cell.pos.x - 1, cell.pos.y))
            cell.energy -= 0.3
        elif action == 5:  # CONSUME
            res = self.resource_field[cell.pos.y, cell.pos.x]
            if res > 0.1:
                gain = min(5.0, res)
                cell.energy += gain
                self.resource_field[cell.pos.y, cell.pos.x] -= gain
                cell.energy = min(100.0, cell.energy)
        elif action == 6:  # EMIT_SIGNAL
            if signal is not None:
                cell.signal_output = signal.copy()
                cell.energy -= 0.2
            else:
                # Default: emit current signal (no change)
                pass

        # Check death
        if cell.energy <= 0:
            cell.alive = False

    def regen_resources(self):
        """Regenerate resources over time."""
        regen = self.rng.random((self.height, self.width)) * 0.1
        self.resource_field += regen
        self.resource_field = np.minimum(self.resource_field, RESOURCE_MAX)

    def run_step(self, policy_fn):
        """
        Run one simulation step.
        policy_fn(cell, observation) -> (action, optional_signal)
        """
        self.step_count += 1

        # Process each living cell
        for cell in list(self.cells.values()):
            if not cell.alive:
                continue

            obs = self.observe(cell)
            action, signal = policy_fn(cell, obs)
            self.apply_action(cell, action, signal)

        # Resource regeneration
        self.regen_resources()

    def alive_count(self) -> int:
        return sum(1 for c in self.cells.values() if c.alive)

    def snapshot(self) -> dict:
        """Capture full world state."""
        return {
            "step": self.step_count,
            "cells": {id: c.to_dict() for id, c in self.cells.items()},
            "resource_field": self.resource_field.tolist(),
        }
