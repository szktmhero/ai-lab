from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List

import numpy as np


class Action(enum.IntEnum):
    STAY = 0
    MOVE_N = 1
    MOVE_S = 2
    MOVE_E = 3
    MOVE_W = 4
    CONSUME = 5
    EMIT_SIGNAL = 6


@dataclass
class Vec2:
    x: int
    y: int

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


CELL_STATE_DIM = 12
SIGNAL_DIM = 16

# Energy costs
BASE_METABOLISM = 0.5
MOVE_COST = 0.3
SIGNAL_COST = 0.2
RESOURCE_GAIN = 5.0
INITIAL_ENERGY = 50.0
MAX_ENERGY = 100.0


@dataclass
class CellState:
    id: int
    pos: Vec2
    energy: float = INITIAL_ENERGY
    age: int = 0
    alive: bool = True
    internal_state: np.ndarray = field(default_factory=lambda: np.zeros(CELL_STATE_DIM, dtype=np.float32))
    signal_output: np.ndarray = field(default_factory=lambda: np.zeros(SIGNAL_DIM, dtype=np.float32))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pos": [self.pos.x, self.pos.y],
            "energy": float(self.energy),
            "age": self.age,
            "alive": self.alive,
            "internal_state": self.internal_state.tolist(),
            "signal_output": self.signal_output.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CellState:
        return cls(
            id=d["id"],
            pos=Vec2(d["pos"][0], d["pos"][1]),
            energy=d["energy"],
            age=d["age"],
            alive=d["alive"],
            internal_state=np.array(d["internal_state"], dtype=np.float32),
            signal_output=np.array(d["signal_output"], dtype=np.float32),
        )
