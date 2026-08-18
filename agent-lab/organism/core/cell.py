from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .types import Vec2, CELL_STATE_DIM, SIGNAL_DIM, INITIAL_ENERGY


@dataclass
class Cell:
    id: int
    pos: Vec2
    energy: float = INITIAL_ENERGY
    age: int = 0
    alive: bool = True
    internal_state: np.ndarray = None
    signal_output: np.ndarray = None

    def __post_init__(self):
        if self.internal_state is None:
            self.internal_state = np.zeros(CELL_STATE_DIM, dtype=np.float32)
        if self.signal_output is None:
            self.signal_output = np.zeros(SIGNAL_DIM, dtype=np.float32)

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
    def from_dict(cls, d: dict) -> Cell:
        return cls(
            id=d["id"],
            pos=Vec2(d["pos"][0], d["pos"][1]),
            energy=d["energy"],
            age=d["age"],
            alive=d["alive"],
            internal_state=np.array(d["internal_state"], dtype=np.float32),
            signal_output=np.array(d["signal_output"], dtype=np.float32),
        )
