from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from .base import CellPolicy
from ..core.cell import Cell
from ..core.types import SIGNAL_DIM


class RuleBasedPolicy(CellPolicy):
    """
    Simple rule-based policy.
    
    Rules (in priority order):
    1. If energy < 20 and local_resource > 0.3: consume
    2. If energy < 10: move toward the richest reachable adjacent position
    3. If energy > 40 and neighbors exist: emit signal (aggregate neighbor signals)
    4. Otherwise: random movement or stay
    """

    def __init__(self, signal_dim: int = SIGNAL_DIM, seed: int = 42):
        self.signal_dim = signal_dim
        self.rng = np.random.default_rng(seed)

    def decide(self, cell: Cell, observation: dict) -> Tuple[int, Optional[np.ndarray]]:
        energy = cell.energy
        local_resource = observation["local_resource"]
        neighbors = observation["neighbors"]

        # Rule 1: Consume if low energy and resource available
        if energy < 20 and local_resource > 0.3:
            return 5, None  # CONSUME

        # Rule 2: Move toward resource if very low energy
        if energy < 10:
            resources = observation["directional_resource"]
            best_value = max(resources.values())
            best_actions = [action for action, value in resources.items() if value == best_value]
            return int(self.rng.choice(best_actions)), None

        # Rule 3: Emit aggregate signal if energy high
        if energy > 40 and neighbors:
            avg_signal = np.mean([n["signal"] for n in neighbors], axis=0)
            # Add small noise
            noise = self.rng.standard_normal(self.signal_dim).astype(np.float32) * 0.1
            signal = avg_signal + noise
            norm = np.linalg.norm(signal)
            if norm > 0:
                signal = signal / max(norm, 1.0)
            return 6, signal  # EMIT_SIGNAL

        # Rule 4: Random movement or stay
        action = self.rng.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
        return int(action), None

    def reset(self):
        pass
