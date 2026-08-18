from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from .base import CellPolicy
from ..core.cell import Cell
from ..core.types import SIGNAL_DIM


class RandomPolicy(CellPolicy):
    """Baseline: completely random actions."""

    def __init__(self, signal_dim: int = SIGNAL_DIM, seed: int = 42):
        self.signal_dim = signal_dim
        self.rng = np.random.default_rng(seed)

    def decide(self, cell: Cell, observation: dict) -> Tuple[int, Optional[np.ndarray]]:
        action = self.rng.integers(0, 7)
        signal = None
        if action == 6:  # EMIT_SIGNAL
            signal = self.rng.standard_normal(self.signal_dim).astype(np.float32)
            # Normalize to [-1, 1]
            norm = np.linalg.norm(signal)
            if norm > 0:
                signal = signal / max(norm, 1.0)
        return action, signal

    def reset(self):
        pass
