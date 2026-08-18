from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from .base import CellPolicy
from ..core.cell import Cell
from ..core.types import SIGNAL_DIM


class ModelPolicy(CellPolicy):
    """
    Stub for model-based policy.
    
    This provides the interface for connecting external models:
    - Ollama
    - llama.cpp
    - Small transformer
    - Tiny neural network
    
    The decide() method formats the observation as input and
    expects the model to return an action and optional signal.
    """

    def __init__(self, signal_dim: int = SIGNAL_DIM):
        self.signal_dim = signal_dim
        self.model = None  # Placeholder for loaded model

    def load_model(self, model_path: str):
        """Load a model from path."""
        raise NotImplementedError("ModelPolicy.load_model() not yet implemented")

    def decide(self, cell: Cell, observation: dict) -> Tuple[int, Optional[np.ndarray]]:
        if self.model is None:
            # Fallback to random if no model loaded
            rng = np.random.default_rng()
            action = rng.integers(0, 7)
            signal = None
            if action == 6:
                signal = rng.standard_normal(self.signal_dim).astype(np.float32)
            return action, signal

        # Model inference would go here:
        # 1. Format observation as model input
        # 2. Run model.forward(input)
        # 3. Parse output as (action, signal)
        raise NotImplementedError("Model inference not yet implemented")

    def format_observation(self, cell: Cell, observation: dict) -> np.ndarray:
        """Format observation as flat vector for model input."""
        self_state = observation["self_state"]
        local_res = np.array([observation["local_resource"]], dtype=np.float32)

        # Flatten neighbor info (max 8 neighbors)
        neighbor_vecs = []
        for n in observation["neighbors"][:8]:
            vec = np.array([
                n["relative_pos"][0] / 32.0,
                n["relative_pos"][1] / 32.0,
                n["energy"],
                n["distance"] / 32.0,
            ], dtype=np.float32)
            vec = np.concatenate([vec, n["signal"]])
            neighbor_vecs.append(vec)

        # Pad to 8 neighbors
        while len(neighbor_vecs) < 8:
            neighbor_vecs.append(np.zeros(4 + self.signal_dim, dtype=np.float32))

        return np.concatenate([self_state, local_res] + neighbor_vecs)

    def reset(self):
        pass
