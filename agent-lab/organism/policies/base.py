from __future__ import annotations

import abc
from typing import Tuple, Optional
import numpy as np

from ..core.cell import Cell


class CellPolicy(abc.ABC):
    """Abstract base class for cell decision-making policies."""

    @abc.abstractmethod
    def decide(self, cell: Cell, observation: dict) -> Tuple[int, Optional[np.ndarray]]:
        """
        Given a cell and its observation, return (action, optional_signal).
        
        Args:
            cell: The cell making the decision
            observation: Dict with 'self_state', 'neighbors', 'local_resource'
        
        Returns:
            action: int (0-6)
            signal: Optional[np.ndarray] of shape (signal_dim,) if emitting
        """
        ...

    def reset(self):
        """Reset any internal state between experiments."""
        pass
