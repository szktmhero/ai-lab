"""Brain architecture experiment package."""

from .core.system import BrainConfig, BrainSystem
from .tasks.generator import generate_task_suite

__all__ = ["BrainConfig", "BrainSystem", "generate_task_suite"]
