"""Replaceable inference backends."""

from .base import ModelAdapter
from .deterministic import DeterministicModelAdapter

__all__ = ["DeterministicModelAdapter", "ModelAdapter"]
