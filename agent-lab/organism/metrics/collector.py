from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import numpy as np

from ..core.world import World
from ..core.cell import Cell


class MetricsCollector:
    """Collects and records simulation metrics."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def collect(self, world: World) -> Dict[str, Any]:
        """Compute metrics for current world state."""
        alive_cells = [c for c in world.cells.values() if c.alive]
        dead_cells = [c for c in world.cells.values() if not c.alive]

        alive_count = len(alive_cells)
        total_count = len(world.cells)

        # Energy metrics
        if alive_cells:
            energies = [c.energy for c in alive_cells]
            avg_energy = float(np.mean(energies))
            energy_var = float(np.var(energies))
        else:
            avg_energy = 0.0
            energy_var = 0.0

        # Cluster analysis (connected components via flood fill)
        largest_cluster, num_clusters = self._analyze_clusters(world)

        # Neighbor count
        neighbor_counts = []
        for cell in alive_cells:
            neighbors = world.get_neighbors(cell)
            neighbor_counts.append(len(neighbors))
        avg_neighbors = float(np.mean(neighbor_counts)) if neighbor_counts else 0.0

        # Signal metrics
        if alive_cells:
            signals = np.array([c.signal_output for c in alive_cells])
            avg_signal_mag = float(np.mean(np.linalg.norm(signals, axis=1)))
            signal_var = float(np.mean(np.var(signals, axis=0)))
        else:
            avg_signal_mag = 0.0
            signal_var = 0.0

        # Distance between cells
        if len(alive_cells) > 1:
            positions = np.array([[c.pos.x, c.pos.y] for c in alive_cells])
            # Exact average pairwise distance is inexpensive for 128 cells.
            dists = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = abs(positions[i][0] - positions[j][0])
                    dy = abs(positions[i][1] - positions[j][1])
                    # Toroidal distance
                    dx = min(dx, world.width - dx)
                    dy = min(dy, world.height - dy)
                    dists.append(np.sqrt(dx * dx + dy * dy))
            avg_distance = float(np.mean(dists)) if dists else 0.0
        else:
            avg_distance = 0.0

        total_resource = float(np.sum(world.resource_field))

        # Spatial entropy (simplified)
        spatial_entropy = self._compute_spatial_entropy(world, alive_cells)

        # Signal diversity (entropy of signal distribution)
        signal_diversity = self._compute_signal_diversity(alive_cells)

        # Cell state diversity
        state_diversity = self._compute_state_diversity(alive_cells)

        record = {
            "step": world.step_count,
            "alive_cells": alive_count,
            "dead_cells": total_count - alive_count,
            "average_energy": round(avg_energy, 4),
            "energy_variance": round(energy_var, 4),
            "average_neighbor_count": round(avg_neighbors, 4),
            "largest_cluster_size": largest_cluster,
            "number_of_clusters": num_clusters,
            "average_signal_magnitude": round(avg_signal_mag, 4),
            "signal_variance": round(signal_var, 4),
            "average_distance_between_cells": round(avg_distance, 4),
            "resource_consumption": round(world.resource_consumed, 4),
            "resource_remaining": round(total_resource, 4),
            "resource_regenerated": round(world.resource_regenerated, 4),
            "spatial_entropy": round(spatial_entropy, 4),
            "signal_diversity": round(signal_diversity, 4),
            "cell_state_diversity": round(state_diversity, 4),
        }

        self.records.append(record)
        return record

    def _analyze_clusters(self, world: World) -> tuple:
        """Find connected components of living cells."""
        alive_cells = {(c.pos.x, c.pos.y): c for c in world.cells.values() if c.alive}
        visited = set()
        clusters = []

        for pos, cell in alive_cells.items():
            if pos in visited:
                continue
            # BFS
            cluster = []
            queue = [pos]
            while queue:
                p = queue.pop(0)
                if p in visited or p not in alive_cells:
                    continue
                visited.add(p)
                cluster.append(p)
                # Use the same Moore neighborhood as cell perception.
                for dx, dy in [
                    (-1, -1), (0, -1), (1, -1), (-1, 0),
                    (1, 0), (-1, 1), (0, 1), (1, 1),
                ]:
                    np_ = ((p[0] + dx) % world.width, (p[1] + dy) % world.height)
                    if np_ not in visited and np_ in alive_cells:
                        queue.append(np_)
            clusters.append(len(cluster))

        if not clusters:
            return 0, 0
        return max(clusters), len(clusters)

    def _compute_spatial_entropy(self, world: World, alive_cells: list) -> float:
        """Compute spatial entropy of cell distribution."""
        if not alive_cells:
            return 0.0

        # Bin positions into grid cells
        bin_size = 4
        bins = Counter()
        for cell in alive_cells:
            bx = cell.pos.x // bin_size
            by = cell.pos.y // bin_size
            bins[(bx, by)] += 1

        total = len(alive_cells)
        entropy = 0.0
        for count in bins.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _compute_signal_diversity(self, alive_cells: list) -> float:
        """Compute diversity of cell signals."""
        if not alive_cells or len(alive_cells) < 2:
            return 0.0

        signals = np.array([c.signal_output for c in alive_cells])
        signals = signals[np.linalg.norm(signals, axis=1) > 1e-12]
        if len(signals) < 2:
            return 0.0
        # Compute pairwise cosine distances
        norms = np.linalg.norm(signals, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = signals / norms

        # Average pairwise cosine similarity
        sim_matrix = normalized @ normalized.T
        n = len(signals)
        avg_sim = (np.sum(sim_matrix) - n) / (n * (n - 1)) if n > 1 else 0.0
        return float(1.0 - avg_sim)  # Diversity = 1 - similarity

    def _compute_state_diversity(self, alive_cells: list) -> float:
        """Compute diversity of cell internal states."""
        if not alive_cells or len(alive_cells) < 2:
            return 0.0

        states = np.array([c.internal_state for c in alive_cells])
        # Compute variance across cells for each dimension
        dim_var = np.var(states, axis=0)
        return float(np.mean(dim_var))

    def save_csv(self, path: Path):
        """Save all records to CSV."""
        if not self.records:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

    def save_json(self, path: Path):
        """Save all records to JSON."""
        with open(path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all records."""
        if not self.records:
            return {}

        df = {}
        for key in self.records[0]:
            values = [r[key] for r in self.records]
            df[key] = {
                "min": min(values),
                "max": max(values),
                "mean": float(np.mean(values)),
                "final": values[-1],
            }
        return df
