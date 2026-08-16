"""
Perception module (Section III-A).

At each epoch, converts raw telemetry into a structured state snapshot:
- Per-UAV: energy, position
- Occupied-cell set O(t)
- Per-cell: travel distances, search statistics (n_k, mu_k, tau_k)
"""

import numpy as np
from .grid_map import GridMap
from .config import GCS_CELL, CELL_SIZE


class PerceptionModule:
    """Computes the structured state snapshot at each decision epoch."""

    def __init__(self, grid_map: GridMap):
        self.grid = grid_map
        self.base_pos = grid_map.cell_centre(*GCS_CELL)

    def compute_state(self, uavs, search_stats, current_time):
        """
        Args:
            uavs: list of UAV objects
            search_stats: dict (r,c) -> {"count": int, "last_time": float}
            current_time: current simulation time in seconds

        Returns:
            dict with keys:
                idle_uavs: list of idle UAV objects
                occupied_cells: set of (r,c) currently assigned to active UAVs
                cell_stats: dict (r,c) -> {n_k, mu_k, tau_k, p_d, p_f}
                distances: dict (uav_id, (r,c)) -> d_ik
                base_distances: dict (r,c) -> d_k_base
        """
        idle_uavs = [u for u in uavs if u.status == "idle"]
        occupied = set()
        for u in uavs:
            if u.status == "active" and u.assigned_cell is not None:
                occupied.add(u.assigned_cell)

        # Per-cell statistics
        cell_stats = {}
        traversable = self.grid.get_traversable_cells()
        for rc in traversable:
            p_d, p_f = self.grid.sensor_params(*rc)
            stats = search_stats.get(rc, {"count": 0, "last_time": 0.0})
            n_k = stats["count"]
            mu_k = (1.0 - p_d) ** n_k  # cumulative miss factor
            tau_k = current_time - stats["last_time"] if n_k > 0 else current_time
            cell_stats[rc] = {
                "n_k": n_k,
                "mu_k": mu_k,
                "tau_k": tau_k,
                "p_d": p_d,
                "p_f": p_f,
            }

        # Distances from each idle UAV to each traversable cell
        distances = {}
        base_distances = {}
        for rc in traversable:
            cell_pos = self.grid.cell_centre(*rc)
            base_distances[rc] = float(np.linalg.norm(cell_pos - self.base_pos))
            for u in idle_uavs:
                d = float(np.linalg.norm(u.position - cell_pos))
                distances[(u.id, rc)] = d

        return {
            "idle_uavs": idle_uavs,
            "occupied_cells": occupied,
            "cell_stats": cell_stats,
            "distances": distances,
            "base_distances": base_distances,
            "current_time": current_time,
        }
