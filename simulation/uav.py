"""
UAV model aligned with Section II-A of the paper.

Each UAV has position, energy, and status (idle/active).
Energy consumption: E_fly(d) = P_fly * d / v_c,  E_search = P_search * T_search.
"""

import numpy as np
from .config import (
    CRUISE_SPEED, P_FLY, P_SEARCH, E_INIT, E_RESERVE,
    T_SEARCH_BY_TERRAIN, T_SEARCH_DEFAULT,
)


class UAV:
    """Single UAV state."""

    def __init__(self, uav_id: int, position: np.ndarray):
        self.id = uav_id
        self.position = position.copy().astype(np.float64)  # (x, y) in metres
        self.energy = E_INIT
        self.status = "idle"          # "idle" or "active"
        self.assigned_cell = None     # (row, col) or None
        self.arrival_time = None      # sim time when UAV finishes current task
        # Once energy is too low to search any remaining cell, the UAV
        # is permanently grounded at base. Setting retired=True removes
        # it from the simulator's batch-dispatch accounting so the sim
        # does not wait for the impossible "full batch idle" condition.
        self.retired = False

    # ── Energy calculations (Eq. 2) ──────────────────────────────────
    @staticmethod
    def energy_fly(distance: float) -> float:
        """E^fly(d) = P_fly * d / v_c."""
        return P_FLY * distance / CRUISE_SPEED

    @staticmethod
    def energy_search(terrain_type: str = None) -> float:
        """E^search(c) = P_search * T_search(c), terrain-dependent."""
        t_s = T_SEARCH_BY_TERRAIN.get(terrain_type, T_SEARCH_DEFAULT) if terrain_type else T_SEARCH_DEFAULT
        return P_SEARCH * t_s

    @staticmethod
    def search_time_for_terrain(terrain_type: str) -> float:
        """T_search(c) determined by cell terrain complexity."""
        return T_SEARCH_BY_TERRAIN.get(terrain_type, T_SEARCH_DEFAULT)

    @staticmethod
    def travel_time(distance: float) -> float:
        """t_ik = d_ik / v_c."""
        return distance / CRUISE_SPEED

    # ── Feasibility check (Eq. 5 / constraint 2b) ───────────────────
    def can_reach_and_return(self, dist_to_cell: float,
                             dist_cell_to_base: float,
                             terrain_type: str = None) -> bool:
        """Check energy constraint:
        E_i - E_fly(d_ik) - E_search(c) >= E_fly(d_k,base) + E_reserve.
        """
        cost = (self.energy_fly(dist_to_cell)
                + self.energy_search(terrain_type)
                + self.energy_fly(dist_cell_to_base)
                + E_RESERVE)
        return self.energy >= cost

    # ── Actions ──────────────────────────────────────────────────────
    def dispatch(self, cell_rc: tuple, cell_centre: np.ndarray,
                 current_time: float, terrain_type: str = None) -> float:
        """Dispatch UAV to a cell.  Returns the time when search completes."""
        dist = float(np.linalg.norm(self.position - cell_centre))
        fly_time = self.travel_time(dist)
        t_search = self.search_time_for_terrain(terrain_type) if terrain_type else T_SEARCH_DEFAULT
        total_time = fly_time + t_search

        self.status = "active"
        self.assigned_cell = cell_rc
        self.arrival_time = current_time + total_time

        # Energy consumed
        self.energy -= self.energy_fly(dist) + self.energy_search(terrain_type)
        # Position updated to cell centre after task
        self.position = cell_centre.copy()

        return self.arrival_time

    def complete_task(self):
        """Mark task as complete; UAV becomes idle."""
        self.status = "idle"
        self.assigned_cell = None
        self.arrival_time = None

    def recall_to_base(self, base_position: np.ndarray, current_time: float):
        """Recall UAV to GCS (energy depleted). UAV is permanently grounded."""
        dist = float(np.linalg.norm(self.position - base_position))
        self.energy -= self.energy_fly(dist)
        self.position = base_position.copy()
        self.status = "idle"
        self.assigned_cell = None
        self.arrival_time = None
        self.retired = True

    def __repr__(self):
        return (f"UAV(id={self.id}, pos=({self.position[0]:.0f},"
                f"{self.position[1]:.0f}), E={self.energy:.0f}J, "
                f"status={self.status})")
