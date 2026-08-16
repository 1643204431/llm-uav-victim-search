"""
Target (lost person) movement model — pre-planned continuous trajectories.

Each movement unit generates a complete trajectory at simulation start:
  Phase 1: Walk from trail fork toward destination along weighted-shortest path
  Phase 2: Wander near destination in small radius

All 5 targets belong to 3 movement units; members of the same unit share
one trajectory (same position at all times).

The full route starts from GCS, goes along the main trail to trail fork,
then each unit diverges to its destination. Since the group was together
before the trail fork, trajectory generation starts from the trail fork
(the separation point).
"""

import heapq
import numpy as np

from .config import (
    CELL_SIZE, GRID_ROWS, GRID_COLS,
    MOVEMENT_UNITS, TARGET_PROFILES, NUM_TARGETS,
    TRAJECTORY_DT, MAX_SIM_TIME,
    TRAIL_WEIGHT, NON_TRAIL_WEIGHT,
)
from .grid_map import GridMap, TERRAIN_TRAIL


# ══════════════════════════════════════════════════════════════════════
#  Weighted A* pathfinding (people prefer trails)
# ══════════════════════════════════════════════════════════════════════

def _cell_cost(grid_map: GridMap, r: int, c: int) -> float:
    """Movement cost for entering cell (r, c)."""
    if grid_map.terrain(r, c) == TERRAIN_TRAIL:
        return TRAIL_WEIGHT
    return NON_TRAIL_WEIGHT


def weighted_astar(grid_map: GridMap, start: tuple, goal: tuple) -> list:
    """A* search with terrain-weighted costs. Returns list of (r,c) cells."""
    open_set = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    goal_pos = np.array(grid_map.cell_centre(*goal))

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for nr, nc in grid_map.neighbours(*current):
            cost = _cell_cost(grid_map, nr, nc)
            tentative_g = g_score[current] + cost

            if tentative_g < g_score.get((nr, nc), float("inf")):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                # Heuristic: Euclidean distance in cell units
                h = np.linalg.norm(
                    np.array(grid_map.cell_centre(nr, nc)) - goal_pos
                ) / CELL_SIZE
                heapq.heappush(open_set, (tentative_g + h, (nr, nc)))

    return []  # no path found


# ══════════════════════════════════════════════════════════════════════
#  Trajectory generation
# ══════════════════════════════════════════════════════════════════════

def _cells_to_waypoints(grid_map: GridMap, cell_path: list) -> np.ndarray:
    """Convert cell path to array of (x, y) waypoints (cell centres)."""
    waypoints = np.array([grid_map.cell_centre(*rc) for rc in cell_path],
                         dtype=np.float64)
    return waypoints


def _sample_trajectory_along_path(waypoints: np.ndarray, speed: float,
                                   dt: float) -> tuple:
    """
    Sample (x, y) positions at intervals of dt seconds along waypoints.

    Returns:
        positions: ndarray of shape (N, 2)
        arrival_time: float, time when the last waypoint is reached
    """
    # Compute cumulative distances along waypoints
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_dist = cum_dist[-1]
    arrival_time = total_dist / speed if speed > 0 else 0.0

    positions = []
    t = 0.0
    while t <= arrival_time:
        d = speed * t
        # Find which segment we're on
        idx = np.searchsorted(cum_dist, d, side="right") - 1
        idx = min(idx, len(seg_lengths) - 1)
        idx = max(idx, 0)

        if seg_lengths[idx] > 0:
            frac = (d - cum_dist[idx]) / seg_lengths[idx]
            frac = np.clip(frac, 0.0, 1.0)
        else:
            frac = 0.0

        pos = waypoints[idx] + frac * diffs[idx]
        positions.append(pos)
        t += dt

    return np.array(positions, dtype=np.float64), arrival_time


def _generate_wander(centre: np.ndarray, radius: float, speed: float,
                     dt: float, duration: float,
                     grid_map: GridMap,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate wandering trajectory around a centre point."""
    positions = []
    pos = centre.copy()
    t = 0.0

    while t < duration:
        # Random direction
        angle = rng.uniform(0, 2 * np.pi)
        step = speed * dt
        new_pos = pos + step * np.array([np.cos(angle), np.sin(angle)])

        # Stay within wander radius
        if np.linalg.norm(new_pos - centre) > radius:
            # Reflect back toward centre
            new_pos = pos + step * (centre - pos) / max(
                np.linalg.norm(centre - pos), 1e-6)

        # Stay within grid bounds
        new_pos[0] = np.clip(new_pos[0], 0, GRID_COLS * CELL_SIZE)
        new_pos[1] = np.clip(new_pos[1], 0, GRID_ROWS * CELL_SIZE)

        # Check traversability of the cell we'd enter
        c = int(new_pos[0] / CELL_SIZE)
        r = int(new_pos[1] / CELL_SIZE)
        c = min(c, GRID_COLS - 1)
        r = min(r, GRID_ROWS - 1)
        if grid_map.is_traversable(r, c):
            pos = new_pos
        # else: stay at current pos

        positions.append(pos.copy())
        t += dt

    return np.array(positions, dtype=np.float64) if positions else centre.reshape(1, 2)


def generate_unit_trajectory(grid_map: GridMap, unit_id: int,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Generate the full trajectory for a movement unit.

    Returns:
        trajectory: ndarray of shape (T, 2), sampled at TRAJECTORY_DT intervals.
                    trajectory[i] = (x, y) position at time i * TRAJECTORY_DT.
    """
    unit = MOVEMENT_UNITS[unit_id]
    speed = unit["speed"]
    dest_name = unit["destination"]
    wander_radius = unit["wander_radius"]
    wander_speed = unit["wander_speed"]

    # Start at trail fork (separation point)
    trail_fork = grid_map.landmarks["trail_fork"]
    dest_rc = grid_map.landmarks[dest_name]

    # Find weighted path from trail fork to destination
    cell_path = weighted_astar(grid_map, trail_fork, dest_rc)
    if not cell_path:
        # Fallback: stay at trail fork
        n_samples = int(MAX_SIM_TIME / TRAJECTORY_DT) + 1
        pos = grid_map.cell_centre(*trail_fork)
        return np.tile(pos, (n_samples, 1))

    waypoints = _cells_to_waypoints(grid_map, cell_path)

    # Phase 1: travel along path
    phase1_traj, arrival_time = _sample_trajectory_along_path(
        waypoints, speed, TRAJECTORY_DT
    )

    # Phase 2: wander near destination
    remaining_time = MAX_SIM_TIME - arrival_time
    if remaining_time > 0:
        dest_pos = grid_map.cell_centre(*dest_rc)
        phase2_traj = _generate_wander(
            dest_pos, wander_radius, wander_speed,
            TRAJECTORY_DT, remaining_time, grid_map, rng
        )
    else:
        phase2_traj = np.empty((0, 2))

    # Concatenate phases
    trajectory = np.concatenate([phase1_traj, phase2_traj], axis=0)
    return trajectory


# ══════════════════════════════════════════════════════════════════════
#  Target class
# ══════════════════════════════════════════════════════════════════════

class Target:
    """A single lost person. Position is looked up from the unit trajectory."""

    def __init__(self, target_id: int, unit_id: int,
                 trajectory: np.ndarray):
        self.id = target_id
        self.unit_id = unit_id
        self.name = TARGET_PROFILES[target_id]["name"]
        self.trajectory = trajectory  # shape (T, 2)
        self.found = False
        self.discovery_time = None

    def position_at(self, t: float) -> np.ndarray:
        """Return (x, y) position at time t via linear interpolation."""
        idx_f = t / TRAJECTORY_DT
        idx = int(idx_f)
        max_idx = len(self.trajectory) - 1

        if idx >= max_idx:
            return self.trajectory[max_idx].copy()

        frac = idx_f - idx
        return (1.0 - frac) * self.trajectory[idx] + frac * self.trajectory[idx + 1]

    def cell_at(self, t: float) -> tuple:
        """Return (row, col) cell at time t."""
        pos = self.position_at(t)
        c = int(pos[0] / CELL_SIZE)
        r = int(pos[1] / CELL_SIZE)
        c = min(max(c, 0), GRID_COLS - 1)
        r = min(max(r, 0), GRID_ROWS - 1)
        return (r, c)

    def __repr__(self):
        status = f"FOUND@{self.discovery_time:.0f}s" if self.found else "unfound"
        return f"Target({self.id}, {self.name}, {status})"


# ══════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════

def create_targets(grid_map: GridMap, rng: np.random.Generator) -> list:
    """Create all M=5 targets with pre-computed trajectories."""
    # Generate one trajectory per movement unit
    unit_trajectories = {}
    for uid in MOVEMENT_UNITS:
        unit_trajectories[uid] = generate_unit_trajectory(grid_map, uid, rng)

    # Create targets, each pointing to its unit's trajectory
    targets = []
    for tid in range(NUM_TARGETS):
        uid = TARGET_PROFILES[tid]["unit"]
        traj = unit_trajectories[uid]
        targets.append(Target(tid, uid, traj))

    return targets
