"""
28×28 grid map for the SAR simulation.

Terrain layout reconstructed from the paper figure (grid_map_v3.png).
Coordinate convention: (row, col) with row 0 = south, row 27 = north.
Origin (0,0) is the south-west corner.

Terrain types:
  M  = meadow (open ground)
  S  = sparse vegetation
  F  = dense forest
  R  = rocky terrain
  X  = impassable zone
  T  = trail
  W  = creek (water)
  P  = pond
"""

import numpy as np
from .config import (
    GRID_ROWS, GRID_COLS, CELL_SIZE,
    TERRAIN_MEADOW, TERRAIN_SPARSE_VEG, TERRAIN_DENSE_FOREST,
    TERRAIN_ROCKY, TERRAIN_TRAIL, TERRAIN_CREEK,
    TERRAIN_IMPASSABLE, TERRAIN_POND, SENSOR_PARAMS,
)

# ── Shorthand ────────────────────────────────────────────────────────
M = TERRAIN_MEADOW
S = TERRAIN_SPARSE_VEG
F = TERRAIN_DENSE_FOREST
R = TERRAIN_ROCKY
X = TERRAIN_IMPASSABLE
T = TERRAIN_TRAIL
W = TERRAIN_CREEK
P = TERRAIN_POND

# ── Terrain grid ─────────────────────────────────────────────────────
# Each row is south-to-north (row 0 = bottom of map).
# Carefully reconstructed from the figure:
#   - Rocky ridge along the north (top rows)
#   - Dense forest patches in the west and central-north
#   - Impassable zones: rock face & crevasses (~col 12-17, row 24-27),
#     collapsed cliff face (~col 10-14, row 18-22),
#     flooded ravine (~col 3-7, row 10-13),
#     landslide debris (~col 22-26, row 10-13)
#   - Main trail from GCS (row 0-1, col ~13-14) north to trail fork (~row 16)
#   - West branch trail toward cabin (~row 25, col 7)
#   - East branch trail toward overlook (~row 24, col 21)
#   - Perennial creek runs roughly col 15-16 from south, bending west
#   - Seasonal creek in NW corner
#   - Pond at ~(row 14, col 9-10)
#   - Clearing at ~(row 18, col 1-2)

_TERRAIN_ROWS = [
    # Row 0 (southernmost)
    [S, S, S, S, S, S, S, S, S, S, S, S, T, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 1 — GCS entry area
    [S, S, S, S, S, S, S, S, S, S, S, S, T, T, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 2
    [S, S, S, S, S, S, S, S, S, S, S, S, T, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 3
    [S, S, S, S, S, S, S, S, S, S, S, T, T, S, S, W, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 4
    [S, S, S, S, S, M, M, M, M, S, S, T, S, S, S, W, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 5
    [M, M, M, M, S, M, M, M, M, S, S, T, S, S, S, W, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 6
    [M, M, M, M, S, S, S, S, S, S, T, T, S, S, S, S, W, S, S, S, S, S, S, S, S, S, S, S],
    # Row 7
    [M, M, M, M, S, S, S, S, S, S, T, S, S, S, S, S, W, S, S, S, S, S, S, S, S, S, S, S],
    # Row 8
    [S, S, S, S, S, S, S, S, S, T, T, S, S, S, S, S, W, S, S, S, S, S, S, S, S, S, S, S],
    # Row 9
    [S, S, S, S, S, S, S, S, S, T, S, S, S, S, S, S, W, S, S, S, S, S, S, S, S, S, S, S],
    # Row 10
    [S, S, S, X, X, X, S, S, S, T, S, S, S, S, S, W, S, S, S, S, S, S, X, X, X, S, S, S],
    # Row 11 — flooded ravine (west), landslide debris (east)
    [S, S, S, X, X, X, X, S, S, T, T, S, S, S, S, W, S, S, S, S, S, S, X, X, X, X, S, S],
    # Row 12
    [S, S, S, X, X, X, X, S, S, T, S, S, S, S, W, S, S, S, S, S, S, X, X, X, X, X, S, S],
    # Row 13
    [S, S, S, S, X, X, S, S, S, T, S, S, S, S, W, S, S, S, S, S, S, S, X, X, X, S, S, S],
    # Row 14 — pond area
    [S, S, S, S, S, S, S, S, P, P, T, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 15
    [S, S, S, S, S, S, S, S, P, P, T, T, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S],
    # Row 16 — trail fork; east branch extended to col 17
    [S, S, S, F, F, F, S, S, S, T, T, T, T, T, T, T, T, T, S, S, S, S, S, S, S, S, S, S],
    # Row 17 — east branch turns south along col 17
    [S, S, F, F, F, F, F, S, S, T, S, S, X, X, S, S, S, T, S, S, S, S, S, S, S, S, S, S],
    # Row 18 — clearing (west), collapsed cliff face (centre)
    [M, M, S, F, F, F, F, S, T, T, S, X, X, X, X, S, S, T, S, S, S, S, S, S, S, S, S, S],
    # Row 19
    [M, M, S, S, F, F, F, S, T, S, X, X, X, X, S, S, S, T, S, S, S, S, S, S, S, S, S, S],
    # Row 20 — campsite clearing on east branch route
    [S, S, S, S, F, F, F, T, T, S, X, X, X, S, S, S, S, T, M, M, X, X, S, S, S, S, S, S],
    # Row 21
    [S, S, S, S, S, F, F, T, S, S, X, X, S, S, S, S, S, T, S, X, X, X, T, T, S, S, S, S],
    # Row 22
    [S, S, S, S, S, F, T, T, S, F, F, X, X, S, S, S, S, T, S, X, X, T, T, S, S, S, S, R],
    # Row 23 — seasonal creek zone; east branch meets the NE trail here
    [S, S, W, W, S, T, T, F, F, F, F, X, X, X, S, S, S, T, T, T, T, T, S, S, S, S, R, R],
    # Row 24 — overlook (24,21); (24,20) bridges to the NE corner trail
    [R, S, S, W, W, T, F, F, F, F, F, X, X, X, X, S, S, T, T, S, T, S, S, S, S, S, R, R],
    # Row 25 — cabin area (west), rock face (centre), overlook (east)
    [R, R, S, S, W, T, T, T, F, F, X, X, X, X, X, X, T, T, S, S, S, T, T, S, S, S, R, R],
    # Row 26
    [R, R, R, S, S, S, S, F, F, X, X, X, X, X, X, T, T, S, S, S, T, T, S, S, S, R, R, R],
    # Row 27 (northernmost — rocky ridge)
    [R, R, R, R, S, S, F, F, X, X, X, X, X, X, T, T, S, S, S, T, T, S, S, S, R, R, R, R],
]


# ── Elevation map (approximate, in metres) ──────────────────────────
# Valley floor ~500 m, ridge ~900 m, trail fork ~600 m
def _build_elevation():
    """Simple elevation model: rises toward north and east edges."""
    elev = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Base: rises linearly northward
            base = 500.0 + (r / GRID_ROWS) * 300.0
            # East ridge bump
            if c >= 22:
                base += 80.0
            # West forest area is slightly lower
            if c <= 6 and r >= 16:
                base -= 40.0
            # Impassable zones (cliff faces) are higher
            if _TERRAIN_ROWS[r][c] == X:
                base += 100.0
            elev[r, c] = base
    return elev


# ── Named landmarks (row, col) ──────────────────────────────────────
LANDMARKS = {
    "gcs":          (1, 13),
    "trail_fork":   (16, 10),
    "overlook":     (24, 21),
    "caves":        (25, 27),
    "pond":         (14, 9),
    "clearing":     (18, 1),
    "campsite":     (20, 18),
}


class GridMap:
    """Immutable 28×28 search-region representation."""

    def __init__(self):
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        self.cell_size = CELL_SIZE
        self.num_cells = GRID_ROWS * GRID_COLS

        # terrain_grid[r][c] -> terrain string
        self.terrain_grid = _TERRAIN_ROWS  # list of lists

        # elevation[r, c] in metres
        self.elevation = _build_elevation()

        # Pre-compute cell centres in metres (origin at SW corner)
        # p_k = ((col + 0.5) * cell_size, (row + 0.5) * cell_size)
        self.cell_centres = np.zeros((GRID_ROWS, GRID_COLS, 2), dtype=np.float32)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                self.cell_centres[r, c] = [
                    (c + 0.5) * CELL_SIZE,
                    (r + 0.5) * CELL_SIZE,
                ]

        self.landmarks = LANDMARKS

    # ── Helpers ──────────────────────────────────────────────────────
    def terrain(self, r, c) -> str:
        return self.terrain_grid[r][c]

    def is_traversable(self, r, c) -> bool:
        return self.terrain_grid[r][c] != X

    def sensor_params(self, r, c):
        """Return (p_d, p_f) for cell (r, c)."""
        return SENSOR_PARAMS[self.terrain_grid[r][c]]

    def cell_centre(self, r, c) -> np.ndarray:
        """Return (x, y) in metres."""
        return self.cell_centres[r, c]

    def cell_index(self, r, c) -> int:
        """Flat index k = r * cols + c."""
        return r * self.cols + c

    def index_to_rc(self, k) -> tuple:
        return divmod(k, self.cols)

    def distance(self, rc1, rc2) -> float:
        """Euclidean distance between cell centres in metres."""
        p1 = self.cell_centre(*rc1)
        p2 = self.cell_centre(*rc2)
        return float(np.linalg.norm(p1 - p2))

    def neighbours(self, r, c):
        """Return list of traversable (r, c) neighbours (4-connected)."""
        nbrs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.is_traversable(nr, nc):
                    nbrs.append((nr, nc))
        return nbrs

    def get_trail_cells(self):
        """Return list of (r, c) that are trail cells."""
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols)
                if self.terrain_grid[r][c] == T]

    def get_creek_cells(self):
        """Return list of (r, c) that are creek or pond cells."""
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols)
                if self.terrain_grid[r][c] in (W, P)]

    def get_traversable_cells(self):
        """Return list of all traversable (r, c)."""
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols)
                if self.is_traversable(r, c)]

    def terrain_summary(self) -> str:
        """Compact text summary for LLM prompt.

        Each row is prefixed with its row index (27=north, 0=south) and
        column indices are printed as an aligned header so the LLM can
        look up any (row, col) without counting across the grid.
        """
        header = "Col:    " + " ".join(f"{c:2d}" for c in range(self.cols))
        lines = [header]
        for r in range(self.rows - 1, -1, -1):  # north first
            row_str = " ".join(
                f" {self.terrain_grid[r][c][0].upper()}"
                for c in range(self.cols)
            )
            lines.append(f"Row {r:2d}: {row_str}")
        return "\n".join(lines)

    def cells_by_terrain_text(self) -> str:
        """List cells grouped by terrain type (sparse_vegetation is the default)."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in range(self.rows):
            for c in range(self.cols):
                buckets[self.terrain_grid[r][c]].append((r, c))
        order = ["trail", "impassable", "dense_forest", "rocky",
                 "meadow", "creek", "pond"]
        lines = ["Cells by terrain (all other cells are sparse vegetation):"]
        for t in order:
            if t not in buckets:
                continue
            cells = buckets[t]
            coords = ", ".join(f"({r},{c})" for r, c in cells)
            pretty = t.replace("_", " ")
            lines.append(f"  {pretty} ({len(cells)}): {coords}")
        return "\n".join(lines)

    def elevation_summary(self) -> str:
        """Elevation grid (metres), aligned with ``terrain_summary``."""
        header = "Col:    " + " ".join(f"{c:>3d}" for c in range(self.cols))
        lines = [header]
        for r in range(self.rows - 1, -1, -1):
            row_str = " ".join(
                f"{int(round(self.elevation[r, c])):>3d}"
                for c in range(self.cols)
            )
            lines.append(f"Row {r:2d}: {row_str}")
        return "\n".join(lines)
