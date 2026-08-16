"""
Configuration for Multi-UAV SAR Simulation.
All parameters aligned with the paper formulation.
"""
import os

import numpy as np


# ── Grid / Search Region ─────────────────────────────────────────────
# Paper: search area discretised into K = GRID_ROWS × GRID_COLS cells.
GRID_ROWS = 28
GRID_COLS = 28
CELL_SIZE = 250.0          # metres per cell edge
REGION_SIZE = GRID_ROWS * CELL_SIZE   # 7 000 m  (paper: 7 km × 7 km)

# ── Terrain types ────────────────────────────────────────────────────
TERRAIN_MEADOW      = "meadow"
TERRAIN_SPARSE_VEG  = "sparse_vegetation"
TERRAIN_DENSE_FOREST = "dense_forest"
TERRAIN_ROCKY       = "rocky"
TERRAIN_TRAIL       = "trail"
TERRAIN_CREEK       = "creek"
TERRAIN_IMPASSABLE  = "impassable"
TERRAIN_POND        = "pond"

# Sensor parameters per terrain type: (p_d, p_f)
# Trail and creek use meadow parameters (as stated in paper)
SENSOR_PARAMS = {
    TERRAIN_MEADOW:       (0.95, 0.02),
    TERRAIN_SPARSE_VEG:   (0.80, 0.05),
    TERRAIN_DENSE_FOREST: (0.55, 0.10),
    TERRAIN_ROCKY:        (0.65, 0.12),
    TERRAIN_TRAIL:        (0.95, 0.02),
    TERRAIN_CREEK:        (0.95, 0.02),
    TERRAIN_IMPASSABLE:   (0.00, 0.00),   # cannot search
    TERRAIN_POND:         (0.95, 0.02),
}

# ── UAV Fleet ────────────────────────────────────────────────────────
# Paper symbol ↔ code identifier:
#   N       ↔ NUM_UAVS
#   B       ↔ BATCH_SIZE          (drones-per-cycle dispatch batch)
#   v_c     ↔ CRUISE_SPEED
#   P_f     ↔ P_FLY
#   P_d     ↔ P_SEARCH
#   E_init  ↔ E_INIT
#   E_buf   ↔ E_RESERVE            (safety-margin energy reserved for return)
NUM_UAVS        = 10       # N
BATCH_SIZE      = 10       # B (main configuration: dispatch full fleet)
CRUISE_SPEED    = 15.0     # v_c  [m/s]
P_FLY           = 200.0    # P_f, flight power   [W]
P_SEARCH        = 150.0    # P_d, search power   [W]
E_INIT          = 500_000.0  # E_init, initial energy  [J]  (500 kJ)
E_RESERVE       = 30_000.0   # E_buf,  safety reserve  [J]  (30 kJ)

# Search time per cell T_s(c) [s], determined by terrain complexity
T_SEARCH_BY_TERRAIN = {
    "meadow":            60.0,   # open ground, fast sweep
    "sparse_vegetation": 80.0,   # light brush, slightly slower
    "dense_forest":     120.0,   # thick canopy, slow careful scan
    "rocky":            100.0,   # uneven ground, moderate
    "trail":             50.0,   # narrow corridor, quick pass
    "creek":             70.0,   # water feature, moderate
    "pond":              60.0,   # open water edge, fast
    "impassable":         0.0,   # cannot search
}
T_SEARCH_DEFAULT = 90.0    # fallback for unknown terrain
EPSILON         = 0.01     # numerical stability for search value
ALPHA           = 1.0      # priority exponent in search value
BETA            = 0.1      # distance exponent in search value (lower = less distance penalty)

# ── GCS / Base ───────────────────────────────────────────────────────
# Ground Control Station at the southern trailhead.
# Paper symbol: p_base ↔ centre of GCS_CELL.
GCS_CELL = (1, 13)  # (row, col), 0-indexed

# ── Targets ──────────────────────────────────────────────────────────
# Paper uses "victim"; code uses "target". M ↔ NUM_TARGETS.
NUM_TARGETS = 5            # M

# Movement units: members of the same unit share one trajectory.
# Paper §IV-A names these "Family A", "Family B", "Youth C".
MOVEMENT_UNITS = {
    0: {
        "name": "Family A",
        "members": [0, 1],          # Father A + Child A
        "speed": 0.5,               # m/s — limited by child
        "destination": "pond",      # (14, 9)
        "wander_radius": 300.0,     # metres, after arrival
        "wander_speed": 0.2,        # m/s, after arrival
    },
    1: {
        "name": "Family B",
        "members": [2, 3],          # Mother B + Father B
        "speed": 0.8,               # m/s
        "destination": "campsite",  # (20, 18) — clearing on east branch route
        "wander_radius": 400.0,
        "wander_speed": 0.3,
    },
    2: {
        "name": "Youth C",
        "members": [4],             # Youth C
        "speed": 1.0,               # m/s
        "destination": "overlook",  # (24, 21)
        "wander_radius": 500.0,
        "wander_speed": 0.4,
    },
}

# Target profiles: individual person info (for LLM context)
TARGET_PROFILES = {
    0: {"name": "Father A",      "unit": 0},
    1: {"name": "Child A (age 7)", "unit": 0},
    2: {"name": "Mother B",      "unit": 1},
    3: {"name": "Father B",      "unit": 1},
    4: {"name": "Youth C",       "unit": 2},
}

# ─── Initial contextual clue at t=0 ──────────────────────────────────
# Paper symbol: L (the contextual-clue set provided to the LLM agent at
# mission start). The narrative below is the §IV-A scenario briefing.
INITIAL_CLUE = (
    "A hiking group of 5 people departed from the southern trailhead this morning at 08:30, "
    "heading for the overlook in the northeast of the region as their shared destination. "
    "They were expected back by 5 PM but have not returned and are now "
    "unreachable by phone. "
    "Group composition: Family A (a parent in their 40s with a 7-year-old child; "
    "the young child limits how far the family can travel), "
    "Family B (a married couple in their early 30s, moderately fit), "
    "and Youth C (a 22-year-old experienced solo hiker with strong stamina). "
    "Last visual sighting: a passing hiker saw all five at the trail fork around 10:00 AM; "
    "no sightings since."
)

# ── Trajectory generation ────────────────────────────────────────────
TRAJECTORY_DT = 10.0       # sampling interval for trajectories [s]
MAX_SIM_TIME = 120000.0    # 2000 minutes
TRAIL_WEIGHT = 1.0         # path cost for trail cells
NON_TRAIL_WEIGHT = 5.0     # path cost for other traversable cells

# ── Simulation timing ────────────────────────────────────────────
SIM_START_OFFSET = 3600.0  # seconds (60 min): UAVs dispatched this long
                           # after targets begin moving

# ── LLM ──────────────────────────────────────────────────────────────
LLM_PROVIDER = "deepseek"     # "deepseek" | "qwen" | "hunyuan"

# The default/anchor run uses DeepSeek-V4-Pro without reasoning effort.
# DeepSeek-V4-Flash is evaluated separately in Exp. 4; both formal DeepSeek
# configurations run in non-thinking mode.
LLM_MODEL_DEEPSEEK = "deepseek-v4-pro"
LLM_THINKING_DEEPSEEK = False
LLM_REASONING_EFFORT_DEEPSEEK = "high"
DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Qwen (Aliyun Dashscope OpenAI-compatible endpoint).
# Pin the official snapshot instead of the rolling ``qwen-plus`` alias so
# every run records the exact server-side version requested.
LLM_MODEL_QWEN = "qwen-plus-2025-12-01"
LLM_THINKING_QWEN = False
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Hunyuan (Tencent OpenAI-compatible endpoint).
LLM_MODEL_HUNYUAN = "hunyuan-turbo"
HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"

# Max number of priority cells the LLM is asked to output per cycle.
# Paper symbol: K' (K' << K). Cells not listed are treated as w_k = 0.
# A lower value shortens LLM response length (faster API, cheaper
# tokens) and forces the LLM to focus on the highest-priority cells;
# too low and the optimizer may run out of feasible candidates.
TOP_K_PRIORITY_CELLS = 50  # K'

# ── Experiment ───────────────────────────────────────────────────────
NUM_RUNS = 10
METHODS = [
    "lawnmower",
    "greedy_coverage",
    "bayesian_search",
    "clue_prior_sweep",
    "clue_prior_greedy",
    "clue_prior_bayes",
    "ours",
]
SEED_BASE = 42
