"""
Baseline methods for comparison (Section IV).

Each baseline has its own independent assignment logic, without using
the proposed method's optimizer (solve_assignment).

Methods used in the formal comparison:
1. greedy_coverage    — nearest feasible unsearched cell
2. lawnmower          — fixed boustrophedon sweep
3. bayesian_search    — uniform prior + motion/observation updates
4. clue_prior_sweep   — clue prior orders the sweep strips once
5. clue_prior_greedy  — fixed clue prior + greedy dispatch
6. clue_prior_bayes   — clue prior + motion/observation updates

All six methods are deterministic, non-LLM baselines.  The clue-derived
prior uses only information exposed in the scenario briefing: the last
sighting, group mobility categories, elapsed time, terrain, water-seeking
heuristics, and the stated shared destination.  It never reads the simulated
targets' hidden destinations or trajectories.
"""

import numpy as np

from simulation.config import (
    GRID_ROWS, GRID_COLS, CELL_SIZE, CRUISE_SPEED,
    INITIAL_CLUE, GCS_CELL, SIM_START_OFFSET, T_SEARCH_BY_TERRAIN,
)
from simulation.grid_map import GridMap
from simulation.uav import UAV


def _is_feasible(uav: UAV, cell_rc: tuple, grid_map: GridMap,
                 base_pos: np.ndarray) -> bool:
    """Check if UAV can fly to cell, search, and return to base."""
    cell_pos = grid_map.cell_centre(*cell_rc)
    d_to = float(np.linalg.norm(uav.position - cell_pos))
    d_back = float(np.linalg.norm(cell_pos - base_pos))
    terrain_type = grid_map.terrain(*cell_rc)
    return uav.can_reach_and_return(d_to, d_back, terrain_type)


def greedy_coverage(grid_map: GridMap, state: dict, uavs_list) -> dict:
    """Greedy: each UAV picks the nearest unsearched feasible cell.

    UAVs are assigned one by one. Once a cell is assigned, it is
    removed from the candidate set to avoid duplication.
    """
    occupied = set(state["occupied_cells"])
    cell_stats = state["cell_stats"]
    base_pos = grid_map.cell_centre(1, 13)  # GCS

    # Candidate cells: traversable, not occupied, never searched (or low confidence)
    candidates = []
    for rc in grid_map.get_traversable_cells():
        if rc in occupied:
            continue
        candidates.append(rc)

    # Sort candidates by search count (prefer unsearched)
    candidates.sort(key=lambda rc: cell_stats.get(rc, {"n_k": 0})["n_k"])

    assigned_cells = set()
    assignments = {}

    for uav in uavs_list:
        best_cell = None
        best_dist = float("inf")

        for rc in candidates:
            if rc in assigned_cells:
                continue
            if not _is_feasible(uav, rc, grid_map, base_pos):
                continue

            d = float(np.linalg.norm(
                uav.position - grid_map.cell_centre(*rc)))

            # Prefer unsearched cells; among those, pick nearest
            n_k = cell_stats.get(rc, {"n_k": 0})["n_k"]
            # Effective distance: penalise already-searched cells
            eff_dist = d + n_k * 10000.0

            if eff_dist < best_dist:
                best_dist = eff_dist
                best_cell = rc

        if best_cell is not None:
            assignments[uav.id] = best_cell
            assigned_cells.add(best_cell)

    return assignments


def lawnmower(grid_map: GridMap, state: dict, uavs_list,
              sweep_state: dict = None) -> dict:
    """Systematic row-by-row sweep from south to north (boustrophedon).

    sweep_state tracks which cells remain to be swept.
    """
    if sweep_state is None:
        sweep_state = {"next_cells": []}

    occupied = set(state["occupied_cells"])
    base_pos = grid_map.cell_centre(1, 13)

    # Generate full sweep order on first call
    if not sweep_state["next_cells"]:
        cells = []
        for r in range(GRID_ROWS):
            row_cells = []
            for c in range(GRID_COLS):
                if grid_map.is_traversable(r, c):
                    row_cells.append((r, c))
            # Alternate direction (boustrophedon)
            if r % 2 == 1:
                row_cells.reverse()
            cells.extend(row_cells)
        sweep_state["next_cells"] = cells

    # Assign UAVs to the next cells in sweep order
    assignments = {}
    assigned_cells = set()

    cell_idx = 0
    for uav in uavs_list:
        while cell_idx < len(sweep_state["next_cells"]):
            rc = sweep_state["next_cells"][cell_idx]
            cell_idx += 1

            if rc in occupied or rc in assigned_cells:
                continue
            if not _is_feasible(uav, rc, grid_map, base_pos):
                continue

            assignments[uav.id] = rc
            assigned_cells.add(rc)
            break

    # Remove assigned cells from queue
    sweep_state["next_cells"] = [
        c for c in sweep_state["next_cells"]
        if c not in assigned_cells
    ]

    return assignments


# ═══════════════════════════════════════════════════════════════════════
#  Static clue prior and Bayesian-filter baselines
# ══════════════════════════════════════════════════════════════════════

# These profiles come from the category-specific mobility ranges supplied to
# the proposed agent.  Mid-range speeds are used by the deliberately simple
# classical prior; no hidden simulated trajectory parameter is consulted.
_MOBILITY_PROFILES = (
    {
        "name": "child_limited_family",
        "keywords": ("child", "7-year", "age 7"),
        "members": 2,
        "trail_speed": 0.55,
        "offtrail_speed": 0.30,
        "stamina_km": 4.5,
        "water_weight": 0.14,
    },
    {
        "name": "average_adult_pair",
        "keywords": ("married", "couple", "moderately fit"),
        "members": 2,
        "trail_speed": 0.85,
        "offtrail_speed": 0.50,
        "stamina_km": 7.5,
        "water_weight": 0.08,
    },
    {
        "name": "experienced_hiker",
        "keywords": ("experienced", "solo", "strong stamina"),
        "members": 1,
        "trail_speed": 1.10,
        "offtrail_speed": 0.75,
        "stamina_km": 22.5,
        "water_weight": 0.04,
    },
)

_PRIOR_TERRAIN_WEIGHT = {
    "trail": 1.45,
    "meadow": 1.12,
    "creek": 1.05,
    "pond": 0.75,
    "sparse_vegetation": 1.00,
    "dense_forest": 0.68,
    "rocky": 0.72,
    "impassable": 0.0,
}

_TRANSITION_TERRAIN_WEIGHT = {
    "trail": 2.00,
    "meadow": 1.25,
    "creek": 0.85,
    "pond": 0.35,
    "sparse_vegetation": 1.00,
    "dense_forest": 0.55,
    "rocky": 0.60,
    "impassable": 0.0,
}

# Both Bayesian baselines use this same generic between-cycle motion model.
# Their only methodological difference is the initial spatial prior.
_BAYES_WALKING_SPEED = 0.70


def _normalise_map(values: np.ndarray, grid_map: GridMap) -> np.ndarray:
    """Normalise a non-negative map over traversable cells."""
    out = np.asarray(values, dtype=np.float64).copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if not grid_map.is_traversable(r, c):
                out[r, c] = 0.0
    out = np.maximum(out, 0.0)
    total = float(out.sum())
    if total <= 0.0:
        for rc in grid_map.get_traversable_cells():
            out[rc] = 1.0
        total = float(out.sum())
    return out / total


def _profiles_from_clue(clue_text: str) -> list:
    """Return mobility categories explicitly mentioned in the clue."""
    text = (clue_text or "").lower()
    profiles = [dict(profile) for profile in _MOBILITY_PROFILES
                if any(token in text for token in profile["keywords"])]
    if profiles:
        return profiles
    # A clue with no demographic detail still supports a generic adult
    # mobility model; its spatial prior remains clue-agnostic otherwise.
    fallback = dict(_MOBILITY_PROFILES[1])
    fallback["name"] = "generic_adult_group"
    fallback["members"] = 1
    return [fallback]


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray,
                               end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 0.0:
        return float(np.linalg.norm(point - start))
    alpha = float(np.dot(point - start, segment) / denom)
    alpha = min(1.0, max(0.0, alpha))
    return float(np.linalg.norm(point - (start + alpha * segment)))


def build_clue_group_priors(grid_map: GridMap,
                            clue_text: str = None) -> tuple:
    """Build one static spatial prior for every clue-described subgroup.

    The model combines a reachable-distance ring around the last sighting,
    a broad corridor toward the stated destination, terrain preference, and
    a weak proximity-to-water heuristic.  It does not use target identities,
    destinations from ``MOVEMENT_UNITS``, or generated trajectories.
    """
    clue_text = INITIAL_CLUE if clue_text is None else clue_text
    profiles = _profiles_from_clue(clue_text)

    text = clue_text.lower()
    start_rc = (grid_map.landmarks.get("trail_fork", GCS_CELL)
                if "fork" in text else GCS_CELL)
    goal_rc = (grid_map.landmarks.get("overlook", start_rc)
               if "overlook" in text else start_rc)
    start = grid_map.cell_centre(*start_rc).astype(np.float64)
    goal = grid_map.cell_centre(*goal_rc).astype(np.float64)
    water_cells = grid_map.get_creek_cells()
    water_positions = [grid_map.cell_centre(*rc).astype(np.float64)
                       for rc in water_cells]
    goal_distance = float(np.linalg.norm(goal - start))

    priors = []
    for profile in profiles:
        # A weighted on/off-trail midpoint represents an unknown route choice.
        walking_speed = (0.65 * profile["trail_speed"]
                         + 0.35 * profile["offtrail_speed"])
        expected_distance = min(
            walking_speed * SIM_START_OFFSET,
            profile["stamina_km"] * 1000.0,
        )
        radial_sigma = max(1.5 * CELL_SIZE, 0.32 * expected_distance)
        corridor_sigma = 3.0 * CELL_SIZE
        goal_sigma = 2.0 * CELL_SIZE
        goal_reachability = np.exp(
            -0.5 * ((goal_distance - expected_distance) / radial_sigma) ** 2)

        prior = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
        for rc in grid_map.get_traversable_cells():
            pos = grid_map.cell_centre(*rc).astype(np.float64)
            radial_distance = float(np.linalg.norm(pos - start))
            radial = np.exp(
                -0.5 * ((radial_distance - expected_distance)
                        / radial_sigma) ** 2)
            corridor_distance = _point_to_segment_distance(pos, start, goal)
            corridor = np.exp(
                -0.5 * (corridor_distance / corridor_sigma) ** 2)
            goal_bump = np.exp(
                -0.5 * (float(np.linalg.norm(pos - goal)) / goal_sigma) ** 2)

            if water_positions:
                water_distance = min(
                    float(np.linalg.norm(pos - water_pos))
                    for water_pos in water_positions)
                water_bump = np.exp(
                    -0.5 * (water_distance / (1.5 * CELL_SIZE)) ** 2)
            else:
                water_bump = 0.0

            terrain_weight = _PRIOR_TERRAIN_WEIGHT.get(
                grid_map.terrain(*rc), 1.0)
            motion_mass = radial * (0.55 + 0.45 * corridor)
            prior[rc] = (
                0.84 * motion_mass * terrain_weight
                + 0.12 * goal_reachability * goal_bump
                + profile["water_weight"] * water_bump
                + 1e-6
            )
        priors.append(_normalise_map(prior, grid_map))

    return profiles, priors


def build_clue_prior(grid_map: GridMap,
                     clue_text: str = None) -> np.ndarray:
    """Return the fixed, victim-count-weighted clue prior used by Exp. 1."""
    profiles, priors = build_clue_group_priors(grid_map, clue_text)
    combined = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    for profile, prior in zip(profiles, priors):
        combined += float(profile["members"]) * prior
    return _normalise_map(combined, grid_map)


def _static_prior_greedy(grid_map: GridMap, state: dict, uavs_list,
                         prior: np.ndarray) -> dict:
    """Dispatch by a fixed prior, distance, and a binary searched mask."""
    occupied = set(state["occupied_cells"])
    base_pos = grid_map.cell_centre(*GCS_CELL)
    all_candidates = [rc for rc in grid_map.get_traversable_cells()
                      if rc not in occupied]
    unsearched = [rc for rc in all_candidates
                  if state["cell_stats"].get(rc, {"n_k": 0})["n_k"] == 0]
    candidates = unsearched or all_candidates

    assignments = {}
    assigned_cells = set()
    distance_scale = 750.0
    for uav in uavs_list:
        best_cell, best_score = None, -1.0
        for rc in candidates:
            if rc in assigned_cells:
                continue
            if not _is_feasible(uav, rc, grid_map, base_pos):
                continue
            distance = float(np.linalg.norm(
                uav.position - grid_map.cell_centre(*rc)))
            score = float(prior[rc]) * np.exp(-distance / distance_scale)
            if score > best_score:
                best_cell, best_score = rc, score
        if best_cell is not None:
            assignments[uav.id] = best_cell
            assigned_cells.add(best_cell)
    return assignments


def clue_prior_greedy(grid_map: GridMap, state: dict, uavs_list,
                      cache: dict = None,
                      clue_text: str = None) -> dict:
    """Fixed clue prior + greedy feasible dispatch; no posterior update."""
    clue_text = INITIAL_CLUE if clue_text is None else clue_text
    if cache is None:
        cache = {}
    if cache.get("clue") != clue_text or "prior" not in cache:
        cache["prior"] = build_clue_prior(grid_map, clue_text)
        cache["clue"] = clue_text
    return _static_prior_greedy(
        grid_map, state, uavs_list, cache["prior"])


def clue_prior_sweep(grid_map: GridMap, state: dict, uavs_list,
                     sweep_state: dict = None,
                     clue_text: str = None) -> dict:
    """Use the fixed clue prior only to order boustrophedon row strips."""
    if sweep_state is None:
        sweep_state = {}
    if not sweep_state.get("next_cells"):
        prior = build_clue_prior(grid_map, clue_text)
        row_order = sorted(
            range(GRID_ROWS),
            key=lambda row: (-float(prior[row, :].sum()), row),
        )
        cells = []
        for strip_index, row in enumerate(row_order):
            row_cells = [(row, col) for col in range(GRID_COLS)
                         if grid_map.is_traversable(row, col)]
            if strip_index % 2 == 1:
                row_cells.reverse()
            cells.extend(row_cells)
        sweep_state["next_cells"] = cells

    occupied = set(state["occupied_cells"])
    base_pos = grid_map.cell_centre(*GCS_CELL)
    assignments = {}
    assigned_cells = set()
    cell_index = 0
    for uav in uavs_list:
        while cell_index < len(sweep_state["next_cells"]):
            rc = sweep_state["next_cells"][cell_index]
            cell_index += 1
            if rc in occupied or rc in assigned_cells:
                continue
            if not _is_feasible(uav, rc, grid_map, base_pos):
                continue
            assignments[uav.id] = rc
            assigned_cells.add(rc)
            break
    sweep_state["next_cells"] = [
        rc for rc in sweep_state["next_cells"] if rc not in assigned_cells
    ]
    return assignments


def _uniform_group_priors(grid_map: GridMap, profiles: list) -> list:
    uniform = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    for rc in grid_map.get_traversable_cells():
        uniform[rc] = 1.0
    uniform = _normalise_map(uniform, grid_map)
    return [uniform.copy() for _ in profiles]


def _transition_once(belief: np.ndarray, grid_map: GridMap,
                     move_fraction: float) -> np.ndarray:
    """Apply one terrain-aware four-neighbour Markov transition."""
    move_fraction = min(1.0, max(0.0, float(move_fraction)))
    if move_fraction <= 0.0:
        return belief
    predicted = np.zeros_like(belief)
    for rc in grid_map.get_traversable_cells():
        mass = float(belief[rc])
        neighbours = grid_map.neighbours(*rc)
        if not neighbours:
            predicted[rc] += mass
            continue
        predicted[rc] += mass * (1.0 - move_fraction)
        weights = np.array([
            _TRANSITION_TERRAIN_WEIGHT.get(grid_map.terrain(*nbr), 1.0)
            for nbr in neighbours
        ], dtype=np.float64)
        weights /= float(weights.sum())
        for nbr, weight in zip(neighbours, weights):
            predicted[nbr] += mass * move_fraction * float(weight)
    return _normalise_map(predicted, grid_map)


def _predict_belief(belief: np.ndarray, grid_map: GridMap,
                    walking_speed: float, elapsed: float) -> np.ndarray:
    """Diffuse a group belief according to distance walkable in ``elapsed``."""
    cell_steps = max(0.0, walking_speed * elapsed / CELL_SIZE)
    # Each full cell interval moves 65% of mass to neighbours.  The cap is a
    # numerical guard for unusually large gaps, not an experiment shortcut.
    full_steps = min(64, int(cell_steps))
    out = belief
    for _ in range(full_steps):
        out = _transition_once(out, grid_map, 0.65)
    fractional = cell_steps - int(cell_steps)
    if full_steps < 64 and fractional > 0.0:
        out = _transition_once(out, grid_map, 0.65 * fractional)
    return out


def _negative_observation_update(belief: np.ndarray, cell_rc: tuple,
                                 grid_map: GridMap) -> np.ndarray:
    """Bayesian update after a searched cell contains no unfound target."""
    p_d, p_f = grid_map.sensor_params(*cell_rc)
    updated = belief * (1.0 - p_f)
    updated[cell_rc] = belief[cell_rc] * (1.0 - p_d)
    return _normalise_map(updated, grid_map)


def _initialise_bayes_state(grid_map: GridMap, state: dict,
                            clue_prior: bool,
                            clue_text: str = None) -> dict:
    clue_text = INITIAL_CLUE if clue_text is None else clue_text
    profiles = _profiles_from_clue(clue_text)
    if clue_prior:
        profiles, beliefs = build_clue_group_priors(grid_map, clue_text)
    else:
        beliefs = _uniform_group_priors(grid_map, profiles)
    return {
        "profiles": profiles,
        "beliefs": beliefs,
        "active": [True] * len(profiles),
        "observation_index": 0,
        "processed_find_ids": set(),
        "last_time": float(state["current_time"]),
    }


def _remove_confirmed_group(filter_state: dict, cell_rc: tuple,
                            found_count: int):
    candidates = [i for i, active in enumerate(filter_state["active"])
                  if active]
    exact = [i for i in candidates
             if filter_state["profiles"][i]["members"] == found_count]
    pool = exact or candidates
    if not pool:
        return
    chosen = max(pool, key=lambda i: filter_state["beliefs"][i][cell_rc])
    filter_state["active"][chosen] = False


def _advance_bayes_filter(grid_map: GridMap, memory, state: dict,
                          filter_state: dict):
    """Advance motion beliefs and assimilate newly completed searches."""
    observations = memory.observation_history[
        filter_state["observation_index"]:]
    for observation in observations:
        observation_time = float(observation["time"])
        elapsed = max(0.0, observation_time - filter_state["last_time"])
        for i, active in enumerate(filter_state["active"]):
            if not active:
                continue
            filter_state["beliefs"][i] = _predict_belief(
                filter_state["beliefs"][i], grid_map,
                _BAYES_WALKING_SPEED, elapsed)
        filter_state["last_time"] = observation_time

        cell_rc = tuple(observation["cell"])
        new_find_ids = [
            target_id for target_id, info in memory.confirmed_finds.items()
            if target_id not in filter_state["processed_find_ids"]
            and tuple(info["cell"]) == cell_rc
            and abs(float(info["time"]) - observation_time) < 1e-6
        ]
        if new_find_ids:
            _remove_confirmed_group(
                filter_state, cell_rc, len(new_find_ids))
            filter_state["processed_find_ids"].update(new_find_ids)

        # A completed search without a newly confirmed subgroup is a
        # no-target observation for each remaining subgroup.  After a find,
        # the simulator registers every co-located victim, so the same update
        # is valid for the subgroups that remain active.
        for i, active in enumerate(filter_state["active"]):
            if active:
                filter_state["beliefs"][i] = _negative_observation_update(
                    filter_state["beliefs"][i], cell_rc, grid_map)

    filter_state["observation_index"] = len(memory.observation_history)

    current_time = float(state["current_time"])
    elapsed = max(0.0, current_time - filter_state["last_time"])
    for i, active in enumerate(filter_state["active"]):
        if not active:
            continue
        filter_state["beliefs"][i] = _predict_belief(
            filter_state["beliefs"][i], grid_map,
            _BAYES_WALKING_SPEED, elapsed)
    filter_state["last_time"] = current_time


def bayesian_search(grid_map: GridMap, memory, state: dict, uavs_list,
                    cache: dict = None, *, clue_prior: bool = False,
                    clue_text: str = None) -> dict:
    """Bayesian search with either a uniform or clue-derived initial prior."""
    if cache is None:
        cache = {}
    key = "clue" if clue_prior else "uniform"
    if cache.get("mode") != key or "filter" not in cache:
        cache["filter"] = _initialise_bayes_state(
            grid_map, state, clue_prior, clue_text)
        cache["mode"] = key
    filter_state = cache["filter"]
    _advance_bayes_filter(grid_map, memory, state, filter_state)

    expected_victims = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    for profile, belief, active in zip(
            filter_state["profiles"], filter_state["beliefs"],
            filter_state["active"]):
        if active:
            expected_victims += float(profile["members"]) * belief

    occupied = set(state["occupied_cells"])
    candidates = [rc for rc in grid_map.get_traversable_cells()
                  if rc not in occupied]
    base_pos = grid_map.cell_centre(*GCS_CELL)
    assignments = {}
    assigned_cells = set()
    for uav in uavs_list:
        best_cell, best_score = None, -1.0
        for rc in candidates:
            if rc in assigned_cells:
                continue
            if not _is_feasible(uav, rc, grid_map, base_pos):
                continue
            distance = float(np.linalg.norm(
                uav.position - grid_map.cell_centre(*rc)))
            travel_time = distance / CRUISE_SPEED
            search_time = T_SEARCH_BY_TERRAIN.get(
                grid_map.terrain(*rc), 90.0)
            p_d, _ = grid_map.sensor_params(*rc)
            score = (float(expected_victims[rc]) * p_d
                     / max(1.0, travel_time + search_time))
            if score > best_score:
                best_cell, best_score = rc, score
        if best_cell is not None:
            assignments[uav.id] = best_cell
            assigned_cells.add(best_cell)
    return assignments


def random_assignment(grid_map: GridMap, state: dict, uavs_list,
                      rng: np.random.Generator) -> dict:
    """Random: each UAV assigned to a random feasible unoccupied cell."""
    occupied = set(state["occupied_cells"])
    base_pos = grid_map.cell_centre(1, 13)

    # Build candidate pool
    candidates = [
        rc for rc in grid_map.get_traversable_cells()
        if rc not in occupied
    ]
    rng.shuffle(candidates)

    assignments = {}
    assigned_cells = set()
    candidate_idx = 0

    for uav in uavs_list:
        while candidate_idx < len(candidates):
            rc = candidates[candidate_idx]
            candidate_idx += 1

            if rc in assigned_cells:
                continue
            if not _is_feasible(uav, rc, grid_map, base_pos):
                continue

            assignments[uav.id] = rc
            assigned_cells.add(rc)
            break

    return assignments
