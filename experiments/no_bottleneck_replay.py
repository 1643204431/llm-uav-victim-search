"""Controlled replay for the ``w/o Bottleneck`` ablation.

Each cycle starts from the UAV positions recorded by the matching ``Ours``
run and keeps its selected cells.  Only the one-to-one UAV--cell matching is
replaced by a seeded random permutation.  Counterfactual UAV positions and
search outcomes are not propagated to the next cycle.
"""

import copy
import math

import numpy as np

from simulation.config import E_INIT, GCS_CELL
from simulation.grid_map import GridMap
from simulation.target import create_targets
from simulation.uav import UAV


_TIME_TOLERANCE = 1e-6


def _group_dispatches(dispatch_log):
    """Return consecutive dispatch batches in cycle order."""
    groups = []
    for row in dispatch_log:
        dispatch_time = float(row["t"])
        if not groups or groups[-1][0] != dispatch_time:
            groups.append((dispatch_time, []))
        groups[-1][1].append(row)
    return groups


def _priority_order(entries, priority_snapshot, grid_map):
    """Recover the selected-cell order used by Stage 2."""
    selected = {tuple(row["cell"]) for row in entries}
    ordered = [cell for cell in grid_map.get_traversable_cells()
               if cell in selected]
    if len(ordered) != len(selected):
        raise ValueError("reference dispatch contains a non-traversable cell")

    weights = np.asarray(priority_snapshot["w"], dtype=float)
    ordered.sort(key=lambda cell: weights[cell[0], cell[1]], reverse=True)
    return ordered


def _discovery_cell(source_time, target_id, candidates, targets):
    """Resolve the searched cell responsible for one logged discovery."""
    if len(candidates) == 1:
        return tuple(candidates[0]["cell"])

    target_cell = targets[int(target_id)].cell_at(float(source_time))
    matching = [row for row in candidates
                if tuple(row["cell"]) == target_cell]
    if len(matching) != 1:
        raise ValueError(
            f"cannot identify discovery cell for target {target_id} at "
            f"t={source_time}")
    return target_cell


def replay_no_bottleneck(source_result, *, assignment_seed):
    """Build a controlled ``w/o Bottleneck`` result from an ``Ours`` run.

    Non-matching mission state is inherited from ``source_result``.  Random
    matching changes the counterfactual dispatch completion times and energy,
    while the reference UAV positions are restored before every next cycle.
    """
    source = copy.deepcopy(source_result)
    grid_map = GridMap()
    groups = _group_dispatches(source.get("dispatch_log", []))
    snapshots = source.get("priority_snapshots", [])
    if len(groups) != int(source.get("total_epochs", 0)):
        raise ValueError("reference dispatch log does not cover every cycle")
    if len(snapshots) != len(groups):
        raise ValueError("reference priorities do not cover every cycle")

    num_uavs = int(source["num_uavs_actual"])
    base = grid_map.cell_centre(*GCS_CELL)
    reference_positions = {uid: base.copy() for uid in range(num_uavs)}
    rng = np.random.default_rng(int(assignment_seed))

    random_dispatch_log = []
    random_by_cycle_cell = {}
    replay_cycles = []
    total_energy = 0.0

    for cycle_index, ((dispatch_time, entries), snapshot) in enumerate(
            zip(groups, snapshots), start=1):
        uav_ids = sorted(int(row["uav_id"]) for row in entries)
        selected_cells = _priority_order(entries, snapshot, grid_map)
        if len(uav_ids) != len(set(uav_ids)):
            raise ValueError("reference cycle assigns one UAV more than once")
        if len(uav_ids) != len(selected_cells):
            raise ValueError("reference cycle is not one-to-one")

        source_cells_by_uav = {
            int(row["uav_id"]): tuple(row["cell"]) for row in entries
        }
        source_outbound = []
        for uid, cell in source_cells_by_uav.items():
            destination = grid_map.cell_centre(*cell)
            distance = float(np.linalg.norm(
                reference_positions[uid] - destination))
            source_outbound.append(UAV.travel_time(distance))

        random_uav_order = rng.permutation(len(uav_ids))
        random_outbound = []
        cycle_energy = 0.0
        for cell_index, uav_index in enumerate(random_uav_order):
            uid = uav_ids[int(uav_index)]
            cell = selected_cells[cell_index]
            destination = grid_map.cell_centre(*cell)
            distance = float(np.linalg.norm(
                reference_positions[uid] - destination))
            travel_time = UAV.travel_time(distance)
            search_time = UAV.search_time_for_terrain(
                grid_map.terrain(*cell))
            completion_time = dispatch_time + travel_time + search_time
            energy = (UAV.energy_fly(distance)
                      + UAV.energy_search(grid_map.terrain(*cell)))

            row = {
                "t": float(dispatch_time),
                "uav_id": int(uid),
                "cell": [int(cell[0]), int(cell[1])],
                "completion_time": float(completion_time),
            }
            random_dispatch_log.append(row)
            random_by_cycle_cell[(cycle_index, cell)] = row
            random_outbound.append(float(travel_time))
            cycle_energy += float(energy)

        replay_cycles.append({
            "cycle": cycle_index,
            "dispatch_time": float(dispatch_time),
            "ours_worst_case_outbound_travel_time": (
                max(source_outbound) if source_outbound else 0.0),
            "random_worst_case_outbound_travel_time": (
                max(random_outbound) if random_outbound else 0.0),
            "random_energy_consumed": cycle_energy,
        })
        total_energy += cycle_energy

        # Restore the actual Ours trajectory before the next replayed cycle.
        for uid, cell in source_cells_by_uav.items():
            reference_positions[uid] = grid_map.cell_centre(*cell)

    # Map each Ours discovery event to the completion of the same searched
    # cell under random matching.  Search outcomes themselves stay fixed.
    targets = create_targets(grid_map, np.random.default_rng(
        int(source["seed"])))
    new_discovery_times = {}
    for target_id, source_time in source.get("discovery_times", {}).items():
        candidates = [
            row for row in source.get("dispatch_log", [])
            if math.isclose(float(row["completion_time"]),
                            float(source_time), abs_tol=_TIME_TOLERANCE)
        ]
        if not candidates:
            raise ValueError(
                f"no dispatch matches discovery time {source_time}")
        cell = _discovery_cell(source_time, target_id, candidates, targets)

        cycle_matches = []
        for cycle_index, (_, entries) in enumerate(groups, start=1):
            if any(tuple(row["cell"]) == cell and math.isclose(
                    float(row["completion_time"]), float(source_time),
                    abs_tol=_TIME_TOLERANCE) for row in entries):
                cycle_matches.append(cycle_index)
        if len(cycle_matches) != 1:
            raise ValueError(
                f"cannot identify discovery cycle for target {target_id}")
        new_discovery_times[str(target_id)] = float(
            random_by_cycle_cell[(cycle_matches[0], cell)]["completion_time"])

    result = source
    result["dispatch_log"] = random_dispatch_log
    result["discovery_times"] = new_discovery_times

    discovery_values = list(new_discovery_times.values())
    max_time = float(source["max_time"])
    num_found = int(source["num_found"])
    num_targets = num_found + int(source["num_unfound"])
    if discovery_values:
        result["max_discovery_time"] = max(discovery_values)
        result["mean_discovery_time"] = float(np.mean(discovery_values))
        result["time_to_first_target"] = min(discovery_values)
    else:
        result["max_discovery_time"] = max_time
        result["mean_discovery_time"] = max_time
        result["time_to_first_target"] = max_time

    result["mission_completion_time"] = (
        result["max_discovery_time"] if num_found == num_targets else max_time)
    halfway = int(math.ceil(num_targets / 2.0))
    result["time_to_50pct_targets"] = (
        sorted(discovery_values)[halfway - 1]
        if len(discovery_values) >= halfway else max_time)

    budget = float(source.get("time_budget", max_time))
    found_within = sum(t <= budget for t in discovery_values)
    result["num_found_within_budget"] = found_within
    result["success_rate_within_budget"] = (
        found_within / float(num_targets) if num_targets else 0.0)

    total_budget = num_uavs * E_INIT
    result["total_energy_consumed"] = total_energy
    result["energy_utilisation"] = total_energy / total_budget
    result["energy_per_found_target"] = (
        total_energy / num_found if num_found else float("inf"))
    result["solver_latencies"] = []
    result["controlled_replay"] = {
        "source_run_id": source["run_id"],
        "assignment_seed": int(assignment_seed),
        "counterfactual_state_propagated": False,
        "cycles": replay_cycles,
    }
    return result
