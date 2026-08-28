"""Checks for the controlled random-matching Stage-2 ablation."""

import pathlib
import sys
import unittest

import numpy as np


CODE_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = CODE_ROOT / "experiments"
for path in (CODE_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import agent.decision as decision  # noqa: E402
from exp2_ablation import build_specs  # noqa: E402
from no_bottleneck_replay import replay_no_bottleneck  # noqa: E402
from simulation.config import GCS_CELL, SIM_START_OFFSET  # noqa: E402
from simulation.grid_map import GridMap  # noqa: E402
from simulation.perception import PerceptionModule  # noqa: E402
from simulation.uav import UAV  # noqa: E402


class NoBottleneckAblationTests(unittest.TestCase):
    def setUp(self):
        decision.set_cache_enabled(True)
        decision.set_force_refresh(False)
        self.grid = GridMap()
        base = self.grid.cell_centre(*GCS_CELL)
        self.uavs = [UAV(i, base) for i in range(10)]
        self.state = PerceptionModule(self.grid).compute_state(
            self.uavs, {}, SIM_START_OFFSET)
        self.w = np.zeros((self.grid.rows, self.grid.cols), dtype=float)
        for priority, rc in enumerate(
                self.grid.get_traversable_cells(), start=1):
            self.w[rc] = float(priority)

    def tearDown(self):
        decision.configure_ablation(
            use_rag=True, use_cot=True, use_energy_filter=True)
        decision.set_cache_enabled(True)
        decision.set_force_refresh(False)
        decision.reset_stats()

    def test_exp2_registers_seeded_no_bottleneck_runs(self):
        specs = build_specs(conditions=["no_bottleneck"], seeds=[42, 43])
        self.assertEqual(len(specs), 2)
        for spec, seed in zip(specs, [42, 43]):
            self.assertEqual(spec["tag"]["condition"], "no_bottleneck")
            self.assertEqual(
                spec["ablation_kwargs"]["matching_policy"], "random")
            self.assertTrue(spec["dump_priorities"])
            self.assertEqual(spec["ablation_kwargs"]["assignment_seed"],
                             seed)
            self.assertIn(
                f"exp1_ours_N10_B10_s{seed}_",
                spec["ablation_kwargs"]["trajectory_replay_run_id"])

    def test_controlled_replay_restores_ours_state_each_cycle(self):
        traversable = self.grid.get_traversable_cells()
        cells = [traversable[index] for index in
                 (0, len(traversable) // 4,
                  len(traversable) // 2, len(traversable) - 1)]
        first_cells = cells[:2]
        second_cells = cells[2:]
        base = self.grid.cell_centre(*GCS_CELL)
        source_positions = {0: base.copy(), 1: base.copy()}
        dispatch_log = []
        snapshots = []

        for cycle, (dispatch_time, selected) in enumerate(
                [(100.0, first_cells), (500.0, second_cells)]):
            weights = np.zeros(
                (self.grid.rows, self.grid.cols), dtype=float)
            for rank, cell in enumerate(selected, start=1):
                weights[cell] = float(len(selected) - rank + 1)
            snapshots.append({"t": dispatch_time, "w": weights.tolist()})
            for uid, cell in enumerate(selected):
                destination = self.grid.cell_centre(*cell)
                distance = float(np.linalg.norm(
                    source_positions[uid] - destination))
                completion = (
                    dispatch_time + UAV.travel_time(distance)
                    + UAV.search_time_for_terrain(
                        self.grid.terrain(*cell)))
                dispatch_log.append({
                    "t": dispatch_time,
                    "uav_id": uid,
                    "cell": list(cell),
                    "completion_time": completion,
                })
                source_positions[uid] = destination

        source = {
            "run_id": "exp1_ours_test",
            "seed": 42,
            "num_uavs_actual": 2,
            "total_epochs": 2,
            "dispatch_log": dispatch_log,
            "priority_snapshots": snapshots,
            "discovery_times": {"0": dispatch_log[-1]["completion_time"]},
            "num_found": 1,
            "num_unfound": 0,
            "max_time": 36000.0,
            "time_budget": 36000.0,
            "llm_input_tokens": [123],
        }

        replay = replay_no_bottleneck(source, assignment_seed=42)
        replay_again = replay_no_bottleneck(source, assignment_seed=42)
        self.assertEqual(replay["dispatch_log"],
                         replay_again["dispatch_log"])
        for dispatch_time in (100.0, 500.0):
            source_cells = {
                tuple(row["cell"]) for row in source["dispatch_log"]
                if row["t"] == dispatch_time
            }
            replay_cells = {
                tuple(row["cell"]) for row in replay["dispatch_log"]
                if row["t"] == dispatch_time
            }
            self.assertEqual(replay_cells, source_cells)

        second_cycle = [row for row in replay["dispatch_log"]
                        if row["t"] == 500.0]
        for row in second_cycle:
            uid = row["uav_id"]
            source_start = self.grid.cell_centre(*first_cells[uid])
            destination = self.grid.cell_centre(*tuple(row["cell"]))
            expected = (500.0
                        + UAV.travel_time(float(np.linalg.norm(
                            source_start - destination)))
                        + UAV.search_time_for_terrain(
                            self.grid.terrain(*tuple(row["cell"]))))
            self.assertAlmostEqual(row["completion_time"], expected)

        self.assertEqual(replay["num_found"], source["num_found"])
        self.assertEqual(replay["total_epochs"], source["total_epochs"])
        self.assertEqual(replay["llm_input_tokens"],
                         source["llm_input_tokens"])
        self.assertFalse(
            replay["controlled_replay"]["counterfactual_state_propagated"])


if __name__ == "__main__":
    unittest.main()
