"""Offline tests for computation-aware event-clock scheduling."""

import heapq
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import agent.decision as decision
import simulation.simulator as simulator_module
from agent import get_decision_fn
from simulation.grid_map import GridMap
from simulation.simulator import Event, Simulator
from simulation.uav import UAV


class _Grid:
    def cell_centre(self, row, col):
        return np.array([float(col * 250), float(row * 250)])

    def terrain(self, row, col):
        return "trail"

    def get_traversable_cells(self):
        return [(0, 0), (1, 0)]

    def sensor_params(self, row, col):
        return 1.0, 0.0


class _Perception:
    def compute_state(self, uavs, search_stats, current_time):
        return {"current_time": current_time}


class _Memory:
    def __init__(self):
        self.observations = []
        self.finds = []

    def add_observation(self, uav_id, cell, detection, timestamp):
        self.observations.append((uav_id, cell, detection, timestamp))

    def record_find(self, target_id, cell, timestamp):
        self.finds.append((target_id, cell, timestamp))


class _Target:
    def __init__(self, target_id, cell=(1, 0), found=False):
        self.id = target_id
        self.cell = cell
        self.found = found
        self.discovery_time = None

    def cell_at(self, timestamp):
        return self.cell


def _make_simulator(targets):
    sim = Simulator.__new__(Simulator)
    sim.grid = _Grid()
    sim.rng = np.random.default_rng(42)
    sim.current_time = 100.0
    sim.base_pos = np.array([0.0, 0.0])

    idle = UAV(0, sim.base_pos)
    active = UAV(1, sim.base_pos)
    active.status = "active"
    active.assigned_cell = (1, 0)
    active.arrival_time = 105.0
    active.position = sim.grid.cell_centre(1, 0)
    sim.uavs = [idle, active]

    sim.targets = targets
    sim.perception = _Perception()
    sim.memory = _Memory()
    sim.search_stats = {}
    sim.event_queue = [
        Event(105.0, "UAV_COMPLETE", {"uav_id": 1, "cell": (1, 0)})
    ]
    heapq.heapify(sim.event_queue)
    sim.idle_count = 1
    sim.epoch_count = 0
    sim.all_found = False
    sim.discovery_times = {}
    sim.total_cells_searched = 0
    sim.agent_computation_times = []
    sim.dispatch_log = []
    sim._max_time = 1000.0
    sim._max_epochs = 1
    return sim


class SimulatorTimingTests(unittest.TestCase):
    def test_max_epochs_drains_the_last_dispatched_batch(self):
        grid = GridMap()
        sim = Simulator(grid, np.random.default_rng(42))
        sim._detect = lambda cell: 0

        result = sim.run(
            get_decision_fn("greedy_coverage"),
            max_time=10000.0,
            max_epochs=2,
        )

        self.assertEqual(result["total_epochs"], 2)
        self.assertEqual(result["total_cells_searched"], 20)
        self.assertEqual(len(result["dispatch_log"]), 20)
        self.assertEqual(len(result["agent_computation_times"]), 2)
        self.assertEqual(sim.event_queue, [])
        self.assertTrue(all(uav.status == "idle" for uav in sim.uavs))

    def test_recomputes_batch_after_infeasible_uav_is_retired(self):
        sim = _make_simulator([_Target(0)])
        feasible = sim.uavs[1]
        infeasible = sim.uavs[0]
        feasible.status = "idle"
        feasible.assigned_cell = None
        feasible.arrival_time = None
        sim.event_queue = []
        sim.idle_count = 2
        sim._max_epochs = 40

        infeasible.can_reach_and_return = mock.Mock(return_value=False)
        feasible.can_reach_and_return = mock.Mock(return_value=True)
        batches = []

        def decide(grid, memory, state, batch, num_unfound):
            batches.append([u.id for u in batch])
            if len(batches) == 1:
                return {}
            return {feasible.id: (0, 0)}

        with mock.patch.object(
                decision, "resolve_epoch_computation_time",
                return_value=0.0):
            sim._trigger_epoch(decide)

        self.assertEqual(batches, [[0, 1], [1]])
        self.assertTrue(infeasible.retired)
        self.assertEqual(feasible.status, "active")
        self.assertEqual(len(sim.event_queue), 1)

    def test_active_uav_completes_while_idle_uav_waits_for_decision(self):
        sim = _make_simulator([_Target(0)])
        sim._detect = lambda cell: 0

        with mock.patch.object(
                decision, "resolve_epoch_computation_time",
                return_value=10.0):
            sim._trigger_epoch(
                lambda grid, memory, state, batch, num_unfound: {0: (0, 0)})

        self.assertEqual(sim.search_stats[(1, 0)]["last_time"], 105.0)
        self.assertEqual(sim.total_cells_searched, 1)
        self.assertEqual(sim.dispatch_log[0]["t"], 110.0)
        self.assertEqual(sim.current_time, 110.0)
        self.assertEqual(sim.agent_computation_times, [10.0])

    def test_last_target_found_during_decision_prevents_stale_dispatch(self):
        target = _Target(0)
        sim = _make_simulator([target])
        sim._detect = lambda cell: 1

        with mock.patch.object(
                decision, "resolve_epoch_computation_time",
                return_value=10.0):
            sim._trigger_epoch(
                lambda grid, memory, state, batch, num_unfound: {0: (0, 0)})

        self.assertTrue(sim.all_found)
        self.assertEqual(sim.discovery_times, {0: 105.0})
        self.assertEqual(sim.current_time, 105.0)
        self.assertEqual(sim.dispatch_log, [])
        self.assertEqual(sim.uavs[0].status, "idle")

    def test_mission_time_is_not_agent_time_sum(self):
        target = _Target(0, found=True)
        target.discovery_time = 123.0
        sim = _make_simulator([target])
        sim.all_found = True
        sim.discovery_times = {0: 123.0}
        sim.agent_computation_times = [10.0, 20.0]

        with mock.patch.object(simulator_module, "NUM_TARGETS", 1), \
                mock.patch.object(simulator_module, "NUM_UAVS", 2):
            result = sim._compile_results(1000.0)

        self.assertEqual(result["mission_completion_time"], 123.0)
        self.assertEqual(result["total_agent_computation_time"], 30.0)
        self.assertNotIn("physical_mission_completion_time", result)


class EpochTimingCheckpointTests(unittest.TestCase):
    def test_replay_restores_original_end_to_end_epoch_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir, \
                mock.patch.object(decision, "_runs_dir",
                                  pathlib.Path(tmp_dir)), \
                mock.patch.object(decision, "_current_run_id",
                                  "exp1_ours_s42"), \
                mock.patch.object(decision, "_use_cache", True), \
                mock.patch.object(decision, "_force_refresh", False):
            original = decision.resolve_epoch_computation_time(
                0, 7.5, [{"from_cache": False}])
            replay = decision.resolve_epoch_computation_time(
                0, 0.2, [{"from_cache": True}])

            timing_path = (pathlib.Path(tmp_dir) / "exp1_ours_s42" /
                           "checkpoint" / "epoch_0000_timing.json")
            with open(timing_path) as f:
                stored = json.load(f)

        self.assertEqual(original, 7.5)
        self.assertEqual(replay, 7.5)
        self.assertEqual(set(stored), {
            "run_id", "epoch_index", "llm_call_count", "elapsed_seconds"
        })


if __name__ == "__main__":
    unittest.main()
