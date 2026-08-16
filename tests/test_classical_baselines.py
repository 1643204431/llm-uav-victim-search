"""Offline checks for the six formal non-LLM baselines."""

import pathlib
import sys
import unittest

import numpy as np


CODE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from agent import get_decision_fn  # noqa: E402
from agent.baselines import (  # noqa: E402
    bayesian_search, build_clue_group_priors, build_clue_prior,
)
from simulation.config import GCS_CELL, SIM_START_OFFSET  # noqa: E402
from simulation.grid_map import GridMap  # noqa: E402
from simulation.memory import MemoryModule  # noqa: E402
from simulation.perception import PerceptionModule  # noqa: E402
from simulation.uav import UAV  # noqa: E402


FORMAL_BASELINES = (
    "lawnmower",
    "greedy_coverage",
    "bayesian_search",
    "clue_prior_sweep",
    "clue_prior_greedy",
    "clue_prior_bayes",
)


class ClassicalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.grid = GridMap()
        base = self.grid.cell_centre(*GCS_CELL)
        self.uavs = [UAV(i, base) for i in range(10)]
        self.memory = MemoryModule()
        self.state = PerceptionModule(self.grid).compute_state(
            self.uavs, {}, SIM_START_OFFSET)

    def test_each_formal_baseline_returns_unique_feasible_assignments(self):
        for method in FORMAL_BASELINES:
            with self.subTest(method=method):
                fn = get_decision_fn(method)
                assignments = fn(
                    self.grid, self.memory, self.state, self.uavs, 5)
                self.assertEqual(len(assignments), 10)
                self.assertEqual(len(set(assignments.values())), 10)
                for uav_id, rc in assignments.items():
                    self.assertIn(uav_id, range(10))
                    self.assertTrue(self.grid.is_traversable(*rc))

    def test_clue_prior_is_normalised_and_has_three_group_components(self):
        profiles, group_priors = build_clue_group_priors(self.grid)
        combined = build_clue_prior(self.grid)

        self.assertEqual([p["members"] for p in profiles], [2, 2, 1])
        self.assertEqual(len(group_priors), 3)
        for prior in [*group_priors, combined]:
            self.assertAlmostEqual(float(prior.sum()), 1.0, places=12)
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if not self.grid.is_traversable(r, c):
                        self.assertEqual(float(prior[r, c]), 0.0)

    def test_bayesian_filter_assimilates_each_observation_once(self):
        cache = {}
        bayesian_search(
            self.grid, self.memory, self.state, self.uavs, cache=cache)
        before = [belief.copy() for belief in cache["filter"]["beliefs"]]

        searched = (1, 13)
        self.memory.add_observation(0, searched, 0, SIM_START_OFFSET + 60.0)
        next_state = dict(self.state)
        next_state["current_time"] = SIM_START_OFFSET + 60.0
        bayesian_search(
            self.grid, self.memory, next_state, self.uavs, cache=cache)
        after = cache["filter"]["beliefs"]

        self.assertEqual(cache["filter"]["observation_index"], 1)
        self.assertTrue(any(
            not np.allclose(old, new)
            for old, new in zip(before, after)
        ))


if __name__ == "__main__":
    unittest.main()
