"""Offline checks for the fixed-fleet dispatch-batch sensitivity study."""

import pathlib
import sys
import unittest


EXPERIMENTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import _common  # noqa: E402
from exp3_scaling import (  # noqa: E402
    B_SWEEP_VALUES, build_specs, max_epochs_for_batch,
)
from manifest import build_manifest, validate_manifest  # noqa: E402
from simulation import config  # noqa: E402
import simulation.simulator as simulator_module  # noqa: E402


class BatchSweepTests(unittest.TestCase):
    def tearDown(self):
        _common._reset_overrides_to_defaults()

    def test_all_batch_values_and_equalised_assignment_budget(self):
        specs = _common.normalise_specs(build_specs(seeds=[42]))

        self.assertEqual(B_SWEEP_VALUES, [3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual([spec["batch_size"] for spec in specs],
                         B_SWEEP_VALUES)
        for spec in specs:
            batch_size = spec["batch_size"]
            self.assertEqual(spec["num_uavs"], 10)
            self.assertEqual(spec["max_epochs"],
                             max_epochs_for_batch(batch_size, 200))
            self.assertGreaterEqual(spec["max_epochs"] * batch_size, 200)
            self.assertLess(spec["max_epochs"] * batch_size, 210)
            self.assertIn("_N10_", spec["run_id"])
            self.assertIn(f"_B{batch_size}_", spec["run_id"])

    def test_batch_override_reaches_simulator_and_resets(self):
        _common.override_config(BATCH_SIZE=3)
        self.assertEqual(config.BATCH_SIZE, 3)
        self.assertEqual(simulator_module.BATCH_SIZE, 3)

        _common._reset_overrides_to_defaults()
        self.assertEqual(config.BATCH_SIZE, 10)
        self.assertEqual(simulator_module.BATCH_SIZE, 10)

    def test_complete_manifest_contains_116_unique_runs(self):
        specs = build_manifest()
        summary = validate_manifest(specs)

        self.assertEqual(summary["total"], 116)
        self.assertEqual(summary["counts"]["1"], 35)
        self.assertEqual(summary["counts"]["2"], 20)
        self.assertEqual(summary["counts"]["3"], 40)
        self.assertTrue(all("_N10_B" in spec["run_id"] for spec in specs))

    def test_batch_size_changes_run_and_checkpoint_identity(self):
        common = {
            "method": "ours",
            "seed": 42,
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "thinking": False,
            "num_uavs": 10,
            "tag": {"exp": "1"},
        }
        b5 = _common.stable_run_id({**common, "batch_size": 5})
        b10 = _common.stable_run_id({**common, "batch_size": 10})

        self.assertIn("_N10_B5_", b5)
        self.assertIn("_N10_B10_", b10)
        self.assertNotEqual(b5, b10)
        self.assertNotEqual(_common.run_directory(b5),
                            _common.run_directory(b10))
        self.assertNotEqual(_common.run_directory(b5) / "checkpoint",
                            _common.run_directory(b10) / "checkpoint")


if __name__ == "__main__":
    unittest.main()
