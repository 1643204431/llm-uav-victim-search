"""Offline checks for the qualitative cross-backbone visualization."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


CODE_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = CODE_ROOT / "experiments"
for path in (CODE_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze  # noqa: E402
from analyze import (  # noqa: E402
    BACKBONE_ORDER,
    _recreate_targets_for_row, _unfound_target_cells,
    _priority_visualization_for_run, _record_at_time,
    plot_backbone_alignment,
)
from exp1_main_comparison import build_specs as build_exp1  # noqa: E402
from exp4_llm_sensitivity import build_specs as build_exp4  # noqa: E402
from exp5_case_study import build_specs as build_exp5  # noqa: E402
from simulation.config import GRID_COLS, GRID_ROWS  # noqa: E402


class BackboneVisualizationTests(unittest.TestCase):
    @staticmethod
    def _aligned_row(backbone, run_id):
        _, targets = _recreate_targets_for_row({"seed": 42})
        snapshots = []
        for timestamp in (3600.0, 3660.0, 3720.0):
            weights = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
            for rank, cell in enumerate(
                    _unfound_target_cells(targets, {}, timestamp)):
                weights[cell] = 1.0 - rank * 0.01
            snapshots.append({"t": timestamp, "w": weights.tolist()})
        return {
            "run_id": run_id,
            "method": "ours",
            "seed": 42,
            "backbone": backbone,
            "discovery_times": {},
            "priority_snapshots": snapshots,
        }

    def test_snapshot_specs_cover_anchor_and_every_backbone(self):
        exp1 = build_exp1(seeds=[42])
        enabled = {spec["method"] for spec in exp1
                   if spec.get("dump_priorities")}
        self.assertEqual(enabled, {"ours"})
        self.assertTrue(all(spec.get("dump_priorities")
                            for spec in build_exp4(seeds=[42])))
        self.assertTrue(build_exp5(seed=42)[0]["dump_priorities"])

    def test_one_cell_is_scored_per_unfound_movement_unit(self):
        _, targets = _recreate_targets_for_row({"seed": 42})
        cells = _unfound_target_cells(targets, {}, 3600.0)
        self.assertEqual(len(cells), 3)

        first_unit_ids = {target.id for target in targets
                          if target.unit_id == targets[0].unit_id}
        discovered = {target_id: 100.0 for target_id in first_unit_ids}
        remaining = _unfound_target_cells(targets, discovered, 3600.0)
        self.assertEqual(len(remaining), 2)

    def test_visual_record_preserves_top_cells_and_subgroup_locations(self):
        _, targets = _recreate_targets_for_row({"seed": 42})
        snapshots = []
        for timestamp in (3600.0, 3660.0):
            weights = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
            target_cells = _unfound_target_cells(
                targets, {}, timestamp)
            for rank, cell in enumerate(target_cells):
                weights[cell] = 1.0 - rank * 0.01
            snapshots.append({"t": timestamp, "w": weights.tolist()})

        row = {
            "run_id": "offline_visualization_test",
            "seed": 42,
            "provider": "deepseek",
            "llm_model": "deepseek-v4-pro",
            "llm_thinking": False,
            "discovery_times": {},
            "priority_snapshots": snapshots,
        }
        visual = _priority_visualization_for_run(row, top_b=10)
        self.assertEqual(len(visual["records"]), 2)
        for record in visual["records"]:
            self.assertTrue(
                set(map(tuple, record["target_cells"])).issubset(
                    set(map(tuple, record["top_cells"]))))
            self.assertNotIn("coverage", record)
            self.assertNotIn("progress", record)

        initial = _record_at_time(visual, 3600.0)
        self.assertEqual(initial["index"], 0)
        with self.assertRaises(ValueError):
            _record_at_time(visual, 3690.0)

    def test_full_figure_renders_from_offline_snapshots(self):
        rows = [self._aligned_row(backbone, f"run_{index}")
                for index, backbone in enumerate(BACKBONE_ORDER)]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            exp1_path = temp / "exp1.jsonl"
            exp4_path = temp / "exp4.jsonl"
            exp1_path.write_text(json.dumps(rows[0]) + "\n")
            exp4_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[1:]))
            figure_dir = temp / "figures"
            figure_dir.mkdir()
            with mock.patch.object(analyze, "FIG_DIR", figure_dir):
                plot_backbone_alignment(exp1_path, exp4_path)

            self.assertGreater(
                (figure_dir / "exp4_backbone_alignment.pdf").stat().st_size,
                1000)
            self.assertGreater(
                (figure_dir / "exp4_backbone_alignment.svg").stat().st_size,
                1000)
            self.assertFalse(
                (figure_dir /
                 "exp4_backbone_alignment_metrics.json").exists())

    def test_exp5_case_dir_comes_from_current_summary_not_name_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            results = root / "results"
            runs = results / "runs"
            current_id = "exp5_ours_N10_B10_s42_current"
            current_dir = runs / current_id
            legacy_dir = runs / "exp5_zzzz_legacy"
            current_dir.mkdir(parents=True)
            legacy_dir.mkdir(parents=True)

            summary = results / "exp5_case_study_current.jsonl"
            summary.write_text(json.dumps({
                "run_id": current_id,
                "status": "completed",
            }) + "\n")
            (current_dir / "result.json").write_text(json.dumps({
                "run_id": current_id,
            }))
            (legacy_dir / "result.json").write_text(json.dumps({
                "run_id": legacy_dir.name,
            }))

            current_files = dict(analyze.CURRENT_RESULT_FILES)
            current_files["exp5"] = summary
            with mock.patch.object(analyze, "_ROOT", root), \
                    mock.patch.object(
                        analyze, "CURRENT_RESULT_FILES", current_files):
                selected = analyze.resolve_exp5_case_dir()

            self.assertEqual(selected, current_dir)


if __name__ == "__main__":
    unittest.main()
