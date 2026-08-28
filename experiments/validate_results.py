"""Validate the complete 116-run experiment result package.

This command is offline: it reads only local status/result artifacts and
never imports an API key or calls a model endpoint.
"""

import json
import math
import pathlib

import _common
from manifest import build_manifest, validate_manifest
from simulation.config import GRID_COLS, GRID_ROWS


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

AGGREGATES = {
    "exp1_main_current.jsonl": 35,
    "exp2_ablation_current.jsonl": 20,
    "exp3_batch_sweep_current.jsonl": 40,
    "exp4_llm_sensitivity_current.jsonl": 20,
    "exp5_case_study_current.jsonl": 1,
}


def _read_json(path: pathlib.Path):
    with open(path) as handle:
        return json.load(handle)


def _jsonl_count(path: pathlib.Path) -> int:
    with open(path) as handle:
        return sum(1 for line in handle if line.strip())


def validate_complete_results() -> dict:
    specs = build_manifest(max_epochs=40)
    manifest_summary = validate_manifest(specs)
    expected_ids = {spec["run_id"] for spec in specs}

    runs_dir = RESULTS / "runs"
    actual_ids = ({path.name for path in runs_dir.iterdir() if path.is_dir()}
                  if runs_dir.exists() else set())
    missing_dirs = sorted(expected_ids - actual_ids)
    ignored_dirs = sorted(actual_ids - expected_ids)
    if missing_dirs:
        raise ValueError(
            f"missing manifest run directories: {missing_dirs}")

    completed = 0
    for spec in specs:
        run_id = spec["run_id"]
        if not _common.is_completed(spec):
            raise ValueError(f"run is not validly completed: {run_id}")

        status = _read_json(_common.status_path(run_id))
        result = _read_json(_common.result_path(run_id))
        _common.validate_anonymous_payload(status)
        _common.validate_anonymous_payload(result)

        times = result.get("agent_computation_times")
        total = result.get("total_agent_computation_time")
        mission_time = result.get("mission_completion_time")
        if not isinstance(times, list) or total is None or mission_time is None:
            raise ValueError(f"missing current timing fields: {run_id}")
        if not math.isclose(sum(times), float(total),
                            rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"inconsistent agent timing fields: {run_id}")
        if int(result.get("total_epochs", -1)) != len(times):
            raise ValueError(f"cycle/timing length mismatch: {run_id}")
        if spec.get("dump_priorities"):
            snapshots = result.get("priority_snapshots")
            if not isinstance(snapshots, list) or not snapshots:
                raise ValueError(f"missing priority snapshots: {run_id}")
            if len(snapshots) != int(result.get("total_epochs", -1)):
                raise ValueError(
                    f"cycle/snapshot length mismatch: {run_id}")
            for snapshot in snapshots:
                weights = snapshot.get("w")
                if (snapshot.get("t") is None
                        or not isinstance(weights, list)
                        or len(weights) != GRID_ROWS
                        or any(not isinstance(row, list)
                               or len(row) != GRID_COLS for row in weights)):
                    raise ValueError(
                        f"invalid priority snapshot shape: {run_id}")
        completed += 1

    aggregate_counts = {}
    for filename, expected_count in AGGREGATES.items():
        path = RESULTS / filename
        if not path.exists():
            raise ValueError(f"missing aggregate: {filename}")
        count = _jsonl_count(path)
        if count != expected_count:
            raise ValueError(
                f"aggregate {filename} has {count} rows; "
                f"expected {expected_count}")
        aggregate_counts[filename] = count

    summary = {
        "status": "valid",
        "total_runs": manifest_summary["total"],
        "completed_runs": completed,
        "experiment_counts": manifest_summary["counts"],
        "aggregate_rows": aggregate_counts,
        "ignored_non_manifest_run_directories": len(ignored_dirs),
        "default_max_epochs": 40,
        "exp3_assignment_budget": 200,
    }
    _common.persist_json_artifact(
        RESULTS / "validation_summary.json", summary)
    return summary


def main():
    summary = validate_complete_results()
    print(
        "VALID: "
        f"{summary['completed_runs']}/{summary['total_runs']} runs completed; "
        f"counts={summary['experiment_counts']}")


if __name__ == "__main__":
    main()
