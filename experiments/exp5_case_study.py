"""
Experiment 5 — Qualitative case study.

Runs the proposed method ("ours") ONCE with priority-vector snapshotting
enabled, then dumps everything needed for the figure to its stable
``results/runs/{run_id}/`` directory:

    result.json            full simulation result (with priority_snapshots
                           and dispatch_log already embedded by the
                           simulator)
    target_trajectories.json  (row, col) of each target at a coarse time
                              grid — for overlaying ground-truth paths on
                              the heatmaps
    scenario.json          dict with landmarks, grid dims, cell size

The plotting (terrain base + priority heatmap + UAV trajectory + target
paths) is done by analyze.py --exp 5.

Usage:
    python experiments/exp5_case_study.py            # 1 run, default seed
    python experiments/exp5_case_study.py --seed 43 --max-epochs 40
"""

import argparse
import time

import numpy as np

import _common
import agent.decision as _decision
from simulation import config
from simulation.grid_map import GridMap
from simulation.simulator import Simulator
from agent import get_decision_fn
from simulation.config import (
    SEED_BASE, GRID_ROWS, GRID_COLS, CELL_SIZE,
    TRAJECTORY_DT, NUM_TARGETS,
)


def dump_target_trajectories(sim: Simulator,
                             t_grid: np.ndarray) -> list:
    """Sample each target's (row, col) at each time in t_grid."""
    out = []
    for t in sim.targets:
        samples = []
        for tc in t_grid:
            r, c = t.cell_at(float(tc))
            samples.append([int(tc), int(r), int(c)])
        out.append({
            "id": t.id,
            "name": t.name,
            "unit_id": t.unit_id,
            "samples": samples,
            "found": t.found,
            "discovery_time": t.discovery_time,
        })
    return out


def build_specs(seed=SEED_BASE, *, provider=None, max_epochs=40,
                max_time=36000.0) -> list:
    """Build the single deterministic Exp. 5 run specification."""
    return [{
        "method": "ours",
        "seed": seed,
        "provider": provider,
        "max_epochs": max_epochs,
        "max_time": max_time,
        "time_budget": max_time,
        "verbose": False,
        "dump_priorities": True,
        "tag": {"exp": "5", "replicate": 0},
    }]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: qualitative case study")
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="Default 40 — usually enough to find all 5")
    parser.add_argument("--max-time", type=float, default=36000.0)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Skip cache lookup but still write back — "
                             "use to overwrite stale cached responses")
    parser.add_argument("--output", type=str, default=None,
                        help="Derived JSONL path (default deterministic)")
    _common.add_run_control_args(parser)
    args = parser.parse_args()

    specs = _common.normalise_specs(build_specs(
        args.seed, provider=args.provider, max_epochs=args.max_epochs,
        max_time=args.max_time))
    spec = specs[0]
    run_id = spec["run_id"]
    if args.run_id and args.run_id != [run_id]:
        unknown = sorted(set(args.run_id) - {run_id})
        if unknown:
            raise ValueError("unknown --run-id: " + ", ".join(unknown))
    if args.list_runs:
        state = "completed" if _common.is_completed(spec) else "pending"
        print(f"{run_id}\t{state}")
        return

    out_path = (args.output if args.output else
                _common.default_output_path("exp5_case_study"))
    if _common.is_completed(spec) and not args.rerun_completed:
        print(f"  [{run_id}] SKIP completed")
        rows = _common.rebuild_jsonl(out_path, specs)
        print(f"  [aggregate] {rows} completed rows -> "
              f"{_common.public_path(out_path)}")
        return

    _decision.set_cache_enabled(not args.no_cache)
    _decision.set_force_refresh(args.force_refresh)
    _decision.set_dump_priorities(True)  # enable w(t) snapshotting
    # Register the stable run identity before any LLM call.
    _decision.set_current_seed(args.seed)
    _decision.set_current_run_id(run_id)
    out_dir = _common.run_directory(run_id)
    _common.mark_run_status(spec, "running")
    _common.rebuild_jsonl(out_path, specs)
    print(f"[exp5] run: {run_id}")

    try:
        # Run
        rng = np.random.default_rng(args.seed)
        grid_map = GridMap()
        sim = Simulator(grid_map, rng)
        fn = get_decision_fn("ours", rng=rng,
                             provider=args.provider, verbose=False)

        t0 = time.time()
        result = sim.run(fn, max_time=args.max_time,
                         max_epochs=args.max_epochs)
        elapsed = time.time() - t0
        result["wall_time"] = elapsed
        result["seed"] = args.seed
        result["method"] = "ours"
        result["exp"] = "5"
        result["replicate"] = 0
        result["max_epochs"] = args.max_epochs
        result["max_time"] = args.max_time
        result["time_budget"] = args.max_time
        result["num_uavs_actual"] = config.NUM_UAVS
        result["batch_size_actual"] = config.BATCH_SIZE
        result["priority_snapshots_enabled"] = True
        actual_provider = args.provider or config.LLM_PROVIDER
        result["provider"] = actual_provider
        if actual_provider == "qwen":
            result["llm_model"] = config.LLM_MODEL_QWEN
            result["llm_thinking"] = bool(config.LLM_THINKING_QWEN)
        else:
            result["llm_model"] = config.LLM_MODEL_DEEPSEEK
            result["llm_thinking"] = bool(config.LLM_THINKING_DEEPSEEK)
        result["run_id"] = run_id
        result["status"] = "completed"

        # Sample target trajectories at a coarse grid (every 60 s)
        dt_coarse = 60.0
        t_grid = np.arange(
            0.0,
            min(result.get("max_discovery_time", args.max_time) + 60.0,
                args.max_time),
            dt_coarse)
        target_traj = dump_target_trajectories(sim, t_grid)

        # Scenario metadata
        scenario = {
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "cell_size_m": CELL_SIZE,
            "landmarks": {k: list(v) for k, v in grid_map.landmarks.items()},
            "initial_clue": config.INITIAL_CLUE,
            "num_targets": NUM_TARGETS,
            "num_uavs": len(sim.uavs),
        }

        # Additional artifacts are written first. ``status=completed`` is
        # committed only after all files are present.
        _common.persist_json_artifact(
            out_dir / "target_trajectories.json", target_traj)
        _common.persist_json_artifact(out_dir / "scenario.json", scenario)
        _common.persist_completed_result(result)

        print(f"[exp5] done: {result['num_found']}/{NUM_TARGETS} found, "
              f"epochs={result['total_epochs']}, wall={elapsed:.1f}s, "
              f"priority_snapshots={len(result['priority_snapshots'])}, "
              f"dispatches={len(result['dispatch_log'])}")
        print(f"[exp5] files: {sorted(p.name for p in out_dir.iterdir())}")
    except Exception as exc:
        _common.mark_run_status(spec, "failed", error=exc)
        print(f"[exp5] FAILED: {_common.sanitise_error_message(exc)}")
        raise SystemExit(1)
    finally:
        _decision.set_dump_priorities(False)

    rows = _common.rebuild_jsonl(out_path, specs)
    print(f"  [aggregate] {rows} completed rows -> "
          f"{_common.public_path(out_path)}")


if __name__ == "__main__":
    main()
