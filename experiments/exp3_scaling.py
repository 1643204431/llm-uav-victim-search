"""
Experiment 3 — Dispatch-batch sensitivity (paper §IV-B, Fig. 3).

Fixes the fleet at N=10 and sweeps every integer batch size B from 3
(the number of victim sub-groups) through 10 (the full fleet).  All eight
settings are run inside Exp. 3 because its 200-assignment budget gives
B=10 a 20-cycle cap, whereas the canonical Exp. 1 run uses 40 cycles.

To compare batch sizes under the same search-work budget, each setting
uses ``max_epochs = ceil(200 / B)``.  Thus every setting can dispatch at
most approximately 200 UAV-cell assignments, while mission time, energy,
map, targets, model, and seeds remain fixed.

Methods : "ours"
Seeds   : SEED_BASE + 0..runs-1 (default 5, aligned with Exp. 1).

Usage:
    python experiments/exp3_scaling.py
    python experiments/exp3_scaling.py --runs 3
    python experiments/exp3_scaling.py --batch-sizes 3 4 10
    python experiments/exp3_scaling.py --parallel 4
    python experiments/exp3_scaling.py --force-refresh
"""

import argparse
import math

from _common import (
    run_suite, default_output_path, decision_cache_setup,
    add_run_control_args, public_path,
)
from simulation.config import SEED_BASE


FLEET_SIZE = 10
ALL_BATCH_SIZES = list(range(3, 11))
B_SWEEP_VALUES = list(ALL_BATCH_SIZES)
DEFAULT_ASSIGNMENT_BUDGET = 200


def max_epochs_for_batch(batch_size: int,
                         assignment_budget: int =
                         DEFAULT_ASSIGNMENT_BUDGET) -> int:
    """Return the cycle cap that equalises maximum dispatch work."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if assignment_budget <= 0:
        raise ValueError("assignment_budget must be positive")
    return int(math.ceil(assignment_budget / batch_size))


def build_specs(batch_sizes=None, seeds=None, *, provider=None,
                assignment_budget=DEFAULT_ASSIGNMENT_BUDGET,
                max_time=36000.0, time_budget=36000.0) -> list:
    """Build deterministic Exp. 3 run specifications."""
    batch_sizes = list(batch_sizes or B_SWEEP_VALUES)
    seeds = list(seeds or [SEED_BASE + i for i in range(5)])
    invalid = sorted(set(batch_sizes) - set(ALL_BATCH_SIZES))
    if invalid:
        raise ValueError(f"unsupported batch sizes: {invalid}")
    return [
        {
            "method": "ours",
            "seed": seed,
            "provider": provider,
            "max_epochs": max_epochs_for_batch(b, assignment_budget),
            "max_time": max_time,
            "time_budget": time_budget,
            "verbose": False,
            "num_uavs": FLEET_SIZE,
            "batch_size": b,
            "tag": {
                "exp": "3",
                "N": FLEET_SIZE,
                "B": b,
                "assignment_budget": assignment_budget,
                "replicate": i,
            },
        }
        for b in batch_sizes
        for i, seed in enumerate(seeds)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: dispatch-batch-size sensitivity")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Exact seeds; overrides --runs")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        choices=B_SWEEP_VALUES,
                        help="Selected batch sizes from 3 through 10")
    parser.add_argument("--assignment-budget", type=int,
                        default=DEFAULT_ASSIGNMENT_BUDGET,
                        help="Approximate maximum UAV-cell assignments per "
                             "setting (default 200)")
    parser.add_argument("--max-time", type=float, default=36000.0)
    parser.add_argument("--time-budget", type=float, default=36000.0)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Worker processes for seed-level parallelism "
                             "(default 1 = serial)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Skip cache lookup but still write back — "
                             "use to overwrite stale cached responses")
    parser.add_argument("--output", type=str, default=None)
    add_run_control_args(parser)
    args = parser.parse_args()

    decision_cache_setup(use_cache=not args.no_cache,
                         force_refresh=args.force_refresh)

    out_path = (args.output if args.output else
                default_output_path("exp3_batch_sweep"))
    cache_state = ("OFF" if args.no_cache
                   else "REFRESH" if args.force_refresh else "ON")
    selected_b = args.batch_sizes or B_SWEEP_VALUES
    seeds = (args.seeds if args.seeds is not None else
             [SEED_BASE + i for i in range(args.runs)])
    epoch_caps = {
        b: max_epochs_for_batch(b, args.assignment_budget)
        for b in ALL_BATCH_SIZES
    }

    print(f"[exp3] writing to {public_path(out_path)}")
    print(f"[exp3] fixed N={FLEET_SIZE}; B values={selected_b}")
    print(f"[exp3] assignment budget={args.assignment_budget}; "
          f"epoch caps={epoch_caps}")
    print(f"[exp3] seeds: {seeds}")
    print(f"[exp3] runs: {len(selected_b) * len(seeds)}, "
          f"cache: {cache_state}, parallel: {args.parallel}")

    all_specs = build_specs(
        B_SWEEP_VALUES, seeds, provider=args.provider,
        assignment_budget=args.assignment_budget,
        max_time=args.max_time, time_budget=args.time_budget)
    specs = [spec for spec in all_specs if spec["tag"]["B"] in selected_b]
    run_suite(specs, out_path, parallel=args.parallel,
              use_cache=not args.no_cache,
              force_refresh=args.force_refresh,
              aggregate_specs=all_specs,
              selected_run_ids=args.run_id,
              list_only=args.list_runs,
              rerun_completed=args.rerun_completed)

    print(f"\n[exp3] done. {public_path(out_path)}")


if __name__ == "__main__":
    main()
