"""
Experiment 1 — Main comparison (Claim 1: method effectiveness).

Methods compared:
  lawnmower, greedy_coverage, bayesian_search, clue_prior_sweep,
  clue_prior_greedy, clue_prior_bayes, ours

Seeds: SEED_BASE + 0..runs-1 (default 5 seeds).

Each run is stored under ``results/runs/{run_id}/``. The experiment JSONL
is rebuilt from completed per-run results and never triggers API calls.

Expected DeepSeek call count (first run, cold cache):
    ≈ runs * avg_epochs (only ours calls an LLM)
On cached reruns: ~0 API calls.

Usage:
    python experiments/exp1_main_comparison.py
    python experiments/exp1_main_comparison.py --runs 3 --max-epochs 30
    python experiments/exp1_main_comparison.py --methods ours clue_prior_bayes
    python experiments/exp1_main_comparison.py --parallel 4
    python experiments/exp1_main_comparison.py --no-cache       # latency run
"""

import argparse
import pathlib

from _common import (
    run_suite, default_output_path, decision_cache_setup, RESULTS_DIR,
    add_run_control_args, public_path,
)
from simulation.config import SEED_BASE


# Sentinel file written by exp1 after it picks its output path. run_all.sh
# reads this to learn the canonical ``ours`` JSONL path without parsing
# exp1's stdout. Deterministic handoff, decoupled from log format.
CANONICAL_SENTINEL = RESULTS_DIR / ".canonical_ours_path"


DEFAULT_METHODS = [
    "lawnmower",
    "greedy_coverage",
    "bayesian_search",
    "clue_prior_sweep",
    "clue_prior_greedy",
    "clue_prior_bayes",
    "ours",          # paper's method
]


def build_specs(methods=None, seeds=None, *, provider=None,
                max_epochs=40, max_time=36000.0,
                time_budget=36000.0) -> list:
    """Build deterministic Exp. 1 run specifications."""
    methods = list(methods or DEFAULT_METHODS)
    seeds = list(seeds or [SEED_BASE + i for i in range(5)])
    return [
        {
            "method": method,
            "seed": seed,
            "provider": provider,
            "max_epochs": max_epochs,
            "max_time": max_time,
            "time_budget": time_budget,
            "verbose": False,
            "dump_priorities": method == "ours",
            "tag": {"exp": "1", "replicate": i},
        }
        for method in methods
        for i, seed in enumerate(seeds)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: main method comparison")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=DEFAULT_METHODS,
                        help="Run only selected Exp. 1 methods")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of seeds per method (default 5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Exact seeds; overrides --runs")
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="Cap on decision epochs per run (default 40)")
    parser.add_argument("--max-time", type=float, default=36000.0,
                        help="Sim time cutoff in seconds (default 10h)")
    parser.add_argument("--time-budget", type=float, default=36000.0,
                        help="Budget for success_rate_within_budget metric")
    parser.add_argument("--provider", type=str, default=None,
                        help="LLM provider (default from config)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Worker processes for seed-level parallelism "
                             "(default 1 = serial)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the LLM disk cache — use for latency runs")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Skip cache lookup but still write back — "
                             "use to overwrite stale cached responses")
    parser.add_argument("--output", type=str, default=None,
                        help="Derived JSONL path (default deterministic)")
    add_run_control_args(parser)
    args = parser.parse_args()

    decision_cache_setup(use_cache=not args.no_cache,
                         force_refresh=args.force_refresh)

    out_path = (args.output if args.output
                else default_output_path("exp1_main"))
    if not args.list_runs:
        # Advertise the canonical path via a sentinel file so run_all.sh
        # does not have to parse log output.
        CANONICAL_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        CANONICAL_SENTINEL.write_text(public_path(out_path) + "\n")

    cache_state = ("OFF" if args.no_cache
                   else "REFRESH" if args.force_refresh else "ON")
    print(f"[exp1] writing to {public_path(out_path)}")
    print(f"[exp1] canonical sentinel: {public_path(CANONICAL_SENTINEL)}")
    selected_methods = args.methods or DEFAULT_METHODS
    seeds = (args.seeds if args.seeds is not None else
             [SEED_BASE + i for i in range(args.runs)])
    print(f"[exp1] methods: {selected_methods}")
    print(f"[exp1] seeds: {seeds}")
    print(f"[exp1] runs: {args.runs}, max_epochs: {args.max_epochs}, "
          f"cache: {cache_state}, parallel: {args.parallel}")

    all_specs = build_specs(
        DEFAULT_METHODS, seeds, provider=args.provider,
        max_epochs=args.max_epochs, max_time=args.max_time,
        time_budget=args.time_budget)
    specs = [spec for spec in all_specs
             if spec["method"] in selected_methods]
    run_suite(specs, out_path, parallel=args.parallel,
              use_cache=not args.no_cache,
              force_refresh=args.force_refresh,
              aggregate_specs=all_specs,
              selected_run_ids=args.run_id,
              list_only=args.list_runs,
              rerun_completed=args.rerun_completed)

    print(f"\n[exp1] done. Results: {public_path(out_path)}")


if __name__ == "__main__":
    main()
