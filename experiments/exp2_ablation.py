"""
Experiment 2 — LLM-direct comparison and ablation study (Claim 2).

Conditions:

    llm_direct    : LLM outputs UAV assignments directly
    no_rag        : system prompt omits the SAR domain knowledge block
    no_cot        : system prompt removes STEP 1 / STEP 2 framing

The "full" baseline (all components ON) is NOT re-run here — it is the
canonical ``ours`` row in exp1's output. analyze.py pulls it from the
exp1 JSONL via ``--exp1-path`` and injects it into the ablation plot.

Scenario: default (INITIAL_CLUE=DETAILED, NUM_UAVS=10, NUM_TARGETS=5).
Seeds: SEED_BASE + 0..runs-1 (default 5 seeds, aligned with exp 1).

Each run is stored under ``results/runs/{run_id}/``. The experiment JSONL
is rebuilt from completed per-run results.
"""

import argparse

from _common import (
    run_suite, default_output_path, decision_cache_setup,
    add_run_control_args, public_path,
)
from simulation.config import SEED_BASE


# label → ablation_kwargs (what to flip off)
ABLATIONS = {
    "no_rag":        dict(use_rag=False, use_cot=True,  use_energy_filter=True),
    "no_cot":        dict(use_rag=True,  use_cot=False, use_energy_filter=True),
}


DEFAULT_CONDITIONS = ["llm_direct", "no_rag", "no_cot"]


def build_specs(conditions=None, seeds=None, *, provider=None,
                max_epochs=40, max_time=36000.0,
                time_budget=36000.0) -> list:
    """Build deterministic Exp. 2 run specifications."""
    conditions = list(conditions or DEFAULT_CONDITIONS)
    seeds = list(seeds or [SEED_BASE + i for i in range(5)])
    specs = []
    for condition in conditions:
        if condition == "llm_direct":
            method = "llm_direct"
            ablation_kwargs = None
        else:
            method = "ours_ablated"
            ablation_kwargs = ABLATIONS[condition]
        for i, seed in enumerate(seeds):
            spec = {
                "method": method,
                "seed": seed,
                "provider": provider,
                "max_epochs": max_epochs,
                "max_time": max_time,
                "time_budget": time_budget,
                "verbose": False,
                "tag": {"exp": "2", "condition": condition,
                        "variant": condition, "replicate": i},
            }
            if ablation_kwargs is not None:
                spec["ablation_kwargs"] = ablation_kwargs
            specs.append(spec)
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: LLM-direct comparison and ablations")
    parser.add_argument("--conditions", "--variants", nargs="+", default=None,
                        choices=DEFAULT_CONDITIONS,
                        help="Run selected Exp. 2 conditions")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Exact seeds; overrides --runs")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--max-time", type=float, default=36000.0)
    parser.add_argument("--time-budget", type=float, default=36000.0)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Worker processes for seed-level parallelism "
                             "(default 1 = serial)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the LLM disk cache (for latency runs)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Skip cache lookup but still write back — "
                             "use to overwrite stale cached responses")
    parser.add_argument("--output", type=str, default=None)
    add_run_control_args(parser)
    args = parser.parse_args()

    decision_cache_setup(use_cache=not args.no_cache,
                         force_refresh=args.force_refresh)

    out_path = (args.output if args.output
                else default_output_path("exp2_ablation"))
    cache_state = ("OFF" if args.no_cache
                   else "REFRESH" if args.force_refresh else "ON")
    print(f"[exp2] writing to {public_path(out_path)}")
    selected_conditions = args.conditions or DEFAULT_CONDITIONS
    seeds = (args.seeds if args.seeds is not None else
             [SEED_BASE + i for i in range(args.runs)])
    print(f"[exp2] conditions: {selected_conditions}")
    print(f"[exp2] seeds: {seeds}")
    print(f"[exp2] runs: {args.runs}, max_epochs: {args.max_epochs}, "
          f"cache: {cache_state}, parallel: {args.parallel}")

    all_specs = build_specs(
        DEFAULT_CONDITIONS, seeds, provider=args.provider,
        max_epochs=args.max_epochs, max_time=args.max_time,
        time_budget=args.time_budget)
    specs = [spec for spec in all_specs
             if spec["tag"]["condition"] in selected_conditions]

    run_suite(specs, out_path, parallel=args.parallel,
              use_cache=not args.no_cache,
              force_refresh=args.force_refresh,
              aggregate_specs=all_specs,
              selected_run_ids=args.run_id,
              list_only=args.list_runs,
              rerun_completed=args.rerun_completed)

    print(f"\n[exp2] done. Results: {public_path(out_path)}")


if __name__ == "__main__":
    main()
