"""
Experiment runner for Multi-UAV SAR Simulation.

Usage:
    python run_experiment.py                          # run all methods, all runs
    python run_experiment.py --method proposed         # single method
    python run_experiment.py --method proposed --provider deepseek
    python run_experiment.py --runs 3                  # fewer runs
    python run_experiment.py --method random --runs 5  # quick test
"""

import argparse
import json
import time
import os
import numpy as np

from simulation.config import NUM_RUNS, METHODS, SEED_BASE, LLM_PROVIDER
from simulation.grid_map import GridMap
from simulation.simulator import Simulator
from agent import get_decision_fn
import agent.decision as _decision


def run_single(method: str, run_id: int, seed: int,
               provider: str = None, max_epochs: int = None,
               verbose: bool = False) -> dict:
    """Run a single simulation with the given method and seed."""
    rng = np.random.default_rng(seed)
    grid_map = GridMap()
    sim = Simulator(grid_map, rng)

    decision_fn = get_decision_fn(method, rng=rng, provider=provider,
                                  verbose=verbose)

    print(f"  [{method}] run {run_id}, seed {seed} ... ", end="", flush=True)
    t0 = time.time()
    results = sim.run(decision_fn, max_epochs=max_epochs)
    elapsed = time.time() - t0
    results["wall_time"] = elapsed
    results["method"] = method
    results["run_id"] = run_id
    results["seed"] = seed

    status = "ALL FOUND" if results["all_found"] else f"{results['num_found']}/{results['num_found']+results['num_unfound']} found"
    import math as _m
    e_util_pct = results.get("energy_utilisation", 0.0) * 100.0
    e_per_tgt = results.get("energy_per_found_target", float("inf"))
    e_str = "inf" if _m.isinf(e_per_tgt) else f"{e_per_tgt / 1000.0:.0f}kJ"
    illegal = results.get("llm_illegal_outputs", 0)
    fallback = results.get("llm_fallback_count", 0)
    print(f"{status}, max_T={results['max_discovery_time']:.0f}s, "
          f"epochs={results['total_epochs']}, "
          f"illegal={illegal}, fallback={fallback}, "
          f"E_util={e_util_pct:.1f}%, E/tgt={e_str}, "
          f"wall={elapsed:.1f}s")
    return results


def run_experiment(methods: list, num_runs: int, provider: str = None,
                   max_epochs: int = None):
    """Run the full experiment matrix."""
    os.makedirs("results", exist_ok=True)

    all_results = []
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        for run_id in range(num_runs):
            seed = SEED_BASE + run_id
            result = run_single(method, run_id, seed, provider=provider,
                                max_epochs=max_epochs,
                                verbose=(method == "proposed"))
            all_results.append(result)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"results/experiment_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {result_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<20} {'Found':>6} {'MaxT(s)':>10} {'MeanT(s)':>10} "
          f"{'Epochs':>8} {'Cells':>8} {'E_util':>8}")
    print(f"{'-'*80}")
    for method in methods:
        method_results = [r for r in all_results if r["method"] == method]
        if not method_results:
            continue
        avg_found = np.mean([r["num_found"] for r in method_results])
        avg_maxT = np.mean([r["max_discovery_time"] for r in method_results])
        avg_meanT = np.mean([r["mean_discovery_time"] for r in method_results])
        avg_epochs = np.mean([r["total_epochs"] for r in method_results])
        avg_cells = np.mean([r["total_cells_searched"] for r in method_results])
        avg_energy = np.mean([r["energy_utilisation"] for r in method_results])
        print(f"{method:<20} {avg_found:>6.1f} {avg_maxT:>10.0f} {avg_meanT:>10.0f} "
              f"{avg_epochs:>8.0f} {avg_cells:>8.0f} {avg_energy:>8.2f}")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR Simulation Experiment")
    parser.add_argument("--method", type=str, default=None,
                        help="Single method to run (default: all)")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="List of methods to run")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"Number of runs per method (default: {NUM_RUNS})")
    parser.add_argument("--provider", type=str, default=None,
                        help=f"LLM provider for proposed method (default: {LLM_PROVIDER})")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Max decision epochs per run (default: unlimited)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the on-disk LLM response cache. "
                             "Use this for latency-measurement runs.")
    args = parser.parse_args()

    # Cache policy: ON by default; --no-cache disables both reads and writes.
    _decision.set_cache_enabled(not args.no_cache)

    if args.method:
        methods = [args.method]
    elif args.methods:
        methods = args.methods
    else:
        methods = METHODS

    run_experiment(methods, args.runs, provider=args.provider,
                   max_epochs=args.max_epochs)
