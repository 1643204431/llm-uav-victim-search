"""
Agent module — wraps each method into a unified decision function
compatible with Simulator.run().

Each method's decision function has signature:
    fn(grid_map, memory, state, uavs, num_unfound) -> dict {uav_id: (r,c)}
"""

from simulation.grid_map import GridMap
from simulation.memory import MemoryModule
from .decision import (
    make_decision, make_decision_llm_direct, configure_ablation,
)
from .baselines import (
    greedy_coverage, lawnmower, random_assignment,
    clue_prior_greedy, clue_prior_sweep, bayesian_search,
)
import numpy as np


def build_proposed_fn(provider: str = None, verbose: bool = False):
    """Build the proposed LLM-based decision function."""
    def fn(grid_map, memory, state, uavs, num_unfound):
        return make_decision(grid_map, memory, state, uavs, num_unfound,
                             provider=provider, verbose=verbose)
    return fn


def build_greedy_fn():
    """Build the greedy coverage decision function."""
    def fn(grid_map, memory, state, uavs, num_unfound):
        return greedy_coverage(grid_map, state, uavs)
    return fn


def build_lawnmower_fn():
    """Build the lawnmower sweep decision function."""
    sweep_state = {"next_cells": []}
    def fn(grid_map, memory, state, uavs, num_unfound):
        return lawnmower(grid_map, state, uavs, sweep_state)
    return fn


def build_random_fn(rng: np.random.Generator):
    """Build the random assignment decision function."""
    def fn(grid_map, memory, state, uavs, num_unfound):
        return random_assignment(grid_map, state, uavs, rng)
    return fn


def build_clue_prior_greedy_fn():
    """Build the fixed clue-prior + greedy baseline."""
    cache = {}
    def fn(grid_map, memory, state, uavs, num_unfound):
        return clue_prior_greedy(grid_map, state, uavs, cache=cache)
    return fn


def build_clue_prior_sweep_fn():
    """Build the fixed clue-prior ordered sweep baseline."""
    sweep_state = {"next_cells": []}
    def fn(grid_map, memory, state, uavs, num_unfound):
        return clue_prior_sweep(
            grid_map, state, uavs, sweep_state=sweep_state)
    return fn


def build_bayesian_fn(*, clue_prior: bool):
    """Build a stateful Bayesian-search baseline."""
    cache = {}
    def fn(grid_map, memory, state, uavs, num_unfound):
        return bayesian_search(
            grid_map, memory, state, uavs, cache=cache,
            clue_prior=clue_prior)
    return fn


def build_llm_direct_fn(provider: str = None, verbose: bool = False):
    """Build the LLM-direct baseline (end-to-end, no two-stage)."""
    def fn(grid_map, memory, state, uavs, num_unfound):
        return make_decision_llm_direct(grid_map, memory, state, uavs,
                                        num_unfound, provider=provider,
                                        verbose=verbose)
    return fn


def build_ours_ablated_fn(provider: str = None, verbose: bool = False,
                          *, use_rag: bool = True, use_cot: bool = True,
                          use_energy_filter: bool = True):
    """Build the proposed method with one or more components ablated.

    Flags match ``decision.configure_ablation``. The flags are applied
    ONCE at factory time; the simulator calls ``reset_stats`` in its
    __init__ which does NOT touch ablation state, so the configuration
    survives the whole run.
    """
    configure_ablation(use_rag=use_rag, use_cot=use_cot,
                       use_energy_filter=use_energy_filter)

    def fn(grid_map, memory, state, uavs, num_unfound):
        # Re-apply in case a previous variant in the same process changed
        # the flags — costs a few nanoseconds, keeps exp 2 deterministic
        # no matter the iteration order.
        configure_ablation(use_rag=use_rag, use_cot=use_cot,
                           use_energy_filter=use_energy_filter)
        return make_decision(grid_map, memory, state, uavs, num_unfound,
                             provider=provider, verbose=verbose)
    return fn


def get_decision_fn(method: str, rng: np.random.Generator = None,
                    provider: str = None, verbose: bool = False,
                    **ablation_kwargs):
    """Factory: return the decision function for a given method name.

    Method names:
      - "proposed" / "ours" : paper's method (LLM priority → bottleneck)
      - "ours_ablated"      : proposed method with ablation flags
                              (pass use_rag/use_cot/use_energy_filter
                              as keyword args)
      - "greedy_coverage"   : nearest-unsearched baseline
      - "lawnmower"         : boustrophedon coverage
      - "bayesian_search"   : uniform prior + Bayesian filtering
      - "clue_prior_sweep"  : clue-prior ordered boustrophedon sweep
      - "clue_prior_greedy" : fixed clue prior + greedy dispatch
      - "clue_prior_bayes"  : clue prior + Bayesian filtering
      - "random"            : random feasible cell
      - "llm_direct"        : LLM outputs assignment directly (no two-stage)

    ablation_kwargs:
        use_rag, use_cot, use_energy_filter — used only when
        method == "ours_ablated"; otherwise ignored.
    """
    if method in ("proposed", "ours"):
        # Reset to defaults in case a prior ablation variant left flags
        # in a non-default state within the same process.
        configure_ablation(use_rag=True, use_cot=True,
                           use_energy_filter=True)
        return build_proposed_fn(provider=provider, verbose=verbose)
    elif method == "ours_ablated":
        return build_ours_ablated_fn(provider=provider, verbose=verbose,
                                     **ablation_kwargs)
    elif method == "greedy_coverage":
        return build_greedy_fn()
    elif method == "lawnmower":
        return build_lawnmower_fn()
    elif method == "bayesian_search":
        return build_bayesian_fn(clue_prior=False)
    elif method == "clue_prior_sweep":
        return build_clue_prior_sweep_fn()
    elif method == "clue_prior_greedy":
        return build_clue_prior_greedy_fn()
    elif method == "clue_prior_bayes":
        return build_bayesian_fn(clue_prior=True)
    elif method == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        return build_random_fn(rng)
    elif method == "llm_direct":
        return build_llm_direct_fn(provider=provider, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")
