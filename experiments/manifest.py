"""Offline manifest for the complete experiment matrix.

This module only constructs and validates run specifications.  It never
creates a simulator, reads an API key, or calls a model endpoint.
"""

from collections import Counter

from _common import normalise_specs
from exp1_main_comparison import build_specs as build_exp1
from exp2_ablation import build_specs as build_exp2
from exp3_scaling import (
    B_SWEEP_VALUES, DEFAULT_ASSIGNMENT_BUDGET,
    build_specs as build_exp3, max_epochs_for_batch,
)
from exp4_llm_sensitivity import build_specs as build_exp4
from exp5_case_study import build_specs as build_exp5
from simulation.config import BATCH_SIZE, SEED_BASE


EXPECTED_COUNTS = {"1": 35, "2": 20, "3": 40, "4": 20, "5": 1}


def build_manifest(max_epochs: int = 40,
                   exp3_assignment_budget: int =
                   DEFAULT_ASSIGNMENT_BUDGET) -> list:
    """Return all 116 run specs in deterministic experiment order."""
    seeds = [SEED_BASE + i for i in range(5)]
    specs = []
    specs.extend(build_exp1(seeds=seeds, max_epochs=max_epochs))
    specs.extend(build_exp2(seeds=seeds, max_epochs=max_epochs))
    specs.extend(build_exp3(
        seeds=seeds, assignment_budget=exp3_assignment_budget))
    specs.extend(build_exp4(seeds=seeds, max_epochs=max_epochs))
    specs.extend(build_exp5(max_epochs=max_epochs))
    return normalise_specs(specs)


def validate_manifest(specs: list) -> dict:
    """Raise on a duplicated, misplaced, or incomplete experiment run."""
    if len(specs) != 116:
        raise ValueError(f"expected 116 runs, found {len(specs)}")

    counts = Counter(spec["tag"]["exp"] for spec in specs)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"experiment counts {dict(counts)} != {EXPECTED_COUNTS}")

    run_ids = [spec["run_id"] for spec in specs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("manifest contains duplicate run IDs")
    for spec in specs:
        if spec["tag"]["exp"] == "3":
            expected_cap = max_epochs_for_batch(
                spec["tag"]["B"], spec["tag"]["assignment_budget"])
            if spec.get("max_epochs") != expected_cap:
                raise ValueError(
                    f"Exp. 3 B={spec['tag']['B']} uses max_epochs="
                    f"{spec.get('max_epochs')}, expected {expected_cap}")
        elif spec.get("max_epochs") != 40:
            raise ValueError("non-Exp. 3 runs must use max_epochs=40")

    exp1_methods = {
        spec["method"] for spec in specs if spec["tag"]["exp"] == "1"
    }
    expected_exp1 = {
        "lawnmower", "greedy_coverage", "bayesian_search",
        "clue_prior_sweep", "clue_prior_greedy", "clue_prior_bayes",
        "ours",
    }
    if exp1_methods != expected_exp1:
        raise ValueError(f"Exp. 1 methods are {sorted(exp1_methods)}")

    exp2_conditions = {
        spec["tag"]["condition"]
        for spec in specs if spec["tag"]["exp"] == "2"
    }
    if exp2_conditions != {
            "llm_direct", "no_rag", "no_cot", "no_bottleneck"}:
        raise ValueError(
            f"Exp. 2 conditions are {sorted(exp2_conditions)}")

    if any(spec["method"] == "llm_direct"
           and spec["tag"]["exp"] != "2" for spec in specs):
        raise ValueError("LLM-Direct must belong only to Exp. 2")
    if any(spec["tag"].get("condition") == "no_energy_flt"
           for spec in specs):
        raise ValueError("no_energy_flt is not in the approved matrix")

    exp3_batches = {
        spec["tag"]["B"]
        for spec in specs if spec["tag"]["exp"] == "3"
    }
    if exp3_batches != set(B_SWEEP_VALUES):
        raise ValueError(
            f"Exp. 3 batch sizes are {sorted(exp3_batches)}")
    if any(spec.get("num_uavs") != 10
           for spec in specs if spec["tag"]["exp"] == "3"):
        raise ValueError("Exp. 3 must keep N=10 fixed")

    if BATCH_SIZE != 10:
        raise ValueError(f"main BATCH_SIZE is {BATCH_SIZE}, expected 10")
    if any(spec.get("batch_size", BATCH_SIZE) != 10
           for spec in specs if spec["tag"]["exp"] != "3"):
        raise ValueError("Exp. 1/2/4/5 must use the main B=10 setting")

    exp1_snapshot_methods = {
        spec["method"] for spec in specs
        if spec["tag"]["exp"] == "1" and spec.get("dump_priorities")
    }
    if exp1_snapshot_methods != {"ours"}:
        raise ValueError(
            "Exp. 1 must retain priority snapshots only for ours")
    if any(not spec.get("dump_priorities")
           for spec in specs if spec["tag"]["exp"] in {"4", "5"}):
        raise ValueError(
            "Exp. 4/5 must retain priority snapshots for visual analysis")

    exp4_backbones = {
        spec["tag"]["backbone"]
        for spec in specs if spec["tag"]["exp"] == "4"
    }
    expected_exp4 = {
        "deepseek:deepseek-v4-flash:non-thinking",
        "qwen:qwen-flash:non-thinking",
        "qwen:qwen-plus-2025-12-01:non-thinking",
        "qwen:qwen3-max-2026-01-23:non-thinking",
    }
    if exp4_backbones != expected_exp4:
        raise ValueError(
            f"Exp. 4 backbones are {sorted(exp4_backbones)}")

    return {"total": len(specs), "counts": dict(counts)}


def main():
    specs = build_manifest()
    summary = validate_manifest(specs)
    for spec in specs:
        print(spec["run_id"])
    print(f"VALID: {summary['total']} unique runs; "
          f"counts={summary['counts']}")


if __name__ == "__main__":
    main()
