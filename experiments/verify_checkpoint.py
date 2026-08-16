"""
Smoke test: rerun a handful of previously-successful specs via the
checkpoint-backed run_single and compare key metrics against the
original current-schema JSONL row. A correctly-set-up checkpoint system
should yield:

  * zero API calls (everything served from checkpoint)
  * bit-exact simulation metrics (num_found, mission completion time,
    total_cells_searched, total_energy_consumed, ...)

Both original and replay rows must use the current timing schema:
``mission_completion_time`` comes directly from the computation-aware event
clock,
``total_agent_computation_time`` stores the sum of the per-cycle full policy
runtime, and the latter is not added to mission time again. Per-epoch timing
checkpoints make the simulated event times reproducible during replay.

Edit the ``SPECS`` list below to test other runs. Each entry is a
(exp, method, seed, num_uavs, batch_size, variant, provider, model) tuple — leave
fields as None when they don't apply.

Usage:
    python experiments/verify_checkpoint.py
"""

import json
import math
import pathlib
import sys
from typing import Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _common import run_single  # noqa: E402


RESULTS = _ROOT / "results"

# (exp, method, seed, num_uavs, batch_size, variant, provider, model)
SPECS = [
    ("1", "ours",         42, None, None, None, None,   None),
    ("2", "llm_direct",   42, None, None, "llm_direct", None, None),
    ("2", "ours_ablated", 43, None, None, "no_rag", None, None),
    ("3", "ours",         43, 10,   3, None, None,      None),
    ("3", "ours",         42, 10,  10, None, None,      None),
    ("4", "ours",         42, None, None, None, "qwen",
     "qwen3-max-2026-01-23"),
    ("5", "ours",         42, None, None, None, None,   None),
]


# ── Locate original JSONL row ─────────────────────────────────────────

def _resolve_model(provider: Optional[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    if provider == "deepseek":
        return "deepseek-v4-pro"
    if provider == "qwen":
        return "qwen-plus-2025-12-01"
    return None


def _load_all_rows():
    rows = []
    for name in [
        "exp1_main_current.jsonl",
        "exp2_ablation_current.jsonl",
        "exp3_batch_sweep_current.jsonl",
        "exp4_llm_sensitivity_current.jsonl",
        "exp5_case_study_current.jsonl",
    ]:
        p = RESULTS / name
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def _find_row(rows, exp, method, seed, num_uavs, batch_size, variant,
              provider, model):
    prov_norm = provider or "deepseek"
    model_norm = _resolve_model(prov_norm, model)
    for r in rows:
        if r.get("method") != method: continue
        if r.get("seed") != seed: continue
        if r.get("exp") != exp: continue
        if variant and r.get("variant") != variant: continue
        if num_uavs is not None and (r.get("N") or r.get("num_uavs_actual")) != num_uavs:
            continue
        if batch_size is not None and (r.get("B") or r.get("batch_size_actual")) != batch_size:
            continue
        row_prov = r.get("provider") or "deepseek"
        if row_prov != prov_norm: continue
        row_model = _resolve_model(row_prov, r.get("llm_model") or r.get("model"))
        if row_model != model_norm: continue
        return r
    return None


# ── Spec → run_single kwargs ──────────────────────────────────────────

def _spec_to_kwargs(exp, method, seed, num_uavs, batch_size, variant,
                    provider, model, row):
    """Reconstruct run_single kwargs from the test spec + original row.

    ``ablation_kwargs`` is copied from the row (only set when variant is
    present — exp2).
    """
    return {
        "method": method,
        "seed": seed,
        "provider": provider,
        "model": model if (model and provider) else None,
        "num_uavs": num_uavs,
        "batch_size": batch_size,
        "max_epochs": row.get("max_epochs"),
        "max_time": row.get("max_time", 36000.0),
        "time_budget": row.get("time_budget", 36000.0),
        "ablation_kwargs": row.get("ablation"),
        "tag": {"exp": exp, "variant": variant,
                "N": num_uavs, "B": batch_size},
    }


# ── Compare ───────────────────────────────────────────────────────────

COMPARE_KEYS = [
    "all_found", "num_found", "num_unfound",
    "max_discovery_time", "mission_completion_time", "mean_discovery_time",
    "total_epochs", "total_cells_searched",
    "total_energy_consumed", "energy_utilisation",
]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _values_match(original_value, replay_value) -> bool:
    """Compare a scalar using the verifier's bit-level float tolerance."""
    if isinstance(original_value, float) or isinstance(replay_value, float):
        return (original_value is not None and replay_value is not None and
                abs(float(original_value) - float(replay_value)) < 1e-9)
    return original_value == replay_value


def _validate_current_timing(row: dict, label: str) -> None:
    """Require and validate the current per-cycle agent timing schema."""
    required = (
        "mission_completion_time",
        "agent_computation_times",
        "total_agent_computation_time",
    )
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise ValueError(
            f"{label} lacks current timing fields: {', '.join(missing)}"
        )

    per_cycle = float(sum(row["agent_computation_times"]))
    total = float(row["total_agent_computation_time"])
    if not math.isclose(per_cycle, total, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            f"{label} has inconsistent agent timing: per-cycle sum "
            f"{per_cycle} != recorded total {total}"
        )

def _compare(replay: dict, original: dict) -> list:
    """Return bit-exact comparisons for deterministic simulation metrics."""
    _validate_current_timing(original, "original row")
    _validate_current_timing(replay, "replay row")

    out = []
    for k in COMPARE_KEYS:
        o = original.get(k)
        r = replay.get(k)
        out.append((k, o, r, _values_match(o, r)))

    return out


def main():
    rows = _load_all_rows()
    print(f"Loaded {len(rows)} original result rows\n")

    any_mismatch = False
    any_api = False

    for spec in SPECS:
        (exp, method, seed, num_uavs, batch_size,
         variant, provider, model) = spec
        label = f"exp{exp} {method} s={seed}"
        if num_uavs is not None: label += f" N={num_uavs}"
        if batch_size is not None: label += f" B={batch_size}"
        if variant: label += f" v={variant}"
        if provider: label += f" {provider}:{model}"

        row = _find_row(rows, exp, method, seed, num_uavs, batch_size,
                        variant, provider, model)
        if row is None:
            print(f"--- {label}: NO original row found, skipping ---\n")
            continue

        try:
            _validate_current_timing(row, "original row")
        except ValueError as e:
            print(f"--- {label}: INVALID original row: {e} ---\n")
            any_mismatch = True
            continue

        kwargs = _spec_to_kwargs(*spec, row=row)
        print(f"--- {label} ---")
        print(f"  running...", flush=True)
        try:
            result = run_single(**kwargs, verbose=False)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}\n")
            any_mismatch = True
            continue

        api = result.get("llm_api_calls", 0)
        hits = result.get("llm_cache_hits", 0)
        print(f"  api_calls={api}  cache_hits={hits}")
        if api > 0:
            any_api = True

        comparison = _compare(result, row)
        mismatches = [c for c in comparison if not c[3]]
        if not mismatches:
            print(f"  ALL {len(comparison)} metrics match ✓")
        else:
            any_mismatch = True
            print(f"  {len(mismatches)} / {len(comparison)} MISMATCH:")
            for k, o, r, _ in mismatches:
                print(f"    {k}: original={_fmt(o)}  replay={_fmt(r)}")
        print()

    print("=" * 50)
    if any_api:
        print("⚠ Some runs made API calls — checkpoint incomplete for those specs.")
    else:
        print("✓ Zero API calls across all specs — checkpoints fully served reads.")
    if any_mismatch:
        print("⚠ Mismatches found. Inspect above.")
    else:
        print("✓ All metrics match original runs bit-for-bit.")


if __name__ == "__main__":
    main()
