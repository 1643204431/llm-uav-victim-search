"""
Shared helpers for the experiment drivers.

- run_single(...) wraps Simulator.run with full telemetry.
- Every run owns ``results/runs/{run_id}/result.json`` and ``status.json``.
- Experiment JSONLs are deterministic views rebuilt from completed runs.
- Helpers to override mutable config values at runtime (NUM_UAVS,
  BATCH_SIZE, INITIAL_CLUE) via monkey-patching, since config.py values are
  imported eagerly by many modules.
"""

import importlib
import json
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time

import numpy as np

# Make the code root importable when called as `python experiments/xxx.py`
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation import config  # noqa: E402
import agent.decision as _decision  # noqa: E402
from simulation.grid_map import GridMap  # noqa: E402
from simulation.simulator import Simulator  # noqa: E402
from agent import get_decision_fn  # noqa: E402


RESULTS_DIR = pathlib.Path(_ROOT / "results")


# Snapshot module-level defaults at import time so ``run_single`` can
# revert any prior ``override_config`` / ``set_model_for_provider``
# overrides before applying the current spec's values. Without this
# reset, running Exp. 3 (which changes BATCH_SIZE) before Exp. 4 would
# leak the previous batch size into the next simulation.
_DEFAULT_NUM_UAVS = config.NUM_UAVS
_DEFAULT_BATCH_SIZE = config.BATCH_SIZE
_DEFAULT_INITIAL_CLUE = config.INITIAL_CLUE
_DEFAULT_LLM_MODEL_DEEPSEEK = config.LLM_MODEL_DEEPSEEK
_DEFAULT_LLM_THINKING_DEEPSEEK = config.LLM_THINKING_DEEPSEEK
_DEFAULT_LLM_REASONING_EFFORT_DEEPSEEK = config.LLM_REASONING_EFFORT_DEEPSEEK
_DEFAULT_LLM_MODEL_QWEN = config.LLM_MODEL_QWEN
_DEFAULT_LLM_THINKING_QWEN = config.LLM_THINKING_QWEN
_DEFAULT_LLM_MODEL_HUNYUAN = config.LLM_MODEL_HUNYUAN


def override_config(**overrides):
    """Monkey-patch a small set of config values across dependent modules.

    NUM_UAVS, BATCH_SIZE, and INITIAL_CLUE are supported experiment knobs.
    """
    for key, value in overrides.items():
        if key == "NUM_UAVS":
            setattr(config, "NUM_UAVS", value)
            # Modules that imported NUM_UAVS by name need rebinding.
            # After the simulation/agent restructure, modules live under
            # the ``simulation`` package; "run_experiment" is still at
            # repo root.
            for mod_name in ("simulation.simulator", "simulation.uav",
                             "run_experiment"):
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    if hasattr(mod, "NUM_UAVS"):
                        setattr(mod, "NUM_UAVS", value)
        elif key == "BATCH_SIZE":
            setattr(config, "BATCH_SIZE", value)
            for mod_name in ("simulation.simulator", "run_experiment"):
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    if hasattr(mod, "BATCH_SIZE"):
                        setattr(mod, "BATCH_SIZE", value)
        elif key == "INITIAL_CLUE":
            setattr(config, "INITIAL_CLUE", value)
            for mod_name in ("agent.decision", "agent.sar_knowledge",
                             "agent.baselines"):
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    if hasattr(mod, "INITIAL_CLUE"):
                        setattr(mod, "INITIAL_CLUE", value)
        else:
            raise ValueError(f"override_config: unsupported key {key!r}")


def _reset_overrides_to_defaults():
    """Revert NUM_UAVS / BATCH_SIZE / INITIAL_CLUE / provider models to the
    values captured at module import.

    Idempotent and cheap — call at the top of every ``run_single`` so
    state set by a prior spec cannot leak into the current one.
    """
    override_config(NUM_UAVS=_DEFAULT_NUM_UAVS,
                    BATCH_SIZE=_DEFAULT_BATCH_SIZE,
                    INITIAL_CLUE=_DEFAULT_INITIAL_CLUE)
    config.LLM_MODEL_DEEPSEEK = _DEFAULT_LLM_MODEL_DEEPSEEK
    _decision.LLM_MODEL_DEEPSEEK = _DEFAULT_LLM_MODEL_DEEPSEEK
    config.LLM_THINKING_DEEPSEEK = _DEFAULT_LLM_THINKING_DEEPSEEK
    _decision.LLM_THINKING_DEEPSEEK = _DEFAULT_LLM_THINKING_DEEPSEEK
    config.LLM_REASONING_EFFORT_DEEPSEEK = (
        _DEFAULT_LLM_REASONING_EFFORT_DEEPSEEK)
    _decision.LLM_REASONING_EFFORT_DEEPSEEK = (
        _DEFAULT_LLM_REASONING_EFFORT_DEEPSEEK)
    config.LLM_MODEL_QWEN = _DEFAULT_LLM_MODEL_QWEN
    _decision.LLM_MODEL_QWEN = _DEFAULT_LLM_MODEL_QWEN
    config.LLM_THINKING_QWEN = _DEFAULT_LLM_THINKING_QWEN
    _decision.LLM_THINKING_QWEN = _DEFAULT_LLM_THINKING_QWEN
    config.LLM_MODEL_HUNYUAN = _DEFAULT_LLM_MODEL_HUNYUAN
    _decision.LLM_MODEL_HUNYUAN = _DEFAULT_LLM_MODEL_HUNYUAN


LLM_METHODS = {
    "ours", "ours_ablated", "llm_direct",
}


def _resolve_model(provider: str, explicit: str = None) -> str:
    """Return the model name that ``_call_llm`` will actually use.

    Mirrors the provider-to-model-constant dispatch inside ``_call_llm``
    so callers can build a run-id string that matches exactly what the
    decision module will log.
    """
    if explicit is not None:
        return explicit
    if provider == "deepseek":
        return config.LLM_MODEL_DEEPSEEK
    if provider == "qwen":
        return config.LLM_MODEL_QWEN
    if provider == "hunyuan":
        return config.LLM_MODEL_HUNYUAN
    raise ValueError(f"Unknown provider: {provider!r}")


def _run_id_from_spec(method: str, seed: int, provider: str, model: str,
                      num_uavs: int, batch_size: int, tag: dict,
                      thinking: bool = None) -> str:
    """Deterministic run identifier that matches the subdir naming used
    by ``results/runs/``.

    Format: ``exp{exp}_{method}_N{N}_B{B}[_v{variant}]_s{seed}``
    ``[_{provider}_{model}_{mode}]``
      * N and B are always explicit, including the main N=10, B=10 setup
      * ``v{variant}`` included when a variant tag is present (exp2)
      * provider/model/mode suffix included only for LLM-using methods
    """
    tag = tag or {}
    exp = tag.get("exp")
    variant = tag.get("variant")
    effective_num_uavs = (
        int(num_uavs) if num_uavs is not None else int(_DEFAULT_NUM_UAVS))
    tagged_batch_size = tag.get("B")
    effective_batch_size = (
        int(batch_size) if batch_size is not None
        else int(tagged_batch_size) if tagged_batch_size is not None
        else int(_DEFAULT_BATCH_SIZE))

    parts = []
    if exp:
        parts.append(f"exp{exp}")
    parts.append(method)
    parts.append(f"N{effective_num_uavs}")
    parts.append(f"B{effective_batch_size}")
    if variant and variant != method:
        parts.append(f"v{variant}")
    parts.append(f"s{seed}")
    if method in LLM_METHODS:
        prov = provider or config.LLM_PROVIDER
        mod = _resolve_model(prov, model)
        parts.append(prov)
        if mod:
            parts.append(mod)
        if thinking is None:
            if prov == "deepseek":
                thinking = bool(config.LLM_THINKING_DEEPSEEK)
            elif prov == "qwen":
                thinking = bool(config.LLM_THINKING_QWEN)
            else:
                thinking = False
        parts.append("thinking" if thinking else "nonthinking")
    return "_".join(parts)


def stable_run_id(spec: dict) -> str:
    """Return the deterministic public identifier for one run spec."""
    return _run_id_from_spec(
        spec["method"], spec["seed"], spec.get("provider"),
        spec.get("model"), spec.get("num_uavs"), spec.get("batch_size"),
        spec.get("tag"),
        thinking=spec.get("thinking"),
    )


def set_model_for_provider(provider: str, model: str,
                           thinking: bool = None,
                           reasoning_effort: str = None):
    """Override the per-provider model and inference mode.

    Lives here (rather than in a single driver) so parallel workers can
    apply a per-spec model override without each driver reimplementing
    it. Safe to call before every ``run_single`` — it just rebinds a
    module attribute in the current process.
    """
    if provider == "deepseek":
        config.LLM_MODEL_DEEPSEEK = model
        _decision.LLM_MODEL_DEEPSEEK = model
        if thinking is not None:
            config.LLM_THINKING_DEEPSEEK = bool(thinking)
            _decision.LLM_THINKING_DEEPSEEK = bool(thinking)
        if reasoning_effort is not None:
            config.LLM_REASONING_EFFORT_DEEPSEEK = reasoning_effort
            _decision.LLM_REASONING_EFFORT_DEEPSEEK = reasoning_effort
    elif provider == "qwen":
        config.LLM_MODEL_QWEN = model
        _decision.LLM_MODEL_QWEN = model
        if thinking is not None:
            config.LLM_THINKING_QWEN = bool(thinking)
            _decision.LLM_THINKING_QWEN = bool(thinking)
    elif provider == "hunyuan":
        config.LLM_MODEL_HUNYUAN = model
        _decision.LLM_MODEL_HUNYUAN = model
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def run_single(method: str, seed: int, *,
               run_id: str = None,
               provider: str = None,
               model: str = None,
               thinking: bool = None,
               reasoning_effort: str = None,
               max_epochs: int = None,
               max_time: float = 36000.0,
               time_budget: float = 36000.0,
               verbose: bool = False,
               num_uavs: int = None,
               batch_size: int = None,
               dump_priorities: bool = False,
               tag: dict = None,
               ablation_kwargs: dict = None) -> dict:
    """Run one simulation and return a fully-populated result dict.

    Args:
        method: one of the strings accepted by agent.get_decision_fn
        seed: seeds the numpy RNG used by Simulator + targets + detection
        run_id: stable identifier supplied by the suite runner. If omitted,
                it is derived from the remaining arguments.
        provider: "deepseek" | "qwen" | "hunyuan" (None -> config.LLM_PROVIDER)
        model: optional per-call model override; requires
               provider to be set. Applied before the sim runs so the
               decision module picks up the new model constant.
        thinking: optional explicit thinking-mode switch; requires provider.
        reasoning_effort: DeepSeek thinking effort (e.g. ``high``).
        max_epochs: cap on decision epochs (None -> unlimited)
        max_time: cap on simulated time in seconds
        time_budget: cutoff for success_rate_within_budget metric
        verbose: prints LLM reasoning (only meaningful for LLM methods)
        num_uavs: override config.NUM_UAVS for this run
        batch_size: override config.BATCH_SIZE for this run (Exp. 3)
        dump_priorities: retain the per-cycle priority grids for offline
                         spatial-quality analysis
        tag: free-form dict merged into the result (e.g. {"exp": "1"})
    """
    # Reset any leaked overrides from a prior run before applying this
    # spec's. Prevents one sensitivity setting leaking into another run.
    _reset_overrides_to_defaults()

    # Apply any runtime overrides BEFORE constructing the simulator so
    # that the decision module picks up the new values.
    if num_uavs is not None:
        override_config(NUM_UAVS=num_uavs)
    if batch_size is not None:
        override_config(BATCH_SIZE=batch_size)
    if model is not None or thinking is not None or reasoning_effort is not None:
        if provider is None:
            raise ValueError("run_single: model/mode overrides require provider")
        resolved_model = model or _resolve_model(provider)
        set_model_for_provider(provider, resolved_model,
                               thinking=thinking,
                               reasoning_effort=reasoning_effort)

    # Register seed + run identifier with the decision module. The
    # run_id determines which ``results/runs/{run_id}/checkpoint/``
    # directory this run's LLM calls read from / write to.
    if run_id is None:
        run_id = _run_id_from_spec(
            method, seed, provider, model, num_uavs, batch_size, tag,
            thinking=thinking)
    _decision.set_current_seed(seed)
    _decision.set_current_run_id(run_id)
    _decision.set_dump_priorities(dump_priorities)

    try:
        rng = np.random.default_rng(seed)
        grid_map = GridMap()
        sim = Simulator(grid_map, rng)

        kw = dict(ablation_kwargs or {})
        fn = get_decision_fn(method, rng=rng, provider=provider,
                             verbose=verbose, **kw)

        t0 = time.time()
        results = sim.run(fn, max_time=max_time, max_epochs=max_epochs,
                          time_budget=time_budget)
        elapsed = time.time() - t0
    finally:
        # Worker processes can execute several specs sequentially.  Never
        # leak snapshot collection into the next method in that process.
        _decision.set_dump_priorities(False)

    results["wall_time"] = elapsed
    results["method"] = method
    results["seed"] = seed
    results["provider"] = provider or config.LLM_PROVIDER
    results["num_uavs_actual"] = config.NUM_UAVS
    results["batch_size_actual"] = config.BATCH_SIZE
    results["max_epochs"] = max_epochs
    results["max_time"] = max_time
    results["time_budget"] = time_budget
    results["priority_snapshots_enabled"] = bool(dump_priorities)
    actual_provider = provider or config.LLM_PROVIDER
    if method in LLM_METHODS:
        results["llm_model"] = _resolve_model(actual_provider)
        if actual_provider == "deepseek":
            results["llm_thinking"] = bool(config.LLM_THINKING_DEEPSEEK)
            results["llm_reasoning_effort"] = (
                config.LLM_REASONING_EFFORT_DEEPSEEK
                if config.LLM_THINKING_DEEPSEEK else None)
        elif actual_provider == "qwen":
            results["llm_thinking"] = bool(config.LLM_THINKING_QWEN)
    if ablation_kwargs:
        results["ablation"] = dict(ablation_kwargs)
    if tag:
        results.update(tag)
    # These fields are controlled by the runner and must not be overwritten
    # by a free-form tag.
    results["run_id"] = run_id
    results["status"] = "completed"
    return results


def format_summary(r: dict) -> str:
    """One-line human-readable summary for console output."""
    status = ("ALL" if r.get("all_found")
              else f"{r['num_found']}/{r['num_found']+r['num_unfound']}")
    cache_hits = r.get("llm_cache_hits", 0)
    illegal = r.get("llm_illegal_outputs", 0)
    fallback = r.get("llm_fallback_count", 0)
    e_util_pct = r.get("energy_utilisation", 0.0) * 100.0
    e_per_tgt = r.get("energy_per_found_target", float("inf"))
    # kJ unit (J / 1000); show "inf" if no targets found
    import math as _m
    e_str = "inf" if _m.isinf(e_per_tgt) else f"{e_per_tgt / 1000.0:.0f}kJ"
    return (f"{status} found, max_T={r['max_discovery_time']:.0f}s, "
            f"epochs={r['total_epochs']}, "
            f"api={r.get('llm_api_calls', 0)}/hits={cache_hits}, "
            f"illegal={illegal}, fallback={fallback}, "
            f"E_util={e_util_pct:.1f}%, E/tgt={e_str}, "
            f"wall={r['wall_time']:.1f}s")


def default_output_path(prefix: str) -> pathlib.Path:
    """Return the deterministic aggregate path for an experiment.

    The JSONL is a derived view rebuilt from per-run ``result.json`` files;
    it is never the source of truth and rebuilding it never calls an API.
    """
    return RESULTS_DIR / f"{prefix}_current.jsonl"


def public_path(path) -> str:
    """Return a log-safe path without a username or absolute prefix."""
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(_ROOT.resolve()))
    except (OSError, ValueError):
        return p.name


def add_run_control_args(parser):
    """Add common exact-rerun and inspection flags to a driver parser."""
    parser.add_argument(
        "--run-id", action="append", default=None,
        help="Run only this stable run ID; repeat for multiple runs")
    parser.add_argument(
        "--list-runs", action="store_true",
        help="List selected run IDs and exit without running simulations")
    parser.add_argument(
        "--rerun-completed", action="store_true",
        help="Run selected IDs even when a completed result already exists")


def run_directory(run_id: str) -> pathlib.Path:
    """Return the isolated directory for one stable run ID."""
    return RESULTS_DIR / "runs" / run_id


def result_path(run_id: str) -> pathlib.Path:
    return run_directory(run_id) / "result.json"


def status_path(run_id: str) -> pathlib.Path:
    return run_directory(run_id) / "status.json"


def _atomic_write_json(path: pathlib.Path, payload: dict):
    """Atomically replace one JSON artifact."""
    validate_anonymous_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)
        f.write("\n")
    os.replace(tmp, path)


def persist_json_artifact(path, payload):
    """Persist an additional anonymous per-run JSON artifact atomically."""
    _atomic_write_json(pathlib.Path(path), payload)


def _sanitise_error_message(value) -> str:
    """Remove secrets and local absolute paths from persisted errors."""
    text = str(value)
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted-key>", text)
    text = re.sub(r"/Users/[^\s:'\"]+", "<local-path>", text)
    text = re.sub(r"/home/[^\s:'\"]+", "<local-path>", text)
    text = re.sub(r"/private/(?:tmp|var)/[^\s:'\"]+",
                  "<temporary-path>", text)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\s:'\"]+",
                  "<local-path>", text)
    text = re.sub(r"\bwxid_[A-Za-z0-9_]+", "<redacted-id>", text)
    return text[:1000]


def sanitise_error_message(value) -> str:
    """Public wrapper used by standalone experiment drivers."""
    return _sanitise_error_message(value)


def validate_anonymous_payload(payload):
    """Reject result artifacts containing common secret/personal markers."""
    text = json.dumps(payload, default=str)
    forbidden = {
        "API key": r"sk-[A-Za-z0-9_-]{8,}",
        "macOS user path": r"/Users/[^/\s]+/",
        "Linux user path": r"/home/[^/\s]+/",
        "Windows user path": r"[A-Za-z]:\\Users\\[^\\\s]+\\",
        "WeChat identifier": r"\bwxid_[A-Za-z0-9_]+",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, text):
            raise ValueError(f"anonymous artifact check failed: {label}")


def _public_spec_fields(spec: dict) -> dict:
    """Return anonymous, result-relevant fields for a status record."""
    tag = dict(spec.get("tag") or {})
    fields = {
        "run_id": spec["run_id"],
        "exp": tag.get("exp"),
        "method": spec.get("method"),
        "seed": spec.get("seed"),
    }
    for key in ("variant", "condition", "N", "B", "backbone"):
        if tag.get(key) is not None:
            fields[key] = tag[key]
    return fields


def normalise_specs(specs: list) -> list:
    """Attach stable IDs and reject duplicate run definitions."""
    out = []
    seen = set()
    for original in specs:
        spec = dict(original)
        rid = spec.get("run_id") or stable_run_id(spec)
        if rid in seen:
            raise ValueError(f"duplicate run_id in experiment matrix: {rid}")
        seen.add(rid)
        spec["run_id"] = rid
        out.append(spec)
    return out


def is_completed(spec: dict) -> bool:
    """True only when status and result agree on a completed run."""
    rid = spec["run_id"]
    try:
        with open(status_path(rid)) as f:
            status = json.load(f)
        with open(result_path(rid)) as f:
            result = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        status.get("status") == "completed"
        and status.get("run_id") == rid
        and result.get("status") == "completed"
        and result.get("run_id") == rid
        and result.get("method") == spec.get("method")
        and result.get("seed") == spec.get("seed")
        and result.get("max_epochs") == spec.get("max_epochs")
        and result.get("max_time") == spec.get("max_time")
        and result.get("time_budget") == spec.get("time_budget")
        and result.get("num_uavs_actual") == (
            spec.get("num_uavs")
            if spec.get("num_uavs") is not None
            else _DEFAULT_NUM_UAVS)
        and result.get("batch_size_actual") == (
            spec.get("batch_size")
            if spec.get("batch_size") is not None
            else _DEFAULT_BATCH_SIZE)
        and bool(result.get("priority_snapshots_enabled", False)) ==
            bool(spec.get("dump_priorities", False))
    )


def mark_run_status(spec: dict, status: str, error=None):
    """Write an anonymous status record for one run."""
    record = _public_spec_fields(spec)
    record["status"] = status
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error_message"] = _sanitise_error_message(error)
    _atomic_write_json(status_path(spec["run_id"]), record)


def persist_completed_result(result: dict):
    """Persist the authoritative result before marking the run complete."""
    rid = result["run_id"]
    result = dict(result)
    result["status"] = "completed"
    _atomic_write_json(result_path(rid), result)
    status = {
        "run_id": rid,
        "exp": result.get("exp"),
        "method": result.get("method"),
        "seed": result.get("seed"),
        "status": "completed",
    }
    for key in ("variant", "condition", "N", "B", "backbone"):
        if result.get(key) is not None:
            status[key] = result[key]
    _atomic_write_json(status_path(rid), status)


def rebuild_jsonl(out_path, specs: list) -> int:
    """Rebuild an aggregate JSONL solely from completed per-run results."""
    out_path = pathlib.Path(out_path)
    normalised = normalise_specs(specs)
    rows = []
    for spec in normalised:
        if not is_completed(spec):
            continue
        with open(result_path(spec["run_id"])) as f:
            rows.append(json.load(f))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        for row in rows:
            validate_anonymous_payload(row)
            f.write(json.dumps(row, default=str) + "\n")
    os.replace(tmp, out_path)
    return len(rows)


def decision_cache_setup(use_cache: bool, force_refresh: bool = False):
    """Configure the decision-module LLM cache.

    Args:
        use_cache: master switch. When False, no disk cache is consulted
                   or written; every call goes to the API.
        force_refresh: when True, cache lookup is skipped but writes still
                       happen (so cached entries get overwritten with fresh
                       responses). Ignored when use_cache is False.
    """
    _decision.set_cache_enabled(use_cache)
    _decision.set_force_refresh(force_refresh)


# ══════════════════════════════════════════════════════════════════════
#  Parallel execution
# ══════════════════════════════════════════════════════════════════════

def _init_worker(use_cache: bool, force_refresh: bool):
    """Pool initializer — runs ONCE per spawned worker process.

    On macOS / spawn-based multiprocessing, the child starts with a
    fresh import of ``decision`` and does not inherit parent state, so
    cache policy must be re-applied here. Each worker then runs specs
    sequentially, applying per-spec overrides (seed, ablation, model,
    N, B) inside ``run_single``.
    """
    _decision.set_cache_enabled(use_cache)
    _decision.set_force_refresh(force_refresh)


def _run_worker(spec: dict) -> dict:
    """Top-level worker: execute one spec, persist/return the result.

    Must be a module-level function so multiprocessing can pickle it.

    If ``spec`` carries a ``_result_path`` key (set by ``run_suite`` for
    parallel runs), the full result dict is written to that temp file
    and only a tiny handshake dict is returned through the Pool queue.
    This is necessary because each result can be 100-300 KB (``llm_calls``
    trace, dispatch_log, token lists); putting that through macOS's
    default 16 KB pipe buffer under contention causes BrokenPipeError.

    Exceptions are captured and returned as ``{"_error": ...}`` records
    so one bad seed does not abort the whole suite.
    """
    spec = dict(spec)
    method = spec.pop("method")
    seed = spec.pop("seed")
    result_path = spec.pop("_result_path", None)
    run_id = spec.get("run_id")
    tag = dict(spec.get("tag") or {})
    try:
        result = run_single(method, seed, **spec)
    except Exception as e:
        result = {
            "_error": _sanitise_error_message(e),
            "_error_type": type(e).__name__,
            "run_id": run_id,
            "exp": tag.get("exp"),
            "method": method,
            "seed": seed,
        }
        for key in ("variant", "condition", "N", "B", "backbone"):
            if tag.get(key) is not None:
                result[key] = tag[key]

    if result_path is None:
        # Serial path (no Pool) — caller expects the full dict back.
        return result

    # Parallel path — persist the heavy payload to disk and return a
    # tiny handshake dict so the Pool queue stays clear.
    try:
        with open(result_path, "w") as f:
            json.dump(result, f, default=str)
    except Exception as e:
        return {"_error": f"failed to persist result: "
                          f"{type(e).__name__}: "
                          f"{_sanitise_error_message(e)}",
                "_error_type": type(e).__name__,
                "run_id": run_id, "exp": tag.get("exp"),
                "method": method, "seed": seed}
    return {"_result_path": result_path,
            "run_id": run_id, "exp": tag.get("exp"),
            "method": method, "seed": seed,
            "is_error": "_error" in result}


def _persist_one(r: dict):
    """Print/persist one worker result; return True only on success."""
    if "_error" in r:
        rid = r.get("run_id")
        safe_error = _sanitise_error_message(r["_error"])
        spec = {
            "run_id": rid,
            "method": r.get("method"),
            "seed": r.get("seed"),
            "tag": {key: r.get(key) for key in
                    ("exp", "variant", "condition", "N", "B", "backbone")},
        }
        error = RuntimeError(safe_error)
        mark_run_status(spec, "failed", error=error)
        # Preserve the original exception class separately from the wrapper
        # used to pass the already-sanitised message to mark_run_status.
        failed_status = json.loads(status_path(rid).read_text())
        failed_status["error_type"] = r.get("_error_type", "RuntimeError")
        _atomic_write_json(status_path(rid), failed_status)
        print(f"  [{rid}] FAILED: {safe_error}")
        return False
    label = r.get("method", "?")
    # Annotate the console line with the condition so parallel output
    # is still readable (rows complete in arbitrary order).
    extras = []
    if r.get("variant") is not None:
        extras.append(f"variant={r['variant']}")
    if r.get("N") is not None:
        extras.append(f"N={r['N']}")
    if r.get("B") is not None:
        extras.append(f"B={r['B']}")
    if r.get("backbone") is not None:
        extras.append(f"bb={r['backbone']}")
    suffix = f"  [{' '.join(extras)}]" if extras else ""
    print(f"  [{r['run_id']}]{suffix} {format_summary(r)}")
    persist_completed_result(r)
    return True


def run_suite(specs: list, out_path,
              parallel: int = 1,
              use_cache: bool = True,
              force_refresh: bool = False,
              aggregate_specs: list = None,
              selected_run_ids: list = None,
              list_only: bool = False,
              rerun_completed: bool = False):
    """Execute isolated runs and rebuild a derived aggregate JSONL.

    Args:
        specs: selected run specs. Each must contain ``method`` and ``seed``.
        out_path: deterministic, derived JSONL aggregate path.
        parallel: worker count. ``<=1`` → serial execution in-process
                  (preserves the pre-parallel behaviour exactly).
        aggregate_specs: full experiment matrix used to rebuild ``out_path``;
                         defaults to ``specs``.
        selected_run_ids: optional exact stable-ID filter.
        list_only: print run IDs without executing or writing artifacts.
        rerun_completed: execute even when a valid completed result exists.

    Seeds are independent, so completion order under ``imap_unordered``
    is undefined. The JSONL row order therefore may not match ``specs``
    order — downstream analysis groups rows by tag fields, not order.

    Parallel mode uses **file-based result passing**: each worker writes
    its full result dict to a per-spec temp JSON, then returns a small
    (~200-byte) handshake dict via the Pool queue. This avoids pushing
    100-300 KB pickled payloads through macOS's default 16 KB pipe
    buffer, which otherwise causes BrokenPipeError when several workers
    finish around the same time.
    """
    out_path = pathlib.Path(out_path)
    execution_specs = normalise_specs(specs)
    aggregate_specs = normalise_specs(
        aggregate_specs if aggregate_specs is not None else specs)

    if selected_run_ids:
        requested = set(selected_run_ids)
        available = {spec["run_id"] for spec in execution_specs}
        missing = sorted(requested - available)
        if missing:
            raise ValueError("unknown --run-id: " + ", ".join(missing))
        execution_specs = [spec for spec in execution_specs
                           if spec["run_id"] in requested]

    if list_only:
        for spec in execution_specs:
            state = "completed" if is_completed(spec) else "pending"
            print(f"{spec['run_id']}\t{state}")
        return {"listed": len(execution_specs), "executed": 0}

    pending = []
    skipped = 0
    for spec in execution_specs:
        if is_completed(spec) and not rerun_completed:
            print(f"  [{spec['run_id']}] SKIP completed")
            skipped += 1
            continue
        mark_run_status(spec, "running")
        pending.append(spec)

    parallel = max(1, int(parallel))
    if not pending:
        rows = rebuild_jsonl(out_path, aggregate_specs)
        print(f"  [aggregate] {rows} completed rows -> {public_path(out_path)}")
        return {"executed": 0, "skipped": skipped, "failed": 0,
                "aggregated": rows}

    # Remove any formerly-completed row from the derived view before a
    # deliberate rerun begins. If the process is interrupted, status.json
    # and the aggregate still agree that the run is not currently valid.
    rebuild_jsonl(out_path, aggregate_specs)

    if parallel == 1:
        # Serial path: use the same module state that the driver set up.
        failures = 0
        for spec in pending:
            if not _persist_one(_run_worker(spec)):
                failures += 1
        rows = rebuild_jsonl(out_path, aggregate_specs)
        print(f"  [aggregate] {rows} completed rows -> {public_path(out_path)}")
        if failures:
            raise RuntimeError(
                f"{failures} run(s) failed; rerun their stable IDs")
        return {"executed": len(pending), "skipped": skipped,
                "failed": 0, "aggregated": rows}

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="runsuite_"))
    pool = None
    failures = 0
    try:
        enriched = []
        for i, spec in enumerate(pending):
            s = dict(spec)
            s["_result_path"] = str(tmp_dir / f"r_{i:05d}.json")
            enriched.append(s)

        # maxtasksperchild=5: each worker process exits after 5 specs and
        # is replaced by a fresh one. Prevents accumulation of leaked
        # semaphores / memory / cached Python objects over long runs.
        # Cost: one extra spawn per 5 specs (~2-3s on macOS). Negligible
        # against multi-minute LLM spec walltimes.
        pool = mp.Pool(parallel,
                       initializer=_init_worker,
                       initargs=(use_cache, force_refresh),
                       maxtasksperchild=5)

        # chunksize=1 so each handshake streams back as soon as its
        # worker finishes (rather than being batched inside Pool).
        for meta in pool.imap_unordered(_run_worker, enriched, chunksize=1):
            result_path = meta.get("_result_path")
            if result_path and os.path.exists(result_path):
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                except Exception as e:
                    ok = _persist_one({
                        "_error": f"failed to read result file: "
                                  f"{type(e).__name__}: "
                                  f"{_sanitise_error_message(e)}",
                        "_error_type": type(e).__name__,
                        "run_id": meta.get("run_id"),
                        "exp": meta.get("exp"),
                        "method": meta.get("method"),
                        "seed": meta.get("seed"),
                    })
                else:
                    ok = _persist_one(result)
                finally:
                    try:
                        os.unlink(result_path)
                    except OSError:
                        pass
                if not ok:
                    failures += 1
            else:
                # Worker couldn't even write its result; the handshake
                # dict itself carries the error.
                if not _persist_one(meta):
                    failures += 1
    finally:
        # Explicit three-step teardown so workers are guaranteed dead
        # before the next experiment's Pool is created. ``with mp.Pool``
        # only calls terminate() which does not wait for workers to die
        # — that can leak semaphores and leave zombie processes that
        # pollute the next Pool.
        if pool is not None:
            try:
                pool.close()    # stop accepting new tasks
            except Exception:
                pass
            try:
                pool.terminate()  # SIGTERM any stragglers
            except Exception:
                pass
            try:
                pool.join()     # wait for worker processes to truly exit
            except Exception:
                pass
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass

    rows = rebuild_jsonl(out_path, aggregate_specs)
    print(f"  [aggregate] {rows} completed rows -> {public_path(out_path)}")
    if failures:
        raise RuntimeError(
            f"{failures} run(s) failed; rerun their stable IDs")
    return {"executed": len(pending), "skipped": skipped,
            "failed": 0, "aggregated": rows}
