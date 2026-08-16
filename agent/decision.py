"""
Per-cycle policy (paper Section III).

Two-stage pipeline:
  Stage 1 (paper §III-A, LLM):       mobility-pattern reasoning + cell
                                     prioritisation → priority vector w(t).
  Stage 2 (paper §III-B, Optimizer): bottleneck assignment over the top-B
                                     candidate cells, solved by binary
                                     search on travel times + bipartite
                                     maximum matching.

Supports DeepSeek, Qwen (Dashscope), and Hunyuan (Tencent) as LLM
backends via their OpenAI-compatible endpoints. All providers are called
with temperature=0 for reproducibility.

Per-call LLM responses are checkpointed at
``results/runs/{run_id}/checkpoint/call_{N:04d}.json`` (N is the
sequential call index within the run) so a finished run can be replayed
with zero API cost. ``set_force_refresh(True)`` bypasses the checkpoint
on read while still writing; ``set_cache_enabled(False)`` disables it
entirely.

Telemetry: ``reset_stats()`` / ``get_stats()`` expose per-run token,
latency, and parse-quality aggregates without a class refactor.
"""

import json
import os
import pathlib
import re
import time
import httpx
import numpy as np
from openai import APIConnectionError, APITimeoutError

from simulation.config import (
    EPSILON, ALPHA, BETA, CRUISE_SPEED, GRID_ROWS, GRID_COLS,
    LLM_PROVIDER,
    LLM_MODEL_DEEPSEEK, LLM_THINKING_DEEPSEEK,
    LLM_REASONING_EFFORT_DEEPSEEK, DEEPSEEK_BASE_URL,
    LLM_MODEL_QWEN, LLM_THINKING_QWEN, QWEN_BASE_URL,
    LLM_MODEL_HUNYUAN, HUNYUAN_BASE_URL,
    NUM_TARGETS, INITIAL_CLUE, TOP_K_PRIORITY_CELLS,
)
from simulation.grid_map import GridMap
from simulation.memory import MemoryModule
from simulation.perception import PerceptionModule
from .sar_knowledge import retrieve_knowledge


# ══════════════════════════════════════════════════════════════════════
#  Per-run telemetry (cleared at the start of each simulation run)
# ══════════════════════════════════════════════════════════════════════

_llm_latencies = []          # per-call latencies in seconds (real API AND cache hits)
_llm_input_tokens = []       # per-call input/prompt tokens (real API AND cache hits)
_llm_output_tokens = []      # per-call output/completion tokens (real API AND cache hits)
_llm_cache_hits = 0          # number of cache hits this run
_llm_api_calls = 0           # number of true API calls this run
_llm_calls = []              # per-call trace:
                             #   [{"idx": int, "epoch": int, "from_cache": bool,
                             #     "tokens_in": int, "tokens_out": int,
                             #     "latency": float}, ...]
_llm_illegal_outputs = 0     # # of responses where the LLM assigned a
                             # priority to at least one impassable cell
_llm_impassable_drops = 0    # total # of impassable-cell priorities dropped
                             # (sum across responses; can be > illegal_outputs)
_llm_fallback_count = 0      # # of decisions where every parse-retry
                             # attempt failed and the method-specific
                             # fallback actually ended up being used.
_llm_parse_retry_events = [] # per-failed-attempt trace for the parse
                             # retry loop. Each entry:
                             #   {"call_index": int (matches checkpoint),
                             #    "attempt":    int (0-based within decision),
                             #    "kind":       "max_tokens" | "tiny_response"
                             #                 | "multi_json_block" | "no_json"
                             #                 | "json_decode_error",
                             #    "parser":     "priorities" | "direct"}
                             # Useful for experiment-level reporting (e.g.
                             # "how often did a model hit max_tokens?").
_solver_latencies = []       # seconds for each solve_assignment call
_priority_snapshots = []     # [{"t": float, "w": list-of-lists}] — dump_priorities only
_dump_priorities = False     # toggled by the agent factory for case-study runs

# Cache policy (toggled via set_cache_enabled; default ON)
_use_cache = True
_force_refresh = False       # when True, skip checkpoint lookup but still write
_runs_dir = pathlib.Path("results/runs")   # per-run checkpoint root

# Current seed (set by experiment driver before sim starts).
_current_seed = None

# Current run identifier (set via set_current_run_id). Determines the
# per-run checkpoint subdirectory: results/runs/{_current_run_id}/checkpoint/.
# Checkpoint storage is skipped when this is None (ad-hoc / test runs).
_current_run_id = None

# Monotonic LLM call counter within the current run. Reset by
# reset_stats() at the start of each sim. Every ``_call_llm`` invocation
# writes to ``call_{_call_counter:04d}.json`` before incrementing, so
# identical prompts across epochs produce distinct checkpoint files.
_call_counter = 0

# ── Ablation flags (Exp 2) ───────────────────────────────────────────
# All default to True, reproducing the paper's proposed method.
# configure_ablation(...) sets them before each run.
_use_rag = True
_use_cot = True
_use_energy_filter = True


def configure_ablation(use_rag: bool = None, use_cot: bool = None,
                       use_energy_filter: bool = None):
    """Set ablation flags; fields left as None are unchanged.

    Invalidates the cached system prompt because use_rag/use_cot affect
    the system message content.
    """
    global _use_rag, _use_cot, _use_energy_filter
    global _cached_system_msg
    if use_rag is not None:
        _use_rag = bool(use_rag)
    if use_cot is not None:
        _use_cot = bool(use_cot)
    if use_energy_filter is not None:
        _use_energy_filter = bool(use_energy_filter)
    _cached_system_msg = None  # force rebuild on next epoch


def get_ablation() -> dict:
    """Return the current ablation flag snapshot (for logging)."""
    return {
        "use_rag": _use_rag,
        "use_cot": _use_cot,
        "use_energy_filter": _use_energy_filter,
    }


def reset_stats():
    """Clear per-run telemetry AND invalidate the cached system prompt.

    Must be called at the start of each simulation run so stats do not leak
    between runs and so changes to INITIAL_CLUE / provider take effect.
    """
    global _llm_latencies, _llm_cache_hits, _llm_api_calls
    global _llm_input_tokens, _llm_output_tokens, _llm_calls
    global _llm_illegal_outputs, _llm_impassable_drops, _llm_fallback_count
    global _llm_parse_retry_events
    global _solver_latencies, _priority_snapshots
    global _cached_knowledge, _cached_system_msg
    global _call_counter
    _llm_latencies = []
    _llm_input_tokens = []
    _llm_output_tokens = []
    _llm_cache_hits = 0
    _llm_api_calls = 0
    _llm_calls = []
    _llm_illegal_outputs = 0
    _llm_impassable_drops = 0
    _llm_fallback_count = 0
    _llm_parse_retry_events = []
    _solver_latencies = []
    _priority_snapshots = []
    _cached_knowledge = None
    _cached_system_msg = None
    _call_counter = 0


def get_stats() -> dict:
    """Return a copy of the per-run telemetry for merging into sim results."""
    return {
        "llm_latencies": list(_llm_latencies),
        "llm_input_tokens": list(_llm_input_tokens),
        "llm_output_tokens": list(_llm_output_tokens),
        "llm_cache_hits": _llm_cache_hits,
        "llm_api_calls": _llm_api_calls,
        "llm_calls": list(_llm_calls),
        "llm_illegal_outputs": _llm_illegal_outputs,
        "llm_impassable_drops": _llm_impassable_drops,
        "llm_fallback_count": _llm_fallback_count,
        "llm_parse_retry_events": list(_llm_parse_retry_events),
        "solver_latencies": list(_solver_latencies),
        "priority_snapshots": list(_priority_snapshots) if _dump_priorities else [],
    }


def set_cache_enabled(enabled: bool):
    """Enable or disable the on-disk LLM response cache."""
    global _use_cache
    _use_cache = bool(enabled)


def set_force_refresh(enabled: bool):
    """When True, every LLM call skips cache lookup and still writes back.

    Use this to overwrite cache entries with fresh responses (e.g. after
    a prompt change).
    """
    global _force_refresh
    _force_refresh = bool(enabled)


def set_current_seed(seed):
    """Register the seed for the upcoming simulation run."""
    global _current_seed
    _current_seed = seed


def set_current_run_id(run_id):
    """Register the run identifier for checkpoint storage.

    Every ``_call_llm`` invocation during this run writes to
    ``results/runs/{run_id}/checkpoint/call_{N:04d}.json`` where N is
    the 0-based call counter (reset to 0 at the start of each run).

    Pass None to disable checkpoint storage (ad-hoc / test runs).
    """
    global _current_run_id
    _current_run_id = run_id


def _log_context(call_index=None) -> str:
    """Return a stable run/call prefix for concurrent console output."""
    run_id = _current_run_id or "unscoped"
    if call_index is None:
        call_index = max(0, _call_counter - 1)
    return f"[{run_id}][call={int(call_index):04d}]"


def set_dump_priorities(enabled: bool):
    """Enable/disable per-epoch priority vector snapshots (case-study use)."""
    global _dump_priorities
    _dump_priorities = bool(enabled)


# ══════════════════════════════════════════════════════════════════════
#  RAG Knowledge Cache (static, computed once from clue set L)
# ══════════════════════════════════════════════════════════════════════

_cached_knowledge = None


def init_knowledge():
    """Retrieve SAR domain knowledge from the clue.

    Called once at the start of each simulation run. Caches the result
    of ``retrieve_knowledge(INITIAL_CLUE)`` in ``_cached_knowledge`` so
    subsequent system-prompt builds reuse it.
    """
    global _cached_knowledge
    _cached_knowledge = retrieve_knowledge(INITIAL_CLUE)


# ══════════════════════════════════════════════════════════════════════
#  Per-run LLM checkpoint storage
# ══════════════════════════════════════════════════════════════════════
#
#  Not a content-addressed cache — a call-indexed replay log. Every
#  invocation of ``_call_llm`` writes ``call_{_call_counter:04d}.json``
#  under ``results/runs/{_current_run_id}/checkpoint/`` before
#  incrementing the counter. On replay, call N reads back call_N.json
#  regardless of prompt content — so two epochs with identical prompts
#  still produce (and read back) two independent entries, preserving
#  exact per-call reproducibility even under LLM non-determinism.

def _checkpoint_path(call_index: int) -> pathlib.Path:
    """Path to the checkpoint file for the given call index within the
    current run. Requires ``_current_run_id`` to be set.
    """
    return (_runs_dir / _current_run_id / "checkpoint"
            / f"call_{call_index:04d}.json")


def _epoch_timing_path(epoch_index: int) -> pathlib.Path:
    """Path of the end-to-end policy time for one decision epoch."""
    return (_runs_dir / _current_run_id / "checkpoint"
            / f"epoch_{epoch_index:04d}_timing.json")


def _checkpoint_load(call_index: int):
    """Return the stored record for this call index, or None if absent.

    The record has keys {response_text, input_tokens, output_tokens,
    latency, ...}; only the first four are read by the LLM loop.
    """
    if _current_run_id is None:
        return None
    p = _checkpoint_path(call_index)
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            data = json.load(f)
        if "response_text" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _checkpoint_store(call_index: int, response_text: str,
                      provider: str, model: str, thinking: bool,
                      max_tokens: int, finish_reason: str,
                      tokens_in: int, tokens_out: int, latency: float):
    """Write the checkpoint file for this call index. No-op if
    ``_current_run_id`` is unset (writes require a run context).
    """
    if _current_run_id is None:
        return
    p = _checkpoint_path(call_index)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "response_text": response_text,
                "provider": provider,
                "model": model,
                "thinking": bool(thinking),
                "max_tokens": int(max_tokens),
                "finish_reason": finish_reason,
                "seed": _current_seed,
                "run_id": _current_run_id,
                "call_index": call_index,
                "input_tokens": int(tokens_in),
                "output_tokens": int(tokens_out),
                "latency": float(latency),
                "timestamp": time.time(),
            }, f)
    except OSError:
        pass


def resolve_epoch_computation_time(epoch_index: int, measured_time: float,
                                   llm_calls: list) -> float:
    """Persist or restore one epoch's end-to-end policy-call duration.

    A replayed LLM response returns immediately, but the simulator must advance
    by the duration observed in the original run.  When all LLM calls in this
    epoch came from checkpoint (or the policy made no LLM call), a matching
    epoch timing record is therefore authoritative.  Fresh or mixed API runs
    use ``measured_time`` and replace the timing record.

    The record contains only anonymous experiment identifiers and numeric
    timing metadata.  With no run context, or when checkpointing is disabled,
    this function is a no-op and returns the measured value.
    """
    elapsed = max(0.0, float(measured_time))
    if _current_run_id is None or not _use_cache:
        return elapsed

    call_count = len(llm_calls)
    fully_replayed = (call_count == 0 or
                      all(bool(call.get("from_cache"))
                          for call in llm_calls))
    path = _epoch_timing_path(epoch_index)

    if fully_replayed and not _force_refresh and path.exists():
        try:
            with open(path, "r") as f:
                saved = json.load(f)
            if (saved.get("run_id") == _current_run_id
                    and int(saved.get("epoch_index")) == int(epoch_index)
                    and int(saved.get("llm_call_count")) == call_count):
                restored = float(saved["elapsed_seconds"])
                if restored >= 0.0:
                    return restored
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError):
            pass

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "run_id": _current_run_id,
                "epoch_index": int(epoch_index),
                "llm_call_count": call_count,
                "elapsed_seconds": elapsed,
            }, f)
    except OSError:
        pass
    return elapsed


# ══════════════════════════════════════════════════════════════════════
#  LLM Interface
# ══════════════════════════════════════════════════════════════════════

_cached_system_msg = None


def _build_system_msg(grid_map: GridMap) -> str:
    """Build the static system message (called once, cached).

    Respects the ablation flags:
      * If _use_cot is False, the STEP 1 / STEP 2 framing is collapsed
        into a single direct-instruction sentence (no CoT scaffolding).
      * If _use_rag is False, the domain-knowledge block is omitted.
    """
    terrain_text = grid_map.terrain_summary()
    cells_text = grid_map.cells_by_terrain_text()
    elev_text = grid_map.elevation_summary()
    lm_text = "\n".join(
        f"  {name}: ({rc[0]},{rc[1]})"
        for name, rc in grid_map.landmarks.items()
    )

    intro = (
        "You are an expert SAR (Search and Rescue) coordinator AI. "
        "Your task is to analyse clues about lost persons and assign a "
        "priority score w_k ∈ [0,1] to each cell in the search grid, "
        "indicating how likely each cell is to contain an unfound target.\n\n"
    )

    if _use_cot:
        reasoning_block = (
            "You must reason in two steps:\n"
            "STEP 1 — MOBILITY PATTERN INFERENCE: For each unfound "
            "target, infer their likely current location. Use every "
            "relevant input: the clue (who is lost, last sighting, "
            "elapsed time), the map (terrain types, elevation, trail "
            "network, impassable zones, landmarks, map structure), the "
            "search history (confirmed finds plus past observations), "
            "the behavior principles, and the SAR domain knowledge "
            "(speeds, stamina, and the always-on heuristics). Reason "
            "about the map semantically — how the trail network "
            "connects, where impassable zones funnel movement, which "
            "landmarks and rest-friendly cells each sub-unit can "
            "realistically reach given their stamina and the elapsed "
            "time — rather than reading letters in isolation. For "
            "each sub-unit, compute the total trail distance from the "
            "trailhead via the last sighting to each candidate rest "
            "or destination cell, and discard candidates whose total "
            "distance exceeds the sub-unit's stamina range.\n"
            "STEP 2 — CELL PRIORITISATION: Map your inferences onto the "
            "grid to produce a priority score for every traversable "
            "cell.\n\n"
        )
    else:
        reasoning_block = (
            "Directly output priority scores for traversable cells. "
            "Do not explain your reasoning; output only the JSON block.\n\n"
        )

    constraints = (
        "IMPORTANT:\n"
        "- Scores should reflect ONLY how likely a cell is to contain a target.\n\n"
    )

    output_format_block = (
        f"=== OUTPUT FORMAT ===\n"
        f"End your response with EXACTLY ONE fenced ```json block containing "
        f"a dict with key \"priorities\" that maps \"row,col\" → w_k ∈ [0,1] "
        f"for ONLY the top {TOP_K_PRIORITY_CELLS} cells by priority. Any "
        f"cell not listed is treated as w_k = 0, so focus priority mass on "
        f"the cells that matter most. Do not emit any other ```json block "
        f"anywhere in the response.\n"
        f"Example:\n"
        f"```json\n"
        f"{{\"priorities\": {{\"16,10\": 0.9, \"14,9\": 0.3}}}}\n"
        f"```\n"
    )

    behavior_block = (
        "=== BEHAVIOR PRINCIPLES ===\n"
        "1. Impassable cells (marked 'I') cannot contain targets — a "
        "lost person cannot walk into these cells. Assign w = 0 to "
        "every impassable cell. Impassable zones also funnel movement "
        "into the adjacent passable corridors.\n"
        "2. Sub-units stay together — the members of each sub-unit "
        "named in the clue occupy the same cell throughout the "
        "incident. Locating one member locates the others.\n"
        "3. A sub-unit's reach is bounded by its weakest member's "
        "stamina. If the group's intended destination is within "
        "reach, the sub-unit arrives there and stays within a small "
        "radius. If the destination is beyond reach, the sub-unit "
        "stops at the nearest reasonably comfortable rest (a "
        "clearing, shade, flat open ground near a trail, a water "
        "edge) — this may be forward along the route, at the point "
        "where stamina ran out, or a short retreat to a rest passed "
        "earlier — and stays within a small radius.\n\n"
    )

    map_struct_text = (
        "Map structure:\n"
        "- The trail network is a single connected corridor. From the "
        "south entry at (0,12) it runs north along cols 9-13 past GCS "
        "(1,13) and the pond, reaching the trail_fork at (16,10). At "
        "the fork it branches:\n"
        "    * A NW arm continues north through cols 5-10 up to (25,5-7).\n"
        "    * An east arm extends from the fork along row 16 to "
        "(16,17), turns south along col 17 from row 17 to row 23, and "
        "joins a NE branch that covers rows 21-27 (cols 14-23), passes "
        "overlook (24,21), and extends to cells near caves (25,27).\n"
        "- Impassable zones split the passable ground:\n"
        "    * Large central-north block: rows 17-27, cols 8-15.\n"
        "    * West block: rows 10-13, cols 3-6.\n"
        "    * Two east blocks: rows 10-13 cols 22-25; and rows 20-22 "
        "cols 19-21.\n"
        "- Water features:\n"
        "    * Pond at (14,8), (14,9), (15,8), (15,9), on the trail "
        "just south of the fork.\n"
        "    * Main creek runs north-south along cols 14-16, rows 3-13.\n"
        "    * A smaller creek near (23,2)-(25,4) in the NW.\n"
        "- Rocky terrain rings the NE corner (rows 22-27, cols 24-27) "
        "and the NW corner (rows 24-27, cols 0-3).\n"
        "- Meadows: a large SW patch (rows 4-7, cols 0-8), a small "
        "patch at the clearing (18,1), and a small group of meadow "
        "cells around campsite (20,18).\n\n"
    )

    region_block = (
        f"=== SEARCH REGION ===\n"
        f"Grid: {GRID_ROWS}x{GRID_COLS}, cell size 250m\n"
        f"Terrain map (row 27=north, row 0=south). "
        f"Legend: M=meadow, S=sparse vegetation, D=dense forest, "
        f"R=rocky, T=trail, C=creek, I=impassable, P=pond.\n"
        f"{terrain_text}\n\n"
        f"{cells_text}\n\n"
        f"Elevation (metres):\n{elev_text}\n\n"
        f"{map_struct_text}"
        f"Landmarks:\n{lm_text}\n\n"
        f"Sensor model by terrain (p_d = detect a present target; "
        f"p_f = false alarm when no target is present). Impassable cells "
        f"are never searched because a target cannot physically occupy "
        f"them, so they do not appear in this table:\n"
        f"                           p_d    p_f\n"
        f"  meadow/trail/creek/pond  0.95   0.02\n"
        f"  sparse vegetation        0.80   0.05\n"
        f"  rocky                    0.65   0.12\n"
        f"  dense forest             0.55   0.10\n"
        f"A 'no detection' at a cell is not proof the target is absent — "
        f"even after multiple no-detections the cell may still hide someone, "
        f"but the probability a target is present there decreases with each "
        f"additional no-detection. Weigh this against exploring unsearched "
        f"cells.\n"
        f"'DETECTION' entries in the observation history are raw sensor "
        f"hits and may be false alarms. Only 'Confirmed finds' are verified "
        f"discoveries — a DETECTION in history without a matching entry in "
        f"Confirmed finds should be treated as a false alarm.\n\n"
        f"=== CONTEXTUAL CLUES ===\n{INITIAL_CLUE}\n\n"
    )

    if _use_rag:
        knowledge = (_cached_knowledge
                     if _cached_knowledge is not None
                     else retrieve_knowledge(INITIAL_CLUE))
        if knowledge:
            rag_block = f"=== SAR DOMAIN KNOWLEDGE ===\n{knowledge}"
        else:
            rag_block = ""   # clue triggered nothing; omit the header too
    else:
        rag_block = ""  # ablation: no retrieved domain knowledge

    return (intro + reasoning_block + constraints
            + behavior_block + region_block + rag_block
            + output_format_block)


def _build_prompt(grid_map: GridMap, memory: MemoryModule,
                  state: dict, num_unfound: int) -> list:
    """Build the LLM prompt (system + user) for Steps 1 & 2."""
    global _cached_system_msg

    # System message: static, build once and cache
    if _cached_system_msg is None:
        _cached_system_msg = _build_system_msg(grid_map)
    system_msg = _cached_system_msg

    # User message: dynamic, changes every epoch
    finds_text = memory.get_finds_text()
    history_text = memory.get_history_text()

    # CoT-gated tail: keep the STEP 1/STEP 2 call-to-action only when CoT
    # is enabled; otherwise the assistant is asked to output JSON directly.
    tail = ("Now perform STEP 1 (mobility inference) and STEP 2 (cell priorities)."
            if _use_cot else
            "Now output the priority JSON directly.")

    user_msg = (
        f"=== CURRENT STATUS (t={state['current_time']:.0f}s) ===\n"
        f"Unfound targets: {num_unfound} / {NUM_TARGETS}\n\n"
        f"Confirmed finds:\n{finds_text}\n\n"
        f"Observation history:\n"
        f"{history_text}\n\n"
        f"{tail}"
    )

    return system_msg, user_msg


def _max_tokens_for(provider: str, model: str,
                    thinking: bool = False) -> int:
    """Return the per-model output-token cap (i.e. the value passed as
    ``max_tokens`` to the provider SDK).

    These are experiment-side budgets, not the much larger provider
    ceilings.  A response that reaches the budget is marked invalid
    instead of entering the result aggregation.
    """
    override = _MODEL_MAX_TOKENS_OVERRIDE.get((model, bool(thinking)))
    if override is None:
        override = _MODEL_MAX_TOKENS_OVERRIDE.get(model)
    if override is not None:
        return override
    # Uniform experiment-side fallback for any model not explicitly pinned
    # below.  Formal configurations therefore share the same output budget.
    return 16384


# Explicit budgets for the pinned experiment models.
_MODEL_MAX_TOKENS_OVERRIDE: dict = {
    ("deepseek-v4-flash", False): 16384,
    ("deepseek-v4-flash", True): 16384,
    ("deepseek-v4-pro", False): 16384,
    ("deepseek-v4-pro", True): 16384,
    "qwen-flash": 16384,
    "qwen-plus-2025-12-01": 16384,
    "qwen3-max-2026-01-23": 16384,
}


def _is_retryable_api_error(error: Exception) -> bool:
    """Return True for transient transport, throttling, and server errors."""
    if isinstance(error, (APIConnectionError, APITimeoutError,
                          httpx.TimeoutException, httpx.NetworkError)):
        return True
    status_code = getattr(error, "status_code", None)
    return (status_code in {408, 409, 425, 429}
            or isinstance(status_code, int) and status_code >= 500)


def _call_llm(system_msg: str, user_msg: str,
              provider: str = None, max_retries: int = 300) -> str:
    """Call the LLM and return the raw response text.

    - Sets temperature=0 for both providers (reproducibility).
    - Uses the experiment output budget from ``_max_tokens_for``. A response
      stopped at that budget is passed to the existing parse-retry layer; it
      does not terminate the run directly.
    - Checkpoint storage: each invocation is keyed by its sequential
      ``_call_counter`` within the current run (``set_current_run_id``),
      NOT by prompt content. Identical prompts across epochs therefore
      produce separate checkpoint files so that each call is
      independently reproducible even under LLM non-determinism.
    - Cache hits DO contribute to per-call tokens and latency (read from
      the checkpoint file that was written when the call first executed).
      Every call — real or cached — is logged in ``_llm_calls`` with a
      ``from_cache`` flag so token spend is fully traceable.
    - ``--force-refresh`` (``set_force_refresh``) skips checkpoint
      lookup but still writes back, overwriting any previous entry.
    - Retries transient timeout, connection, throttling, and server errors up
      to 300 attempts. Backoff is 1,2,4,8,16s and remains capped at 16s.
    """
    global _llm_cache_hits, _llm_api_calls
    global _llm_input_tokens, _llm_output_tokens, _llm_calls
    global _call_counter

    provider = provider or LLM_PROVIDER
    if provider == "deepseek":
        model = LLM_MODEL_DEEPSEEK
        thinking = bool(LLM_THINKING_DEEPSEEK)
        reasoning_effort = LLM_REASONING_EFFORT_DEEPSEEK
        base_url = DEEPSEEK_BASE_URL
        api_key_env = "DEEPSEEK_API_KEY"
    elif provider == "qwen":
        model = LLM_MODEL_QWEN
        thinking = bool(LLM_THINKING_QWEN)
        reasoning_effort = None
        base_url = QWEN_BASE_URL
        api_key_env = "DASHSCOPE_API_KEY"
    elif provider == "hunyuan":
        model = LLM_MODEL_HUNYUAN
        thinking = False
        reasoning_effort = None
        base_url = HUNYUAN_BASE_URL
        api_key_env = "HUNYUAN_API_KEY"
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r} "
                         f"(expected deepseek | qwen | hunyuan)")

    max_tokens = _max_tokens_for(provider, model, thinking)

    # Snapshot the call index for this invocation. We increment
    # _call_counter only after the call resolves (checkpoint hit or API
    # success) so a mid-call crash doesn't skip an index.
    call_index = _call_counter

    # ── Checkpoint lookup (unless force-refresh) ──
    if _use_cache and not _force_refresh:
        cached = _checkpoint_load(call_index)
        if cached is not None:
            profile_matches = (
                cached.get("provider") == provider
                and cached.get("model") == model
                and cached.get("thinking") == thinking
                and cached.get("max_tokens") == max_tokens
            )
            if not profile_matches:
                cached = None
        if cached is not None:
            _llm_cache_hits += 1
            t_in = int(cached.get("input_tokens", 0) or 0)
            t_out = int(cached.get("output_tokens", 0) or 0)
            lat = float(cached.get("latency", 0.0) or 0.0)
            _llm_latencies.append(lat)
            _llm_input_tokens.append(t_in)
            _llm_output_tokens.append(t_out)
            _llm_calls.append({
                "idx": len(_llm_calls),
                "from_cache": True,
                "provider": provider,
                "model": model,
                "thinking": thinking,
                "max_tokens": max_tokens,
                "finish_reason": cached.get("finish_reason"),
                "tokens_in": t_in,
                "tokens_out": t_out,
                "latency": lat,
            })
            _call_counter += 1
            return cached["response_text"]

    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            in_toks = out_toks = 0

            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get(api_key_env),
                base_url=base_url,
                timeout=httpx.Timeout(600.0, connect=30.0),
            )
            request_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            }
            if provider == "deepseek":
                request_kwargs["extra_body"] = {
                    "thinking": {
                        "type": "enabled" if thinking else "disabled"
                    }
                }
                if thinking:
                    request_kwargs["reasoning_effort"] = reasoning_effort
            elif provider == "qwen":
                request_kwargs["extra_body"] = {
                    "enable_thinking": thinking
                }

            response = client.chat.completions.create(
                **request_kwargs,
            )
            text = response.choices[0].message.content
            finish_reason = getattr(response.choices[0],
                                    "finish_reason", None)
            usage = getattr(response, "usage", None)
            if usage is not None:
                in_toks = getattr(usage, "prompt_tokens", 0) or 0
                out_toks = getattr(usage, "completion_tokens", 0) or 0

            latency = time.perf_counter() - t0
            _llm_latencies.append(latency)
            _llm_input_tokens.append(int(in_toks))
            _llm_output_tokens.append(int(out_toks))
            _llm_api_calls += 1
            _llm_calls.append({
                "idx": len(_llm_calls),
                "from_cache": False,
                "provider": provider,
                "model": model,
                "thinking": thinking,
                "max_tokens": max_tokens,
                "finish_reason": finish_reason,
                "tokens_in": int(in_toks),
                "tokens_out": int(out_toks),
                "latency": latency,
            })

            if _use_cache:
                _checkpoint_store(call_index, text, provider, model,
                                  thinking, max_tokens, finish_reason,
                                  int(in_toks), int(out_toks), latency)

            _call_counter += 1
            return text

        except Exception as e:
            if not _is_retryable_api_error(e):
                raise
            if attempt >= max_retries - 1:
                raise
            wait = 2 ** min(attempt, 4)  # 1, 2, 4, 8, then capped at 16s
            print(
                f"\n  {_log_context(call_index)}[LLM retry {attempt + 1}/"
                f"{max_retries}] {type(e).__name__}: {e}. "
                f"Waiting {wait}s...", flush=True)
            time.sleep(wait)


def _parse_priorities(response_text: str, grid_map: GridMap):
    """Extract priority vector w(t) from LLM response.

    Impassable cells receive w = 0 regardless of what the LLM returned,
    on the grounds that a target cannot physically occupy them.

    Telemetry side-effects (this response only):
      * _llm_impassable_drops += # of impassable-cell priorities discarded
      * _llm_illegal_outputs  += 1 if this response assigned any priority
                                 to an impassable cell.

    Does NOT perform the uniform-prior fallback or increment
    ``_llm_fallback_count`` — those are the caller's responsibility via
    ``_call_llm_and_parse_priorities``, which retries the API when
    parsing fails and only falls back once retries are exhausted.

    Returns:
        (w, ok) where w is (GRID_ROWS, GRID_COLS), and ok=True iff at
        least one passable cell received a positive priority.
    """
    global _llm_illegal_outputs, _llm_impassable_drops

    w = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)

    # Try ```json ... ``` first (greedy — captures full nested JSON)
    json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else None

    dropped_impassable = 0
    if json_str:
        try:
            data = json.loads(json_str)
            priorities = data.get("priorities", data)
            if not isinstance(priorities, dict):
                return w, False

            # Parse transactionally: a malformed value must invalidate the
            # whole response instead of leaving an already-written partial
            # priority grid.  Numeric strings are accepted, but non-numeric
            # and non-finite values are sent through the normal parse-retry
            # path in _call_llm_and_parse_priorities.
            parsed_priorities = []
            for key, val in priorities.items():
                parts = key.split(",")
                if len(parts) == 2:
                    r, c = int(parts[0].strip()), int(parts[1].strip())
                    if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                        numeric_val = float(val)
                        if not np.isfinite(numeric_val):
                            return np.zeros_like(w), False
                        parsed_priorities.append((r, c, numeric_val))

            for r, c, numeric_val in parsed_priorities:
                if not grid_map.is_traversable(r, c):
                    dropped_impassable += 1
                    continue
                w[r, c] = float(np.clip(numeric_val, 0.0, 1.0))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError,
                AttributeError):
            return np.zeros_like(w), False
    if dropped_impassable:
        _llm_impassable_drops += dropped_impassable
        _llm_illegal_outputs += 1
        print(f"  {_log_context()}[parse] dropped {dropped_impassable} impassable-cell "
              f"priorities from LLM output", flush=True)

    return w, (w.sum() > 0)


def _classify_parse_failure(response_text: str,
                            provider: str = None) -> str:
    """Bucket a parse-failed response into one of the reportable kinds.

    Resolution order:
      * ``max_tokens``       : provider cap reached (output truncated)
      * ``tiny_response``    : < 100 output tokens (API glitch / refusal)
      * ``multi_json_block`` : two or more ```json blocks confused the
                               greedy regex in _parse_priorities / _parse_direct_assignments
      * ``no_json``          : no ```json block AND no outer ``{...}``
                               candidate — model ignored JSON format
      * ``json_decode_error``: candidate JSON string found but json.loads
                               raised (single-block case; multi-block
                               caught by ``multi_json_block``)
    """
    prov = provider or LLM_PROVIDER
    if prov == "deepseek":
        model = LLM_MODEL_DEEPSEEK
    elif prov == "qwen":
        model = LLM_MODEL_QWEN
    elif prov == "hunyuan":
        model = LLM_MODEL_HUNYUAN
    else:
        model = None
    if prov == "deepseek":
        thinking = bool(LLM_THINKING_DEEPSEEK)
    elif prov == "qwen":
        thinking = bool(LLM_THINKING_QWEN)
    else:
        thinking = False
    cap = _max_tokens_for(prov, model, thinking) if model else 0
    out_tok = _llm_output_tokens[-1] if _llm_output_tokens else 0
    if cap and out_tok >= cap:
        return "max_tokens"
    if out_tok < 100:
        return "tiny_response"
    fenced = re.findall(r'```json\s*.*?\s*```', response_text, re.DOTALL)
    if len(fenced) >= 2:
        return "multi_json_block"
    if len(fenced) == 0 and not re.search(r'\{.*\}', response_text, re.DOTALL):
        return "no_json"
    return "json_decode_error"


def _last_llm_response_hit_output_limit() -> bool:
    """Return whether the most recent LLM response exhausted its budget."""
    if not _llm_calls:
        return False
    call = _llm_calls[-1]
    finish_reason = str(call.get("finish_reason") or "").lower()
    max_tokens = int(call.get("max_tokens", 0) or 0)
    output_tokens = int(call.get("tokens_out", 0) or 0)
    return (finish_reason in {"length", "max_tokens"}
            or (max_tokens > 0 and output_tokens >= max_tokens))


def _uniform_priorities(grid_map: GridMap) -> np.ndarray:
    """Uniform priors (0.1) over all traversable cells — used as the
    final fallback when every parse-retry attempt has failed."""
    w = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid_map.is_traversable(r, c):
                w[r, c] = 0.1
    return w


def _call_llm_and_parse_priorities(system_msg: str, user_msg: str,
                                   grid_map: GridMap,
                                   provider: str,
                                   max_parse_retries: int = 2):
    """Call ``_call_llm`` and parse the priority JSON; retry on parse
    failure up to ``max_parse_retries`` attempts total.

    Each attempt consumes its own ``_call_counter`` slot, so the
    checkpoint layout is deterministically replayable: call N is
    attempt 1, call N+1 is attempt 2, etc.

    If every attempt produces an unparseable response, the uniform
    fallback is installed and ``_llm_fallback_count`` is incremented
    once (per decision, not per attempt).

    Returns:
        (w, response_text) where response_text is the FINAL attempt's
        raw response (even if that attempt still failed to parse).
    """
    global _llm_fallback_count, _llm_parse_retry_events

    response_text = ""
    w = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    for attempt in range(max_parse_retries):
        call_index = _call_counter     # snapshot before _call_llm bumps it
        response_text = _call_llm(system_msg, user_msg, provider=provider)
        w, ok = _parse_priorities(response_text, grid_map)
        hit_output_limit = _last_llm_response_hit_output_limit()
        if ok and not hit_output_limit:
            return w, response_text
        kind = ("max_tokens" if hit_output_limit else
                _classify_parse_failure(response_text, provider))
        _llm_parse_retry_events.append({
            "call_index": call_index, "attempt": attempt,
            "kind": kind, "parser": "priorities",
        })
        if attempt < max_parse_retries - 1:
            print(f"  {_log_context(call_index)}[parse-retry "
                  f"{attempt + 1}/{max_parse_retries}] "
                  f"{kind} — calling API again", flush=True)

    _llm_fallback_count += 1
    print(f"  {_log_context()}[parse] {max_parse_retries} attempts exhausted — "
          f"uniform-prior fallback installed", flush=True)
    return _uniform_priorities(grid_map), response_text


# ══════════════════════════════════════════════════════════════════════
#  Step 3: Priority Selection + Bottleneck Assignment
# ══════════════════════════════════════════════════════════════════════

def _has_perfect_matching(adj, n_left, n_right):
    """Check if bipartite graph has a perfect matching (Hopcroft-Karp style).

    adj: dict {left_node: set of right_nodes}
    Returns True if a matching of size min(n_left, n_right) exists.
    """
    match_l = [-1] * n_left
    match_r = [-1] * n_right

    def dfs(u, visited):
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                if match_r[v] == -1 or dfs(match_r[v], visited):
                    match_l[u] = v
                    match_r[v] = u
                    return True
        return False

    matching_size = 0
    for u in range(n_left):
        visited = set()
        if dfs(u, visited):
            matching_size += 1

    return matching_size >= min(n_left, n_right)


def _bottleneck_assignment(travel_times, n_uavs, n_cells):
    """Solve bottleneck assignment via binary search on sorted travel times.

    Args:
        travel_times: ndarray (n_uavs, n_cells) of travel times
        n_uavs, n_cells: dimensions

    Returns:
        assignments: list of (uav_idx, cell_idx) pairs, or [] if infeasible
    """
    # Collect and sort all unique travel times
    all_times = sorted(set(travel_times.ravel()))

    # Binary search for the smallest threshold admitting a perfect matching
    lo, hi = 0, len(all_times) - 1
    best_threshold = None

    while lo <= hi:
        mid = (lo + hi) // 2
        t_star = all_times[mid]

        # Build bipartite adjacency with edges where t_ik <= t_star
        adj = {}
        for i in range(n_uavs):
            adj[i] = set()
            for j in range(n_cells):
                if travel_times[i, j] <= t_star:
                    adj[i].add(j)

        if _has_perfect_matching(adj, n_uavs, n_cells):
            best_threshold = t_star
            hi = mid - 1
        else:
            lo = mid + 1

    if best_threshold is None:
        return []

    # Reconstruct the actual assignment at best_threshold
    adj = {}
    for i in range(n_uavs):
        adj[i] = set()
        for j in range(n_cells):
            if travel_times[i, j] <= best_threshold:
                adj[i].add(j)

    match_l = [-1] * n_uavs
    match_r = [-1] * n_cells

    def dfs(u, visited):
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                if match_r[v] == -1 or dfs(match_r[v], visited):
                    match_l[u] = v
                    match_r[v] = u
                    return True
        return False

    for u in range(n_uavs):
        dfs(u, set())

    return [(i, match_l[i]) for i in range(n_uavs) if match_l[i] != -1]


def _prepare_priority_assignment(w: np.ndarray, state: dict,
                                 grid_map: GridMap, uavs_list):
    """Build the feasible top-priority assignment problem for Ours."""
    idle_uavs = uavs_list
    occupied = state["occupied_cells"]
    distances = state["distances"]
    base_distances = state["base_distances"]

    if not idle_uavs:
        return [], [], np.empty((0, 0), dtype=np.float64)

    B = len(idle_uavs)
    traversable = grid_map.get_traversable_cells()
    candidate_cells = [rc for rc in traversable if rc not in occupied]

    # Step 1: Feasible intersection — cells reachable by ALL UAVs
    # Ablation: _use_energy_filter False skips the reach-and-return check.
    feasible_intersection = []
    for rc in candidate_cells:
        d_kb = base_distances.get(rc, 0.0)
        all_feasible = True
        for uav in idle_uavs:
            d_ik = distances.get((uav.id, rc))
            if d_ik is None:
                all_feasible = False
                break
            if _use_energy_filter and not uav.can_reach_and_return(
                    d_ik, d_kb, grid_map.terrain(*rc)):
                all_feasible = False
                break
        if all_feasible:
            feasible_intersection.append(rc)

    if not feasible_intersection:
        return idle_uavs, [], np.empty((0, 0), dtype=np.float64)

    # Step 2: Select top-B cells by priority w_k → C*
    feasible_intersection.sort(key=lambda rc: w[rc[0], rc[1]], reverse=True)
    n_dispatch = min(B, len(feasible_intersection))
    selected_cells = feasible_intersection[:n_dispatch]

    # Step 3: Build the shared travel-time matrix for final matching.
    travel_times = np.zeros((n_dispatch, n_dispatch), dtype=np.float64)
    for i in range(n_dispatch):
        uav = idle_uavs[i]
        for j, rc in enumerate(selected_cells):
            d_ik = distances.get((uav.id, rc), 0.0)
            travel_times[i, j] = d_ik / CRUISE_SPEED

    return idle_uavs, selected_cells, travel_times


def solve_assignment(w: np.ndarray, state: dict,
                     grid_map: GridMap, uavs_list) -> dict:
    """
    Two-stage assignment (bottleneck):
      1. Feasible intersection: cells reachable by ALL idle UAVs
      2. Select top-B cells by priority w_k
      3. Bottleneck assignment: minimise max travel time via binary search

    Returns:
        assignments: dict {uav_id: (row, col)}
    """
    idle_uavs, selected_cells, travel_times = _prepare_priority_assignment(
        w, state, grid_map, uavs_list)
    if not selected_cells:
        return {}

    n_dispatch = len(selected_cells)

    t_solve = time.perf_counter()
    pairs = _bottleneck_assignment(travel_times, n_dispatch, n_dispatch)
    _solver_latencies.append(time.perf_counter() - t_solve)

    assignments = {}
    for i, j in pairs:
        assignments[idle_uavs[i].id] = selected_cells[j]

    return assignments


# ══════════════════════════════════════════════════════════════════════
#  Full Decision Pipeline
# ══════════════════════════════════════════════════════════════════════

def _reason_cell_priorities(grid_map: GridMap, memory: MemoryModule,
                            state: dict, num_unfound: int,
                            provider: str = None,
                            verbose: bool = False) -> np.ndarray:
    """Run the shared LLM reasoning stage and return the priority grid."""
    system_msg, user_msg = _build_prompt(grid_map, memory, state, num_unfound)
    w, response_text = _call_llm_and_parse_priorities(
        system_msg, user_msg, grid_map, provider)

    if _dump_priorities:
        _priority_snapshots.append({
            "t": float(state.get("current_time", 0.0)),
            "w": w.tolist(),
        })

    if verbose:
        print(f"\n{'─'*60}")
        print("[LLM Full Response]")
        print(response_text[-1500:])
        top_cells = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if w[r, c] > 0:
                    top_cells.append(((r, c), w[r, c]))
        top_cells.sort(key=lambda x: -x[1])
        print("\n[Top-10 priority cells]")
        for (r, c), value in top_cells[:10]:
            print(f"  ({r},{c}) w={value:.2f} "
                  f"terrain={grid_map.terrain(r,c)}")

    return w


def make_decision(grid_map: GridMap, memory: MemoryModule,
                  state: dict, uavs_list, num_unfound: int,
                  provider: str = None, verbose: bool = False) -> dict:
    """
    Full two-stage decision pipeline:
      Stage 1 (LLM): Steps 1 & 2 → priority vector w(t)
      Stage 2 (Optimizer): Step 3 → top-B selection + bottleneck assignment

    Returns:
        assignments: dict {uav_id: (row, col)}
    """
    w = _reason_cell_priorities(
        grid_map, memory, state, num_unfound,
        provider=provider, verbose=verbose)

    # Stage 2: bottleneck assignment
    assignments = solve_assignment(w, state, grid_map, uavs_list)

    if verbose:
        print(f"\n[Assignments]")
        for uid, rc in assignments.items():
            print(f"  UAV {uid} -> ({rc[0]},{rc[1]}) terrain={grid_map.terrain(*rc)}")
        print(f"{'─'*60}")

    return assignments


# ══════════════════════════════════════════════════════════════════════
#  LLM-Direct (end-to-end) — Exp. 2 comparison
# ══════════════════════════════════════════════════════════════════════
#
# Bypasses the two-stage pipeline (priority vector + bottleneck
# assignment). The LLM is given the RAG knowledge, clue, terrain,
# landmarks, memory, AND a list of idle UAVs with their positions and
# energy; it is asked to output a direct mapping {uav_id: "row,col"}.
# This tests whether the structural decomposition in the proposed
# method adds value on top of just giving the LLM everything.

def _build_llm_direct_prompt(grid_map: GridMap, memory: MemoryModule,
                             state: dict, num_unfound: int,
                             uavs_list) -> tuple:
    """Prompt for the LLM-direct baseline.

    System message re-uses the RAG knowledge block and terrain summary
    from the proposed method, but replaces the two-step CoT instructions
    with an end-to-end assignment format.
    """
    knowledge = _cached_knowledge or retrieve_knowledge(INITIAL_CLUE)
    terrain_text = grid_map.terrain_summary()
    cells_text = grid_map.cells_by_terrain_text()
    elev_text = grid_map.elevation_summary()
    lm_text = "\n".join(
        f"  {name}: ({rc[0]},{rc[1]})"
        for name, rc in grid_map.landmarks.items()
    )

    map_struct_text = (
        "Map structure:\n"
        "- The trail network is a single connected corridor. From the "
        "south entry at (0,12) it runs north along cols 9-13 past GCS "
        "(1,13) and the pond, reaching the trail_fork at (16,10). At "
        "the fork it branches:\n"
        "    * A NW arm continues north through cols 5-10 up to (25,5-7).\n"
        "    * An east arm extends from the fork along row 16 to "
        "(16,17), turns south along col 17 from row 17 to row 23, and "
        "joins a NE branch that covers rows 21-27 (cols 14-23), passes "
        "overlook (24,21), and extends to cells near caves (25,27).\n"
        "- Impassable zones split the passable ground:\n"
        "    * Large central-north block: rows 17-27, cols 8-15.\n"
        "    * West block: rows 10-13, cols 3-6.\n"
        "    * Two east blocks: rows 10-13 cols 22-25; and rows 20-22 "
        "cols 19-21.\n"
        "- Water features:\n"
        "    * Pond at (14,8), (14,9), (15,8), (15,9), on the trail "
        "just south of the fork.\n"
        "    * Main creek runs north-south along cols 14-16, rows 3-13.\n"
        "    * A smaller creek near (23,2)-(25,4) in the NW.\n"
        "- Rocky terrain rings the NE corner (rows 22-27, cols 24-27) "
        "and the NW corner (rows 24-27, cols 0-3).\n"
        "- Meadows: a large SW patch (rows 4-7, cols 0-8), a small "
        "patch at the clearing (18,1), and a small group of meadow "
        "cells around campsite (20,18).\n\n"
    )

    system_msg = (
        "You are an expert SAR (Search and Rescue) coordinator AI. "
        "Your task is to directly assign each idle UAV to a search cell "
        "so that all lost targets are found as quickly as possible.\n\n"
        "You will be told the missing persons' clue, the terrain, a list "
        "of idle UAVs (with positions and remaining energy), and the "
        "observation history. Decide WHERE each UAV should go. You do "
        "not need to compute distances or priorities separately — make "
        "the final assignment directly.\n\n"
        "Constraints:\n"
        f"- Every cell must be traversable in a {GRID_ROWS}x{GRID_COLS} "
        "grid (0-indexed).\n"
        "- Do NOT assign two UAVs to the same cell.\n"
        "- A UAV must have enough energy to fly to the cell, search it, "
        "and return to base (GCS at row 1, col 13). Its remaining "
        "energy is reported below; assume flight costs 200 W / 15 m/s, "
        "search costs 150 W for terrain-dependent seconds, and keep a "
        "reserve of 30 kJ.\n"
        "- Prefer cells likely to contain targets based on the clue and "
        "SAR domain knowledge.\n\n"
        "Output format: a JSON block with key \"assignments\" mapping "
        "each UAV id (as string) to a \"row,col\" string. Assign every "
        "listed idle UAV that has a feasible cell; do not return an empty "
        "mapping when at least one feasible dispatch exists.\n"
        "Example: {\"assignments\": {\"0\": \"16,10\", \"1\": \"14,9\"}}\n\n"
        "=== BEHAVIOR PRINCIPLES ===\n"
        "1. Impassable cells (marked 'I') cannot contain targets — a "
        "lost person cannot walk into these cells. Never dispatch a UAV "
        "to an impassable cell. Impassable zones also funnel movement "
        "into adjacent passable corridors.\n"
        "2. Sub-units stay together — the members of each sub-unit named "
        "in the clue occupy the same cell throughout the incident. "
        "Locating one member locates the others.\n"
        "3. A sub-unit's reach is bounded by its weakest member's "
        "stamina. If the group's intended destination is within "
        "reach, the sub-unit arrives there and stays within a small "
        "radius. If the destination is beyond reach, the sub-unit "
        "stops at the nearest reasonably comfortable rest "
        "(clearing, shade, flat open ground near a trail, water's "
        "edge) — this may be forward along the route, at the point "
        "where stamina ran out, or a short retreat to a rest passed "
        "earlier — and stays within a small radius.\n\n"
        f"=== SEARCH REGION ===\n"
        f"Grid: {GRID_ROWS}x{GRID_COLS}, cell size 250m\n"
        f"Terrain map (row 27=north, row 0=south). "
        f"Legend: M=meadow, S=sparse vegetation, D=dense forest, "
        f"R=rocky, T=trail, C=creek, I=impassable, P=pond.\n"
        f"{terrain_text}\n\n"
        f"{cells_text}\n\n"
        f"Elevation (metres):\n{elev_text}\n\n"
        f"{map_struct_text}"
        f"Landmarks:\n{lm_text}\n\n"
        f"Sensor model by terrain (p_d = detect a present target; "
        f"p_f = false alarm when no target is present). Impassable cells "
        f"are never searched because a target cannot physically occupy "
        f"them, so they do not appear in this table:\n"
        f"                           p_d    p_f\n"
        f"  meadow/trail/creek/pond  0.95   0.02\n"
        f"  sparse vegetation        0.80   0.05\n"
        f"  rocky                    0.65   0.12\n"
        f"  dense forest             0.55   0.10\n"
        f"A 'no detection' at a cell is not proof the target is absent — "
        f"even after multiple no-detections the cell may still hide someone, "
        f"but the probability a target is present there decreases with each "
        f"additional no-detection. Weigh this against exploring unsearched "
        f"cells.\n"
        f"'DETECTION' entries in the observation history are raw sensor "
        f"hits and may be false alarms. Only 'Confirmed finds' are verified "
        f"discoveries — a DETECTION in history without a matching entry in "
        f"Confirmed finds should be treated as a false alarm.\n\n"
        f"=== CONTEXTUAL CLUES ===\n{INITIAL_CLUE}\n\n"
        f"=== SAR DOMAIN KNOWLEDGE ===\n{knowledge}"
    )

    # UAV status block (dynamic per call)
    uav_lines = []
    for uav in uavs_list:
        # Position in (row, col) estimation
        c_col = int(uav.position[0] / 250.0)
        r_row = int(uav.position[1] / 250.0)
        c_col = max(0, min(GRID_COLS - 1, c_col))
        r_row = max(0, min(GRID_ROWS - 1, r_row))
        uav_lines.append(
            f"  UAV {uav.id}: at ({r_row},{c_col}), "
            f"energy {uav.energy/1000.0:.1f} kJ"
        )
    uav_block = "\n".join(uav_lines) if uav_lines else "  (none)"

    finds_text = memory.get_finds_text()
    history_text = memory.get_history_text()

    user_msg = (
        f"=== CURRENT STATUS (t={state['current_time']:.0f}s) ===\n"
        f"Unfound targets: {num_unfound} / {NUM_TARGETS}\n\n"
        f"Idle UAVs:\n{uav_block}\n\n"
        f"Confirmed finds:\n{finds_text}\n\n"
        f"Observation history:\n{history_text}\n\n"
        f"Now produce the direct UAV-to-cell assignment."
    )

    return system_msg, user_msg


def _parse_direct_assignments(response_text: str,
                              grid_map: GridMap,
                              valid_uav_ids: set):
    """Extract {uav_id: (row, col)} from an LLM-direct response.

    Silently drops entries for unknown UAV ids, non-traversable cells,
    or out-of-range coordinates.

    Telemetry side-effects (this response only):
      * _llm_impassable_drops += # of impassable-cell assignments discarded
      * _llm_illegal_outputs  += 1 if this response assigned any UAV to
                                 an impassable cell.

    Does NOT increment ``_llm_fallback_count`` — that counter tracks
    decisions where every retry failed; the caller
    (``_call_llm_and_parse_direct_assignments``) updates it.

    Returns:
        (assignments, ok, failure_kind). ``ok`` is true only when at least
        one usable assignment remains. An empty or fully filtered mapping is
        an invalid decision and therefore enters the single parse-retry path.
    """
    global _llm_illegal_outputs, _llm_impassable_drops

    json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else None

    if not json_str:
        print(f"  {_log_context()}[parse] llm_direct: no JSON in response",
              flush=True)
        return {}, False, "no_json"

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        print(f"  {_log_context()}[parse] llm_direct: JSON decode error",
              flush=True)
        return {}, False, "json_decode_error"

    if not isinstance(data, dict):
        print(f"  {_log_context()}[parse] llm_direct: JSON is not an object",
              flush=True)
        return {}, False, "json_not_object"

    raw = data.get("assignments", data)
    if not isinstance(raw, dict):
        print(f"  {_log_context()}[parse] llm_direct: no 'assignments' dict",
              flush=True)
        return {}, False, "assignments_not_object"

    assignments = {}
    dropped_impassable = 0
    used_cells = set()
    for k, v in raw.items():
        try:
            uid = int(k)
        except (TypeError, ValueError):
            continue
        if uid not in valid_uav_ids:
            continue
        if not isinstance(v, str):
            continue
        parts = v.split(",")
        if len(parts) != 2:
            continue
        try:
            r, c = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            continue
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
            continue
        if not grid_map.is_traversable(r, c):
            dropped_impassable += 1
            continue
        if (r, c) in used_cells:
            continue
        assignments[uid] = (r, c)
        used_cells.add((r, c))

    if dropped_impassable:
        _llm_impassable_drops += dropped_impassable
        _llm_illegal_outputs += 1
        print(f"  {_log_context()}[parse] llm_direct: dropped {dropped_impassable} "
              f"impassable-cell assignment(s)", flush=True)

    if not assignments:
        return {}, False, "no_valid_assignment"
    return assignments, True, None


def _filter_direct_assignments(assignments: dict, grid_map: GridMap,
                               state: dict, uavs_list) -> dict:
    """Keep only unoccupied assignments each UAV can search and return from."""
    occupied = set(state.get("occupied_cells", []))
    base_distances = state.get("base_distances", {})
    distances = state.get("distances", {})
    uavs_by_id = {u.id: u for u in uavs_list}
    filtered = {}
    for uid, rc in assignments.items():
        if rc in occupied:
            continue
        uav = uavs_by_id.get(uid)
        if uav is None:
            continue
        d_ik = distances.get((uid, rc))
        d_kb = base_distances.get(rc)
        cell_pos = grid_map.cell_centre(*rc)
        if d_ik is None:
            d_ik = float(np.linalg.norm(uav.position - cell_pos))
        if d_kb is None:
            d_kb = float(np.linalg.norm(
                cell_pos - grid_map.cell_centre(1, 13)))
        if uav.can_reach_and_return(d_ik, d_kb, grid_map.terrain(*rc)):
            filtered[uid] = rc
    return filtered


def _nearest_direct_fallback(grid_map: GridMap, state: dict,
                             uavs_list) -> dict:
    """Deterministic nearest-search fallback for LLM-Direct.

    This deliberately uses no clue-derived priority and no global matching.
    UAVs are processed by id; each selects its nearest feasible cell while
    preferring cells with the lowest search count. Row/column traversal order
    deterministically breaks equal-distance ties.
    """
    from .baselines import greedy_coverage
    return greedy_coverage(
        grid_map, state, sorted(uavs_list, key=lambda uav: uav.id))


def _call_llm_and_parse_direct_assignments(system_msg: str, user_msg: str,
                                           grid_map: GridMap,
                                           valid_uav_ids: set,
                                           provider: str,
                                           max_parse_retries: int = 2,
                                           assignment_filter=None,
                                           fallback_fn=None):
    """Call ``_call_llm`` and parse direct assignments; retry on JSON
    decode failure up to ``max_parse_retries`` attempts.

    Returns:
        (assignments, response_text). After both attempts fail, the caller's
        method-specific deterministic fallback is used.
    """
    global _llm_fallback_count, _llm_parse_retry_events

    response_text = ""
    for attempt in range(max_parse_retries):
        call_index = _call_counter
        response_text = _call_llm(system_msg, user_msg, provider=provider)
        assignments, ok, failure_kind = _parse_direct_assignments(
            response_text, grid_map, valid_uav_ids)
        hit_output_limit = _last_llm_response_hit_output_limit()
        if ok and assignment_filter is not None:
            assignments = assignment_filter(assignments)
            if not assignments:
                ok = False
                failure_kind = "no_feasible_assignment"
        if ok and not hit_output_limit:
            return assignments, response_text
        kind = ("max_tokens" if hit_output_limit else
                failure_kind or _classify_parse_failure(
                    response_text, provider))
        _llm_parse_retry_events.append({
            "call_index": call_index, "attempt": attempt,
            "kind": kind, "parser": "direct",
        })
        if attempt < max_parse_retries - 1:
            print(f"  {_log_context(call_index)}[parse-retry "
                  f"{attempt + 1}/{max_parse_retries}] "
                  f"{kind} — calling API again", flush=True)

    _llm_fallback_count += 1
    fallback = fallback_fn() if fallback_fn is not None else {}
    print(f"  {_log_context()}[parse] llm_direct: "
          f"{max_parse_retries} attempts exhausted — "
          f"nearest-search fallback assigned {len(fallback)} UAV(s)",
          flush=True)
    return fallback, response_text


def make_decision_llm_direct(grid_map: GridMap, memory: MemoryModule,
                             state: dict, uavs_list, num_unfound: int,
                             provider: str = None,
                             verbose: bool = False) -> dict:
    """LLM-direct baseline: LLM outputs the assignment directly.

    A normal LLM mapping is feasibility-filtered and used directly. If both
    allowed responses are invalid, a deterministic nearest-search fallback
    keeps the mission running without borrowing the proposed optimizer.
    """
    system_msg, user_msg = _build_llm_direct_prompt(
        grid_map, memory, state, num_unfound, uavs_list
    )
    valid_ids = {u.id for u in uavs_list}
    assignments, response_text = _call_llm_and_parse_direct_assignments(
        system_msg, user_msg, grid_map, valid_ids, provider,
        assignment_filter=lambda raw: _filter_direct_assignments(
            raw, grid_map, state, uavs_list),
        fallback_fn=lambda: _nearest_direct_fallback(
            grid_map, state, uavs_list))

    if verbose:
        print(f"\n{'─'*60}")
        print(f"[LLM-direct Response]")
        print(response_text[-1500:])
        print(f"\n[Assignments (LLM-direct)]")
        for uid, rc in assignments.items():
            print(f"  UAV {uid} -> ({rc[0]},{rc[1]}) terrain={grid_map.terrain(*rc)}")
        print(f"{'─'*60}")

    return assignments
