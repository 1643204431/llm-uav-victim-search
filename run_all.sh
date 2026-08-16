#!/bin/bash
# Run all experiments (exp1 → exp2 → exp3 → exp5 → exp4)
# sequentially and then render figures.
#
# Usage from an activated virtual environment:
#
#   ./run_all.sh --preflight
#   EXP_PARALLEL=16 ./run_all.sh
#
# Required env vars:
#   DEEPSEEK_API_KEY  — used by exp1/2/3/5 (default backbone) + exp4
#                       (deepseek-v4-flash, non-thinking mode)
#   DEEPSEEK_BASE_URL — OpenAI-compatible endpoint that exposes
#                       deepseek-v4-flash and deepseek-v4-pro
#   DASHSCOPE_API_KEY — used by exp4 only, for the Qwen backbones
#                       (qwen-flash + pinned qwen-plus / qwen3-max snapshots)
#
# exp1 is the single source of truth for the canonical ``ours`` baseline.
# exp1 writes results/.canonical_ours_path on startup; this script reads
# that sentinel (robust against log format changes, unlike tee/grep).
#
# Batch-size sweep (Exp. 3) fixes N=10 and independently runs every B from
# 3 through 10 under the same approximate assignment budget. Backbone sweep
# (Exp. 4) skips deepseek:deepseek-v4-pro:non-thinking because that B=10
# anchor is the canonical ``ours`` row from Exp. 1.
#
# On any single-script hard failure the whole pipeline stops (set -e).
# Individual failures are persisted by stable run ID; the driver finishes
# its submitted runs, rebuilds the valid aggregate, then exits non-zero so
# the pipeline stops before consuming incomplete downstream results.

set -euo pipefail
cd "$(dirname "$0")"

# Offline source-package preflight. This branch deliberately runs before
# loading .env or checking credentials, and none of its commands calls an API.
if [[ "${1:-}" == "--preflight" ]]; then
    if [[ "$#" -ne 1 ]]; then
        echo "ERROR: --preflight does not accept additional arguments." >&2
        exit 2
    fi
    PREFLIGHT_CACHE_DIR="${TMPDIR:-/tmp}/llm-victim-policy-preflight"
    mkdir -p "$PREFLIGHT_CACHE_DIR/matplotlib" "$PREFLIGHT_CACHE_DIR/xdg"
    export MPLCONFIGDIR="$PREFLIGHT_CACHE_DIR/matplotlib"
    export XDG_CACHE_HOME="$PREFLIGHT_CACHE_DIR/xdg"
    echo "=== offline preflight start ==="
    python3 -c "import numpy, matplotlib, openai, httpx; print('dependencies: OK')"
    python3 -B -m unittest discover -s tests -q
    python3 experiments/manifest.py | tail -n 1
    python3 experiments/exp1_main_comparison.py --list-runs >/dev/null
    python3 experiments/exp2_ablation.py --list-runs >/dev/null
    python3 experiments/exp3_scaling.py --list-runs >/dev/null
    python3 experiments/exp4_llm_sensitivity.py --list-runs >/dev/null
    python3 experiments/exp5_case_study.py --list-runs >/dev/null
    python3 experiments/analyze.py --help >/dev/null
    echo "PREFLIGHT VALID: dependencies, test suite, 111-run manifest, five drivers, and analysis entry point"
    exit 0
fi

if [[ "$#" -ne 0 ]]; then
    echo "Usage: ./run_all.sh [--preflight]" >&2
    exit 2
fi

# Optional local credentials/config.  This file is git-ignored and must
# never be copied into result artifacts.  A *_API_KEY_FILE entry lets a
# secret remain in its original local file instead of being duplicated.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" && -n "${DEEPSEEK_API_KEY_FILE:-}" ]]; then
    # ``read`` returns 1 for a final line without a newline; accept it when
    # the key itself was read successfully.
    IFS= read -r DEEPSEEK_API_KEY < "$DEEPSEEK_API_KEY_FILE" ||
        [[ -n "$DEEPSEEK_API_KEY" ]]
    export DEEPSEEK_API_KEY
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is required." >&2
    exit 1
fi
if [[ -z "${DEEPSEEK_BASE_URL:-}" ]]; then
    echo "ERROR: DEEPSEEK_BASE_URL must identify an OpenAI-compatible endpoint that exposes deepseek-v4-flash and deepseek-v4-pro." >&2
    exit 1
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "ERROR: DASHSCOPE_API_KEY is required for the Exp. 4 Qwen backbones." >&2
    exit 1
fi

DEFAULT_RUNS=5
DEFAULT_MAX_EPOCHS=40
EXP3_ASSIGNMENT_BUDGET=200
DEFAULT_PARALLEL="${EXP_PARALLEL:-16}"  # run-level parallelism; override
                                        # with EXP_PARALLEL=N before running
EXP4_RUNS=5
SENTINEL=results/.canonical_ours_path

# Clear any stale sentinel from a previous aborted run so a read-after-
# write race cannot pick up an old path.
rm -f "$SENTINEL"

echo "=== exp1 start: $(date) (parallel=$DEFAULT_PARALLEL) ==="
python3 experiments/exp1_main_comparison.py \
    --runs "$DEFAULT_RUNS" --max-epochs "$DEFAULT_MAX_EPOCHS" \
    --parallel "$DEFAULT_PARALLEL"

# Recover the canonical JSONL path via the sentinel file written by exp1.
if [[ ! -f "$SENTINEL" ]]; then
    echo "ERROR: exp1 did not write $SENTINEL." >&2
    exit 1
fi
OURS_PATH=$(head -1 "$SENTINEL")
if [[ -z "$OURS_PATH" || ! -f "$OURS_PATH" ]]; then
    echo "ERROR: sentinel points at missing file: $OURS_PATH" >&2
    exit 1
fi
echo "exp1 canonical ours: $OURS_PATH"

echo "=== exp2 start: $(date) ==="
python3 experiments/exp2_ablation.py \
    --runs "$DEFAULT_RUNS" --max-epochs "$DEFAULT_MAX_EPOCHS" \
    --parallel "$DEFAULT_PARALLEL"

echo "=== exp3 start: $(date) ==="
python3 experiments/exp3_scaling.py \
    --runs "$DEFAULT_RUNS" \
    --assignment-budget "$EXP3_ASSIGNMENT_BUDGET" \
    --parallel "$DEFAULT_PARALLEL"

echo "=== exp5 start: $(date) ==="
python3 experiments/exp5_case_study.py --seed 42 --max-epochs "$DEFAULT_MAX_EPOCHS"

echo "=== exp4 start: $(date) ==="
# Exp. 4 backbones are independent specs and share the same worker bound.
python3 experiments/exp4_llm_sensitivity.py \
    --runs "$EXP4_RUNS" \
    --max-epochs "$DEFAULT_MAX_EPOCHS" \
    --parallel "$DEFAULT_PARALLEL"

echo "=== validate start: $(date) ==="
python3 experiments/validate_results.py

echo "=== analyze start: $(date) ==="
python3 experiments/analyze.py --exp1-path "$OURS_PATH"

echo "=== ALL DONE: $(date) ==="
