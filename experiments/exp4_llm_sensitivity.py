"""
Experiment 4 — LLM backbone sensitivity.

Runs the proposed method with different LLM providers/models to show
that the method is not fragile to the specific backbone.

Configs compared (default — 4 backbones spanning capability tiers).
``deepseek:deepseek-v4-pro:non-thinking`` is NOT swept here because it is the default
backbone already covered by exp1's canonical ``ours`` row; analyze.py
pulls it from the exp1 JSONL via ``--exp1-path`` and stitches it into
the backbone comparison figure.

    deepseek:deepseek-v4-flash:non-thinking
    qwen:qwen-flash:non-thinking
    qwen:qwen-plus-2025-12-01:non-thinking
    qwen:qwen3-max-2026-01-23:non-thinking

Each row of the output carries:
    - mission_completion_time / success_rate / num_found ...
    - llm_latencies (per-call seconds)
    - llm_input_tokens / llm_output_tokens (per-call token counts)
    - llm_illegal_outputs / llm_impassable_drops / llm_fallback_count
    - llm_calls (per-call trace with from_cache flag)

Scenario: default (same as exp 1). Five seeds per configuration.

Usage:
    python experiments/exp4_llm_sensitivity.py
    python experiments/exp4_llm_sensitivity.py --configs \\
        deepseek:deepseek-v4-flash:non-thinking qwen:qwen-plus-2025-12-01:non-thinking
    python experiments/exp4_llm_sensitivity.py --parallel 4
"""

import argparse

from _common import (
    run_suite, default_output_path, decision_cache_setup,
    add_run_control_args, public_path,
)
from simulation.config import SEED_BASE
from simulation import config as _config


DEFAULT_CONFIGS = [
    "deepseek:deepseek-v4-flash:non-thinking",
    "qwen:qwen-flash:non-thinking",
    "qwen:qwen-plus-2025-12-01:non-thinking",
    "qwen:qwen3-max-2026-01-23:non-thinking",
]


def parse_config(spec: str) -> tuple:
    """Parse ``provider:model[:mode]``.

    If no colon is given, treat the whole string as the model name and
    leave the provider at the current config default.
    """
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) == 1:
        return _config.LLM_PROVIDER, parts[0], None
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid model config: {spec!r}")
    provider, model = parts[:2]
    thinking = None
    if len(parts) == 3:
        mode = parts[2].lower()
        if mode in {"thinking", "think"}:
            thinking = True
        elif mode in {"non-thinking", "nonthinking", "no-think"}:
            thinking = False
        else:
            raise ValueError(f"Unknown inference mode in {spec!r}")
    return provider, model, thinking


def build_specs(configs=None, seeds=None, *, max_epochs=40,
                max_time=36000.0, time_budget=36000.0) -> list:
    """Build deterministic Exp. 4 run specifications."""
    configs = list(configs or DEFAULT_CONFIGS)
    seeds = list(seeds or [SEED_BASE + i for i in range(5)])
    specs = []
    for config_spec in configs:
        provider, model, thinking = parse_config(config_spec)
        mode = ("thinking" if thinking else "non-thinking"
                if thinking is not None else "provider-default")
        label = f"{provider}:{model}:{mode}"
        for i, seed in enumerate(seeds):
            specs.append({
                "method": "ours",
                "seed": seed,
                "provider": provider,
                "model": model,
                "thinking": thinking,
                "reasoning_effort": (
                    "high" if provider == "deepseek" and thinking else None),
                "max_epochs": max_epochs,
                "max_time": max_time,
                "time_budget": time_budget,
                "verbose": False,
                "dump_priorities": True,
                "tag": {"exp": "4", "llm_provider": provider,
                        "llm_model": model, "llm_thinking": thinking,
                        "backbone": label, "replicate": i},
            })
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: LLM backbone sensitivity")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="provider:model:mode entries, e.g. "
                             "deepseek:deepseek-v4-flash:non-thinking "
                             "qwen:qwen-plus-2025-12-01:non-thinking")
    parser.add_argument("--runs", type=int, default=5,
                        help="Seeds per model configuration (default 5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Exact seeds; overrides --runs")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--max-time", type=float, default=36000.0)
    parser.add_argument("--time-budget", type=float, default=36000.0)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Worker processes. Each backbone is an "
                             "independent spec — 4-wide parallelism runs "
                             "all configurations concurrently, bounded by "
                             "the slowest one.")
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
                default_output_path("exp4_llm_sensitivity"))
    cache_state = ("OFF" if args.no_cache
                   else "REFRESH" if args.force_refresh else "ON")
    print(f"[exp4] writing to {public_path(out_path)}")
    seeds = (args.seeds if args.seeds is not None else
             [SEED_BASE + i for i in range(args.runs)])
    print(f"[exp4] configs: {args.configs}")
    print(f"[exp4] seeds: {seeds}")
    print(f"[exp4] runs: {args.runs}, max_epochs: {args.max_epochs}, "
          f"cache: {cache_state}, parallel: {args.parallel}")

    all_specs = build_specs(
        DEFAULT_CONFIGS, seeds, max_epochs=args.max_epochs,
        max_time=args.max_time, time_budget=args.time_budget)
    selected_configs = set(args.configs)
    specs = [spec for spec in all_specs
             if spec["tag"]["backbone"] in selected_configs
             or ":".join([
                 spec["provider"], spec["model"],
                 "thinking" if spec["thinking"] else "non-thinking"
             ]) in selected_configs]

    # ``backbone`` is exactly the canonical provider:model:mode string;
    # the explicit fallback above only keeps compatibility with equivalent
    # spellings accepted by ``parse_config``.
    if not specs:
        raise ValueError("no Exp. 4 configurations selected")

    run_suite(specs, out_path, parallel=args.parallel,
              use_cache=not args.no_cache,
              force_refresh=args.force_refresh,
              aggregate_specs=all_specs,
              selected_run_ids=args.run_id,
              list_only=args.list_runs,
              rerun_completed=args.rerun_completed)

    print(f"\n[exp4] done. Results: {public_path(out_path)}")


if __name__ == "__main__":
    main()
