"""
Aggregation + plotting for all experiments.

Single entry point produces the paper's result artefacts:

    Tables:  results/figures/combined_table.md
             (Exp. 1 baselines + Exp. 2 ablations)
             results/figures/cross_backbone_table.md
             (Exp. 4 plus the default Exp. 1 backbone anchor)
    Figures: results/figures/exp3_batch_sweep.pdf
             results/figures/exp4_backbone_alignment.pdf/.svg
             results/figures/exp5_case_study_current.pdf
    Extra:   results/figures/llm_output_quality_by_backbone_table.md
             (per-backbone parse-retry + impassable-cell stats, kept
             separate from the combined table because it describes LLM
             behaviour rather than method performance.)

Usage:
    python experiments/analyze.py              # regenerate everything
    python experiments/analyze.py --only combined
    python experiments/analyze.py --only exp3
    python experiments/analyze.py --only backbone
    python experiments/analyze.py --only exp5
    python experiments/analyze.py --only quality

exp1 is the single source of truth for the canonical ``ours`` baseline
(default config, 5 seeds). Exp. 2 and Exp. 4 do not re-run that
configuration; the combined table stitches the canonical row back in.
Exp. 3 runs its own complete B=3..10 sweep under a common assignment budget.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")        # headless — no display needed
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIG_DIR = _ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_RESULT_FILES = {
    "exp1": _ROOT / "results" / "exp1_main_current.jsonl",
    "exp2": _ROOT / "results" / "exp2_ablation_current.jsonl",
    "exp3": _ROOT / "results" / "exp3_batch_sweep_current.jsonl",
    "exp4": _ROOT / "results" / "exp4_llm_sensitivity_current.jsonl",
    "exp5": _ROOT / "results" / "exp5_case_study_current.jsonl",
}


# ───────────────────────── I/O helpers ─────────────────────────────

def load_jsonl(path: pathlib.Path) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def resolve_result_path(tag: str, override: str | None = None) -> pathlib.Path:
    """Resolve one formal result file without consulting legacy outputs."""
    if tag not in CURRENT_RESULT_FILES:
        raise KeyError(f"Unknown experiment tag: {tag}")
    path = pathlib.Path(override) if override else CURRENT_RESULT_FILES[tag]
    if not path.exists():
        raise FileNotFoundError(f"Missing {tag} result file: {path}")
    return path


def resolve_exp5_case_dir(summary_path: pathlib.Path | None = None
                          ) -> pathlib.Path:
    """Resolve the case-study run named by Exp. 5's current summary.

    The results directory may contain older case-study runs whose names sort
    after the formal run.  The current JSONL is therefore the only authority
    for selecting the run used by the paper figure and quality analysis.
    """
    path = summary_path or CURRENT_RESULT_FILES["exp5"]
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp. 5 result file: {path}")
    rows = load_jsonl(path)
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row in {path}, found {len(rows)}")
    row = rows[0]
    if row.get("status") != "completed":
        raise ValueError(
            f"Exp. 5 summary is not completed: {row.get('status')!r}")
    run_id = row.get("run_id")
    if not run_id:
        raise ValueError(f"Exp. 5 summary lacks run_id: {path}")
    case_dir = _ROOT / "results" / "runs" / run_id
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"Exp. 5 run directory named by {path} is missing: {case_dir}")
    result_path = case_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing Exp. 5 run result: {result_path}")
    with open(result_path) as f:
        detailed = json.load(f)
    if detailed.get("run_id") != run_id:
        raise ValueError(
            f"Exp. 5 run_id mismatch: summary={run_id!r}, "
            f"result={detailed.get('run_id')!r}")
    return case_dir


def mean_std(vals):
    vals = [v for v in vals if v is not None and
            (not isinstance(v, float) or not math.isinf(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def time_stats_cond(rows: list, field: str):
    """Conditional time statistic — mean ± std of ``field`` computed only
    over rows where ``all_found`` is True.

    Conditional statistics avoid mixing completed missions with censored or
    penalised runs. ``mean_discovery_time`` / ``time_to_first_target`` are
    likewise only defined for targets that were actually discovered.

    Returns:
        (mean, std), n_complete, n_total  — ``(nan, nan)`` if n_complete==0.
    """
    complete = [r for r in rows if r.get("all_found")]
    n_total = len(rows)
    n_complete = len(complete)
    if not complete:
        return (float("nan"), float("nan")), n_complete, n_total
    return mean_std([r[field] for r in complete]), n_complete, n_total


def _fmt_mean_std(pair, width: int = 6, dash_width: int = 13) -> str:
    """Format a (mean, std) tuple; emit a dash of fixed width if NaN."""
    mean, std = pair
    if math.isnan(mean):
        return "—".center(dash_width)
    return f"{mean:>{width}.0f}±{std:<{width-2}.0f}".rjust(dash_width)


def agent_time_for_run(row):
    """Return one run's end-to-end agent computation time.

    Timing results must use the current per-cycle policy-call definition.
    Whole-run ``wall_time`` and LLM-only latency are diagnostic telemetry and
    are never substitutes for this metric.
    """
    times = row.get("agent_computation_times")
    total = row.get("total_agent_computation_time")
    if times is None or total is None:
        raise ValueError(
            "Result row lacks current agent timing fields; regenerate it "
            "with per-cycle agent_computation_times and "
            "total_agent_computation_time."
        )

    summed = float(sum(times))
    recorded = float(total)
    if not math.isclose(summed, recorded, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "Inconsistent current timing fields: "
            f"sum(agent_computation_times)={summed} but "
            f"total_agent_computation_time={recorded}."
        )
    return recorded


def agent_time_per_cycle_stats(rows):
    """Mean±std of each run's average end-to-end agent time per cycle."""
    vals = []
    for row in rows:
        cycles = int(row.get("total_epochs", 0) or 0)
        if cycles > 0:
            vals.append(agent_time_for_run(row) / cycles)
    return mean_std(vals) if vals else (float("nan"), float("nan"))


def effective_mission_completion_time(row):
    """Return mission time from the computation-aware event clock."""
    if row.get("mission_completion_time") is None:
        raise ValueError(
            "Result row lacks mission_completion_time; regenerate it with "
            "the computation-aware event-clock simulator."
        )
    # Validate the separately reported agent metric without adding it again.
    agent_time_for_run(row)
    return float(row["mission_completion_time"])


def completion_time_stats(rows):
    """Conditional end-to-end completion-time statistics."""
    complete = [row for row in rows if row.get("all_found")]
    n_total = len(rows)
    n_complete = len(complete)
    if not complete:
        return (float("nan"), float("nan")), n_complete, n_total
    return (mean_std([effective_mission_completion_time(row)
                      for row in complete]),
            n_complete, n_total)


def token_stats(rows):
    """Total tokens per run (input + output) as mean±std.

    Returns (nan, nan) for runs with no LLM calls so callers can show
    ``—`` in the column.
    """
    vals = []
    any_llm = False
    for r in rows:
        ins = r.get("llm_input_tokens", []) or []
        outs = r.get("llm_output_tokens", []) or []
        if ins or outs:
            any_llm = True
        vals.append(sum(ins) + sum(outs))
    if not any_llm:
        return (float("nan"), float("nan"))
    return mean_std(vals)


def load_canonical_ours(exp1_path) -> list:
    """Return the canonical ``ours`` rows from an exp1 JSONL.

    These rows encode the default configuration
    (NUM_UAVS=10, INITIAL_CLUE=DETAILED,
    provider=deepseek:deepseek-v4-pro, non-thinking,
    all ablation flags ON). Downstream experiments reuse them instead
    of re-running the same configuration.
    """
    if exp1_path is None:
        return []
    p = pathlib.Path(exp1_path)
    if not p.exists():
        print(f"  [warn] --exp1-path {p} does not exist; "
              f"canonical ours will be missing")
        return []
    return [r for r in load_jsonl(p) if r.get("method") == "ours"]


# ──────────── Cross-backbone priority-map visualization ────────────

BACKBONE_ORDER = [
    "deepseek:deepseek-v4-pro:non-thinking",
    "deepseek:deepseek-v4-flash:non-thinking",
    "qwen:qwen-flash:non-thinking",
    "qwen:qwen-plus-2025-12-01:non-thinking",
    "qwen:qwen3-max-2026-01-23:non-thinking",
]

BACKBONE_DISPLAY = {
    "deepseek:deepseek-v4-pro:non-thinking": "DeepSeek V4 Pro",
    "deepseek:deepseek-v4-flash:non-thinking": "DeepSeek V4 Flash",
    "qwen:qwen-flash:non-thinking": "Qwen Flash",
    "qwen:qwen-plus-2025-12-01:non-thinking": "Qwen Plus",
    "qwen:qwen3-max-2026-01-23:non-thinking": "Qwen3 Max",
}

def _collect_backbone_rows(exp1_path, exp4_path) -> list:
    """Collect the default anchor and the four Exp. 4 configurations."""
    rows = load_canonical_ours(exp1_path)
    if exp4_path is not None:
        rows.extend(load_jsonl(pathlib.Path(exp4_path)))
    return rows


def _recreate_targets_for_row(row: dict):
    """Recreate deterministic trajectories for the post-hoc visualization."""
    from simulation.grid_map import GridMap
    from simulation.target import create_targets

    grid_map = GridMap()
    rng = np.random.default_rng(int(row["seed"]))
    return grid_map, create_targets(grid_map, rng)


def _unfound_target_cells(targets, discovery_times: dict,
                          timestamp: float) -> list:
    """Return one ground-truth cell per victim subgroup still unfound at t."""
    cells_by_unit = {}
    for target in targets:
        found_at = discovery_times.get(target.id)
        if found_at is not None and float(found_at) <= timestamp:
            continue
        # Members of a movement unit share one trajectory.  Retaining one
        # cell per unit prevents a two-person family from receiving twice
        # the weight of a single-person subgroup.  A unit remains relevant
        # while at least one of its members is still unfound.
        cells_by_unit.setdefault(
            int(target.unit_id), tuple(target.cell_at(timestamp)))
    return [cells_by_unit[unit_id] for unit_id in sorted(cells_by_unit)]


def _top_priority_cells(priority_grid, grid_map, top_b: int) -> list:
    """Return the highest positive-priority traversable cells."""
    w = np.asarray(priority_grid, dtype=float)
    ranked = []
    for rc in grid_map.get_traversable_cells():
        value = float(w[rc])
        if math.isfinite(value) and value > 0.0:
            ranked.append((value, rc))
    ranked.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return [rc for _, rc in ranked[:top_b]]


def _priority_visualization_for_run(row: dict, *, top_b: int = 10) -> dict:
    """Prepare top-priority and ground-truth markers for visualization.

    Ground truth is reconstructed only after the run.  It never enters the
    LLM prompt, optimizer, or simulator decision state, and no derived
    alignment score is computed.
    """
    snapshots = row.get("priority_snapshots") or []
    if not snapshots:
        raise ValueError(
            f"{row.get('run_id', 'unknown run')} has no priority snapshots")

    grid_map, targets = _recreate_targets_for_row(row)
    discovery_times = {
        int(target_id): float(found_at)
        for target_id, found_at in (row.get("discovery_times") or {}).items()
    }
    records = []
    for index, snapshot in enumerate(snapshots):
        timestamp = float(snapshot["t"])
        target_cells = _unfound_target_cells(
            targets, discovery_times, timestamp)
        top_cells = _top_priority_cells(snapshot["w"], grid_map, top_b)

        records.append({
            "index": index,
            "t": timestamp,
            "top_cells": [list(rc) for rc in top_cells],
            "target_cells": [list(rc) for rc in target_cells],
        })

    return {
        "run_id": row.get("run_id"),
        "seed": int(row["seed"]),
        "backbone": _backbone_label(row),
        "top_b": int(top_b),
        "records": records,
    }


def _backbone_visualization_data(exp1_path, exp4_path,
                                 *, top_b: int = 10) -> dict:
    by_backbone = defaultdict(list)
    for row in _collect_backbone_rows(exp1_path, exp4_path):
        key = _backbone_label(row)
        by_backbone[key].append(
            _priority_visualization_for_run(row, top_b=top_b))
    missing = [key for key in BACKBONE_ORDER if not by_backbone.get(key)]
    if missing:
        raise ValueError(
            "missing cross-backbone priority data: " + ", ".join(missing))
    return {key: by_backbone[key] for key in BACKBONE_ORDER}


def _record_at_time(run: dict, timestamp: float) -> dict:
    """Return a priority record captured at the requested simulation time."""
    matches = [
        record for record in run["records"]
        if math.isclose(float(record["t"]), float(timestamp),
                        rel_tol=0.0, abs_tol=1e-6)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{run.get('run_id', 'unknown run')} has {len(matches)} "
            f"priority records at t={timestamp:g}s; expected exactly one")
    return matches[0]


def plot_backbone_alignment(exp1_path, exp4_path, *, top_b: int = 10):
    """Render each backbone's initial seed-42 priority map at t=3600 s.

    The figure is intentionally limited to a qualitative spatial view so it
    can sit directly below the quantitative cross-backbone performance table.
    """
    visual_data = _backbone_visualization_data(
        exp1_path, exp4_path, top_b=top_b)

    fig, heat_axes = plt.subplots(1, 5, figsize=(7.5, 1.95))

    from simulation.grid_map import GridMap
    grid_map = GridMap()
    impassable = np.array([
        [not grid_map.is_traversable(r, c) for c in range(grid_map.cols)]
        for r in range(grid_map.rows)
    ])

    image = None
    source_rows = _collect_backbone_rows(exp1_path, exp4_path)
    for ax, key in zip(heat_axes, BACKBONE_ORDER):
        representative = next(
            (run for run in visual_data[key] if run["seed"] == 42),
            visual_data[key][0],
        )
        record = _record_at_time(representative, 3600.0)
        source_row = next(
            row for row in source_rows
            if row.get("run_id") == representative["run_id"])
        snapshot = source_row["priority_snapshots"][record["index"]]
        w = np.asarray(snapshot["w"], dtype=float)
        image = ax.imshow(w, origin="lower", cmap="cividis",
                          vmin=0.0, vmax=1.0, interpolation="nearest")
        obstacle_layer = np.ma.masked_where(
            ~impassable, np.zeros_like(impassable, dtype=float))
        ax.imshow(obstacle_layer,
                  origin="lower", cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest", alpha=0.90)
        if record["top_cells"]:
            top_rows = [rc[0] for rc in record["top_cells"]]
            top_cols = [rc[1] for rc in record["top_cells"]]
            ax.scatter(top_cols, top_rows, marker="s", s=18,
                       facecolors="none", edgecolors="white",
                       linewidths=0.65)
        if record["target_cells"]:
            target_rows = [rc[0] for rc in record["target_cells"]]
            target_cols = [rc[1] for rc in record["target_cells"]]
            ax.scatter(target_cols, target_rows, marker="*", s=46,
                       facecolors="white", edgecolors="black",
                       linewidths=0.55, zorder=5)
        ax.set_title(BACKBONE_DISPLAY[key], fontsize=8.2, pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])

    if image is not None:
        cbar = fig.colorbar(image, ax=heat_axes, orientation="horizontal",
                            fraction=0.075, pad=0.10, aspect=55)
        cbar.ax.text(-0.025, 0.5, "Cell priority",
                     transform=cbar.ax.transAxes,
                     ha="right", va="center", fontsize=7.6)
        cbar.ax.tick_params(labelsize=6.8, length=2.5)

    fig.subplots_adjust(left=0.015, right=0.995, top=0.86, bottom=0.19,
                        wspace=0.14)

    pdf_path = FIG_DIR / "exp4_backbone_alignment.pdf"
    svg_path = FIG_DIR / "exp4_backbone_alignment.svg"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {pdf_path}")
    print(f"→ {svg_path}")


# ───────────────────── Experiment 3 — batch-size sensitivity ─────────

def plot_exp3_batch_sweep(input_path: pathlib.Path):
    """Plot mission time and total UAV energy against dispatch batch size."""
    rows = load_jsonl(input_path)
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["B"], r["method"])].append(r)

    methods = sorted({k[1] for k in by_key})
    batch_sizes = sorted({k[0] for k in by_key})

    LABEL_FS = 14
    TICK_FS = 12
    LEGEND_FS = 12
    METHOD_COLOURS = {
        "ours": "#2a6fb5",     # primary blue for the headline method
    }
    FALLBACK_COLOURS = ["#d9534f", "#f0ad4e", "#5cb85c", "#9b59b6"]

    fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(7.5, 4.6),
                                   sharex=True,
                                   gridspec_kw={"hspace": 0.10})
    fc_idx = 0
    method_curves = {}
    for m in methods:
        means, stds, energy_means, energy_stds, completions = [], [], [], [], []
        for b in batch_sizes:
            rs = by_key.get((b, m), [])
            mt, n_complete, n_total = completion_time_stats(rs)
            means.append(mt[0])
            stds.append(mt[1])
            energy = mean_std([
                float(r["total_energy_consumed"]) / 1000.0
                for r in rs if r.get("total_energy_consumed") is not None
            ])
            energy_means.append(energy[0])
            energy_stds.append(energy[1])
            completions.append((n_complete, n_total))
        method_curves[m] = (
            means, stds, energy_means, energy_stds, completions)

        col = METHOD_COLOURS.get(m, FALLBACK_COLOURS[fc_idx % len(FALLBACK_COLOURS)])
        if m not in METHOD_COLOURS:
            fc_idx += 1

        # Top: time panel.
        m_arr = np.array(means, dtype=float)
        s_arr = np.array(stds, dtype=float)
        ax.fill_between(batch_sizes, m_arr - s_arr, m_arr + s_arr,
                        color=col, alpha=0.18, linewidth=0)
        ax.plot(batch_sizes, m_arr, color=col, linewidth=2.4,
                zorder=3, label=m)
        ax.plot(batch_sizes, m_arr, marker="o", linestyle="none",
                markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=2.4,
                markersize=10, zorder=4)

        # Bottom: total energy consumed by the full UAV fleet.
        e_arr = np.array(energy_means, dtype=float)
        es_arr = np.array(energy_stds, dtype=float)
        ax_e.fill_between(batch_sizes, e_arr - es_arr, e_arr + es_arr,
                          color=col, alpha=0.18, linewidth=0)
        ax_e.plot(batch_sizes, e_arr, color=col, linewidth=2.4, zorder=3)
        ax_e.plot(batch_sizes, e_arr, marker="o", linestyle="none",
                  markerfacecolor="white",
                  markeredgecolor=col, markeredgewidth=2.4,
                  markersize=10, zorder=4)

        for b, y, (n_complete, n_total) in zip(
                batch_sizes, means, completions):
            if y is not None and not math.isnan(y):
                ax.annotate(
                    f"{n_complete}/{n_total}", (b, y),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=LEGEND_FS - 2,
                    color="#444")

    # Main operating point used by Exp. 1/2/4/5.
    for a in (ax, ax_e):
        a.axvline(10, color="#888", linewidth=1.0, linestyle="--",
                  alpha=0.6, zorder=1)

    # Tight y-limits so the curve shape is visible.
    def _tight_ylim(a, vals):
        v = [x for x in vals if x is not None and not math.isnan(x)]
        if not v:
            return
        lo, hi = min(v), max(v)
        pad = (hi - lo) * 0.18 if hi > lo else max(1.0, hi * 0.1)
        a.set_ylim(lo - pad, hi + pad)

    primary = method_curves.get("ours") or next(iter(method_curves.values()))
    _tight_ylim(ax,   primary[0])           # times
    _tight_ylim(ax_e, primary[2])           # total energy

    for a in (ax, ax_e):
        a.tick_params(axis="both", labelsize=TICK_FS)
        a.grid(axis="y", linestyle=":", alpha=0.45)
        a.set_axisbelow(True)
    ax_e.text(0.02, 0.96, "dashed: main B = 10",
              transform=ax_e.transAxes,
              ha="left", va="top",
              color="#555", fontsize=LEGEND_FS - 1, fontstyle="italic",
              bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="#cccccc",
                        alpha=0.92), zorder=5)

    ax.set_ylabel("Completion\ntime (s)", fontsize=LABEL_FS)
    ax_e.set_ylabel("Total energy\nconsumed (kJ)", fontsize=LABEL_FS)
    ax_e.set_xlabel("Dispatch batch size (B)", fontsize=LABEL_FS)
    ax_e.set_xticks(batch_sizes)

    # tight_layout doesn't play well with axvspan/axvline added after
    # subplot creation; use manual margins instead.
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.13,
                        hspace=0.10)
    out = FIG_DIR / "exp3_batch_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"→ {out}")


# ───────────────────────── Experiment 5 — case-study figure ─────────

# Terrain class → display colour. Trail/creek are drawn as overlay
# polylines so the underlying terrain (meadow / sparse veg / etc.) remains
# visible underneath them; we paint trail/creek cells the same as their
# surrounding meadow rather than highlighting them in the base layer.
# Keys here MUST match the strings stored in ``GridMap.terrain_grid``
# (full lowercase names from config.TERRAIN_*).
_TERRAIN_COLOURS = {
    "meadow":            "#cfe5b8",
    "sparse_vegetation": "#9ecb84",
    "dense_forest":      "#3d6e3d",
    "rocky":             "#b3b3b3",
    "impassable":        "#6d4a2e",
    "trail":             "#cfe5b8",   # base layer = meadow; overlay drawn separately
    "creek":             "#cfe5b8",   # ditto
    "pond":              "#7eb6e0",
}

_LANDMARK_LABEL = {
    "gcs":        "GCS",
    "trail_fork": "Trail Fork",
    "overlook":   "Overlook",
    "caves":      "Caves",
    "pond":       "Pond",
    "clearing":   "Clearing",
    "campsite":   "Campsite",
}



def _render_terrain(ax, grid_rows):
    """Colour-code each cell by terrain type. Trails and creeks are
    drawn as polylines over the top in the caller (so they look like
    paths, not blocks of colour).
    """
    import matplotlib.colors as mcolors
    nrows = len(grid_rows)
    ncols = len(grid_rows[0])
    rgb = np.zeros((nrows, ncols, 3))
    for r in range(nrows):
        for c in range(ncols):
            rgb[r, c] = mcolors.to_rgb(
                _TERRAIN_COLOURS.get(grid_rows[r][c], "#dddddd"))
    ax.imshow(rgb, origin="lower")


# Per-landmark marker styling — reproduces the reference map.
_LANDMARK_STYLE = {
    "gcs":        dict(marker="D", size=14, mfc="#f08c2a", mec="black",
                       mew=1.2, label="GCS / Entry", text_color="#cc6600"),
    "trail_fork": dict(marker="o", size=12, mfc="#c81818", mec="black",
                       mew=1.2, label="Trail Fork",  text_color="#a01010"),
    "cabin":      dict(marker="s", size=12, mfc="#c81818", mec="black",
                       mew=1.2, label="Cabin",       text_color="#a01010"),
    "overlook":   dict(marker="*", size=18, mfc="#f8a528", mec="black",
                       mew=0.9,  label="Overlook",    text_color="#cc6600"),
    "caves":      dict(marker=None,                   label="Caves",
                       text_color="#3a2a8a"),
    # pond + clearing + campsite use no marker — drawn as italic feature
    # labels by ``_draw_reference_map`` directly.
}

def _draw_reference_map(ax, grid_rows, landmarks, *,
                        label_fs=11, axis_label_fs=14, tick_fs=12):
    """Reproduce the reference terrain map on ``ax``.

    Layers (bottom-up):
      1. terrain colour cells
      2. pond outline
      3. perennial creek (solid blue) and seasonal creek (dotted blue)
      4. main trail (solid brown thick) and branch trails (dashed brown)
      5. landmark markers (per-shape) + bold coloured labels

    The figure-level legend (added by ``analyze_exp5``) covers terrain
    swatches, line styles, and the heatmap markers — keeping the map
    itself uncluttered.
    """
    import matplotlib.colors as mcolors

    nrows = len(grid_rows)
    ncols = len(grid_rows[0])

    # ── 1. Terrain colour layer ────────────────────────────────────
    rgb = np.zeros((nrows, ncols, 3))
    for r in range(nrows):
        for c in range(ncols):
            rgb[r, c] = mcolors.to_rgb(
                _TERRAIN_COLOURS.get(grid_rows[r][c], "#dddddd"))
    ax.imshow(rgb, origin="lower")

    # ── 2. Pond — outline the pond-cell block in dark blue ─────────
    pond_cells = [(r, c) for r in range(nrows) for c in range(ncols)
                  if grid_rows[r][c] == "pond"]
    if pond_cells:
        rs = [p[0] for p in pond_cells]; cs = [p[1] for p in pond_cells]
        rmin, rmax = min(rs) - 0.5, max(rs) + 0.5
        cmin, cmax = min(cs) - 0.5, max(cs) + 0.5
        ax.plot([cmin, cmax, cmax, cmin, cmin],
                [rmin, rmin, rmax, rmax, rmin],
                color="#1f5fa0", linewidth=1.4, zorder=3)
        # Faint cross to suggest open water.
        ax.plot([cmin, cmax], [(rmin+rmax)/2, (rmin+rmax)/2],
                color="#1f5fa0", linewidth=0.6, alpha=0.5, zorder=3)
        ax.plot([(cmin+cmax)/2, (cmin+cmax)/2], [rmin, rmax],
                color="#1f5fa0", linewidth=0.6, alpha=0.5, zorder=3)

    # ── 3. Creeks — single style (GridMap doesn't distinguish) ─────
    creek_cells = {(r, c) for r in range(nrows) for c in range(ncols)
                   if grid_rows[r][c] == "creek"}
    for r, c in creek_cells:
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            n = (r + dr, c + dc)
            if n in creek_cells:
                ax.plot([c, n[1]], [r, n[0]], color="#1f5fa0",
                        linewidth=2.0, solid_capstyle="round", zorder=3)

    # ── 4. Trails — single style; one connected network reaching all
    #          passable regions ────────────────────────────────────
    trail_cells = {(r, c) for r in range(nrows) for c in range(ncols)
                   if grid_rows[r][c] == "trail"}
    for r, c in trail_cells:
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            n = (r + dr, c + dc)
            if n in trail_cells:
                ax.plot([c, n[1]], [r, n[0]], color="#5a3a1a",
                        linewidth=2.6, solid_capstyle="round", zorder=3)

    # ── 5. Landmark markers + bold coloured labels ─────────────────
    for key, (r, c) in landmarks.items():
        st = _LANDMARK_STYLE.get(key)
        if st is None:
            continue
        if st.get("marker"):
            ax.plot(c, r,
                    marker=st["marker"], markersize=st["size"],
                    markerfacecolor=st["mfc"],
                    markeredgecolor=st["mec"],
                    markeredgewidth=st["mew"],
                    linestyle="none", zorder=6)
        # Smart offset — keep label inside the panel.
        (dx, dy), ha, va = _smart_label_offset(r, c, nrows, ncols)
        ax.annotate(st["label"], (c, r),
                    xytext=(dx, dy), textcoords="offset points",
                    color=st["text_color"], fontsize=label_fs,
                    fontweight="bold", ha=ha, va=va,
                    zorder=7, clip_on=False)

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.set_xlabel("Column", fontsize=axis_label_fs)
    ax.set_ylabel("Row (from south)", fontsize=axis_label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)


def _build_figure_legend(label_fs=11):
    """Build the shared legend that sits below every subplot.

    Categories match what's actually in GridMap: 6 terrain swatches +
    1 trail line + 1 creek line. The 2 heatmap-only markers (UAV
    dispatch ▲, target path) are not included — they're self-evident
    from the heatmap context.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=_TERRAIN_COLOURS["meadow"], edgecolor="black",
              linewidth=0.4, label="Meadow"),
        Patch(facecolor=_TERRAIN_COLOURS["sparse_vegetation"],
              edgecolor="black", linewidth=0.4, label="Sparse vegetation"),
        Patch(facecolor=_TERRAIN_COLOURS["dense_forest"],
              edgecolor="black", linewidth=0.4, label="Dense forest"),
        Patch(facecolor=_TERRAIN_COLOURS["rocky"], edgecolor="black",
              linewidth=0.4, label="Rocky terrain"),
        Patch(facecolor=_TERRAIN_COLOURS["impassable"], edgecolor="black",
              linewidth=0.4, label="Impassable zone"),
        Patch(facecolor=_TERRAIN_COLOURS["pond"], edgecolor="#1f5fa0",
              linewidth=1.0, label="Pond"),
        Line2D([], [], color="#1f5fa0", linewidth=2.0, label="Creek"),
        Line2D([], [], color="#5a3a1a", linewidth=2.6, label="Trail"),
    ]


def _smart_label_offset(r, c, nrows, ncols):
    """Pick (dx, dy, ha, va) so a landmark label lands inside the panel.

    Labels in the right third flip to left-anchored; in the top third
    flip to bottom-anchored. Default = up-right.
    """
    right = c > ncols * 0.65
    top = r > nrows * 0.75
    dx = -8 if right else 8
    dy = -10 if top else 8
    ha = "right" if right else "left"
    va = "top" if top else "bottom"
    return (dx, dy), ha, va


def analyze_exp5(case_dir: pathlib.Path):
    """Four-panel case-study figure. Leftmost = terrain map with
    landmark labels (the spatial reference). Right 3 = priority heatmap
    snapshots at evenly spaced times, with UAV dispatch markers and
    target ground-truth trajectories overlaid. All panels share x/y
    axes so the reader can map any heatmap cell back to the terrain.
    """
    print(f"\n=== Exp 5: case study ({case_dir}) ===")
    with open(case_dir / "result.json") as f:
        result = json.load(f)
    with open(case_dir / "target_trajectories.json") as f:
        targets = json.load(f)
    with open(case_dir / "scenario.json") as f:
        scen = json.load(f)

    priorities = result.get("priority_snapshots", [])
    dispatches = result.get("dispatch_log", [])
    print(f"  num_found={result['num_found']}/{scen['num_targets']}, "
          f"epochs={result['total_epochs']}, "
          f"snapshots={len(priorities)}, "
          f"dispatches={len(dispatches)}")
    if not priorities:
        print("  no priority snapshots — was dump_priorities set?")
        return

    # Pick 3 evenly-spaced priority snapshots.
    idxs = np.linspace(0, len(priorities) - 1,
                       num=min(3, len(priorities))).astype(int)
    panels = [priorities[i] for i in idxs]

    # Terrain grid loaded directly from grid_map (scenario.json doesn't
    # carry the full terrain matrix; only landmarks + grid dims).
    from simulation.grid_map import GridMap
    grid_rows = GridMap().terrain_grid

    rows = scen["grid_rows"]
    cols = scen["grid_cols"]
    landmarks = scen["landmarks"]

    # Font sizes — paper-readable defaults.
    TICK_FS = 12
    LABEL_FS = 14
    LM_LABEL_FS = 11        # landmark text on terrain panel
    HEAT_TITLE_FS = 13      # "t = ... s" on heatmap panels

    fig, axes = plt.subplots(1, 4,
                             figsize=(4.4 * 4, 5.0),
                             sharex=True, sharey=True)

    # Subplot captions follow academic style: "(a) ...", "(b) ...",
    # placed BELOW each panel in bold. Letters are assigned in display
    # order (terrain panel first, then heatmap panels left-to-right).
    SUBPLOT_LETTERS = ["a", "b", "c", "d"]
    SUBPLOT_TITLES = [None, None, None, None]   # filled in below

    # ── Leftmost panel: reference-style terrain map ────────────────
    ax_terrain = axes[0]
    _draw_reference_map(ax_terrain, grid_rows, landmarks,
                        label_fs=LM_LABEL_FS,
                        axis_label_fs=LABEL_FS,
                        tick_fs=TICK_FS)
    SUBPLOT_TITLES[0] = "Terrain map"

    # ── Right 3 panels: priority heatmaps ──────────────────────────
    for k, snap in enumerate(panels, start=1):
        ax = axes[k]
        w = np.array(snap["w"])
        ax.imshow(w, origin="lower", cmap="hot",
                  vmin=0, vmax=max(1.0, w.max()))
        ax.set_xlabel("Column", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        # Landmark markers (no text; reader looks at leftmost panel).
        for key, (r, c) in landmarks.items():
            ax.plot(c, r, marker="o", markersize=4,
                    markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=0.8)
        # Dispatch points up to this time.
        t = snap["t"]
        for d in dispatches:
            if d["t"] <= t:
                dr, dc = d["cell"]
                ax.plot(dc, dr, "b^", markersize=5)
        # Target ground-truth path up to this time + final position.
        for tt in targets:
            xs, ys = [], []
            for (tc, tr, ttc) in tt["samples"]:
                if tc <= t:
                    xs.append(ttc); ys.append(tr)
            if xs:
                ax.plot(xs, ys, "-", color="cyan", linewidth=1.2,
                        alpha=0.9)
                ax.plot(xs[-1], ys[-1], "c*", markersize=9,
                        markeredgecolor="black", markeredgewidth=0.5)
        SUBPLOT_TITLES[k] = f"t = {snap['t']:.0f} s"

    # ── Subplot captions "(a) ...", "(b) ..." below each panel ─────
    # Standard academic style: bold lower-case letter in parens, panel
    # description in plain weight, placed below the xlabel using axis
    # transform so it always tracks the panel.
    for k, ax in enumerate(axes):
        if SUBPLOT_TITLES[k] is None:
            continue
        ax.text(0.5, -0.18, f"({SUBPLOT_LETTERS[k]}) {SUBPLOT_TITLES[k]}",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=LABEL_FS, fontweight="bold")

    # Layout: top 70% for panels, then xlabel + subplot caption, then a
    # single-row figure legend at the very bottom. Wide enough for all
    # 8 legend entries on one line so it doesn't fight the captions.
    fig.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.28,
                        wspace=0.05)
    fig.legend(handles=_build_figure_legend(label_fs=LM_LABEL_FS),
               loc="lower center", ncol=8,
               fontsize=LM_LABEL_FS, framealpha=0.95,
               edgecolor="#444",
               bbox_to_anchor=(0.5, 0.005),
               handlelength=1.6, handletextpad=0.5,
               columnspacing=1.4)

    out = FIG_DIR / "exp5_case_study_current.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"→ {out}")


# ───────────────────────── Main table (Exp. 1 + Exp. 2) ──────────────────

def _agg_llm_metrics(rows):
    """Aggregate LLM-related metrics over a list of rows. Returns a dict
    with per-call / per-run token means + illegal / fallback counts.
    Returns None if no row in the group made any LLM calls (baselines).
    """
    api = sum(int(r.get("llm_api_calls", 0) or 0) for r in rows)
    hits = sum(int(r.get("llm_cache_hits", 0) or 0) for r in rows)
    total_calls = api + hits
    if total_calls == 0:
        return None  # non-LLM baseline

    in_all, out_all = [], []
    tot_per_run = []
    tot_illegal = 0
    tot_drops = 0
    tot_fallback = 0
    # Per-run "average drops per illegal response" — one data point per
    # seed that had at least one illegal response. Taking mean±std of
    # these gives a sense of both the typical severity AND the spread
    # across seeds (a high-std signals LLM is inconsistent when it does
    # violate the map).
    per_run_drops_per_illegal = []
    for r in rows:
        ins = r.get("llm_input_tokens", []) or []
        outs = r.get("llm_output_tokens", []) or []
        in_all.extend(ins)
        out_all.extend(outs)
        tot_per_run.append(sum(ins) + sum(outs))
        ill = int(r.get("llm_illegal_outputs", 0) or 0)
        drp = int(r.get("llm_impassable_drops", 0) or 0)
        fb = int(r.get("llm_fallback_count", 0) or 0)
        tot_illegal += ill
        tot_drops += drp
        tot_fallback += fb
        if ill > 0:
            per_run_drops_per_illegal.append(drp / ill)

    drops_per_illegal = (mean_std(per_run_drops_per_illegal)
                         if per_run_drops_per_illegal
                         else (float("nan"), float("nan")))

    return {
        "in_tok": mean_std(in_all) if in_all else (0.0, 0.0),
        "out_tok": mean_std(out_all) if out_all else (0.0, 0.0),
        "tot_tok": mean_std(tot_per_run) if tot_per_run else (0.0, 0.0),
        "api_calls": api,
        "cache_hits": hits,
        "total_calls": total_calls,
        "illegal_n": tot_illegal,
        "illegal_drops": tot_drops,
        "drops_per_illegal": drops_per_illegal,  # (mean, std) across seeds
        "fallback_n": tot_fallback,
    }


def _row_tuple(rows, display_name):
    """Compute the common per-section statistics for a group of rows.

    Returns a dict with:
      name, n, found(mean,std), n_complete, n_total, maxT(mean,std),
      epochs(mean,std), e_tgt(mean,std), agent_time(mean,std),
      llm (agg_llm_metrics dict or None)
    """
    found = mean_std([r["num_found"] for r in rows])
    mt, n_c, n_t = completion_time_stats(rows)
    epochs = mean_std([r.get("total_epochs", 0) or 0 for r in rows])
    # E/tgt is in J; convert to kJ. Skip +inf (no targets found).
    e_vals = [r["energy_per_found_target"] / 1000.0 for r in rows
              if not math.isinf(r["energy_per_found_target"])]
    e = mean_std(e_vals) if e_vals else (float("nan"), float("nan"))
    agent_time = agent_time_per_cycle_stats(rows)
    return {
        "name": display_name, "n": len(rows),
        "found": found, "n_complete": n_c, "n_total": n_t,
        "maxT": mt, "epochs": epochs,
        "e_tgt": e, "agent_time": agent_time,
        "llm": _agg_llm_metrics(rows),
    }


def analyze_combined(exp1_path, exp2_path):
    """Emit the main comparison table spanning:

      A. Baselines + proposed (Exp 1, 7 methods × 5 seeds)
      B. LLM-Direct + ablations (Exp 2, 3 conditions × 5 seeds,
         full = Exp 1 ours)
    Cross-backbone evidence is emitted independently by
    :func:`analyze_cross_backbone`.
    """
    exp1_rows = load_jsonl(exp1_path)
    exp2_rows = load_jsonl(exp2_path) if exp2_path else []

    # Section A — baselines + ours (anchor)
    by_method_A = defaultdict(list)
    for r in exp1_rows:
        by_method_A[r["method"]].append(r)
    ours_rows = by_method_A.get("ours", [])

    section_A = []
    baseline_order = [
        "lawnmower",
        "greedy_coverage",
        "bayesian_search",
        "clue_prior_sweep",
        "clue_prior_greedy",
        "clue_prior_bayes",
    ]
    for m in baseline_order:
        rs = by_method_A.get(m, [])
        if rs:
            section_A.append(_row_tuple(rs, m))
    # anchor row last in baselines section
    if ours_rows:
        section_A.append(_row_tuple(ours_rows, "ours (anchor)"))

    # Section B — ablations (variant rows from exp2)
    by_variant_B = defaultdict(list)
    for r in exp2_rows:
        by_variant_B[r.get("variant", "?")].append(r)
    ablation_order = ["llm_direct", "no_rag", "no_cot"]
    section_B = []
    for v in ablation_order:
        rs = by_variant_B.get(v, [])
        if rs:
            label = ("LLM-Direct" if v == "llm_direct"
                     else f"ours − {v.replace('_', ' ')}")
            section_B.append(_row_tuple(rs, label))

    # Print table
    print(f"\n=== Main comparison table (Exp 1 + Exp 2) ===")
    print(f"  Anchor row = canonical ours (default config, "
          f"deepseek-v4-pro non-thinking, N=10, 5 seeds)")
    print(f"  — for baseline rows LLM cols are N/A")
    print(f"  — E/tgt over found targets only; NaN = no seed completed")
    header = (f"{'Row':<32} {'n':>2} {'Found':>9} {'Comp':>6} "
              f"{'Cycles':>9} {'T_comp(s)':>13} {'E/tgt(kJ)':>13} "
              f"{'Tot tok/run':>14} {'Agent/cyc(s)':>13} "
              f"{'Illegal(n)':>11} {'Drops/ill':>13} {'FB(n)':>6}")
    print(header)
    print("-" * len(header))

    def _print_row(row):
        llm = row["llm"]
        tot_tok_str = (f"{llm['tot_tok'][0]:>6.0f}±{llm['tot_tok'][1]:<4.0f}"
                       if llm else "—".center(14))
        if llm:
            ill_n_str = f"{llm['illegal_n']:>4}/{llm['total_calls']:<4}"
            dpi = llm['drops_per_illegal']
            if math.isnan(dpi[0]):
                dpi_str = "—".center(13)
            else:
                dpi_str = f"{dpi[0]:>5.1f}±{dpi[1]:<4.1f}".rjust(13)
            fb_str = f"{llm['fallback_n']:>5}"
        else:
            ill_n_str = "—".center(11)
            dpi_str = "—".center(13)
            fb_str = "—".center(6)
        e_str = (f"{row['e_tgt'][0]:>6.0f}±{row['e_tgt'][1]:<4.0f}"
                 if not math.isnan(row['e_tgt'][0])
                 else "—".center(13))
        ep = row["epochs"]
        ep_str = f"{ep[0]:>4.1f}±{ep[1]:<3.1f}"
        print(f"{row['name']:<32} {row['n']:>2} "
              f"{row['found'][0]:>4.1f}±{row['found'][1]:<3.1f} "
              f"{row['n_complete']:>2}/{row['n_total']:<2} "
              f"{ep_str} "
              f"{_fmt_mean_std(row['maxT'])} "
              f"{e_str} "
              f"{tot_tok_str} "
              f"{_fmt_mean_std(row['agent_time'])} "
              f"{ill_n_str} {dpi_str} {fb_str}")

    print("\n[A] Baselines + proposed (Exp 1)")
    for r in section_A:
        _print_row(r)
    print("\n[B] Ablation (Exp 2; anchor = ``ours`` above)")
    for r in section_B:
        _print_row(r)
    # Write the main-comparison Markdown table.
    md_path = FIG_DIR / "combined_table.md"
    with open(md_path, "w") as f:
        f.write("| Row | n | Found | Complete | Cycles | T_comp (s) | "
                "E/vic (kJ) | Tok/run | T_agent/cycle (s) | "
                "Illegal (n/total) | Drops/illegal | Fallback (n) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

        def _md_row(row, is_header=False):
            if is_header:
                f.write(f"| **{row}** | | | | | | | | | | | |\n")
                return
            llm = row["llm"]
            maxT_md = ("—" if math.isnan(row["maxT"][0])
                       else f"{row['maxT'][0]:.0f} ± {row['maxT'][1]:.0f}")
            e_md = ("—" if math.isnan(row['e_tgt'][0])
                    else f"{row['e_tgt'][0]:.0f} ± {row['e_tgt'][1]:.0f}")
            agent_md = ("—" if math.isnan(row['agent_time'][0])
                        else f"{row['agent_time'][0]:.0f} ± "
                             f"{row['agent_time'][1]:.0f}")
            ep_md = f"{row['epochs'][0]:.1f} ± {row['epochs'][1]:.1f}"
            if llm is None:
                tok_md = ill_md = dpi_md = fb_md = "—"
            else:
                tok_md = f"{llm['tot_tok'][0]:.0f} ± {llm['tot_tok'][1]:.0f}"
                ill_md = f"{llm['illegal_n']}/{llm['total_calls']}"
                dpi = llm['drops_per_illegal']
                dpi_md = ("—" if math.isnan(dpi[0])
                          else f"{dpi[0]:.1f} ± {dpi[1]:.1f}")
                fb_md = f"{llm['fallback_n']}"
            f.write(f"| {row['name']} | {row['n']} | "
                    f"{row['found'][0]:.1f} ± {row['found'][1]:.1f} | "
                    f"{row['n_complete']}/{row['n_total']} | "
                    f"{ep_md} | {maxT_md} | "
                    f"{e_md} | {tok_md} | {agent_md} | "
                    f"{ill_md} | {dpi_md} | {fb_md} |\n")

        _md_row("Baselines + proposed (Exp 1)", is_header=True)
        for r in section_A:
            _md_row(r)
        _md_row("Ablation (Exp 2)", is_header=True)
        for r in section_B:
            _md_row(r)
    print(f"\n→ {md_path}")


# ───────────────────────── Cross-backbone evidence ───────────────────────

def analyze_cross_backbone(exp1_path, exp4_path):
    """Emit the cross-backbone performance table and qualitative figure."""
    all_rows = _collect_backbone_rows(exp1_path, exp4_path)
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[_backbone_label(row)].append(row)

    table_rows = []
    for key in BACKBONE_ORDER:
        rows = grouped.get(key, [])
        if not rows:
            continue
        summary = _row_tuple(rows, BACKBONE_DISPLAY[key])
        table_rows.append(summary)

    print("\n=== Cross-backbone table (Exp. 1 anchor + Exp. 4) ===")
    header = (f"{'Backbone':<22} {'Found':>9} {'Comp':>6} "
              f"{'Cycles':>9} {'T_comp(s)':>13} {'E/tgt(kJ)':>13} "
              f"{'Tok/run':>13} {'Agent/cyc':>12} {'Illegal':>10} "
              f"{'Drops/ill':>12} {'Fb.':>5}")
    print(header)
    print("-" * len(header))
    for row in table_rows:
        llm = row["llm"]
        illegal_rate = (100.0 * llm["illegal_n"] / llm["total_calls"]
                        if llm and llm["total_calls"] else float("nan"))
        print(
            f"{row['name']:<22} "
            f"{row['found'][0]:>4.1f}±{row['found'][1]:<3.1f} "
            f"{row['n_complete']:>2}/{row['n_total']:<2} "
            f"{row['epochs'][0]:>4.1f}±{row['epochs'][1]:<3.1f} "
            f"{_fmt_mean_std(row['maxT'])} "
            f"{_fmt_mean_std(row['e_tgt'])} "
            f"{_fmt_mean_std(llm['tot_tok'])} "
            f"{_fmt_mean_std(row['agent_time'])} "
            f"{illegal_rate:>8.1f}% "
            f"{_fmt_mean_std(llm['drops_per_illegal'])} "
            f"{llm['fallback_n']:>5}"
        )

    md_path = FIG_DIR / "cross_backbone_table.md"
    with open(md_path, "w") as f:
        f.write("| Backbone | n | Found | Complete | Cycles | "
                "T_comp (s) | E/vic (kJ) | Tok/run | "
                "T_agent/cycle (s) | Illegal (%) | Drops/ill. | Fb. |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in table_rows:
            llm = row["llm"]
            illegal_rate = (100.0 * llm["illegal_n"] / llm["total_calls"]
                            if llm and llm["total_calls"] else float("nan"))
            max_t = ("—" if math.isnan(row["maxT"][0])
                     else f"{row['maxT'][0]:.0f} ± {row['maxT'][1]:.0f}")
            energy = ("—" if math.isnan(row["e_tgt"][0])
                      else f"{row['e_tgt'][0]:.0f} ± {row['e_tgt'][1]:.0f}")
            f.write(
                f"| {row['name']} | {row['n']} | "
                f"{row['found'][0]:.1f} ± {row['found'][1]:.1f} | "
                f"{row['n_complete']}/{row['n_total']} | "
                f"{row['epochs'][0]:.1f} ± {row['epochs'][1]:.1f} | "
                f"{max_t} | {energy} | "
                f"{llm['tot_tok'][0]:.0f} ± {llm['tot_tok'][1]:.0f} | "
                f"{row['agent_time'][0]:.0f} ± "
                f"{row['agent_time'][1]:.0f} | "
                f"{illegal_rate:.1f} | "
                f"{llm['drops_per_illegal'][0]:.1f} ± "
                f"{llm['drops_per_illegal'][1]:.1f} | "
                f"{llm['fallback_n']} |\n"
            )
    print(f"\n→ {md_path}")
    plot_backbone_alignment(exp1_path, exp4_path)


# ───────────────────────── LLM output-quality aggregation ─────────────────

# Two flavours of LLM-output problems get reported under the same
# "LLM output quality" subsection of the paper:
#
#   A. Parse failures — the response could not be turned into usable
#      priorities / assignments. Tracked per-attempt in
#      ``llm_parse_retry_events`` (classified by kind) and aggregated
#      into ``llm_fallback_count`` (decisions where every retry
#      attempt failed).
#
#   B. Impassable-cell placements — the response WAS parseable but put
#      priority or assignment weight on cells that are physically
#      unreachable. Tracked as ``llm_illegal_outputs`` (# responses with
#      at least one such cell) and ``llm_impassable_drops`` (total cells
#      dropped across responses).
#
# Both are "the LLM didn't give us a clean answer" events — A is hard
# (need retry), B is soft (can keep the rest of the answer). Reporting
# them together lets the paper compare backbone robustness.

# Stable x-axis order — every kind the classifier can emit.
PARSE_RETRY_KINDS = [
    "max_tokens",
    "multi_json_block",
    "tiny_response",
    "no_json",
    "json_decode_error",
]


def _backbone_label(row: dict) -> str:
    """Normalize the (provider, model) pair into a backbone identifier.

    Defaults to ``deepseek:deepseek-v4-pro`` when the row predates the
    new fields (most rows from the original experiments carry only
    ``provider``, not ``llm_model`` / ``model``).
    """
    if row.get("backbone"):
        return row["backbone"]
    prov = row.get("provider") or "deepseek"
    model = (row.get("llm_model") or row.get("model")
             or ("deepseek-v4-pro" if prov == "deepseek"
                 else "qwen-plus-2025-12-01" if prov == "qwen"
                 else "hunyuan-turbo"))
    label = f"{prov}:{model}"
    if row.get("llm_thinking") is not None:
        label += (":thinking" if row["llm_thinking"]
                  else ":non-thinking")
    return label


def _collect_retry_rows():
    """Pull the five current JSONLs into a flat, traceable row list."""
    out = []
    for tag in ("exp1", "exp2", "exp3", "exp4", "exp5"):
        p = resolve_result_path(tag)
        for r in load_jsonl(p):
            r["_source_exp"] = tag
            out.append(r)
    return out


def analyze_llm_output_quality():
    """Aggregate LLM output-quality issues by backbone:
      * parse failures: kind-level counts + final-fallback count
      * impassable-cell placements: illegal-response rate + drop counts

    Reads the five current experiment JSONLs. Outputs a
    stdout table + a markdown table for paste-into-paper.
    """
    rows = _collect_retry_rows()
    print(f"\n=== LLM output quality "
          f"(across exp1+exp2+exp3+exp4+exp5, {len(rows)} rows) ===")

    # per-backbone aggregates
    by_bb = defaultdict(lambda: {
        "rows": 0, "api_calls": 0, "cache_hits": 0,
        "fallback_count": 0, "retry_events": defaultdict(int),
        "decisions_with_retry": 0,
        "illegal_outputs": 0, "impassable_drops": 0,
    })
    for r in rows:
        bb = _backbone_label(r)
        # Only LLM-using methods contribute meaningful retry stats;
        # baselines (lawnmower / greedy / ...) emit no LLM calls.
        if r.get("method") in (
                "lawnmower", "greedy_coverage", "bayesian_search",
                "clue_prior_sweep", "clue_prior_greedy",
                "clue_prior_bayes", "random"):
            continue
        s = by_bb[bb]
        s["rows"] += 1
        s["api_calls"] += int(r.get("llm_api_calls", 0) or 0)
        s["cache_hits"] += int(r.get("llm_cache_hits", 0) or 0)
        s["fallback_count"] += int(r.get("llm_fallback_count", 0) or 0)
        s["illegal_outputs"] += int(r.get("llm_illegal_outputs", 0) or 0)
        s["impassable_drops"] += int(r.get("llm_impassable_drops", 0) or 0)
        events = r.get("llm_parse_retry_events", []) or []
        # Group events by (call_index) so we can count unique decisions.
        # Each retry loop keeps firing _call_llm at incrementing indices,
        # so events within the same "decision" land on *consecutive*
        # ``call_index`` values. We approximate "decisions with >=1
        # retry" as the number of events (each event = one decision
        # that had at least one retry-triggering attempt, counted once
        # per failed attempt). With the current policy a fallback contains
        # two failed attempts (the initial request and one retry).
        s["decisions_with_retry"] += (1 if events else 0)
        for e in events:
            kind = e.get("kind", "json_decode_error")
            s["retry_events"][kind] += 1

    # Print table — two conceptual groups:
    #   (A) parse failures:   total retries, fallback count, per-kind counts
    #   (B) impassable cells: illegal_resps (per-response), impassable_drops
    # Rates are shown as percent of API calls for apples-to-apples
    # comparison across backbones with different run volumes.
    kinds = PARSE_RETRY_KINDS
    header = (f"{'backbone':<30} {'rows':>5} {'api':>7} "
              f"{'fallback':>9} {'retries':>8} "
              f"{'illegal%':>9} {'drops':>7}")
    for k in kinds:
        header += f" {k:>18}"
    print(header)
    print("-" * len(header))
    bb_sorted = sorted(by_bb.keys())
    totals = defaultdict(int)
    for bb in bb_sorted:
        s = by_bb[bb]
        total_retries = sum(s["retry_events"].values())
        illegal_pct = (100.0 * s["illegal_outputs"] / s["api_calls"]
                       if s["api_calls"] else 0.0)
        line = (f"{bb:<30} {s['rows']:>5} {s['api_calls']:>7} "
                f"{s['fallback_count']:>9} {total_retries:>8} "
                f"{illegal_pct:>7.1f}%  {s['impassable_drops']:>7}")
        for k in kinds:
            line += f" {s['retry_events'][k]:>18}"
        print(line)
        totals["api"] += s["api_calls"]
        totals["retries"] += total_retries
        totals["fallback"] += s["fallback_count"]
        totals["illegal_outputs"] += s["illegal_outputs"]
        totals["impassable_drops"] += s["impassable_drops"]
        for k in kinds:
            totals[k] += s["retry_events"][k]
    # Totals row
    total_illegal_pct = (100.0 * totals["illegal_outputs"] / totals["api"]
                         if totals["api"] else 0.0)
    line = (f"{'TOTAL':<30} {'':>5} {totals['api']:>7} "
            f"{totals['fallback']:>9} {totals['retries']:>8} "
            f"{total_illegal_pct:>7.1f}%  {totals['impassable_drops']:>7}")
    for k in kinds:
        line += f" {totals[k]:>18}"
    print("-" * len(header))
    print(line)

    # Markdown table
    md_path = FIG_DIR / "llm_output_quality_by_backbone_table.md"
    with open(md_path, "w") as f:
        f.write("| Backbone | Rows | API calls | "
                "Final fallback | Total retries | "
                "Illegal resp% | Impassable drops |")
        for k in kinds:
            f.write(f" {k} |")
        f.write("\n|---|---|---|---|---|---|---|")
        f.write("---|" * len(kinds))
        f.write("\n")
        for bb in bb_sorted:
            s = by_bb[bb]
            total_retries = sum(s["retry_events"].values())
            illegal_pct = (100.0 * s["illegal_outputs"] / s["api_calls"]
                           if s["api_calls"] else 0.0)
            f.write(f"| {bb} | {s['rows']} | {s['api_calls']} | "
                    f"{s['fallback_count']} | {total_retries} | "
                    f"{illegal_pct:.1f}% ({s['illegal_outputs']}/{s['api_calls']}) | "
                    f"{s['impassable_drops']} |")
            for k in kinds:
                f.write(f" {s['retry_events'][k]} |")
            f.write("\n")
    print(f"→ {md_path}")
    return by_bb


# ───────────────────────── CLI ─────────────────────────────

def main():
    """Single entry point. Regenerates all paper artefacts:
      1. ``combined_table.md`` (Exp. 1 + Exp. 2 main table)
      2. ``cross_backbone_table.md`` and
         ``exp4_backbone_alignment.pdf/.svg`` (Exp. 4 + anchor)
      3. ``exp3_batch_sweep.pdf`` (batch-size sensitivity figure)
      4. ``exp5_case_study_current.pdf`` (priority heatmap + trajectory figure)
      5. ``llm_output_quality_by_backbone_table.md``
         (per-backbone parse-retry + impassable-cell breakdown)

    Formal ``*_current.jsonl`` files are the default sources. Explicit path
    arguments remain available for deliberate one-off analysis.
    """
    parser = argparse.ArgumentParser(
        description="Regenerate every paper table + figure in one pass")
    parser.add_argument("--only", type=str, default=None,
                        choices=["combined", "backbone", "exp3", "exp5",
                                 "quality"],
                        help="If given, regenerate only that one artefact.")
    parser.add_argument("--exp1-path", type=str, default=None,
                        help="Override exp1 JSONL (default: latest)")
    parser.add_argument("--exp2-path", type=str, default=None,
                        help="Override exp2 JSONL (default: latest)")
    parser.add_argument("--exp3-path", type=str, default=None,
                        help="Override Exp. 3 JSONL (default: latest)")
    parser.add_argument("--exp4-path", type=str, default=None,
                        help="Override exp4 JSONL (default: latest)")
    parser.add_argument("--exp5-dir", type=str, default=None,
                        help="Override exp5 case-study directory")
    args = parser.parse_args()

    e1 = resolve_result_path("exp1", args.exp1_path)
    e2 = resolve_result_path("exp2", args.exp2_path)
    e3 = resolve_result_path("exp3", args.exp3_path)
    e4 = resolve_result_path("exp4", args.exp4_path)
    if args.exp5_dir:
        e5 = pathlib.Path(args.exp5_dir)
        if not e5.is_dir():
            raise FileNotFoundError(f"Missing Exp. 5 directory: {e5}")
    else:
        e5 = resolve_exp5_case_dir()

    tasks = args.only or "all"

    if tasks in ("all", "combined") and e1 is not None:
        analyze_combined(str(e1), str(e2) if e2 else None)

    if tasks in ("all", "backbone"):
        if e1 is None or e4 is None:
            if tasks == "backbone":
                raise FileNotFoundError(
                    "cross-backbone analysis requires Exp. 1 and Exp. 4 "
                    "result files")
        else:
            analyze_cross_backbone(str(e1), str(e4))

    if tasks in ("all", "exp3") and e3 is not None:
        plot_exp3_batch_sweep(e3)

    if tasks in ("all", "exp5") and e5 is not None:
        analyze_exp5(e5)

    if tasks in ("all", "quality"):
        analyze_llm_output_quality()


if __name__ == "__main__":
    main()
