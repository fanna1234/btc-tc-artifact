#!/usr/bin/env python3
"""Generate microarchitectural profiling figure (redesigned).

Layout (figure*, 1×3):
  (a) Warp Cycle Breakdown — horizontal stacked bars (per-method avg, 11 datasets)
  (b) Performance Profile  — radar chart (6 metrics, normalized, 3 method polygons)
  (c) L2 Throughput        — horizontal dumbbell dot plot (per dataset, smoking gun)

Data source: results/ncu/pro6000_{method}_{dataset}_*.raw.csv

Outputs:
  - sc26/paper/figures/microarch_profile.{pdf,png}
  - paper/figures/microarch_profile.{pdf,png}
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_plot_style import apply_paper_style

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_OUT_BASE = REPO_ROOT / "results" / "figures" / "microarch_profile"
SECONDARY_OUT_BASE = REPO_ROOT / "results" / "figures" / "microarch_profile"

METHODS = ["BTC-TC", "ToT", "TRUST", "Polak"]
METHOD_KEY_MAP = {"BTC-TC": "btc128", "ToT": "tot", "TRUST": "trust", "Polak": "polak"}
METHOD_COLORS = {
    "BTC-TC": "#C0392B",
    "ToT": "#4E79A7",
    "TRUST": "#2E7D32",
    "Polak": "#7B3F00",
}

# Kernel name patterns for matching the actual compute kernel
KERNEL_PATTERNS = {
    "BTC-TC": ["btc::kernel_16x128", "btc::kernel_16x32"],
    "ToT": ["tot::tot_kernel"],
    "TRUST": ["dynamic_assign"],
    "Polak": ["CalculateTriangles"],
}

DATASETS = [
    "Ga41As41H72", "Si41Ge41H72", "F1",
    "g7jac140sc",
    "eu-2005", "consph", "cant", "pwtk",
]
DATASET_SHORT = {
    "g7jac140sc": "g7jac",
    "Ga41As41H72": "Ga41As",
    "Si41Ge41H72": "Si41Ge",
    "cant": "cant",
    "F1": "F1",
    "consph": "consph",
    "eu-2005": "eu-05",
    "pwtk": "pwtk",
}

ALL_COLUMNS = {
    # Compute metrics
    "compute_tp": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "ipc": "sm__inst_executed.avg.per_cycle_active",
    # Memory hierarchy
    "l1_hit": "l1tex__t_sector_hit_rate.pct",
    "l1_tp": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "l2_tp": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_tp": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # Warp / scheduling
    "warp_latency": "smsp__average_warp_latency_per_inst_issued.ratio",
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Stall breakdown
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "short_sb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "not_sel": "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "wait": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "lg_throttle": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "selected": "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
}

STALL_CATEGORIES = [
    ("selected", "Executing", "#22c55e"),
    ("long_sb", "Long SB", "#ef4444"),
    ("lg_throttle", "LG Thr.", "#f97316"),
    ("wait", "Wait", "#eab308"),
    ("short_sb", "Short SB", "#8b5cf6"),
    ("not_sel", "Not Sel.", "#06b6d4"),
]

# Radar chart axes: (display_label, metric_key, higher_is_better)
RADAR_AXES = [
    ("Compute\nTP", "compute_tp", True),
    ("IPC", "ipc", True),
    ("L1 Hit\nRate", "l1_hit", True),
    ("L2\nEfficiency", "l2_tp", False),        # lower L2 TP = less congestion
    ("Warp\nEfficiency", "warp_latency", False),  # lower latency = better
    ("Occupancy", "occupancy", True),
]

C_AXIS = "#78909C"
C_PRIMARY = "#2C3E50"
C_GRID = "#D5DBDB"


# --------------- data loading ---------------

def find_latest_csv(method_key: str, dataset: str) -> Path | None:
    pattern = str(
        REPO_ROOT / f"results/ncu/pro6000_{method_key}_{dataset}_2026*.raw.csv"
    )
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def parse_ncu_csv(filepath: Path, method: str = "") -> dict:
    with open(filepath) as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header found in {filepath}")
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    if len(rows) > 1:
        rows = rows[1:]

    patterns = KERNEL_PATTERNS.get(method, [])
    matched_rows = [
        r for r in rows
        if any(pat in r.get("Kernel Name", "") for pat in patterns)
    ] if patterns else []

    candidates = matched_rows if matched_rows else rows

    best, best_time = None, 0.0
    for r in candidates:
        t_str = r.get("gpu__time_duration.sum", "0").replace(",", "")
        try:
            t = float(t_str)
        except ValueError:
            t = 0.0
        if t > best_time:
            best_time = t
            best = r
    row = best if best else rows[0]

    result = {}
    for metric_key, col_name in ALL_COLUMNS.items():
        raw = row.get(col_name, "0")
        raw = raw.replace(",", "") if raw else "0"
        try:
            result[metric_key] = float(raw)
        except ValueError:
            result[metric_key] = 0.0
    return result


def load_all_data() -> dict:
    data = {}
    for method in METHODS:
        mkey = METHOD_KEY_MAP[method]
        for dataset in DATASETS:
            fpath = find_latest_csv(mkey, dataset)
            if fpath is None or not fpath.exists():
                print(f"WARNING: missing CSV for {method} / {dataset}")
                data[(method, dataset)] = {k: 0.0 for k in ALL_COLUMNS}
                continue
            data[(method, dataset)] = parse_ncu_csv(fpath, method=method)
    return data


# --------------- drawing helpers ---------------

def style_ax(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(C_AXIS)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors="#546E7A", width=0.6)


def draw_stall_panel(ax, data):
    """Panel (a): One stacked horizontal bar per method with occupancy annotation."""
    style_ax(ax)

    DISPLAY_CATS = [
        ("selected", "Executing", "#22c55e"),
        ("long_sb", "Long SB", "#ef4444"),
        ("lg_throttle", "LG Thr.", "#f97316"),
        ("short_sb", "Short SB", "#8b5cf6"),
        ("wait", "Wait", "#eab308"),
        ("not_sel", "Not Sel.", "#06b6d4"),
    ]
    n_m = len(METHODS)
    bar_h = 0.55
    y_pos = np.arange(n_m)[::-1] * 1.1

    for m_idx, method in enumerate(METHODS):
        y = y_pos[m_idx]
        left = 0.0
        total = 0.0
        for cat_key, cat_label, cat_color in DISPLAY_CATS:
            per_ds = [data[(method, ds)].get(cat_key, 0.0) for ds in DATASETS]
            avg_val = float(np.mean(per_ds))
            total += avg_val
            ax.barh(
                y, avg_val, bar_h,
                left=left, color=cat_color, edgecolor="white",
                linewidth=0.4, alpha=0.85, zorder=3,
            )
            if avg_val >= 3.0:
                ax.text(
                    left + avg_val / 2, y,
                    f"{avg_val:.0f}" if avg_val >= 10 else f"{avg_val:.1f}",
                    ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold", zorder=5,
                )
            left += avg_val

        # Occupancy per method
        occ_vals = [data[(method, ds)].get("occupancy", 0.0) for ds in DATASETS]
        occ_avg = float(np.mean(occ_vals))

        # Σ total + occupancy annotation
        ax.text(
            left + 1.0, y,
            f"\u03a3{total:.0f}  occ {occ_avg:.0f}%",
            ha="left", va="center", fontsize=6.5,
            color="#666", fontweight="bold", zorder=5,
        )

    # Y-axis: method names in their own colour
    ax.set_yticks(y_pos)
    labels = ax.set_yticklabels(
        METHODS, fontsize=8.5, fontweight="bold",
    )
    for lbl, method in zip(labels, METHODS):
        lbl.set_color(METHOD_COLORS[method])

    ax.set_xlabel("Avg. stalled warps / issue", fontsize=7.5, labelpad=2)
    ax.set_title(
        "(a) Warp Stall Breakdown", fontsize=9, pad=6,
        fontweight="semibold", color=C_PRIMARY, loc="left",
    )
    ax.grid(True, axis="x", alpha=0.15, linestyle="-", linewidth=0.4, color=C_GRID)
    ax.set_ylim(y_pos[-1] - 0.7, y_pos[0] + 0.7)
    ax.set_xlim(left=0)

    cat_handles = [
        mpatches.Patch(facecolor=c, label=l, alpha=0.85)
        for _, l, c in DISPLAY_CATS
    ]
    ax.legend(
        handles=cat_handles, loc="upper right", ncol=3, fontsize=5.5,
        frameon=True, framealpha=0.92, edgecolor="#CFD8DC", facecolor="white",
        handlelength=0.8, handletextpad=0.3, borderpad=0.3,
        labelspacing=0.2, columnspacing=0.6,
    )


def draw_radar_panel(ax, data):
    """Panel (b): Radar chart — 6 normalized axes, 3 method polygons."""

    N = len(RADAR_AXES)

    # Per-method averages across all datasets
    avg = {}
    for method in METHODS:
        avg[method] = {}
        for _, mk, _ in RADAR_AXES:
            vals = [data[(method, ds)][mk] for ds in DATASETS]
            avg[method][mk] = float(np.mean(vals))

    # Normalize: 0 = worst, 1 = best (per metric)
    norm = {m: {} for m in METHODS}
    raw_display = {}  # for annotation: (metric_key -> best_raw_value)
    for _, mk, higher in RADAR_AXES:
        raw_vals = [avg[m][mk] for m in METHODS]
        if higher:
            mx = max(raw_vals)
            for m in METHODS:
                norm[m][mk] = avg[m][mk] / mx if mx > 0 else 0
            raw_display[mk] = mx
        else:
            mn = min(raw_vals)
            for m in METHODS:
                norm[m][mk] = mn / avg[m][mk] if avg[m][mk] > 0 else 0
            raw_display[mk] = mn

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # Draw polygons
    for method in METHODS:
        vals = [norm[method][mk] for _, mk, _ in RADAR_AXES]
        vals_closed = vals + vals[:1]
        ax.plot(
            angles_closed, vals_closed,
            color=METHOD_COLORS[method], linewidth=1.6, label=method, zorder=3,
        )
        ax.fill(
            angles_closed, vals_closed,
            color=METHOD_COLORS[method], alpha=0.10, zorder=2,
        )
        # Dot markers at vertices
        ax.scatter(
            angles, vals, s=18, c=METHOD_COLORS[method],
            edgecolor="white", linewidth=0.3, zorder=4,
        )

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [name for name, _, _ in RADAR_AXES],
        fontsize=6.5, color="#444", fontweight="semibold",
    )

    # Radial grid
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", ""], fontsize=5, color="#aaa")
    ax.spines["polar"].set_visible(False)
    ax.grid(color="#D5DBDB", linewidth=0.5, alpha=0.6)

    # Annotate raw "100%" value at outer edge of each axis
    for i, (name, mk, higher) in enumerate(RADAR_AXES):
        raw = raw_display[mk]
        if raw >= 10:
            txt = f"{raw:.0f}"
        elif raw >= 1:
            txt = f"{raw:.1f}"
        else:
            txt = f"{raw:.2f}"
        # Place just outside the outer ring
        ax.text(
            angles[i], 1.18, txt,
            ha="center", va="center", fontsize=5.5, color="#888",
            fontweight="bold",
        )

    ax.set_title(
        "(b) Performance Profile", fontsize=8.5, pad=18,
        fontweight="semibold", color=C_PRIMARY,
    )

    # Method legend
    method_handles = [
        Line2D(
            [0], [0], color=METHOD_COLORS[m], linewidth=1.5,
            marker="o", markersize=3.5, markerfacecolor=METHOD_COLORS[m],
            markeredgecolor="white", markeredgewidth=0.3, label=m,
        )
        for m in METHODS
    ]
    ax.legend(
        handles=method_handles, loc="lower right",
        bbox_to_anchor=(1.15, -0.08),
        fontsize=6, frameon=True, framealpha=0.92,
        edgecolor="#CFD8DC", facecolor="white",
        handletextpad=0.3, borderpad=0.3, labelspacing=0.2,
    )


def draw_l2_panel(ax, data):
    """Panel (c): L2 throughput per-dataset dumbbell — the smoking gun."""
    style_ax(ax)

    n_ds = len(DATASETS)
    y_pos = np.arange(n_ds)[::-1]

    for d_idx, ds in enumerate(DATASETS):
        vals = {m: data[(m, ds)]["l2_tp"] for m in METHODS}
        y = y_pos[d_idx]

        v_min, v_max = min(vals.values()), max(vals.values())
        ax.plot(
            [v_min, v_max], [y, y], color="#D0D0D0", linewidth=2.5,
            solid_capstyle="round", zorder=1,
        )
        for method in METHODS:
            ax.scatter(
                vals[method], y, s=32, c=METHOD_COLORS[method],
                edgecolor="white", linewidth=0.3, zorder=3,
            )

    # Geomean summary row
    y_sep = -0.35
    y_gm = -0.85
    ax.axhline(y_sep, color=C_GRID, linewidth=0.5, zorder=0)

    gm_vals = {}
    for method in METHODS:
        all_v = np.array([data[(method, ds)]["l2_tp"] for ds in DATASETS])
        # Use arithmetic mean for L2 TP (percentages, some near zero)
        gm_vals[method] = float(np.mean(all_v))

    gm_min, gm_max = min(gm_vals.values()), max(gm_vals.values())
    ax.plot(
        [gm_min, gm_max], [y_gm, y_gm], color="#D0D0D0", linewidth=2.5,
        solid_capstyle="round", zorder=1,
    )
    for method in METHODS:
        ax.scatter(
            gm_vals[method], y_gm, marker="D", s=28,
            c=METHOD_COLORS[method], edgecolor="white",
            linewidth=0.3, zorder=3,
        )

    # Y-axis
    y_ticks = list(y_pos) + [y_gm]
    y_labels = [DATASET_SHORT[ds] for ds in DATASETS] + ["Avg"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_fontstyle("italic")
    ax.get_yticklabels()[-1].set_color("#666")

    ax.set_xlabel("% of peak", fontsize=7.5, labelpad=2)
    ax.set_title(
        "(c) L2 Throughput", fontsize=8.5, pad=4,
        fontweight="semibold", color=C_PRIMARY, loc="left",
    )
    ax.grid(True, axis="x", alpha=0.15, linestyle="-", linewidth=0.4, color=C_GRID)
    ax.set_xlim(left=0)
    ax.set_ylim(-1.4, n_ds - 0.5)

    # "lower is better" note
    ax.text(
        0.97, 0.97, r"$\leftarrow$ lower = less congestion",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=5.5, style="italic", color="#888",
    )


def save_outputs(fig):
    for base in [PRIMARY_OUT_BASE, SECONDARY_OUT_BASE]:
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03,
        )
        fig.savefig(
            base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03, dpi=300,
        )
        print(f"Wrote {base.with_suffix('.pdf')}")


def main() -> None:
    apply_paper_style(font_size=9.0, legend_size=7.5, tick_label_size=8.0)

    print("=== Loading NCU data ===")
    data = load_all_data()

    # Summary
    print(f"\n=== Metric averages across {len(DATASETS)} datasets ===")
    for mk_label, mk in [
        ("Compute TP", "compute_tp"), ("IPC", "ipc"),
        ("L1 Hit %", "l1_hit"), ("L2 TP %", "l2_tp"),
        ("DRAM TP %", "dram_tp"), ("Warp Lat", "warp_latency"),
        ("Occupancy", "occupancy"),
    ]:
        vals = {m: np.mean([data[(m, ds)][mk] for ds in DATASETS]) for m in METHODS}
        print(
            f"  {mk_label:14s}: "
            + "  ".join(f"{m}={vals[m]:6.1f}" for m in METHODS)
        )

    # --- Figure: stall bars | radar | L2 dot ---
    fig = plt.figure(figsize=(7.16, 3.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.1, 1.15, 0.85],
        left=0.06, right=0.98, top=0.92, bottom=0.10,
        wspace=0.42,
    )

    ax_stall = fig.add_subplot(gs[0, 0])
    ax_radar = fig.add_subplot(gs[0, 1], polar=True)
    ax_l2 = fig.add_subplot(gs[0, 2])

    draw_stall_panel(ax_stall, data)
    draw_radar_panel(ax_radar, data)
    draw_l2_panel(ax_l2, data)

    save_outputs(fig)
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
