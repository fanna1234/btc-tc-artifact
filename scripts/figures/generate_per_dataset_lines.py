#!/usr/bin/env python3
"""Generate SC26 per-dataset comparison as a line chart (figure* two-row).

Design — tiered color scheme:
  - BTC-TC:  bold red line + circle markers  (hero)
  - ToT:     orange dashed line + triangle   (main TC competitor)
  - Polak:   teal solid line + square        (strong on large sparse)
  - TRUST:   purple solid line + diamond     (strong on large sparse)
  - Others:  thin gray lines                 (background band)
  - Two rows of 18 datasets, sorted by BTC-TC kernel time

Output:
  - sc26/paper/figures/fig_per_dataset_lines.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper37_baseline_data import (
    METHODS, COLORS as BASELINE_COLORS, REPO_ROOT,
    load_paper_datasets, load_ok_frames, dataset_display_name,
)
from paper_plot_style import apply_paper_style

OUT_PDF = REPO_ROOT / "results" / "figures" / "fig_per_dataset_lines.pdf"
OUT_PNG = OUT_PDF.with_suffix(".png")
# Also write to the canonical name referenced by the .tex file
OUT_COMPAT = REPO_ROOT / "results" / "figures" / "fig_per_dataset_bars.pdf"

# Use the same colors as Fig.1 (teaser) from BASELINE_COLORS
C_BASELINE_GRAY = "#B0BEC5"
C_PRIMARY = "#2C3E50"

# Methods to highlight — colors from teaser palette, markers differentiate
HERO_METHODS = {
    "BTC-TC (Lite)": dict(color=BASELINE_COLORS["BTC-TC (Lite)"], lw=2.0, ls="-", marker="o", ms=14, z=8, alpha=1.0),
    "ToT":           dict(color=BASELINE_COLORS["ToT"],            lw=1.5, ls="--", marker="^", ms=12, z=6, alpha=0.90),
    "Polak":         dict(color=BASELINE_COLORS["Polak"],          lw=1.2, ls="-", marker="s", ms=10, z=5, alpha=0.85),
    "TRUST":         dict(color=BASELINE_COLORS["TRUST"],          lw=1.2, ls="-", marker="D", ms=10, z=4, alpha=0.85),
}


def main() -> None:
    apply_paper_style(font_size=8.8, legend_size=7.5, tick_label_size=7.0)

    datasets = load_paper_datasets()
    frames = load_ok_frames()
    n_methods = len(METHODS)
    n_ds = len(datasets)

    method_labels = [label for label, _ in METHODS]
    kernel_times = np.full((n_ds, n_methods), np.nan)
    for j, (label, _) in enumerate(METHODS):
        df = frames[label]
        if df.empty or "Dataset" not in df.columns:
            continue
        for i, ds in enumerate(datasets):
            row = df[df["Dataset"] == ds]
            if not row.empty:
                kernel_times[i, j] = float(row.iloc[0]["Kernel_ms"])

    # Sort by BTC-TC kernel time
    btc_col = 0
    btc_times = kernel_times[:, btc_col].copy()
    btc_times[~np.isfinite(btc_times)] = 1e9
    order = np.argsort(btc_times)
    datasets_sorted = [datasets[i] for i in order]
    kernel_sorted = kernel_times[order]

    mid = (n_ds + 1) // 2
    halves = [
        (datasets_sorted[:mid], kernel_sorted[:mid]),
        (datasets_sorted[mid:], kernel_sorted[mid:]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 3.7), sharex=False)

    for ax_idx, (ds_names, kt_matrix) in enumerate(halves):
        ax = axes[ax_idx]
        n = len(ds_names)
        x = np.arange(n)

        # Layer 1: Background baselines (thin gray)
        for j in range(n_methods):
            label = method_labels[j]
            if label in HERO_METHODS:
                continue
            vals = kt_matrix[:, j].copy()
            valid = np.isfinite(vals) & (vals > 0)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], vals[valid],
                    color=C_BASELINE_GRAY, linewidth=0.7, alpha=0.35,
                    zorder=2, solid_capstyle="round")

        # Layer 2: Highlighted methods (drawn in reverse priority so BTC on top)
        draw_order = ["TRUST", "Polak", "ToT", "BTC-TC (Lite)"]
        for label in draw_order:
            j = method_labels.index(label)
            props = HERO_METHODS[label]
            vals = kt_matrix[:, j].copy()
            valid = np.isfinite(vals) & (vals > 0)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], vals[valid],
                    color=props["color"], linewidth=props["lw"],
                    linestyle=props["ls"], alpha=props["alpha"],
                    zorder=props["z"], solid_capstyle="round")
            ax.scatter(x[valid], vals[valid],
                       color=props["color"], s=props["ms"],
                       marker=props["marker"], zorder=props["z"] + 1,
                       edgecolor="white", linewidth=0.3,
                       alpha=props["alpha"])

        # Axes styling
        ax.set_yscale("log")
        ax.set_xlim(-0.5, n - 0.5)

        # Clip y-range: use top-4 methods to set upper bound, avoid gray outliers stretching axis
        hero_cols = [method_labels.index(m) for m in HERO_METHODS if m in method_labels]
        hero_vals = kt_matrix[:, hero_cols]
        hero_valid = hero_vals[np.isfinite(hero_vals) & (hero_vals > 0)]
        all_valid = kt_matrix[np.isfinite(kt_matrix) & (kt_matrix > 0)]
        if len(all_valid) > 0 and len(hero_valid) > 0:
            ax.set_ylim(all_valid.min() * 0.4, hero_valid.max() * 8.0)

        ax.set_xticks(x)
        ds_display = [dataset_display_name(d) for d in ds_names]
        ax.set_xticklabels(ds_display, rotation=55, fontsize=5.6,
                           ha="right", va="top", rotation_mode="anchor",
                           linespacing=0.85)
        ax.tick_params(axis="x", length=0, pad=1)
        ax.tick_params(axis="y", labelsize=7.4, length=2.5, width=0.5)

        ax.set_ylabel("Kernel time (ms)", fontsize=7.5, labelpad=2)
        ax.grid(True, axis="y", alpha=0.20, linestyle="--", linewidth=0.35, zorder=0)
        ax.grid(True, axis="x", alpha=0.08, linestyle="-", linewidth=0.3, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_color("#D1D5DB")
        ax.spines["bottom"].set_linewidth(0.5)

    # Legend — 5 entries
    legend_handles = []
    for label, short in [("BTC-TC (Lite)", "BTC-TC"), ("ToT", "ToT"),
                         ("Polak", "Polak"), ("TRUST", "TRUST")]:
        p = HERO_METHODS[label]
        legend_handles.append(
            mlines.Line2D([], [], color=p["color"], linewidth=p["lw"],
                          linestyle=p["ls"], marker=p["marker"],
                          markersize=4, markeredgecolor="white",
                          markeredgewidth=0.3, alpha=p["alpha"], label=short)
        )
    legend_handles.append(
        mlines.Line2D([], [], color=C_BASELINE_GRAY, linewidth=0.7,
                      alpha=0.45, label="Others (9)")
    )

    axes[0].legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        fontsize=6.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="#D1D5DB",
        facecolor="white",
        handlelength=1.3,
        handletextpad=0.25,
        columnspacing=0.7,
        borderpad=0.2,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0], pad=0.3, h_pad=0.5)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(OUT_COMPAT, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_COMPAT} (compat)")
    print(f"{n_ds} datasets × {n_methods} methods")


if __name__ == "__main__":
    main()
