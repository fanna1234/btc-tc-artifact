#!/usr/bin/env python3
"""Generate the teaser scatter figure (full-width, all 13 baselines shown)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper37_baseline_data import COLORS as BASELINE_COLORS, METHODS, REPO_ROOT, load_ok_frames, load_paper_datasets
from paper_plot_style import apply_paper_style

OUT_DIR = REPO_ROOT / "results" / "figures"
SC26_OUT_DIR = REPO_ROOT / "results" / "figures"

# BTC-TC hero styling
C_HERO = BASELINE_COLORS["BTC-TC (Lite)"]
C_HERO_EDGE = "#922B21"

# Distinct markers for each baseline (avoid reuse)
MARKERS = {
    "BTC-TC (Lite)": "o",
    "ToT":           "D",    # diamond
    "TRUST":         "^",    # triangle up
    "Polak":         "s",    # square
    "GroupTC":       "p",    # pentagon
    "Hu":            "P",    # plus (filled)
    "Green":         "*",    # star
    "Tricore":       "v",    # triangle down
    "Bisson":        "X",    # x (filled)
    "Fox":           "d",    # thin diamond
    "LAGraph (CPU)": "<",    # triangle left
    "LAGraph (GPU)": ">",    # triangle right
    "HIndex":        "H",    # hexagon2
}


def save_outputs(fig: plt.Figure, stem: str) -> None:
    for out_dir in (OUT_DIR, SC26_OUT_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = out_dir / stem
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Wrote {out_base.with_suffix('.pdf')}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SC26_OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=9.6, legend_size=6.6, tick_label_size=8.4)

    frames = load_ok_frames()
    paper_ds = set(load_paper_datasets())
    for label in frames:
        if not frames[label].empty:
            frames[label] = frames[label][frames[label]["Dataset"].isin(paper_ds)].copy()

    # Single-column figure for right column of page 1
    fig, ax = plt.subplots(figsize=(3.45, 3.4))

    # --- Pass 1: All baselines (not BTC-TC) ---
    for label, _fname in METHODS:
        if label == "BTC-TC (Lite)":
            continue
        df = frames[label]
        if df.empty:
            continue
        ax.scatter(
            df["Triangles"],
            df["E2E_after_clean_ms"],
            s=20,
            marker=MARKERS[label],
            facecolors=BASELINE_COLORS[label],
            edgecolors="none",
            alpha=0.52,
            zorder=2,
        )

    # --- Pass 2: BTC-TC (hero, most prominent) ---
    df_btc = frames["BTC-TC (Lite)"]
    df_btc_sorted = df_btc.sort_values("Triangles")

    ax.scatter(
        df_btc_sorted["Triangles"],
        df_btc_sorted["E2E_after_clean_ms"],
        s=54,
        marker="o",
        facecolors=C_HERO,
        edgecolors=C_HERO_EDGE,
        linewidth=0.8,
        alpha=0.95,
        zorder=8,
    )

    # --- Legend: BTC-TC first, then all baselines ---
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=C_HERO, markeredgecolor=C_HERO_EDGE,
               markeredgewidth=0.8, markersize=6.2, label="BTC-TC"),
    ]
    for label, _fname in METHODS:
        if label == "BTC-TC (Lite)":
            continue
        legend_handles.append(
            Line2D([0], [0], marker=MARKERS[label], linestyle="None",
                   markerfacecolor=BASELINE_COLORS[label], markeredgecolor="none",
                   markersize=4.5, label=label)
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("End-to-end time (ms)")

    ax.grid(True, which="major", alpha=0.14, linestyle="-", linewidth=0.55, color="#90A4AE")
    ax.grid(True, which="minor", alpha=0.06, linestyle="-", linewidth=0.35, color="#CFD8DC")

    for spine in ax.spines.values():
        spine.set_color("#78909C")
        spine.set_linewidth(0.6)

    ax.tick_params(colors="#546E7A", width=0.6)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.995),
        frameon=True,
        framealpha=0.94,
        edgecolor="#CFD8DC",
        facecolor="white",
        borderpad=0.28,
        handletextpad=0.24,
        borderaxespad=0.15,
        ncol=3,
        columnspacing=0.6,
        labelspacing=0.22,
        fontsize=6.0,
    )

    fig.tight_layout(pad=0.3)

    save_outputs(fig, "teaser")
    plt.close(fig)


if __name__ == "__main__":
    main()
