#!/usr/bin/env python3
"""Generate a combined tau-sensitivity + E2E-breakdown figure for SC26.

Layout:
  (a) 16×128 τ sensitivity
  (b) 16×32 τ sensitivity
  (c) Post-clean E2E time breakdown

Output:
  sc26/paper/figures/tau_e2e_combined.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_plot_style import apply_paper_style

apply_paper_style(font_size=9.2, legend_size=8.0, tick_label_size=7.8)


BASE = Path(__file__).resolve().parent.parent.parent
TAU128_CSV = BASE / "results" / "tau_sweep" / "tau_sweep_128_clean.csv"
TAU32_CSV = BASE / "results" / "tau_sweep" / "tau_sweep_32_clean.csv"
BREAKDOWN_CSV = BASE / "results" / "e2e_breakdown" / "breakdown.csv"
OUT_PDF = BASE / "results" / "figures" / "tau_e2e_combined.pdf"

DEFAULT_TAU_128 = 512
DEFAULT_TAU_32 = 64

DATASET_ORDER = [
    "wiki-Vote",
    "g7jac140sc",
    "consph",
    "cant",
    "pwtk",
    "F1",
    "eu-2005",
    "Ga41As41H72",
]

NAME_MAP = {
    "wiki-Vote": "wikiV",
    "g7jac140sc": "g7jac",
    "consph": "consph",
    "cant": "cant",
    "pwtk": "pwtk",
    "F1": "F1",
    "eu-2005": "eu-05",
    "Ga41As41H72": "Ga41As",
}

TAU_COLORS = {
    "wikiV": "#e67e22",
    "g7jac": "#27ae60",
    "consph": "#3498db",
    "cant": "#e74c3c",
    "pwtk": "#8e44ad",
    "F1": "#c0392b",
    "eu-05": "#2980b9",
    "Ga41As": "#16a085",
}

TAU_MARKERS = {
    "wikiV": "^",
    "g7jac": "D",
    "consph": "s",
    "cant": "*",
    "pwtk": "X",
    "F1": "o",
    "eu-05": "v",
    "Ga41As": "P",
}

BREAKDOWN_COLORS = {
    "Convert": "#3498db",
    "Kernel": "#C0392B",
    "Post": "#95a5a6",
}

PANEL_TITLE_KW = dict(fontsize=8, fontweight="semibold", pad=2.5)
TAU_XTICKS_128 = [64, 256, 512, 1024, 1536, 2048]
TAU_XTICKS_32 = [8, 32, 64, 128, 256, 512]
BREAKDOWN_XTICKS = [0.5, 1, 2, 5, 10, 20, 40]
MIN_LABEL_PERCENT = 20.0
IN_BAR_LABEL_FONTSIZE = 5.4
DATASET_LABEL_FONTSIZE = 6.2
MIN_VISIBLE_SEGMENT_PX = 18.0
LABEL_PAD_PX = 2.5


def _fmt_tick(value: float, _pos: float | None = None) -> str:
    if value >= 1:
        return f"{int(value)}" if float(value).is_integer() else f"{value:g}"
    return f"{value:g}"


def _fmt_tau_tick(value: float, _pos: float | None = None) -> str:
    if value >= 1000:
        return f"{value / 1024.0:g}k"
    return _fmt_tick(value, _pos)


def load_tau_data(csv_path: Path, default_tau: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "correct" in df.columns:
        correct_mask = df["correct"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[correct_mask].copy()

    df = df[df["tau"] > 0].copy()
    ref = (
        df.loc[df["tau"] == default_tau, ["dataset", "kernel_ms"]]
        .drop_duplicates(subset=["dataset"])
        .rename(columns={"kernel_ms": "default_kernel_ms"})
    )
    df = df.merge(ref, on="dataset", how="left", validate="many_to_one")

    missing = df.loc[df["default_kernel_ms"].isna(), "dataset"].unique().tolist()
    if missing:
        raise ValueError(f"Missing default τ={default_tau} reference for datasets: {missing}")

    df["normalized_kernel_ms"] = df["kernel_ms"] / df["default_kernel_ms"]
    return df


def plot_tau_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    default_tau: int,
    title: str,
    xticks: list[int],
) -> tuple[list, list]:
    handles: list = []
    labels: list[str] = []

    for dataset in DATASET_ORDER:
        subset = df[df["dataset"] == dataset].sort_values("tau")
        if subset.empty:
            continue

        short = NAME_MAP[dataset]
        marker = TAU_MARKERS[short]
        markersize = 4.3 if marker == "*" else 3.8
        (line,) = ax.plot(
            subset["tau"],
            subset["normalized_kernel_ms"],
            color=TAU_COLORS[short],
            marker=marker,
            markersize=markersize,
            markerfacecolor=TAU_COLORS[short],
            markeredgecolor="white",
            markeredgewidth=0.35,
            linewidth=1.35,
            alpha=0.98,
            solid_capstyle="round",
            label=short,
            zorder=3,
        )
        handles.append(line)
        labels.append(short)

    ax.axvline(default_tau, color="0.60", linestyle="--", linewidth=0.85, alpha=0.9, zorder=1)
    ax.axhline(1.0, color="0.60", linestyle=":", linewidth=0.8, alpha=0.9, zorder=1)

    ax.set_xscale("symlog", linthresh=10)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tau_tick))
    ax.tick_params(axis="x", labelsize=5.6, pad=1.0, rotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.tick_params(axis="y", labelsize=7, pad=1.2)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.28, zorder=0)
    ax.grid(False, axis="x")
    ax.margins(x=0.04, y=0.08)
    ax.set_title(title, **PANEL_TITLE_KW)
    ax.set_xlabel("Threshold τ (set bits)", labelpad=3.0)

    return handles, labels



def _padded_bbox(bbox: Bbox, pad_x: float, pad_y: float = 0.0) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 - pad_x,
        bbox.y0 - pad_y,
        bbox.x1 + pad_x,
        bbox.y1 + pad_y,
    )



def _segment_visible_bounds(ax: plt.Axes, left: float, width: float) -> tuple[float, float] | None:
    x_min, x_max = ax.get_xlim()
    right = left + width
    visible_left = max(left, x_min)
    visible_right = min(right, x_max)
    if visible_right <= visible_left:
        return None
    return visible_left, visible_right



def _segment_display_center_x(ax: plt.Axes, x0: float, x1: float, y0: float) -> float:
    left_px, y_px = ax.transData.transform((x0, y0))
    right_px, _ = ax.transData.transform((x1, y0))
    center_px = 0.5 * (left_px + right_px)
    return ax.transData.inverted().transform((center_px, y_px))[0]



def _make_text_probe(ax: plt.Axes, x: float, y0: float, label: str, color: str):
    return ax.text(
        x,
        y0,
        label,
        ha="center",
        va="center",
        fontsize=IN_BAR_LABEL_FONTSIZE,
        color=color,
        fontweight="bold",
        zorder=5,
        clip_on=True,
        alpha=0.0,
    )



def _add_label_if_it_fits(
    ax: plt.Axes,
    renderer,
    legend_bbox,
    *,
    left: float,
    width: float,
    y0: float,
    pct: float,
    color: str,
) -> None:
    if pct < MIN_LABEL_PERCENT or width <= 0:
        return

    bounds = _segment_visible_bounds(ax, left, width)
    if bounds is None:
        return
    visible_left, visible_right = bounds

    x_center = _segment_display_center_x(ax, visible_left, visible_right, y0)
    label = f"{pct:.0f}%"

    probe = _make_text_probe(ax, x_center, y0, label, color)
    text_bbox = probe.get_window_extent(renderer=renderer)
    probe.remove()

    seg_left_px, _ = ax.transData.transform((visible_left, y0))
    seg_right_px, _ = ax.transData.transform((visible_right, y0))
    segment_width_px = seg_right_px - seg_left_px
    required_width_px = max(MIN_VISIBLE_SEGMENT_PX, text_bbox.width + 2 * LABEL_PAD_PX)
    if segment_width_px < required_width_px:
        return

    padded_bbox = _padded_bbox(text_bbox, LABEL_PAD_PX, 1.0)
    axes_bbox = ax.get_window_extent(renderer=renderer)
    if not axes_bbox.contains(padded_bbox.x0, padded_bbox.y0) or not axes_bbox.contains(
        padded_bbox.x1, padded_bbox.y1
    ):
        return
    if legend_bbox is not None and padded_bbox.overlaps(legend_bbox):
        return

    ax.text(
        x_center,
        y0,
        label,
        ha="center",
        va="center",
        fontsize=IN_BAR_LABEL_FONTSIZE,
        color=color,
        fontweight="bold",
        zorder=5,
        clip_on=True,
    )



def plot_breakdown_panel(ax: plt.Axes, df: pd.DataFrame):
    """Transposed vertical stacked-bar E2E breakdown (datasets on x-axis)."""
    breakdown = df.copy()
    breakdown["PostClean_ms"] = breakdown["Convert_ms"] + breakdown["Kernel_ms"] + breakdown["Post_ms"]
    breakdown = breakdown.sort_values("PostClean_ms", ascending=True).reset_index(drop=True)

    n = len(breakdown)
    x = np.arange(n)
    convert = breakdown["Convert_ms"].to_numpy()
    kernel = breakdown["Kernel_ms"].to_numpy()
    post = breakdown["Post_ms"].to_numpy()
    postclean = breakdown["PostClean_ms"].to_numpy()

    bar_w = 0.55
    ax.bar(x, convert, width=bar_w, color=BREAKDOWN_COLORS["Convert"],
           label="Convert", zorder=3)
    ax.bar(x, kernel, width=bar_w, bottom=convert,
           color=BREAKDOWN_COLORS["Kernel"], label="Kernel", zorder=3)
    ax.bar(x, post, width=bar_w, bottom=convert + kernel,
           color=BREAKDOWN_COLORS["Post"], label="Post", zorder=3)

    # X-axis: rotated dataset labels
    ax.set_xticks(x)
    ax.set_xticklabels(breakdown["Dataset"].tolist(), rotation=75,
                       fontsize=5.0, ha="right", va="top",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1)
    ax.tick_params(axis="y", labelsize=6.2, pad=1.0)
    ax.set_ylabel("PostClean time (ms)", labelpad=1.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(BREAKDOWN_XTICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))

    ymin = max(postclean.min() * 0.70, 0.20)
    ymax = postclean.max() * 1.25
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.28, zorder=0)
    ax.grid(False, axis="x")
    ax.set_title("(c) E2E breakdown", **PANEL_TITLE_KW)

    legend = ax.legend(
        loc="upper left",
        fontsize=6.8,
        framealpha=0.92,
        handlelength=1.0,
        borderpad=0.25,
        labelspacing=0.2,
        handletextpad=0.35,
    )

    return breakdown, convert, kernel, post, legend



def main() -> None:
    tau128 = load_tau_data(TAU128_CSV, DEFAULT_TAU_128)
    tau32 = load_tau_data(TAU32_CSV, DEFAULT_TAU_32)

    fig, ax_tau = plt.subplots(1, 1, figsize=(3.45, 2.55))

    # --- Merged tau panel: solid=16×128, dashed=16×32 ---
    from matplotlib.lines import Line2D

    ds_handles = []
    ds_labels = []

    for dataset in DATASET_ORDER:
        short = NAME_MAP[dataset]
        color = TAU_COLORS[short]
        marker = TAU_MARKERS[short]
        markersize = 4.3 if marker == "*" else 3.8

        # 16×128 (solid)
        sub128 = tau128[tau128["dataset"] == dataset].sort_values("tau")
        if not sub128.empty:
            ax_tau.plot(
                sub128["tau"], sub128["normalized_kernel_ms"],
                color=color, marker=marker, markersize=markersize,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.35, linewidth=1.3, alpha=0.95,
                linestyle="-", solid_capstyle="round", zorder=3,
            )

        # 16×32 (dashed)
        sub32 = tau32[tau32["dataset"] == dataset].sort_values("tau")
        if not sub32.empty:
            ax_tau.plot(
                sub32["tau"], sub32["normalized_kernel_ms"],
                color=color, marker=marker, markersize=markersize,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=0.7, linewidth=1.3, alpha=0.95,
                linestyle="--", dash_capstyle="round", zorder=3,
            )

        # One handle per dataset (solid line for legend)
        ds_handles.append(Line2D([0], [0], color=color, marker=marker,
                                 markersize=3.5, markerfacecolor=color,
                                 markeredgecolor="white", markeredgewidth=0.3,
                                 linewidth=1.0, linestyle="-"))
        ds_labels.append(short)

    # Default τ lines
    ax_tau.axvline(DEFAULT_TAU_128, color="#2980B9", linestyle="--",
                   linewidth=0.8, alpha=0.7, zorder=1)
    ax_tau.axvline(DEFAULT_TAU_32, color="#C0392B", linestyle="--",
                   linewidth=0.8, alpha=0.7, zorder=1)
    ax_tau.axhline(1.0, color="0.60", linestyle=":", linewidth=0.8, alpha=0.9, zorder=1)

    # Label the default τ lines
    tau_ymin = min(tau128["normalized_kernel_ms"].min(), tau32["normalized_kernel_ms"].min())
    tau_ymax = max(tau128["normalized_kernel_ms"].max(), tau32["normalized_kernel_ms"].max())
    yrange = tau_ymax - tau_ymin
    ax_tau.set_ylim(tau_ymin - 0.06 * yrange, tau_ymax + 0.08 * yrange)
    label_y = tau_ymax + 0.04 * yrange
    ax_tau.text(DEFAULT_TAU_128, label_y, "τ₁₂₈", fontsize=5.5, color="#2980B9",
                ha="center", va="bottom", fontweight="bold")
    ax_tau.text(DEFAULT_TAU_32, label_y, "τ₃₂", fontsize=5.5, color="#C0392B",
                ha="center", va="bottom", fontweight="bold")

    # Merged x-axis: union of ticks, symlog
    merged_ticks = sorted(set(TAU_XTICKS_128) | set(TAU_XTICKS_32))
    # Keep only a readable subset to avoid crowding
    display_ticks = [8, 32, 64, 256, 512, 1024, 2048]
    ax_tau.set_xscale("symlog", linthresh=10)
    ax_tau.xaxis.set_major_locator(FixedLocator(display_ticks))
    ax_tau.xaxis.set_major_formatter(FuncFormatter(_fmt_tau_tick))
    ax_tau.tick_params(axis="x", labelsize=5.6, pad=1.0, rotation=40)
    for label in ax_tau.get_xticklabels():
        label.set_horizontalalignment("right")
    ax_tau.tick_params(axis="y", labelsize=7, pad=1.2)
    ax_tau.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.28, zorder=0)
    ax_tau.grid(False, axis="x")
    ax_tau.margins(x=0.04, y=0.08)
    ax_tau.set_title("Dispatch threshold τ sensitivity", **PANEL_TITLE_KW)
    ax_tau.set_xlabel("Threshold τ (set bits)", labelpad=3.0)
    ax_tau.set_ylabel("Normalized kernel time", labelpad=1.5)

    # Legend: block-size line styles (with marker fill distinction) + dataset colors
    style_handles = [
        Line2D([0], [0], color="0.35", linewidth=1.5, linestyle="-",
               marker="o", markersize=4, markerfacecolor="0.35",
               markeredgecolor="0.35", label="16×128"),
        Line2D([0], [0], color="0.35", linewidth=1.5, linestyle="--",
               marker="o", markersize=4, markerfacecolor="white",
               markeredgecolor="0.35", markeredgewidth=0.8, label="16×32"),
    ]
    all_handles = style_handles + ds_handles
    all_labels = ["16×128", "16×32"] + ds_labels
    ax_tau.legend(
        all_handles, all_labels,
        loc="upper center", ncol=5,
        fontsize=5.2, columnspacing=0.5,
        handlelength=1.8, handletextpad=0.2,
        labelspacing=0.15, borderpad=0.2,
        framealpha=0.92, facecolor="white",
    )

    fig.tight_layout(pad=0.4)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    out_png = OUT_PDF.with_suffix(".png")
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
