#!/usr/bin/env python3
"""
Generate ablation figures — compact summary style.

Panel (a): Hybrid speedup distribution — box plot + strip plot.
Panel (b): Block size policy — scatter plot of 16x32 vs 16x128 kernel time.
Panel (c): MMA shape choice — box plot of kernel-time ratios (8x128, 16x256 vs 16x128).
Panel (d): Vertex reordering effect — scatter of block-count change vs kernel-time change.

Outputs:
  - paper/figures/fig_ablation.{pdf,png}
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import apply_paper_style, format_multiplier

# Professional muted palette
C_PRIMARY = "#2C3E50"    # dark blue-gray
C_ACCENT = "#C0392B"     # deep coral
C_SECONDARY = "#2980B9"  # muted blue
C_GRID = "#D5DBDB"


def load_paper_datasets(repo_root: Path) -> list[str]:
    p = repo_root / "data" / "paper_datasets.txt"
    names: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names


def geomean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def save_fig(fig, *out_bases: Path):
    for out_base in out_bases:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
        fig.savefig(str(out_base.with_suffix(".png")), bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_ax(ax):
    """Apply consistent axis styling."""
    for spine in ax.spines.values():
        spine.set_color("#78909C")
        spine.set_linewidth(0.6)
    ax.tick_params(colors="#546E7A", width=0.6)
    ax.set_facecolor("white")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    paper_fig_dir = repo_root / "results" / "figures"
    sc26_fig_dir = repo_root / "results" / "figures"
    paper_ds = set(load_paper_datasets(repo_root))

    panel_title_fs = 10.0

    # --- Load hybrid ablation data ---
    ab_dir = repo_root / "results" / "ablation" / "csv"
    pure128 = pd.read_csv(ab_dir / "V3_16x128_PureTC.csv")
    hyb128 = pd.read_csv(ab_dir / "V5_16x128_Hybrid.csv")
    pure32 = pd.read_csv(ab_dir / "V3_16x32_PureTC.csv")
    hyb32 = pd.read_csv(ab_dir / "V6_16x32_Hybrid.csv")

    m128 = pure128.merge(hyb128, on="Dataset", suffixes=("_pure", "_hyb"))
    m32 = pure32.merge(hyb32, on="Dataset", suffixes=("_pure", "_hyb"))
    m128 = m128[m128["Dataset"].isin(paper_ds)].copy()
    m32 = m32[m32["Dataset"].isin(paper_ds)].copy()

    sp128 = (m128["Kernel_ms_pure"] / m128["Kernel_ms_hyb"]).to_numpy()
    sp32 = (m32["Kernel_ms_pure"] / m32["Kernel_ms_hyb"]).to_numpy()

    # --- Load block-size data ---
    csv_dir = repo_root / "results" / "pro6000" / "csv"
    dfl = pd.read_csv(csv_dir / "BTC_Lite.csv")
    df32 = pd.read_csv(csv_dir / "BTC_16x32_Adaptive.csv")
    df128 = pd.read_csv(csv_dir / "BTC_16x128_Adaptive.csv")
    mb = (
        dfl[["Dataset", "Triangles", "Kernel_ms"]]
        .merge(df32[["Dataset", "Kernel_ms"]], on="Dataset", suffixes=("_lite", "_32"))
        .merge(df128[["Dataset", "Kernel_ms"]], on="Dataset")
        .rename(columns={"Kernel_ms": "Kernel_ms_128"})
    )
    mb = mb[mb["Dataset"].isin(paper_ds)]

    # --- Load MMA shape data (blocksize_bench) ---
    shape_csv = repo_root / "results" / "blocksize_bench" / "blocksize_bench.csv"
    shape_blocks = ["8x128", "16x128", "16x256"]
    sdf = pd.read_csv(shape_csv)
    sdf = sdf[sdf["status"] == "PASS"].copy()
    counts = sdf.groupby("dataset")["block"].nunique()
    complete = counts[counts >= len(shape_blocks)].index.tolist()
    sdf = sdf[sdf["dataset"].isin(complete)].copy()
    wide_shape = sdf.pivot_table(index="dataset", columns="block", values="time_ms", aggfunc="mean")
    for b in shape_blocks:
        wide_shape[b] = pd.to_numeric(wide_shape[b], errors="coerce")
    wide_shape = wide_shape.dropna(subset=shape_blocks).reset_index()

    ratio_8 = (wide_shape["8x128"] / wide_shape["16x128"]).to_numpy()
    ratio_256 = (wide_shape["16x256"] / wide_shape["16x128"]).to_numpy()

    # --- Load reorder comparison data ---
    reorder_dir = repo_root / "results" / "reorder_compare"
    m0_reorder = pd.read_csv(reorder_dir / "mode0_no_reorder.csv")
    m8_reorder = pd.read_csv(reorder_dir / "mode8_gpu_hashorder.csv")
    mr = m0_reorder[["Dataset", "Kernel_ms", "Blocks"]].merge(
        m8_reorder[["Dataset", "Kernel_ms", "Blocks"]], on="Dataset", suffixes=("_m0", "_m8"))
    mr = mr[mr["Dataset"].isin(paper_ds)].copy()
    mr["block_ratio"] = mr["Blocks_m8"] / mr["Blocks_m0"]
    mr["kernel_ratio"] = mr["Kernel_ms_m8"] / mr["Kernel_ms_m0"]

    # ==================================================================
    # Unified lollipop design: all panels show sorted per-dataset ratios
    # with stems from reference line (1.0) — consistent visual language.
    # ==================================================================

    apply_paper_style(font_size=9.0, legend_size=7.5, tick_label_size=7.5)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.6),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1]})

    STEM_LW = 1.1
    DOT_S = 14
    REF_KW = dict(color="#78909C", linestyle="--", linewidth=0.7, alpha=0.6, zorder=1)
    GRID_KW = dict(axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color=C_GRID)
    TITLE_KW = dict(fontsize=8.5, pad=4, color=C_PRIMARY, fontweight="semibold")

    def draw_lollipop(ax, vals, ref=1.0, colors=None, default_color=C_SECONDARY):
        """Sorted lollipop: stems from ref to each value."""
        order = np.argsort(vals)[::-1]  # descending
        sorted_v = vals[order]
        n = len(sorted_v)
        x = np.arange(n)
        if colors is None:
            c = [default_color] * n
        else:
            c = [colors[i] for i in order]
        ax.vlines(x, ref, sorted_v, linewidth=STEM_LW, colors=c, alpha=0.7, zorder=2)
        ax.scatter(x, sorted_v, s=DOT_S, c=c, alpha=0.85,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(ref, **REF_KW)
        ax.set_xlim(-1, n)
        ax.set_xticks([0, n - 1])
        ax.set_xticklabels(["1", str(n)], fontsize=6.5)
        ax.set_xlabel("Dataset rank", fontsize=7, labelpad=2)
        return sorted_v, order

    # ==================================================================
    # (a) Hybrid Execution: speedup = Pure / Hybrid (>1 = hybrid helps)
    # ==================================================================
    ax = axes[0]
    style_ax(ax)

    per_ds_colors = np.where(sp128 > 1.0, C_SECONDARY, C_ACCENT)
    sorted_v, _ = draw_lollipop(ax, sp128, ref=1.0, colors=per_ds_colors)
    gm = geomean(sp128)
    # GM marker
    gm_x = np.searchsorted(-sorted_v, -gm)  # position in sorted order
    ax.scatter(gm_x, gm, marker="D", s=38, color="white",
               edgecolor=C_PRIMARY, linewidth=1.2, zorder=6)
    ax.annotate(f"GM {format_multiplier(gm)}", xy=(gm_x, gm),
                xytext=(8, 0), textcoords="offset points",
                fontsize=6.5, color=C_PRIMARY, va="center",
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.85))
    n_win = int((sp128 > 1.0).sum())
    ax.text(0.97, 0.97, f"{n_win}/{len(sp128)} faster",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            color=C_SECONDARY, alpha=0.8)
    ax.set_ylabel("Speedup (Hybrid / Pure)")
    ax.set_title("(a) Hybrid Execution", **TITLE_KW)
    ax.grid(**GRID_KW)

    # ==================================================================
    # (b) Block Size: ratio = time_32 / time_128 (>1 = 128 faster)
    # ==================================================================
    ax = axes[1]
    style_ax(ax)

    x_vals = mb["Kernel_ms_128"].to_numpy(dtype=float)
    y_vals = mb["Kernel_ms_32"].to_numpy(dtype=float)
    lite_vals = mb["Kernel_ms_lite"].to_numpy(dtype=float)
    block_ratio = y_vals / x_vals  # >1 means 16x128 faster
    picked_128 = np.abs(lite_vals - x_vals) < np.abs(lite_vals - y_vals)
    bs_colors = np.where(picked_128, C_SECONDARY, C_ACCENT)

    sorted_v, order = draw_lollipop(ax, block_ratio, ref=1.0, colors=bs_colors)
    gm_b = geomean(block_ratio)
    gm_x = np.searchsorted(-sorted_v, -gm_b)
    ax.scatter(gm_x, gm_b, marker="D", s=38, color="white",
               edgecolor=C_PRIMARY, linewidth=1.2, zorder=6)
    ax.annotate(f"GM {format_multiplier(gm_b)}", xy=(gm_x, gm_b),
                xytext=(8, 0), textcoords="offset points",
                fontsize=6.5, color=C_PRIMARY, va="center",
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.85))
    # Region labels
    ax.text(0.03, 0.97, "16×128 faster", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.5, color=C_SECONDARY, alpha=0.8)
    ax.text(0.97, 0.03, "16×32 faster", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.5, color=C_ACCENT, alpha=0.8)
    # Legend
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SECONDARY,
               markersize=4, label="Lite→128"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ACCENT,
               markersize=4, label="Lite→32"),
    ], loc="center right", fontsize=5.5, frameon=True, framealpha=0.9,
       edgecolor="#CFD8DC", borderpad=0.2, handletextpad=0.1)
    ax.set_ylabel("Time ratio (16×32 / 16×128)")
    ax.set_title("(b) Block Size Selection", **TITLE_KW)
    ax.grid(**GRID_KW)

    # ==================================================================
    # (c) MMA Shape: ratio vs 16×128 (>1 = 16×128 faster)
    # ==================================================================
    ax = axes[2]
    style_ax(ax)

    # Interleave 8x128 and 16x256 ratios as two series on same sorted axis
    # Sort by 16x256 ratio (usually larger)
    order_c = np.argsort(ratio_256)[::-1]
    n_c = len(ratio_256)
    x_c = np.arange(n_c)

    ax.vlines(x_c, 1.0, ratio_256[order_c], linewidth=STEM_LW,
              colors=C_ACCENT, alpha=0.6, zorder=2)
    ax.scatter(x_c, ratio_256[order_c], s=DOT_S, c=C_ACCENT, alpha=0.85,
               edgecolor="white", linewidth=0.3, zorder=3, label="16×256")
    ax.scatter(x_c, ratio_8[order_c], s=DOT_S - 2, c=C_SECONDARY, alpha=0.85,
               edgecolor="white", linewidth=0.3, zorder=4, marker="s", label="8×128")
    ax.axhline(1.0, **REF_KW)
    ax.set_xlim(-1, n_c)
    ax.set_xticks([0, n_c - 1])
    ax.set_xticklabels(["1", str(n_c)], fontsize=6.5)
    ax.set_xlabel("Dataset rank", fontsize=7, labelpad=2)

    gm_256 = geomean(ratio_256)
    gm_8 = geomean(ratio_8)
    ax.axhline(gm_256, color=C_ACCENT, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axhline(gm_8, color=C_SECONDARY, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.text(n_c - 1, gm_256 + 0.02, f"GM {format_multiplier(gm_256)}",
            fontsize=5.5, color=C_ACCENT, ha="right", va="bottom")
    ax.text(n_c - 1, gm_8 + 0.02, f"GM {format_multiplier(gm_8)}",
            fontsize=5.5, color=C_SECONDARY, ha="right", va="bottom")

    n_shape = len(wide_shape)
    win_128 = int(((wide_shape["16x128"] <= wide_shape["8x128"]) &
                   (wide_shape["16x128"] <= wide_shape["16x256"])).sum())
    ax.text(0.03, 0.97, f"16×128 wins {win_128}/{n_shape}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6,
            color=C_PRIMARY, alpha=0.8)
    ax.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.9,
              edgecolor="#CFD8DC", borderpad=0.2, handletextpad=0.1,
              handlelength=0.8)
    ax.set_ylabel("Kernel-time ratio (vs. 16×128)")
    ax.set_title("(c) MMA Shape", **TITLE_KW)
    ax.grid(**GRID_KW)

    # ==================================================================
    # (d) Reorder: speedup = original / reordered (>1 = reorder helps)
    # ==================================================================
    ax = axes[3]
    style_ax(ax)

    reorder_speedup = (mr["Kernel_ms_m0"] / mr["Kernel_ms_m8"]).to_numpy()
    rd_colors = np.where(reorder_speedup > 1.0, C_ACCENT, "#90A4AE")

    sorted_v, _ = draw_lollipop(ax, reorder_speedup, ref=1.0, colors=rd_colors)
    gm_r = geomean(reorder_speedup)
    gm_x = np.searchsorted(-sorted_v, -gm_r)
    ax.scatter(gm_x, gm_r, marker="D", s=38, color="white",
               edgecolor=C_PRIMARY, linewidth=1.2, zorder=6)

    n_helps = int((reorder_speedup > 1.0).sum())
    ax.text(0.03, 0.97, f"{n_helps}/{len(reorder_speedup)} faster",
            transform=ax.transAxes, ha="left", va="top", fontsize=6,
            color=C_ACCENT, alpha=0.8)
    ax.text(0.97, 0.97, f"{len(reorder_speedup)-n_helps}/{len(reorder_speedup)} slower",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            color="#78909C", alpha=0.8)
    ax.set_ylabel("Kernel speedup (orig / reordered)")
    ax.set_title("(d) Reorder Effect", **TITLE_KW)
    ax.grid(**GRID_KW)

    fig.tight_layout(pad=0.5, w_pad=0.9)
    save_fig(fig, paper_fig_dir / "fig_ablation", sc26_fig_dir / "fig_ablation")
    print(f"Wrote {paper_fig_dir / 'fig_ablation.pdf'}")
    print(f"Wrote {sc26_fig_dir / 'fig_ablation.pdf'}")


if __name__ == "__main__":
    main()
