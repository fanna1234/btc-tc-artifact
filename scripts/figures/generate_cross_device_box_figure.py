#!/usr/bin/env python3
"""Generate cross-device speedup box+strip figure (BTC_Lite vs ToT, 36 datasets).

Output:
  - sc26/paper/figures/fig_cross_device_all.{pdf,png}

Two-panel box+strip layout:
  (a) Kernel Speedup  — 3 boxes (PRO6000, H100, A800)
  (b) E2E Speedup     — 3 boxes (PRO6000, H100, A800)
Each box has jittered individual dataset points overlaid and geomean diamond.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import apply_paper_style, format_multiplier

REPO_ROOT = Path(__file__).resolve().parents[2]
SC26_FIG_DIR = REPO_ROOT / "results" / "figures"
LEGACY_FIG_DIR = REPO_ROOT / "results" / "figures"

DEVICE_SPECS = [
    {
        "slug": "pro6000",
        "label": "RTX PRO\n6000",
        "short": "PRO 6000",
        "csv_dir": REPO_ROOT / "results" / "csv",
    },
    {
        "slug": "h100",
        "label": "H100",
        "short": "H100",
        "csv_dir": REPO_ROOT / "results/h100" / "csv",
    },
    {
        "slug": "a800",
        "label": "A800",
        "short": "A800",
        "csv_dir": REPO_ROOT / "results/a800" / "csv",
    },
]

PAPER_DATASETS_FILE = REPO_ROOT / "data" / "paper_datasets.txt"

# Colors matching ablation figure style
C_KERNEL = "#2980B9"   # muted blue
C_E2E = "#C0392B"      # deep coral
C_PRIMARY = "#2C3E50"  # dark blue-gray
C_AXIS = "#78909C"


def load_paper_datasets() -> list[str]:
    names: list[str] = []
    for line in PAPER_DATASETS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names


def geomean(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float(np.exp(np.log(vals).mean()))


def build_device_df(csv_dir: Path) -> pd.DataFrame:
    btc = pd.read_csv(csv_dir / "BTC_Lite.csv")
    tot = pd.read_csv(csv_dir / "ToT.csv")

    cols = ["Dataset", "Status", "Kernel_ms", "E2E_after_clean_ms"]
    btc = btc[cols].rename(columns={"Status": "S_B", "Kernel_ms": "K_B", "E2E_after_clean_ms": "E_B"})
    tot = tot[cols].rename(columns={"Status": "S_T", "Kernel_ms": "K_T", "E2E_after_clean_ms": "E_T"})

    df = btc.merge(tot, on="Dataset", how="inner")
    df = df[(df["S_B"] == "OK") & (df["S_T"] == "OK")].copy()
    for c in ["K_B", "E_B", "K_T", "E_T"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["K_B", "E_B", "K_T", "E_T"])

    df["Kernel_Speedup"] = df["K_T"] / df["K_B"]
    df["E2E_Speedup"] = df["E_T"] / df["E_B"]

    # Filter to 36 paper datasets
    paper = load_paper_datasets()
    paper_set = set(paper)
    df = df[df["Dataset"].isin(paper_set)].copy()
    return df


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_color(C_AXIS)
        spine.set_linewidth(0.6)
    ax.tick_params(colors="#546E7A", width=0.6)
    ax.set_facecolor("white")


def draw_panel(ax, all_data, positions, color, panel_title, ylabel):
    """Draw one panel with box + strip + geomean diamonds."""
    style_ax(ax)

    # Box plot
    bp = ax.boxplot(
        all_data, positions=positions, widths=0.50, patch_artist=True,
        showfliers=False, zorder=3,
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(color=C_AXIS, linewidth=0.8),
        capprops=dict(color=C_AXIS, linewidth=0.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)

    # Jittered strip points
    rng = np.random.default_rng(42)
    for pos, vals in zip(positions, all_data):
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(pos + jitter, vals, s=10, color=color, alpha=0.50,
                   edgecolor="white", linewidth=0.3, zorder=4)

    # Geomean diamonds
    for pos, vals in zip(positions, all_data):
        gm = geomean(vals)
        ax.scatter(pos, gm, marker="D", s=40, color="white",
                   edgecolor=color, linewidth=1.2, zorder=5)

    # Reference line at 1x
    ax.axhline(1.0, color=C_AXIS, linestyle="--", linewidth=0.7, alpha=0.6)

    # Annotations: geomean value above each box
    for pos, vals in zip(positions, all_data):
        gm = geomean(vals)
        win = int(np.sum(np.array(vals) > 1.0))
        total = len(vals)
        ax.annotate(
            f"GM {format_multiplier(gm)}\n{win}/{total}",
            xy=(pos, gm), xytext=(0, 14),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=7, color="#34495E",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#CFD8DC", alpha=0.90),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([s["label"] for s in DEVICE_SPECS])
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, fontsize=9.5, pad=6, color=C_PRIMARY)
    ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="#D5DBDB")
    ax.grid(False, axis="x")


def main() -> None:
    SC26_FIG_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=9.5, legend_size=8.0, tick_label_size=8.5)

    # Load data for all devices (skip missing)
    kernel_data = []
    e2e_data = []
    valid_specs = []
    for spec in DEVICE_SPECS:
        csv_dir = spec["csv_dir"]
        if not (csv_dir / "BTC_Lite.csv").exists() or not (csv_dir / "ToT.csv").exists():
            print(f"{spec['short']:>12s}: SKIPPED (missing CSV)")
            continue
        df = build_device_df(csv_dir)
        kernel_data.append(df["Kernel_Speedup"].to_numpy())
        e2e_data.append(df["E2E_Speedup"].to_numpy())
        valid_specs.append(spec)
        n = len(df)
        print(f"{spec['short']:>12s}: {n} datasets, "
              f"Kernel GM={geomean(df['Kernel_Speedup']):.2f}x, "
              f"E2E GM={geomean(df['E2E_Speedup']):.2f}x")

    # --- Single merged panel: grouped box + strip ---
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.85))
    style_ax(ax)

    n_dev = len(valid_specs)
    box_w = 0.34
    gap = 0.46           # gap between kernel and E2E boxes within a group
    group_sep = 1.55      # center-to-center between device groups
    rng = np.random.default_rng(42)
    group_centers = []

    for i in range(n_dev):
        cx = i * group_sep
        group_centers.append(cx)
        pos_k = cx - gap / 2
        pos_e = cx + gap / 2

        for j, (pos, vals, color) in enumerate(
            [(pos_k, kernel_data[i], C_KERNEL), (pos_e, e2e_data[i], C_E2E)]
        ):
            bp = ax.boxplot(
                [vals], positions=[pos], widths=box_w, patch_artist=True,
                showfliers=False, zorder=3,
                medianprops=dict(color="white", linewidth=1.6),
                whiskerprops=dict(color=C_AXIS, linewidth=0.8),
                capprops=dict(color=C_AXIS, linewidth=0.8),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.60)
            bp["boxes"][0].set_edgecolor(color)
            bp["boxes"][0].set_linewidth(0.8)

            # Jittered strip
            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(pos + jitter, vals, s=7, color=color, alpha=0.42,
                       edgecolor="white", linewidth=0.2, zorder=4)

            # Geomean diamond
            gm = geomean(vals)
            ax.scatter(pos, gm, marker="D", s=28, color="white",
                       edgecolor=color, linewidth=1.0, zorder=5)

            # Compact annotation: offset kernel left, E2E right
            win = int(np.sum(vals > 1.0))
            x_off = -6 if j == 0 else 6
            ha = "right" if j == 0 else "left"
            ax.annotate(
                f"{format_multiplier(gm)}\n{win}/{len(vals)}",
                xy=(pos, gm), xytext=(x_off, 10),
                textcoords="offset points", ha=ha, va="bottom",
                fontsize=5.5, color=color,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.85),
            )

    # Reference line
    ax.axhline(1.0, color=C_AXIS, linestyle="--", linewidth=0.7, alpha=0.6)

    # X-axis
    ax.set_xticks(group_centers)
    ax.set_xticklabels([s["short"] for s in valid_specs])
    ax.set_ylabel("Speedup (ToT / BTC-TC)")

    # Log scale
    all_vals = np.concatenate(kernel_data + e2e_data)
    y_lo = max(all_vals[all_vals > 0].min() * 0.7, 0.3)
    y_hi = all_vals.max() * 1.5
    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="#D5DBDB")
    ax.grid(False, axis="x")

    # Legend: right side, compact
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=C_KERNEL, alpha=0.60, edgecolor=C_KERNEL, label="Kernel"),
                    Patch(facecolor=C_E2E, alpha=0.60, edgecolor=C_E2E, label="E2E")]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=6.5,
              frameon=True, framealpha=0.92, edgecolor="#CFD8DC",
              facecolor="white", handletextpad=0.3, borderpad=0.25,
              handlelength=1.0)

    fig.tight_layout(pad=0.6)

    # Save
    for out_dir in (SC26_FIG_DIR, LEGACY_FIG_DIR):
        out_base = out_dir / "fig_cross_device_all"
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
        print(f"Wrote {out_base.with_suffix('.pdf')}")

    plt.close(fig)


if __name__ == "__main__":
    main()
