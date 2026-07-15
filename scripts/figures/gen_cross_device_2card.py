#!/usr/bin/env python3
"""Genuine two-device cross-device figure from NEUTRAL Chameleon hardware.

A100 (sm_80 @ CHI@UC, pin 2d02afe) + H100 (sm_90 @ CHI@TACC, pin 474695a) —
two of the paper's three device classes, each independently measured on a fresh
Chameleon bare-metal node. Same box+strip+geomean style as the paper's Fig 8,
but drawn ONLY from re-measured data (no bundled/synthetic device).

Usage:
  python gen_cross_device_2card.py <A100_csv_dir> <H100_csv_dir> <paper_datasets.txt> <out_dir>

Each csv_dir must contain BTC_Lite.csv + ToT.csv with the standard schema
(Dataset,Status,...,Kernel_ms,E2E_after_clean_ms,...).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# reuse the artifact's exact paper style helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_plot_style import apply_paper_style, format_multiplier  # noqa: E402

C_KERNEL = "#2980B9"
C_E2E = "#C0392B"
C_PRIMARY = "#2C3E50"
C_AXIS = "#78909C"


def geomean(vals) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.exp(np.log(vals).mean())) if vals.size else float("nan")


def load_paper_datasets(p: Path) -> set[str]:
    out = set()
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def build_device_df(csv_dir: Path, paper_set: set[str]) -> pd.DataFrame:
    btc = pd.read_csv(csv_dir / "BTC_Lite.csv")
    tot = pd.read_csv(csv_dir / "ToT.csv")
    cols = ["Dataset", "Status", "Triangles", "Kernel_ms", "E2E_after_clean_ms"]
    btc = btc[cols].rename(columns={"Status": "S_B", "Triangles": "T_B",
                                    "Kernel_ms": "K_B", "E2E_after_clean_ms": "E_B"})
    tot = tot[cols].rename(columns={"Status": "S_T", "Triangles": "T_T",
                                    "Kernel_ms": "K_T", "E2E_after_clean_ms": "E_T"})
    df = btc.merge(tot, on="Dataset", how="inner")
    df = df[(df["S_B"] == "OK") & (df["S_T"] == "OK")].copy()
    for c in ["K_B", "E_B", "K_T", "E_T", "T_B", "T_T"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["K_B", "E_B", "K_T", "E_T"])
    df["Kernel_Speedup"] = df["K_T"] / df["K_B"]
    df["E2E_Speedup"] = df["E_T"] / df["E_B"]
    # correctness: BTC is ground truth; ToT exact where its triangle count matches BTC's
    df["ToT_exact"] = (df["T_B"] == df["T_T"])
    df = df[df["Dataset"].isin(paper_set)].copy()
    if len(df) < len(paper_set):
        missing = sorted(paper_set - set(df["Dataset"]))
        print(f"  WARNING: {csv_dir}: only {len(df)}/{len(paper_set)} paper datasets present "
              f"(missing {len(missing)}: {missing[:5]}{'...' if len(missing)>5 else ''})", file=sys.stderr)
    return df


def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_color(C_AXIS); sp.set_linewidth(0.6)
    ax.tick_params(colors="#546E7A", width=0.6)
    ax.set_facecolor("white")


def main():
    a100_dir, h100_dir, paper_txt, out_dir = (Path(sys.argv[1]), Path(sys.argv[2]),
                                              Path(sys.argv[3]), Path(sys.argv[4]))
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_set = load_paper_datasets(paper_txt)
    apply_paper_style(font_size=9.5, legend_size=8.0, tick_label_size=8.5)

    SPECS = [
        {"short": "A100\n(sm_80)", "dir": a100_dir},
        {"short": "H100\n(sm_90)", "dir": h100_dir},
    ]

    kernel_data, e2e_data, valid = [], [], []
    for spec in SPECS:
        cdir = spec["dir"]
        if not (cdir / "BTC_Lite.csv").exists() or not (cdir / "ToT.csv").exists():
            print(f"{spec['short']!r:>14}: SKIPPED (missing CSV in {cdir})")
            continue
        df = build_device_df(cdir, paper_set)
        kernel_data.append(df["Kernel_Speedup"].to_numpy())
        e2e_data.append(df["E2E_Speedup"].to_numpy())
        valid.append(spec)
        n = len(df); exact_btc = n  # BTC is exact by construction (== ground truth)
        tot_exact = int(df["ToT_exact"].sum())
        print(f"{spec['short'].splitlines()[0]:>6}: n={n}  "
              f"Kernel GM={geomean(df['Kernel_Speedup']):.3f}x  "
              f"E2E GM={geomean(df['E2E_Speedup']):.3f}x  "
              f"BTC exact={exact_btc}/{n}  ToT exact={tot_exact}/{n}")

    if not valid:
        print("no device data present yet — rerun once H100 CSVs land"); return

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.85))
    style_ax(ax)
    n_dev = len(valid)
    box_w = 0.34; gap = 0.46; group_sep = 1.55
    rng = np.random.default_rng(42)
    centers = []
    for i in range(n_dev):
        cx = i * group_sep; centers.append(cx)
        for j, (pos, vals, color) in enumerate(
            [(cx - gap / 2, kernel_data[i], C_KERNEL),
             (cx + gap / 2, e2e_data[i], C_E2E)]):
            bp = ax.boxplot([vals], positions=[pos], widths=box_w, patch_artist=True,
                            showfliers=False, zorder=3,
                            medianprops=dict(color="white", linewidth=1.6),
                            whiskerprops=dict(color=C_AXIS, linewidth=0.8),
                            capprops=dict(color=C_AXIS, linewidth=0.8))
            bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.60)
            bp["boxes"][0].set_edgecolor(color); bp["boxes"][0].set_linewidth(0.8)
            jit = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(pos + jit, vals, s=7, color=color, alpha=0.42,
                       edgecolor="white", linewidth=0.2, zorder=4)
            gm = geomean(vals)
            ax.scatter(pos, gm, marker="D", s=28, color="white",
                       edgecolor=color, linewidth=1.0, zorder=5)
            win = int(np.sum(vals > 1.0))
            x_off = -6 if j == 0 else 6
            ha = "right" if j == 0 else "left"
            ax.annotate(f"{format_multiplier(gm)}\n{win}/{len(vals)}",
                        xy=(pos, gm), xytext=(x_off, 10), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=5.5, color=color,
                        bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                                  edgecolor="none", alpha=0.85))
    ax.axhline(1.0, color=C_AXIS, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xticks(centers)
    ax.set_xticklabels([s["short"] for s in valid])
    ax.set_ylabel("Speedup (ToT / BTC-TC)")
    all_vals = np.concatenate(kernel_data + e2e_data)
    ax.set_yscale("log")
    ax.set_ylim(max(all_vals[all_vals > 0].min() * 0.7, 0.3), all_vals.max() * 1.5)
    ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.4, color="#D5DBDB")
    ax.grid(False, axis="x")
    ax.set_title("Cross-device reproduction on Chameleon (neutral HW)",
                 fontsize=8.5, pad=6, color=C_PRIMARY)
    legend_elems = [Patch(facecolor=C_KERNEL, alpha=0.60, edgecolor=C_KERNEL, label="Kernel"),
                    Patch(facecolor=C_E2E, alpha=0.60, edgecolor=C_E2E, label="E2E")]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=6.5, frameon=True,
              framealpha=0.92, edgecolor="#CFD8DC", facecolor="white",
              handletextpad=0.3, borderpad=0.25, handlelength=1.0)
    fig.tight_layout(pad=0.6)
    base = out_dir / "fig_cross_device_chameleon_2card"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {base.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
