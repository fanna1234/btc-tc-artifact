#!/usr/bin/env python3
"""Shared matplotlib style for paper figures.

Goal: keep all generated plots visually consistent with the paper (serif fonts,
Type-42 embedded fonts in PDF, consistent font sizes and grid defaults).

Usage:
  from paper_plot_style import apply_paper_style
  apply_paper_style()
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


_LIBERTINE_FONT_PATHS = [
    Path("/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf"),
    Path("/usr/share/fonts/opentype/linux-libertine/LinLibertine_RB.otf"),
    Path("/usr/share/fonts/opentype/linux-libertine/LinLibertine_RI.otf"),
    Path("/usr/share/fonts/opentype/linux-libertine/LinLibertine_RBI.otf"),
]


def _try_add_libertine_fonts() -> None:
    for p in _LIBERTINE_FONT_PATHS:
        if not p.exists():
            continue
        try:
            fm.fontManager.addfont(str(p))
        except Exception:
            # Best-effort: fall back to whatever serif fonts are available.
            pass


def format_annotation_value(value: float) -> str:
    """Format annotated numeric values with a shared precision policy."""

    value = float(value)
    if not math.isfinite(value):
        return "--"

    magnitude = abs(value)
    if magnitude >= 10.0:
        return f"{value:.0f}"
    if magnitude >= 2.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def format_multiplier(value: float, *, latex: bool = False) -> str:
    """Format multiplicative values (e.g., speedups/geomeans) consistently."""

    formatted = format_annotation_value(value)
    if formatted == "--":
        return formatted
    suffix = r"$\times$" if latex else "×"
    return f"{formatted}{suffix}"


def apply_paper_style(
    *,
    font_size: float = 9.5,
    axes_label_size: float | None = None,
    legend_size: float | None = None,
    tick_label_size: float | None = None,
) -> None:
    """Apply a consistent plotting style for paper figures."""

    _try_add_libertine_fonts()

    axes_label_size = float(font_size if axes_label_size is None else axes_label_size)
    legend_size = float((font_size - 1.0) if legend_size is None else legend_size)
    tick_label_size = float((font_size - 1.0) if tick_label_size is None else tick_label_size)

    plt.rcParams["font.family"] = "serif"
    # Prefer TrueType serif fonts so the generated PDF avoids Poppler/CFF
    # mismatch warnings while still staying visually close to the paper body.
    plt.rcParams["font.serif"] = [
        "Liberation Serif",
        "DejaVu Serif",
        "Times New Roman",
        "serif",
    ]
    # Use STIX math glyphs (embedded as TrueType in our tests) for a cleaner,
    # publication-safe PDF embedding path.
    plt.rcParams["mathtext.fontset"] = "stix"

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    # Embed TrueType fonts into PDF (avoid Type3 fonts).
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": axes_label_size,
            "legend.fontsize": legend_size,
            "xtick.labelsize": tick_label_size,
            "ytick.labelsize": tick_label_size,
            # Paper-style axes/legend defaults. Individual plots can override.
            "axes.edgecolor": "black",
            "axes.linewidth": 1.1,
            "axes.axisbelow": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
            "legend.edgecolor": "0.85",
            "legend.fancybox": True,
            "lines.linewidth": 1.4,
        }
    )

    # Match the cleaner look used in our multi-baseline bar charts.
    for k, v in (("axes.spines.top", False), ("axes.spines.right", False)):
        if k in plt.rcParams:
            plt.rcParams[k] = v
