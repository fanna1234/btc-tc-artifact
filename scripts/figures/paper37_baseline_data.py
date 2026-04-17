#!/usr/bin/env python3
"""Shared paper37 baseline metadata and summary helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER37_CSV_DIR = Path(os.environ.get("BTC_CSV_DIR", str(REPO_ROOT / "results" / "pro6000" / "csv")))
PAPER_DATASETS_FILE = REPO_ROOT / "data" / "paper_datasets.txt"
LOG_DIR = REPO_ROOT / "results" / "ablation" / "logs"

_RE_INPUT = re.compile(r"\[Preprocess\] Input:\s*(\d+)\s*rows,\s*(\d+)\s*entries")
_RE_SYM = re.compile(r"\[Symmetrize\] Original edges:\s*(\d+),\s*After symmetrize & lower-tri:\s*(\d+)")
_RE_BLOCKS = re.compile(r"Num Blocks:\s*(\d+)")
_RE_TRI = re.compile(r"Triangles\s*\(GPU\):\s*(\d+)")

METHODS = [
    ("BTC-TC (Lite)", "BTC_Lite.csv"),
    ("ToT", "ToT.csv"),
    ("TRUST", "TRUST.csv"),
    ("Polak", "Polak.csv"),
    ("GroupTC", "GroupTC.csv"),
    ("Hu", "Hu.csv"),
    ("Green", "Green.csv"),
    ("Tricore", "Tricore.csv"),
    ("Bisson", "Bisson.csv"),
    ("Fox", "Fox.csv"),
    ("LAGraph (CPU)", "LAGraph.csv"),
    ("LAGraph (GPU)", "LAGraph-gpu.csv"),
    ("HIndex", "HIndex.csv"),
]

COLORS = {
    "BTC-TC": "#C0392B",
    "BTC-TC (Lite)": "#C0392B",
    "ToT": "#F57C00",
    "TRUST": "#2E7D32",
    "Polak": "#00897B",
    "GroupTC": "#5E35B1",
    "Hu": "#3949AB",
    "Green": "#6D4C41",
    "Tricore": "#8E24AA",
    "Bisson": "#039BE5",
    "Fox": "#546E7A",
    "LAGraph (CPU)": "#7CB342",
    "LAGraph (GPU)": "#26A69A",
    "HIndex": "#616161",
}


# Abbreviated display names + |E| for figure tick labels.
# Format: raw_name -> (short_name, edge_count_str)
DATASET_DISPLAY: dict[str, tuple[str, str]] = {
    "shyy41":           ("shyy",     "20K"),
    "spaceStation_13":  ("spSt",     "12K"),
    "bcsstk23":         ("bk23",     "24K"),
    "bcsstm13":         ("bm13",     "12K"),
    "g7jac020":         ("g7j20",    "46K"),
    "lpl1":             ("lpl1",     "180K"),
    "net50":            ("net50",    "481K"),
    "msc04515":         ("msc04",    "51K"),
    "tandem_vtx":       ("tndm",     "136K"),
    "delaunay_n17":     ("del17",    "393K"),
    "ex9":              ("ex9",      "51K"),
    "mac_econ_fwd500":  ("macE",     "1.3M"),
    "torso2":           ("torso",    "1.0M"),
    "wiki-Vote":        ("wkVt",     "104K"),
    "bcsstk24":         ("bk24",     "82K"),
    "mc2depi":          ("mc2d",     "2.1M"),
    "dawson5":          ("daw5",     "531K"),
    "struct3":          ("strt3",    "614K"),
    "g7jac140sc":       ("g7j140",   "566K"),
    "nemeth16":         ("nem16",    "298K"),
    "webbase-1M":       ("wb1M",     "3.1M"),
    "pli":              ("pli",      "687K"),
    "Freescale1":       ("Frsc",     "18.9M"),
    "web-NotreDame":    ("wND",      "1.5M"),
    "cage14":           ("cg14",     "27.1M"),
    "pcrystk03":        ("pcry",     "888K"),
    "pkustk06":         ("pk06",     "1.3M"),
    "bcsstk30":         ("bk30",     "1.0M"),
    "cant":             ("cant",     "2.0M"),
    "consph":           ("cnsp",     "3.1M"),
    "pdb1HYS":          ("pdb1",     "2.2M"),
    "pwtk":             ("pwtk",     "5.9M"),
    "F1":               ("F1",       "13.6M"),
    "eu-2005":          ("eu05",     "19.2M"),
    "Si41Ge41H72":      ("SiGe",     "7.6M"),
    "Ga41As41H72":      ("GaAs",     "9.4M"),
}


def dataset_display_name(raw: str) -> str:
    """Return abbreviated name with |E| for figure tick labels."""
    if raw in DATASET_DISPLAY:
        abbrev, size = DATASET_DISPLAY[raw]
        return f"{abbrev}\n({size})"
    return raw


def load_paper_datasets() -> list[str]:
    names: list[str] = []
    for line in PAPER_DATASETS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names


def geomean(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.exp(np.log(arr).mean()))


@dataclass(frozen=True)
class LogStats:
    num_rows: int
    num_entries: int
    edges_lower: int
    num_blocks: int
    triangles: int | None


def parse_log(path: Path) -> LogStats:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    num_rows = num_entries = edges_lower = num_blocks = triangles = None
    for line in lines:
        m = _RE_INPUT.search(line)
        if m:
            num_rows, num_entries = int(m.group(1)), int(m.group(2))
            continue
        m = _RE_SYM.search(line)
        if m:
            edges_lower = int(m.group(2))
            continue
        m = _RE_BLOCKS.search(line)
        if m:
            num_blocks = int(m.group(1))
            continue
        m = _RE_TRI.search(line)
        if m:
            triangles = int(m.group(1))
            continue
    missing = []
    if num_rows is None or num_entries is None:
        missing.append("Preprocess Input")
    if edges_lower is None:
        missing.append("Symmetrize")
    if num_blocks is None:
        missing.append("Num Blocks")
    if missing:
        raise RuntimeError(f"Failed to parse {path}: missing {', '.join(missing)}")
    return LogStats(num_rows=num_rows, num_entries=num_entries,
                    edges_lower=edges_lower, num_blocks=num_blocks, triangles=triangles)


def pick_16x32_log(dataset: str) -> Path:
    p = LOG_DIR / f"V3_16x32_PureTC_{dataset}.log"
    if p.exists():
        return p
    p = LOG_DIR / f"V6_16x32_Hybrid_{dataset}.log"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing 16x32 log for dataset {dataset}")


def compute_bytes_csr(num_rows: int, edges_lower: int) -> int:
    return (num_rows + 1) * 4 + edges_lower * 4


def compute_bytes_bit_bsr(num_rows: int, num_blocks: int, block_words_u32: int) -> int:
    n_row_blocks = (num_rows + 16 - 1) // 16
    indptr = (n_row_blocks + 1) * 4
    indices = num_blocks * 4
    row_indices = num_blocks * 4
    blocks = num_blocks * block_words_u32 * 4
    result = 8
    return indptr + indices + row_indices + blocks + result


def collect_memory_ratios() -> tuple[np.ndarray, np.ndarray]:
    """Compute Bit-BSR/CSR memory ratios for all paper datasets."""
    datasets = load_paper_datasets()
    ratio_32, ratio_128 = [], []
    for ds in datasets:
        log128 = LOG_DIR / f"V3_16x128_PureTC_{ds}.log"
        if not log128.exists():
            raise FileNotFoundError(f"Missing 16x128 log: {log128}")
        st128 = parse_log(log128)
        st32 = parse_log(pick_16x32_log(ds))
        csr_b = compute_bytes_csr(st128.num_rows, st128.edges_lower)
        bsr32_b = compute_bytes_bit_bsr(st128.num_rows, st32.num_blocks, block_words_u32=16)
        bsr128_b = compute_bytes_bit_bsr(st128.num_rows, st128.num_blocks, block_words_u32=64)
        ratio_32.append(bsr32_b / csr_b)
        ratio_128.append(bsr128_b / csr_b)
    return np.asarray(ratio_32, dtype=np.float64), np.asarray(ratio_128, dtype=np.float64)


def load_ok_frames(csv_dir: Path = PAPER37_CSV_DIR) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for label, fname in METHODS:
        p = csv_dir / fname
        if not p.exists():
            frames[label] = pd.DataFrame()
            continue
        df = pd.read_csv(p)
        frames[label] = df[df["Status"] == "OK"].copy()
    return frames


def load_metric_table(csv_dir: Path = PAPER37_CSV_DIR) -> pd.DataFrame:
    rows = []
    for label, fname in METHODS:
        p = csv_dir / fname
        if not p.exists():
            continue
        full = pd.read_csv(p)
        ok = full[full["Status"] == "OK"].copy()
        rows.append(
            {
                "Method": label,
                "CSV": fname,
                "Success_n": int(len(ok)),
                "Total_n": int(len(full)),
                "Success": f"{len(ok)}/{len(full)}",
                "Kernel_ms": geomean(ok["Kernel_ms"]),
                "E2E_ms": geomean(ok["E2E_after_clean_ms"]),
            }
        )
    return pd.DataFrame(rows)
