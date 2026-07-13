#!/usr/bin/env python3
"""
Benchmark runner (minimal, per-method CSV).

User requirement:
  - A fixed method list (defaults to BTC + its 3 in-house variants).
  - Each method runs ALL datasets and writes ONE CSV per method.
  - Easy to rerun/debug a single method without re-running others.
  - Keep results small: per-dataset converted "work dirs" are deleted by default.
  - Failures are categorized (no opaque "OTHER" by default):
      OOM / OOT / LIMIT / CRASH / ENV

Outputs (under --run-dir; default: results/csv1):
  - csv/<Method>.csv
  - failures/<Method>/<Dataset>.log
  - work/<Method>/<Dataset>/...   (temporary; deleted unless --keep-work)
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import signal
import subprocess
import time
from decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = repo_root()
DATA_DIR = PROJECT_ROOT / "data"


BTC_METHODS: list[str] = [
    "BTC_Lite",
    "BTC_16x128_Adaptive",
    "BTC_16x32_Adaptive",
]

BASELINE_METHODS: list[str] = [
    "ToT",
    "TRUST",
    "LAGraph",
    "LAGraph-gpu",
    "Tricore",
    "Fox",
    "Hu",
    "Green",
    "GroupTC",
    "Bisson",
    "HIndex",
    "Polak",
]

METHOD_SUITES: dict[str, list[str]] = {
    "btc": list(BTC_METHODS),
    "baselines": list(BASELINE_METHODS),
    # Convenience suite: compare LAGraph CPU vs GraphBLAS-CUDA build.
    "lagraph": ["LAGraph", "LAGraph-gpu"],
    "all": list(BTC_METHODS) + list(BASELINE_METHODS),
}

# LAGraph has both CPU and GPU (GraphBLAS-CUDA) build trees in this repo.
# We treat them as separate methods/CSVs: "LAGraph" (CPU) and "LAGraph-gpu" (GPU).
LAGRAPH_CPU_BIN = PROJECT_ROOT / "baselines/LAGraph/build/experimental/benchmark/tcc_demo"
LAGRAPH_GPU_BIN = PROJECT_ROOT / "baselines/LAGraph/build-gpu/experimental/benchmark/tcc_demo"


BINARIES: dict[str, Path] = {
    # Ours (MTX input)
    "BTC_Lite": PROJECT_ROOT / "build/apps/btc_tc_lite",
    "BTC_16x128_Adaptive": PROJECT_ROOT / "build/apps/btc_tc_adaptive_16x128",
    "BTC_16x32_Adaptive": PROJECT_ROOT / "build/apps/btc_tc_adaptive_16x32",
    "ToT": (PROJECT_ROOT / "build/baselines/ToT-TPDS25/apps/tot"
            if (PROJECT_ROOT / "build/baselines/ToT-TPDS25/apps/tot").exists()
            else PROJECT_ROOT / "baselines/ToT-TPDS25/build/apps/tot"),
    "TRUST": PROJECT_ROOT / "baselines/TRUST/Without-graph-partition/trianglecounting.bin",
    "cuSPARSE": PROJECT_ROOT / "build/baselines/cusparse_tc/apps/cusparse_tc_lxlt",
    # Keep legacy name as CPU baseline (matches repo docs and existing CSV naming).
    "LAGraph": LAGRAPH_CPU_BIN,
    # Explicit variants (useful for side-by-side runs / distinct CSVs)
    "LAGraph-gpu": LAGRAPH_GPU_BIN,
    "LAGraph-cpu": LAGRAPH_CPU_BIN,
    "Tricore": PROJECT_ROOT / "baselines/TC-Compare/approach/tricore/tricore",
    "Fox": PROJECT_ROOT / "baselines/TC-Compare/approach/Fox/fox",
    "Hu": PROJECT_ROOT / "baselines/TC-Compare/approach/Hu/hu",
    "Green": PROJECT_ROOT / "baselines/TC-Compare/approach/Green/green",
    "GroupTC": PROJECT_ROOT / "baselines/TC-Compare/approach/GroupTC/grouptc",
    "Bisson": PROJECT_ROOT / "baselines/TC-Compare/approach/Bisson/bisson",
    "HIndex": PROJECT_ROOT / "baselines/TC-Compare/approach/H-INDEX/hindex",
    "Polak": PROJECT_ROOT / "baselines/TC-Compare/approach/polak/polak",
}


# Conversion tools for TC-Compare
TC_COMPARE_MTX2CSR = PROJECT_ROOT / "baselines/TC-Compare/preprocessing/cpu_preprocessing/XXX2CSR/MTX2CSR"
TC_COMPARE_CSR2XXX_DIR = PROJECT_ROOT / "baselines/TC-Compare/preprocessing/cpu_preprocessing/CSR2XXX"
TC_COMPARE_CSR2HU = TC_COMPARE_CSR2XXX_DIR / "CSR2HuEdgeList"
TC_COMPARE_CSR2RIDDCSR = TC_COMPARE_CSR2XXX_DIR / "CSR2RidDCSR"
TC_COMPARE_CSR2POLAK = TC_COMPARE_CSR2XXX_DIR / "CSR2PolakEdgeList"


# TRUST preprocess tools
TRUST_PREPROCESS_DIR = PROJECT_ROOT / "baselines/TRUST/Preprocess"
TRUST_FROM_DIRECT = TRUST_PREPROCESS_DIR / "fromDirectToUndirect"
TRUST_PREPROCESS = TRUST_PREPROCESS_DIR / "preprocess"


def safe_method_filename(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method)


def _re_float(pat: str, s: str) -> float:
    m = re.search(pat, s, re.IGNORECASE | re.MULTILINE)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except Exception:
        return float("nan")


def parse_triangles_generic(out: str) -> int:
    for pat in [
        r"Triangles\s*\(GPU\)\s*:\s*(\d+)",
        r"Triangles\s*:\s*(\d+)",
        r"triangle count\s+(\d+)",
        r"triangle count:\s*(\d+)",
    ]:
        m = re.search(pat, out, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return -1


def parse_trust_triangles_and_time(out: str) -> tuple[int, float]:
    """
    TRUST prints a CSV-like line:
      <folder>,<vertex>,<edge>,<triangles>,<time_s>,<teps>
    """
    for line in (out or "").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 5:
            try:
                return int(parts[3]), float(parts[4])
            except Exception:
                pass
    return -1, float("nan")


def parse_lagraph_triangles_and_time(out: str) -> tuple[int, float]:
    """
    LAGraph prints:
      warmup time X.XXXXXX sec, # triangles: XXXXX  (or 3.05639e+06 in scientific notation)
      threads N trial 0: X.XXXXXX sec
      threads N trial 1: X.XXXXXX sec
      ...
      Avg: TCentrality(1) nthreads: N time: X.XXXXXX matrix: ...
    We extract triangle count from first line and avg time from last line.
    """
    triangles = -1
    avg_time_s = float("nan")

    for line in (out or "").splitlines():
        s = line.strip()
        # Extract triangle count: "# triangles: 54380" or "# triangles: 3.05639e+06"
        # Match both integer and scientific notation.
        m = re.search(r"#\s*triangles:\s*([\d\.eE+\-]+)", s, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            # Prefer integer parsing (newer tcc_demo prints uint64 directly).
            try:
                triangles = int(raw)
            except Exception:
                # Fallback for older outputs that used %g (scientific notation / decimals).
                try:
                    triangles = int(Decimal(raw))
                except Exception:
                    pass
        # Extract avg time: "Avg: ... time: 0.001578 ..."
        m = re.search(r"Avg:.*?time:\s*([\d\.]+)", s, re.IGNORECASE)
        if m:
            try:
                avg_time_s = float(m.group(1))
            except Exception:
                pass

    return triangles, avg_time_s



def parse_kernel_ms(method: str, out: str) -> float:
    if method.startswith("BTC_") or method == "BTC_Lite":
        v = _re_float(r"Compute \(Kernel\):\s*([\d\.]+)\s*ms", out)
        if v == v:
            return v
        return _re_float(r"\[Counting Triangles.*?\]\s*time:\s*([\d\.]+)\s*ms", out)
    if method == "ToT":
        return _re_float(r"\[Kernel\]\s*time:\s*([\d\.]+)\s*ms", out)
    if method == "cuSPARSE":
        return _re_float(r"\[cuSPARSE SpGEMM TC\]\s*time:\s*([\d\.]+)\s*ms", out)
    if method == "TRUST":
        _tri, t_s = parse_trust_triangles_and_time(out)
        return t_s * 1000.0 if t_s == t_s else float("nan")
    if method in ("LAGraph", "LAGraph-gpu", "LAGraph-cpu"):
        _tri, t_s = parse_lagraph_triangles_and_time(out)
        return t_s * 1000.0 if t_s == t_s else float("nan")
    # TC-Compare: avg kernel use X.XXXX s
    v = _re_float(r"avg kernel use\s+([\d\.]+)\s*s", out)
    return v * 1000.0 if v == v else float("nan")


def parse_cleaning_ms(method: str, out: str) -> float:
    if method.startswith("BTC_") or method == "BTC_Lite":
        return _re_float(r"\[Preprocessing\]\s*time:\s*([\d\.]+)\s*ms", out)
    if method == "ToT":
        return _re_float(r"\[Cleaning\]\s*Make Undirected:\s*([\d\.]+)\s*ms", out)
    if method == "cuSPARSE":
        return _re_float(r"\[Preprocessing\]\s*time:\s*([\d\.]+)\s*ms", out)
    return float("nan")


def parse_gpu_e2e_after_clean_ms(method: str, out: str) -> float:
    """
    Only MTX-consuming baselines print a GPU-side after-clean total.
    Others fall back to (Build_ms + Wall_ms).
    """
    if method.startswith("BTC_") or method == "BTC_Lite":
        v = _re_float(r"\[Total Time \(Convert\+Compute\)\]\s*time:\s*([\d\.]+)\s*ms", out)
        if v == v:
            return v
        conv = _re_float(r"\[Time Breakdown\]\s*Convert:\s*([\d\.]+)\s*ms", out)
        ker = _re_float(r"Compute \(Kernel\):\s*([\d\.]+)\s*ms", out)
        post = _re_float(r"\[Time Breakdown\].*?Post:\s*([\d\.]+)\s*ms", out)
        if conv == conv and ker == ker:
            return conv + ker + (post if post == post else 0.0)
        return float("nan")
    if method == "ToT":
        v = _re_float(r"\[Total Time \(Build\+Count\)\]\s*time:\s*([\d\.]+)\s*ms", out)
        if v == v:
            return v
        bmp = _re_float(r"\[Converting to Bitmap\]\s*time:\s*([\d\.]+)\s*ms", out)
        cnt = _re_float(r"\[Counting Triangles\]\s*time:\s*([\d\.]+)\s*ms", out)
        parts = [x for x in (bmp, cnt) if x == x]
        return float(sum(parts)) if parts else float("nan")
    if method == "cuSPARSE":
        v = _re_float(r"\[Total Time \(Build\+Compute\)\]\s*time:\s*([\d\.]+)\s*ms", out)
        if v == v:
            return v
        build = _re_float(r"\[Build\]\s*time:\s*([\d\.]+)\s*ms", out)
        spg = _re_float(r"\[cuSPARSE SpGEMM TC\]\s*time:\s*([\d\.]+)\s*ms", out)
        parts = [x for x in (build, spg) if x == x]
        return float(sum(parts)) if parts else float("nan")
    return float("nan")


def summarize_failure(output: str, timed_out: bool, timeout_s: int, retcode: Optional[int] = None) -> str:
    if timed_out:
        return f"TIMEOUT({timeout_s}s)"
    out = output or ""
    for pat in [
        r"(cudaError[A-Za-z0-9_]+[^\n]*)",
        r"(CUSPARSE_STATUS_[A-Z0-9_]+[^\n]*)",
        r"(cuSPARSE Error:[^\n]*)",
        r"(RUN OUT OF GLOBAL MEMORY!![^\n]*)",
        r"(ERROR![^\n]*)",
        r"(Segmentation fault[^\n]*)",
        r"(terminate called after throwing[^\n]*)",
        r"(what\(\):[^\n]*)",
        r"(error:\s*[^\n]*)",
    ]:
        m = re.search(pat, out, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:200]
    # If the process died from a signal and we didn't match a useful error line, surface that.
    if retcode is not None and retcode < 0:
        sig = -retcode
        try:
            name = signal.Signals(sig).name
        except Exception:
            name = f"SIG{sig}"
        return f"CRASH({name})"
    for line in out.splitlines():
        s = line.strip()
        if s:
            return s[:200]
    return "ERROR"


def classify_failure(output: str, timed_out: bool, retcode: Optional[int] = None) -> str:
    """
    Return one of: OK, OOT, OOM, LIMIT, CRASH, ENV, OTHER.

    Notes:
      - OOT is only used when our runner timed out (subprocess timeout).
      - OOM is a best-effort match on common CUDA/cuSPARSE and baseline messages.
    """
    if timed_out:
        return "OOT"
    out = (output or "").lower()
    # Environment / driver / device access issues should not be reported as OOM.
    env_pats = [
        "no devices supporting cuda",
        "cudaerrornodevice",
        "cudaerrorinvaliddevice",
        "cudaerrorinsufficientdriver",
        "cudaerrorinitializationerror",
        "cudaerroroperatingsystem",
        # SuiteSparse:GraphBLAS CUDA init failures (prints GB_cuda_* lines)
        "gb_cuda_get_device_count",
        "gb_cuda_init",
        "os call failed",
        "operation not supported",
        "driver not initialized",
    ]
    if any(p in out for p in env_pats):
        return "ENV"

    # Hard-coded implementation limits in some baselines.
    limit_pats = [
        "the nodenum is too large",
        "nodenum is too large",
    ]
    if any(p in out for p in limit_pats):
        return "LIMIT"

    oom_pats = [
        "out of memory",
        "out-of-memory",
        "std::bad_alloc",
        "bad_alloc",
        "cudaerrormemoryallocation",
        "cuda error memory allocation",
        "cusparse_status_alloc_failed",
        "cusparse_status_insufficient_resources",
        "cusparse error: 11",
        "alloc_failed",
        "allocation failed",
        "run out of global memory",
        "run out of global memory!!",
    ]
    if any(p in out for p in oom_pats):
        return "OOM"
    if retcode is not None and retcode < 0:
        return "CRASH"
    return "OTHER"


@dataclass
class RunResult:
    output: str
    wall_ms: float
    retcode: int
    timed_out: bool


def run_cmd(cmd: list[str], *, cwd: Optional[Path] = None, timeout_s: int = 300) -> RunResult:
    try:
        t0 = time.time()
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
        t1 = time.time()
        out = (p.stdout or "") + (p.stderr or "")
        return RunResult(output=out, wall_ms=(t1 - t0) * 1000.0, retcode=p.returncode, timed_out=False)
    except subprocess.TimeoutExpired as e:
        def _t(x) -> str:
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return str(x)

        out = _t(getattr(e, "stdout", "")) + _t(getattr(e, "stderr", ""))
        return RunResult(output=(out or "") + "\nTIMEOUT\n", wall_ms=float(timeout_s) * 1000.0, retcode=-1, timed_out=True)


def dataset_list(args: argparse.Namespace) -> list[str]:
    if args.datasets.strip():
        return [x.strip() for x in args.datasets.split(",") if x.strip()]
    if args.datasets_file:
        p = Path(args.datasets_file)
        names: list[str] = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
        return names
    # Default to the paper benchmark set if present (keeps artifact/paper reproducible).
    default_file = DATA_DIR / "paper_datasets.txt"
    if default_file.exists():
        names: list[str] = []
        for line in default_file.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
        return names
    return sorted([p.stem for p in DATA_DIR.glob("*.mtx")])


def mtx_to_edges_txt(mtx: Path, out_txt: Path) -> float:
    t0 = time.time()
    with mtx.open("r", encoding="utf-8", errors="replace") as f_in, out_txt.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line or line.startswith("%"):
                continue
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            # Skip MatrixMarket header: "n n nnz"
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                try:
                    if int(parts[0]) == int(parts[1]):
                        continue
                except Exception:
                    pass
            f_out.write(parts[0] + " " + parts[1] + "\n")
    return (time.time() - t0) * 1000.0


def trust_preprocess_from_mtx(mtx: Path, work_dir: Path, timeout_s: int) -> tuple[Optional[Path], float, float, float, str]:
    if not TRUST_FROM_DIRECT.exists() or not TRUST_PREPROCESS.exists():
        return None, 0.0, 0.0, 0.0, "MISSING_TRUST_PREPROCESS_BINARIES"
    work_dir.mkdir(parents=True, exist_ok=True)
    edges_txt = work_dir / "edges.txt"
    edges_txt_ms = mtx_to_edges_txt(mtx, edges_txt)
    rr1 = run_cmd([str(TRUST_FROM_DIRECT), str(edges_txt)], cwd=work_dir, timeout_s=timeout_s)
    if rr1.retcode != 0:
        return None, edges_txt_ms, rr1.wall_ms, 0.0, rr1.output
    rr2 = run_cmd([str(TRUST_PREPROCESS)], cwd=work_dir, timeout_s=timeout_s)
    ok = (rr2.retcode == 0) and (work_dir / "begin.bin").exists() and (work_dir / "adjacent.bin").exists()
    out = (rr1.output or "") + "\n" + (rr2.output or "")
    return (work_dir if ok else None), edges_txt_ms, rr1.wall_ms, rr2.wall_ms, out


def tc_compare_mtx_to_csr(mtx: Path, out_dir: Path, timeout_s: int) -> tuple[Optional[Path], float, str]:
    if not TC_COMPARE_MTX2CSR.exists():
        return None, 0.0, f"MISSING_CONVERTER:{TC_COMPARE_MTX2CSR}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rr = run_cmd([str(TC_COMPARE_MTX2CSR), str(mtx), str(out_dir) + "/"], timeout_s=timeout_s)
    ok = (rr.retcode == 0) and (out_dir / "begin.bin").exists() and (out_dir / "adjacent.bin").exists()
    return (out_dir if ok else None), rr.wall_ms, rr.output


def tc_compare_csr_to_hu(csr_dir: Path, out_dir: Path, timeout_s: int) -> tuple[Optional[Path], float, str]:
    if not TC_COMPARE_CSR2HU.exists():
        return None, 0.0, f"MISSING_CONVERTER:{TC_COMPARE_CSR2HU}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rr = run_cmd([str(TC_COMPARE_CSR2HU), str(csr_dir) + "/", str(out_dir) + "/"], timeout_s=timeout_s)
    ok = (rr.retcode == 0) and (out_dir / "edges.bin").exists()
    return (out_dir if ok else None), rr.wall_ms, rr.output


def tc_compare_csr_to_rid_dcsr(csr_dir: Path, out_dir: Path, timeout_s: int) -> tuple[Optional[Path], float, str]:
    if not TC_COMPARE_CSR2RIDDCSR.exists():
        return None, 0.0, f"MISSING_CONVERTER:{TC_COMPARE_CSR2RIDDCSR}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rr = run_cmd([str(TC_COMPARE_CSR2RIDDCSR), str(csr_dir) + "/", str(out_dir) + "/"], timeout_s=timeout_s)
    ok = (rr.retcode == 0) and (out_dir / "begin.bin").exists() and (out_dir / "adjacent.bin").exists()
    return (out_dir if ok else None), rr.wall_ms, rr.output


def tc_compare_csr_to_polak(csr_dir: Path, out_file: Path, timeout_s: int) -> tuple[Optional[Path], float, str]:
    if not TC_COMPARE_CSR2POLAK.exists():
        return None, 0.0, f"MISSING_CONVERTER:{TC_COMPARE_CSR2POLAK}"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rr = run_cmd([str(TC_COMPARE_CSR2POLAK), str(csr_dir) + "/", str(out_file)], timeout_s=timeout_s)
    ok = (rr.retcode == 0) and out_file.exists()
    return (out_file if ok else None), rr.wall_ms, rr.output


@dataclass
class Row:
    dataset: str
    status: str
    triangles: int
    kernel_ms: float
    e2e_after_clean_ms: float
    wall_ms: float
    cleaning_ms: float
    build_ms: float
    retcode: int
    reason: str
    log_path: str


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Dataset",
                "Status",
                "Triangles",
                "Kernel_ms",
                "E2E_after_clean_ms",
                "Wall_ms",
                "Cleaning_ms",
                "Build_ms",
                "Retcode",
                "Reason",
                "Log",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.dataset,
                    r.status,
                    r.triangles if r.triangles >= 0 else "",
                    f"{r.kernel_ms:.6f}" if r.kernel_ms == r.kernel_ms else "",
                    f"{r.e2e_after_clean_ms:.6f}" if r.e2e_after_clean_ms == r.e2e_after_clean_ms else "",
                    f"{r.wall_ms:.6f}" if r.wall_ms == r.wall_ms else "",
                    f"{r.cleaning_ms:.6f}" if r.cleaning_ms == r.cleaning_ms else "",
                    f"{r.build_ms:.6f}" if r.build_ms == r.build_ms else "",
                    r.retcode if r.retcode != -1 else "",
                    r.reason,
                    r.log_path,
                ]
            )


def run_one_baseline(
    *,
    method: str,
    run_dir: Path,
    datasets: list[str],
    timeout_s: int,
    iters: int,
    keep_work: bool,
) -> None:
    bin_path = BINARIES[method]
    if not bin_path.exists():
        print(f"WARNING: Binary not found for {method}: {bin_path}, skipping.")
        return

    csv_path = run_dir / "csv" / f"{safe_method_filename(method)}.csv"
    failures_dir = run_dir / "failures" / safe_method_filename(method)
    # Keep failures dir "fresh" per method so old logs don't confuse later plotting/debugging.
    if failures_dir.exists():
        shutil.rmtree(failures_dir, ignore_errors=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []

    for ds in datasets:
        mtx = DATA_DIR / f"{ds}.mtx"
        if not mtx.exists():
            rows.append(
                Row(
                    dataset=ds,
                    status="MISSING",
                    triangles=-1,
                    kernel_ms=float("nan"),
                    e2e_after_clean_ms=float("nan"),
                    wall_ms=float("nan"),
                    cleaning_ms=float("nan"),
                    build_ms=float("nan"),
                    retcode=-1,
                    reason="MISSING_MTX",
                    log_path="",
                )
            )
            write_csv(csv_path, rows)
            print(f"{method:10s} {ds:20s} -> MISSING")
            continue

        work_ds = run_dir / "work" / safe_method_filename(method) / ds
        # Always start from a clean work dir for deterministic runs.
        if work_ds.exists():
            shutil.rmtree(work_ds, ignore_errors=True)
        work_ds.mkdir(parents=True, exist_ok=True)

        triangles = -1
        kernel_ms = float("nan")
        e2e_after_clean_ms = float("nan")
        wall_ms = float("nan")
        cleaning_ms = float("nan")
        build_ms = float("nan")
        retcode = -1
        status = "OTHER"
        reason = ""
        log_path = ""
        output = ""

        # MTX-consuming methods (ours + selected baselines)
        if method in (
            "BTC_Lite",
            "BTC_16x128_Adaptive",
            "BTC_16x32_Adaptive",
            "ToT",
            "cuSPARSE",
            "LAGraph",
            "LAGraph-gpu",
            "LAGraph-cpu",
        ):
            rr = (
                run_cmd([str(bin_path), str(mtx)], timeout_s=timeout_s)
                if method in ("LAGraph", "LAGraph-gpu", "LAGraph-cpu")
                else run_cmd([str(bin_path), "-i", str(mtx)], timeout_s=timeout_s)
            )
            output = rr.output
            wall_ms = rr.wall_ms
            retcode = rr.retcode
            if rr.timed_out or rr.retcode != 0:
                status = classify_failure(rr.output, rr.timed_out, rr.retcode)
                reason = summarize_failure(rr.output, rr.timed_out, timeout_s, rr.retcode)
            else:
                # LAGraph uses its own parser
                if method in ("LAGraph", "LAGraph-gpu", "LAGraph-cpu"):
                    triangles, t_s = parse_lagraph_triangles_and_time(rr.output)
                    kernel_ms = (t_s * 1000.0) if (t_s == t_s) else float("nan")
                    cleaning_ms = float("nan")  # LAGraph doesn't report cleaning time separately
                    e2e_after_clean_ms = wall_ms  # Use wall time as E2E
                else:
                    triangles = parse_triangles_generic(rr.output)
                    kernel_ms = parse_kernel_ms(method, rr.output)
                    cleaning_ms = parse_cleaning_ms(method, rr.output)
                    e2e_after_clean_ms = parse_gpu_e2e_after_clean_ms(method, rr.output)
                if triangles >= 0 and kernel_ms == kernel_ms and e2e_after_clean_ms == e2e_after_clean_ms:
                    status = "OK"
                else:
                    status = "OTHER"
                    reason = "PARSE_ERROR"

        # TRUST: preprocess in work dir then mpirun
        elif method == "TRUST":
            graph_dir, edges_txt_ms, undirect_ms, preprocess_ms, conv_out = trust_preprocess_from_mtx(
                mtx, work_ds / "trust", timeout_s=timeout_s
            )
            cleaning_ms = edges_txt_ms + undirect_ms
            build_ms = preprocess_ms

            if graph_dir is None:
                conv_timed_out = "TIMEOUT" in (conv_out or "")
                status = classify_failure(conv_out, conv_timed_out, None)
                output = conv_out
                reason = summarize_failure(conv_out, conv_timed_out, timeout_s, None)
            else:
                cmd = [
                    "mpirun",
                    "--allow-run-as-root",
                    "-n",
                    "1",
                    str(bin_path),
                    str(graph_dir) + "/",
                    "1",
                    "1024",
                    "1024",
                    "1",
                ]
                rr = run_cmd(cmd, timeout_s=timeout_s)
                output = rr.output
                wall_ms = rr.wall_ms
                retcode = rr.retcode
                if rr.timed_out or rr.retcode != 0 or "ERROR" in (rr.output or ""):
                    status = classify_failure(rr.output, rr.timed_out, rr.retcode)
                    reason = summarize_failure(rr.output, rr.timed_out, timeout_s, rr.retcode)
                else:
                    triangles, t_s = parse_trust_triangles_and_time(rr.output)
                    kernel_ms = (t_s * 1000.0) if (t_s == t_s) else float("nan")
                    e2e_after_clean_ms = build_ms + wall_ms if wall_ms == wall_ms else float("nan")
                    if triangles >= 0 and kernel_ms == kernel_ms and e2e_after_clean_ms == e2e_after_clean_ms:
                        status = "OK"
                    else:
                        status = "OTHER"
                        reason = "PARSE_ERROR"

        # TC-Compare baselines: MTX2CSR + optional CSR->X
        else:
            csr_dir, csr_ms, csr_out = tc_compare_mtx_to_csr(mtx, work_ds / "csr", timeout_s=timeout_s)
            cleaning_ms = csr_ms
            if csr_dir is None:
                csr_timed_out = "TIMEOUT" in (csr_out or "")
                status = classify_failure(csr_out, csr_timed_out, None)
                output = csr_out
                reason = summarize_failure(csr_out, csr_timed_out, timeout_s, None)
            else:
                data_dir: Optional[Path] = None
                polak_file: Optional[Path] = None
                extra_out = ""

                if method in ("Fox", "Hu", "Bisson", "Tricore"):
                    hu_dir, hu_ms, hu_out = tc_compare_csr_to_hu(csr_dir, work_ds / "hu", timeout_s=timeout_s)
                    build_ms = hu_ms
                    data_dir = hu_dir
                    extra_out = hu_out
                elif method == "GroupTC":
                    rid_dir, rid_ms, rid_out = tc_compare_csr_to_rid_dcsr(csr_dir, work_ds / "rid_dcsr", timeout_s=timeout_s)
                    build_ms = rid_ms
                    data_dir = rid_dir
                    extra_out = rid_out
                elif method == "Polak":
                    pf, polak_ms, polak_out = tc_compare_csr_to_polak(csr_dir, work_ds / "polak.bin", timeout_s=timeout_s)
                    build_ms = polak_ms
                    polak_file = pf
                    extra_out = polak_out
                else:
                    # Green / HIndex consume CSR folder.
                    build_ms = 0.0
                    data_dir = csr_dir

                if (method == "Polak" and polak_file is None) or (method != "Polak" and data_dir is None):
                    output = (csr_out or "") + "\n" + (extra_out or "")
                    extra_timed_out = "TIMEOUT" in (output or "")
                    status = classify_failure(output, extra_timed_out, None)
                    reason = summarize_failure(output, extra_timed_out, timeout_s, None)
                else:
                    if method == "Green":
                        cmd = [str(bin_path), str(data_dir) + "/", "0", "256", "128", "8", "1"]
                    elif method == "HIndex":
                        cmd = [str(bin_path), str(data_dir) + "/", "1", "256", "64", "64", "1", "1", "0", str(int(iters))]
                    elif method in ("Fox", "Bisson"):
                        edges = str((data_dir / "edges.bin"))
                        cmd = [str(bin_path), "-f", edges, "0", "0", str(int(iters))]
                    elif method in ("Hu", "Tricore"):
                        edges = str((data_dir / "edges.bin"))
                        cmd = [str(bin_path), "-f", edges, "0", str(int(iters))]
                    elif method == "GroupTC":
                        cmd = [str(bin_path), str(data_dir) + "/", "0", str(int(iters))]
                    elif method == "Polak":
                        cmd = [str(bin_path), str(polak_file), "0", str(int(iters))]
                    else:
                        raise RuntimeError(f"Unknown baseline: {method}")

                    rr = run_cmd(cmd, timeout_s=timeout_s)
                    output = rr.output
                    wall_ms = rr.wall_ms
                    retcode = rr.retcode
                    if rr.timed_out or rr.retcode != 0 or "ERROR" in (rr.output or "") or "error" in (rr.output or "").lower():
                        status = classify_failure(rr.output, rr.timed_out, rr.retcode)
                        reason = summarize_failure(rr.output, rr.timed_out, timeout_s, rr.retcode)
                    else:
                        triangles = parse_triangles_generic(rr.output)
                        kernel_ms = parse_kernel_ms(method, rr.output)
                        e2e_after_clean_ms = build_ms + wall_ms if wall_ms == wall_ms else float("nan")
                        if triangles >= 0 and kernel_ms == kernel_ms and e2e_after_clean_ms == e2e_after_clean_ms:
                            status = "OK"
                        else:
                            status = "OTHER"
                            reason = "PARSE_ERROR"

        if status != "OK":
            log_path = str(failures_dir / f"{ds}.log")
            Path(log_path).write_text(
                f"dataset: {ds}\nmethod: {method}\nretcode: {retcode}\nreason: {reason}\n\n--- output ---\n{output}\n",
                encoding="utf-8",
                errors="replace",
            )

        rows.append(
            Row(
                dataset=ds,
                status=status,
                triangles=triangles,
                kernel_ms=kernel_ms,
                e2e_after_clean_ms=e2e_after_clean_ms,
                wall_ms=wall_ms,
                cleaning_ms=cleaning_ms,
                build_ms=build_ms,
                retcode=retcode,
                reason=reason,
                log_path=log_path,
            )
        )
        write_csv(csv_path, rows)
        print(f"{method:10s} {ds:20s} -> {status}")

        if not keep_work:
            shutil.rmtree(work_ds, ignore_errors=True)

    # Remove empty baseline work dir if keep_work is false.
    if not keep_work:
        top = run_dir / "work" / safe_method_filename(method)
        if top.exists() and not any(top.iterdir()):
            top.rmdir()
        # If the method produced no failures, drop the empty dir to keep results tidy.
        if failures_dir.exists() and not any(failures_dir.iterdir()):
            failures_dir.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default=str(PROJECT_ROOT / "results" / "csv1"),
        help="Run directory (default: results/csv1).",
    )
    ap.add_argument("--timeout-sec", type=int, default=300, help="Per-command timeout.")
    ap.add_argument("--datasets", default="", help="Comma-separated dataset names (stem, without .mtx).")
    ap.add_argument("--datasets-file", default="", help="File with one dataset name per line (without .mtx).")
    ap.add_argument("--iters", type=int, default=1, help="Iters for TC-Compare baselines (default: 1).")
    ap.add_argument("--keep-work", action="store_true", help="Keep per-dataset converted inputs under run_dir/work/.")
    ap.add_argument(
        "--suite",
        default="btc",
        choices=sorted(METHOD_SUITES.keys()),
        help="Which fixed method suite to run (default: btc).",
    )
    ap.add_argument(
        "--methods",
        default="",
        help=("Comma-separated method names to run (overrides --suite). Choices: " + ",".join(sorted(BINARIES.keys()))),
    )
    ap.add_argument(
        "--method",
        default="",
        help=("Run only one method (overrides --methods/--suite). Choices: " + ",".join(sorted(BINARIES.keys()))),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = dataset_list(args)
    if not ds:
        raise SystemExit("No datasets selected.")

    if args.method.strip():
        m = args.method.strip()
        if m not in BINARIES:
            raise SystemExit(f"Unknown method: {m}. Choices: {sorted(BINARIES.keys())}")
        methods = [m]
    elif args.methods.strip():
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
        unknown = [m for m in methods if m not in BINARIES]
        if unknown:
            raise SystemExit(f"Unknown methods: {unknown}. Choices: {sorted(BINARIES.keys())}")
    else:
        methods = list(METHOD_SUITES[args.suite])

    print(f"Run dir: {run_dir}")
    print(f"Methods: {methods}")

    for m in methods:
        run_one_baseline(
            method=m,
            run_dir=run_dir,
            datasets=ds,
            timeout_s=int(args.timeout_sec),
            iters=int(args.iters),
            keep_work=bool(args.keep_work),
        )

    # Remove empty top-level dirs when we delete per-dataset work dirs (default).
    if not args.keep_work:
        work_root = run_dir / "work"
        if work_root.exists() and not any(work_root.iterdir()):
            work_root.rmdir()
    failures_root = run_dir / "failures"
    if failures_root.exists() and not any(failures_root.iterdir()):
        failures_root.rmdir()

    # ---- Coverage summary + exit code -------------------------------------------------
    # A silent success here is how a broken run (e.g. baseline binaries linked against a
    # CUDA the driver rejects, so every row CRASHes) can still look "Done". Print a
    # per-method OK-count and fail the process when results are missing:
    #   * a core BTC method that did not cover every dataset, or
    #   * any requested method that produced ZERO successful rows (a total collapse).
    # A partial baseline shortfall (e.g. one crash/timeout on a huge graph) is reported
    # but NOT fatal.
    import csv as _csv
    csv_dir = run_dir / "csv"
    n_ds = len(ds)
    core = set(BTC_METHODS)
    print("\n=== Coverage summary (OK rows / datasets) ===")
    incomplete_core: list[tuple[str, int]] = []
    collapsed: list[str] = []
    for m in methods:
        p = csv_dir / f"{m}.csv"
        ok = 0
        if p.exists():
            with open(p, newline="") as fh:
                for row in _csv.DictReader(fh):
                    if (row.get("Status") or "").strip().upper() == "OK":
                        ok += 1
        flag = ""
        if ok == 0:
            collapsed.append(m); flag = "  <-- 0 OK (broken?)"
        elif m in core and ok < n_ds:
            incomplete_core.append((m, ok)); flag = "  <-- CORE incomplete"
        elif ok < n_ds:
            flag = f"  (partial: {n_ds - ok} missing)"
        print(f"  {m:24s} {ok:3d}/{n_ds}{flag}")

    print(f"\nDone. CSVs under: {csv_dir}")
    rc = 0
    if collapsed:
        print(f"FAIL: methods produced zero successful rows (likely a build/runtime break): {collapsed}")
        rc = 1
    if incomplete_core:
        print(f"FAIL: core BTC method(s) did not cover all {n_ds} datasets: {incomplete_core}")
        rc = 1
    if rc == 0:
        print("OK: every core method is complete and no method fully collapsed.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
