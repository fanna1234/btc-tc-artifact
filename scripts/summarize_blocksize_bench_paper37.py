#!/usr/bin/env python3
"""Summarize btc_blocksize_bench logs for the 37-dataset paper suite.

Parses per-dataset log files and extracts RESULT_CSV lines emitted by
apps/btc_blocksize_bench.cu.

Outputs:
- A single CSV with one row per (dataset, block-shape).
- A short text summary (geomean time + normalized ratios + win counts).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Row:
    dataset: str
    input_file: str
    block: str
    num_blocks: int
    bytes_total: int
    time_ms: float
    triangles: int
    status: str


def _geomean(values: Iterable[float]) -> float:
    xs = [x for x in values if x > 0.0 and math.isfinite(x)]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def parse_result_csv_lines(log_path: Path) -> List[Row]:
    dataset = log_path.stem
    rows: List[Row] = []

    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith("RESULT_CSV,"):
            continue
        parts = line.strip().split(",")
        if len(parts) != 8:
            # Keep the parser strict to avoid silently mis-parsing.
            raise ValueError(f"Bad RESULT_CSV line in {log_path}: {line}")

        _, input_file, block, num_blocks, bytes_total, time_ms, triangles, status = parts
        rows.append(
            Row(
                dataset=dataset,
                input_file=input_file,
                block=block,
                num_blocks=int(num_blocks),
                bytes_total=int(bytes_total),
                time_ms=float(time_ms),
                triangles=int(triangles),
                status=status,
            )
        )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary)

    logs = sorted(p for p in log_dir.glob("*.log") if p.is_file())
    if not logs:
        raise SystemExit(f"No .log files found under: {log_dir}")

    all_rows: List[Row] = []
    for lp in logs:
        all_rows.extend(parse_result_csv_lines(lp))

    # Write CSV.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "input_file", "block", "num_blocks", "bytes_total", "time_ms", "triangles", "status"])
        for r in all_rows:
            w.writerow([r.dataset, r.input_file, r.block, r.num_blocks, r.bytes_total, f"{r.time_ms:.6f}", r.triangles, r.status])

    # Index by dataset/block.
    by_ds: Dict[str, Dict[str, Row]] = {}
    for r in all_rows:
        by_ds.setdefault(r.dataset, {})[r.block] = r

    expected_blocks = ["8x128", "16x128", "16x256"]
    baseline = "16x128"

    # Coverage.
    complete = [d for d, m in by_ds.items() if all(b in m for b in expected_blocks)]
    incomplete = [d for d in sorted(by_ds.keys()) if d not in set(complete)]

    # Geomean times and ratios (only on complete datasets with PASS rows).
    gm_time: Dict[str, float] = {}
    for b in expected_blocks:
        gm_time[b] = _geomean(
            by_ds[d][b].time_ms
            for d in complete
            if by_ds[d][b].status == "PASS" and by_ds[d][baseline].status == "PASS"
        )

    gm_ratio: Dict[str, float] = {}
    base_gm = gm_time.get(baseline, float("nan"))
    for b in expected_blocks:
        if not math.isfinite(base_gm) or base_gm <= 0.0:
            gm_ratio[b] = float("nan")
        else:
            gm_ratio[b] = gm_time[b] / base_gm if math.isfinite(gm_time[b]) else float("nan")

    # Win counts (fastest time among blocks, per dataset).
    win_counts = {b: 0 for b in expected_blocks}
    for d in complete:
        rows = [by_ds[d][b] for b in expected_blocks]
        rows = [r for r in rows if r.status == "PASS" and r.time_ms > 0.0]
        if len(rows) != len(expected_blocks):
            continue
        best = min(rows, key=lambda r: r.time_ms)
        win_counts[best.block] += 1

    # Summary.
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with out_summary.open("w") as f:
        f.write("Blocksize Bench Summary (paper 37 datasets)\n")
        f.write(f"Generated: {now}\n")
        f.write(f"Log dir: {os.path.abspath(log_dir)}\n")
        f.write(f"Rows: {len(all_rows)}\n")
        f.write(f"Datasets (with any rows): {len(by_ds)}\n")
        f.write(f"Datasets (complete {expected_blocks}): {len(complete)}\n")
        if incomplete:
            f.write("Incomplete datasets:\n")
            for d in incomplete:
                have = sorted(by_ds[d].keys())
                f.write(f"  {d}: {have}\n")
        f.write("\nGeometric-mean kernel time (ms):\n")
        for b in expected_blocks:
            v = gm_time[b]
            if math.isfinite(v):
                f.write(f"  {b}: {v:.6f}  (ratio vs {baseline}: {gm_ratio[b]:.3f}x)\n")
            else:
                f.write(f"  {b}: nan\n")
        f.write("\nWin counts (fastest among blocks, per dataset):\n")
        for b in expected_blocks:
            f.write(f"  {b}: {win_counts[b]}/{len(complete)}\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
