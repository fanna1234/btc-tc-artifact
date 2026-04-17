#!/usr/bin/env bash
#
# Minimal Nsight Compute (ncu) runner to collect KPI metrics (bandwidth/compute/tensor) and
# export a clean CSV for later plotting/analysis.
#
# Usage examples:
#   bash scripts/ncu_profile_metrics.sh --method btc128 --dataset data/web-Google.mtx --tag H100
#   bash scripts/ncu_profile_metrics.sh --method tot    --dataset data/web-Google.mtx --tag H100
#
# Notes:
# - This script must be run on a machine where the NVIDIA driver is accessible to Nsight Compute.
# - Output is saved under: results/ncu/<tag>_<method>_<dataset>_<timestamp>.raw.csv
#
set -euo pipefail

METHOD="btc128"
DATASET="data/web-Google.mtx"
TAG=""
OUT_DIR="results/ncu"
KERNEL=""
TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-application-only}"
LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --kernel) KERNEL="$2"; shift 2;;
    --target-processes) TARGET_PROCESSES="$2"; shift 2;;
    --launch-count) LAUNCH_COUNT="$2"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET" >&2
  exit 1
fi

# Collect the full set of metrics needed for the microarch profile figure.
# Includes: throughput, IPC, warp latency, occupancy, stall breakdown, cache, DRAM.
METRICS_DEFAULT="dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
l1tex__t_sector_hit_rate.pct,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed.avg.per_cycle_active,\
sm__inst_executed.sum.per_cycle_active,\
sm__maximum_warps_avg_per_active_cycle,\
sm__maximum_warps_per_active_cycle_pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
smsp__average_warp_latency_per_inst_issued.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__maximum_warps_avg_per_active_cycle,\
smsp__thread_inst_executed_per_inst_executed.ratio"

NCU_METRICS="${NCU_METRICS:-$METRICS_DEFAULT}"

BIN=""
ARGS=()
case "$METHOD" in
  btc128)
    BIN="./build/apps/btc_tc_adaptive_16x128"
    ARGS=(-i "$DATASET")
    ;;
  btc32)
    BIN="./build/apps/btc_tc_adaptive_16x32"
    ARGS=(-i "$DATASET")
    ;;
  lite)
    BIN="./build/apps/btc_tc_lite"
    ARGS=(-i "$DATASET")
    ;;
  tot)
    BIN="./build/baselines/ToT-TPDS25/apps/tot"
    ARGS=(-i "$DATASET")
    ;;
  trust)
    # TRUST requires preprocessing: mtx -> edges.txt -> undirect -> preprocess -> begin.bin/adjacent.bin
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    TRUST_DIR="${PROJECT_ROOT}/baselines/TRUST"
    TRUST_BIN="${TRUST_DIR}/Without-graph-partition/trianglecounting.bin"
    TRUST_FROM_DIRECT="${TRUST_DIR}/Preprocess/fromDirectToUndirect"
    TRUST_PREPROCESS="${TRUST_DIR}/Preprocess/preprocess"
    BIN="$TRUST_BIN"

    if [[ ! -x "$TRUST_FROM_DIRECT" ]] || [[ ! -x "$TRUST_PREPROCESS" ]]; then
      echo "ERROR: TRUST preprocess binaries not found." >&2
      exit 1
    fi

    # Preprocess into a temp work directory
    TRUST_WORK="$(mktemp -d /tmp/trust_ncu_XXXXXX)"
    trap 'rm -rf "$TRUST_WORK"' EXIT
    echo "[ncu] TRUST preprocessing in $TRUST_WORK ..."

    # mtx -> edges.txt
    python3 -c "
import scipy.io, sys
M = scipy.io.mmread('$DATASET')
from scipy.sparse import issparse, coo_matrix
if issparse(M):
    M = coo_matrix(M)
    with open('$TRUST_WORK/edges.txt', 'w') as f:
        for r, c in zip(M.row, M.col):
            f.write(f'{r} {c}\n')
    print(f'Wrote {M.nnz} edges')
else:
    import numpy as np
    rows, cols = np.nonzero(M)
    with open('$TRUST_WORK/edges.txt', 'w') as f:
        for r, c in zip(rows, cols):
            f.write(f'{r} {c}\n')
    print(f'Wrote {len(rows)} edges')
"
    # undirect + preprocess
    (cd "$TRUST_WORK" && "$TRUST_FROM_DIRECT" edges.txt && "$TRUST_PREPROCESS")

    if [[ ! -f "$TRUST_WORK/begin.bin" ]] || [[ ! -f "$TRUST_WORK/adjacent.bin" ]]; then
      echo "ERROR: TRUST preprocessing failed (begin.bin/adjacent.bin not found)." >&2
      exit 1
    fi
    echo "[ncu] TRUST preprocessing done."
    # TRUST must be launched via mpirun; ncu profiles the child process
    BIN="mpirun"
    ARGS=(-n 1 "$TRUST_BIN" "$TRUST_WORK/" 1 1024 1024 1)
    TARGET_PROCESSES="all"
    ;;
  polak)
    # Polak requires preprocessing: mtx -> CSR (begin.bin/adjacent.bin) -> PolakEdgeList
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    MTX2CSR="${PROJECT_ROOT}/build/baselines/TC-Compare/preprocessing/cpu_preprocessing/MTX2CSR"
    CSR2POLAK="${PROJECT_ROOT}/build/baselines/TC-Compare/preprocessing/cpu_preprocessing/CSR2PolakEdgeList"
    POLAK_BIN="${PROJECT_ROOT}/baselines/TC-Compare/approach/polak/polak"

    if [[ ! -x "$MTX2CSR" ]]; then
      echo "ERROR: MTX2CSR not found: $MTX2CSR" >&2
      exit 1
    fi
    if [[ ! -x "$CSR2POLAK" ]]; then
      echo "ERROR: CSR2PolakEdgeList not found: $CSR2POLAK" >&2
      exit 1
    fi
    if [[ ! -x "$POLAK_BIN" ]]; then
      echo "ERROR: polak binary not found: $POLAK_BIN" >&2
      exit 1
    fi

    POLAK_WORK="$(mktemp -d /tmp/polak_ncu_XXXXXX)"
    trap 'rm -rf "$POLAK_WORK"' EXIT
    echo "[ncu] Polak preprocessing in $POLAK_WORK ..."

    # Step 1: MTX -> CSR
    POLAK_CSR_DIR="$POLAK_WORK/csr"
    mkdir -p "$POLAK_CSR_DIR"
    "$MTX2CSR" "$DATASET" "$POLAK_CSR_DIR/"
    if [[ ! -f "$POLAK_CSR_DIR/begin.bin" ]] || [[ ! -f "$POLAK_CSR_DIR/adjacent.bin" ]]; then
      echo "ERROR: MTX2CSR failed (begin.bin/adjacent.bin not found)." >&2
      exit 1
    fi

    # Step 2: CSR -> PolakEdgeList
    POLAK_EDGES="$POLAK_WORK/polak_edges.bin"
    "$CSR2POLAK" "$POLAK_CSR_DIR/" "$POLAK_EDGES"
    if [[ ! -f "$POLAK_EDGES" ]]; then
      echo "ERROR: CSR2PolakEdgeList failed." >&2
      exit 1
    fi
    echo "[ncu] Polak preprocessing done."

    BIN="$POLAK_BIN"
    ARGS=("$POLAK_EDGES" 0 1)
    ;;
  *)
    echo "ERROR: unknown --method '$METHOD' (use: btc128|btc32|lite|tot|trust|polak)" >&2
    exit 2
    ;;
esac

if [[ ! -x "$BIN" ]] && ! command -v "$BIN" >/dev/null 2>&1; then
  echo "ERROR: binary not found/executable: $BIN" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
ds_name="$(basename "$DATASET")"
ds_stem="${ds_name%.mtx}"
prefix="${OUT_DIR}/"
if [[ -n "$TAG" ]]; then
  prefix+="${TAG}_"
fi
prefix+="${METHOD}_${ds_stem}_${ts}"

report="${prefix}.ncu-rep"
log="${prefix}.ncu_run.log"
csv_raw="${prefix}.raw.csv"
summary_txt="${prefix}.summary.txt"

echo "[ncu] method=$METHOD dataset=$DATASET out=$csv_raw"
echo "[ncu] report=$report"

NCO_ARGS=(
  --metrics "$NCU_METRICS"
  --target-processes "$TARGET_PROCESSES"
  --force-overwrite
  -o "$report"
)
if [[ -n "$KERNEL" ]]; then
  NCO_ARGS+=(-k "$KERNEL")
fi
if [[ -n "$LAUNCH_COUNT" ]]; then
  NCO_ARGS+=(-c "$LAUNCH_COUNT")
fi

set +e
ncu "${NCO_ARGS[@]}" "$BIN" "${ARGS[@]}" >"$log" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "ERROR: ncu failed (exit=$rc). See: $log" >&2
  echo "Tip: If metrics are unsupported, try: NCU_METRICS=... or use --set speedOfLight manually." >&2
  exit $rc
fi

# Export a clean, parseable CSV without application stdout/stderr interleaving.
ncu --import "$report" --csv --page raw >"$csv_raw"
ncu --import "$report" --print-summary per-kernel >"$summary_txt" || true

echo "Wrote:"
echo "  $csv_raw"
echo "  $summary_txt"
echo "  $report"
echo "  $log"
