#!/bin/bash
# One-click reproduction of all SC26 paper results.
# Usage: bash scripts/reproduce_paper.sh [--quick]
#   --quick: only run BTC-TC + ToT + TRUST (skip other baselines)
# Expected runtime: ~15 min (quick) or ~1-1.5 h (full)
set -uo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

echo "============================================"
echo "  BTC-TC SC26 Paper Reproduction Pipeline"
echo "============================================"
echo ""

# Step 0: Check prerequisites
echo "[Step 0] Checking prerequisites..."
if [ ! -x "./build/apps/btc_tc_lite" ]; then
    echo "ERROR: ./build/apps/btc_tc_lite not found. Run: bash scripts/build_all.sh"
    exit 1
fi
TOT="./build/baselines/ToT-TPDS25/apps/tot"
[ ! -x "$TOT" ] && TOT="./baselines/ToT-TPDS25/build/apps/tot"
if [ ! -x "$TOT" ]; then
    echo "WARNING: ToT binary not found (will skip ToT comparisons)"
fi
python3 -c "import matplotlib, pandas, numpy" 2>/dev/null || {
    echo "ERROR: Python dependencies missing. Run: pip install -r requirements.txt"
    exit 1
}
echo "  Prerequisites OK."
echo ""

# Step 1: Smoke test
echo "[Step 1] Running smoke test..."
bash scripts/smoke_test.sh || { echo "Smoke test failed. Aborting."; exit 1; }
echo ""

# Step 2: Benchmark
echo "[Step 2] Running benchmark..."
RESULT_DIR="results-reproduce"
mkdir -p "$RESULT_DIR/csv"

BENCH_RC=0
if [ "$QUICK" -eq 1 ]; then
    echo "  Quick mode: BTC-TC + ToT + TRUST only"
    python3 scripts/bench_baselines.py \
        --methods BTC_Lite,BTC_16x128_Adaptive,BTC_16x32_Adaptive,ToT,TRUST \
        --run-dir "$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/bench.log"
    BENCH_RC=${PIPESTATUS[0]}
else
    echo "  Full mode: all 13 baselines"
    python3 scripts/bench_baselines.py \
        --suite all \
        --run-dir "$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/bench.log"
    BENCH_RC=${PIPESTATUS[0]}
fi
[ "$BENCH_RC" -ne 0 ] && echo "  [!] benchmark reported missing/failed results (exit $BENCH_RC) — see the coverage summary above"
echo ""

# Step 3: Verify key claims
echo "[Step 3] Verifying key claims..."
python3 -c "
import pandas as pd, numpy as np, sys, os

result_dir = '$RESULT_DIR/csv'
if not os.path.isdir(result_dir):
    print('  ERROR: result dir not found — the benchmark produced no output')
    sys.exit(1)

def load(method):
    p = os.path.join(result_dir, f'{method}.csv')
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    return df[df['Status']=='OK']

btc = load('BTC_Lite')
tot = load('ToT')

if btc is None or len(btc) == 0:
    print('  ERROR: BTC_Lite results missing/empty — the core method did not run')
    sys.exit(1)

# Correctness: Status = process success (Run); Exact = Triangles vs the exact TC method (BTC-TC)
print(f'  Run (completed): BTC-TC {len(btc)}/36')
if len(btc) < 36:
    print(f'  ERROR: BTC-TC covered only {len(btc)}/36 datasets (expected 36) — reproduction incomplete')
    sys.exit(1)
if tot is not None:
    _m = btc.merge(tot, on='Dataset', suffixes=('_btc','_tot'))
    _exact = int((_m['Triangles_btc'].astype(str) == _m['Triangles_tot'].astype(str)).sum())
    print(f'  Run (completed): ToT {len(tot)}/36')
    print(f'  Exact triangle match vs BTC-TC: ToT {_exact}/{len(_m)} '
          f'(BTC-TC is exact; cross-check vs CPU LAGraph via CLAIMS.md)')

# Kernel speedup
if tot is not None:
    merged = btc.merge(tot, on='Dataset', suffixes=('_btc','_tot'))
    merged['Kernel_ms_btc'] = pd.to_numeric(merged['Kernel_ms_btc'], errors='coerce')
    merged['Kernel_ms_tot'] = pd.to_numeric(merged['Kernel_ms_tot'], errors='coerce')
    valid = merged.dropna(subset=['Kernel_ms_btc','Kernel_ms_tot'])
    if len(valid) > 0:
        gm_btc = np.exp(np.mean(np.log(valid['Kernel_ms_btc'])))
        gm_tot = np.exp(np.mean(np.log(valid['Kernel_ms_tot'])))
        speedup = gm_tot / gm_btc
        print(f'  Kernel speedup vs ToT: {speedup:.2f}x (expect ~1.9x)')

print('  Verification complete.')
"
VERIFY_RC=$?
echo ""

# Step 4: Ablation experiments (Fig 9)
echo "[Step 4] Running ablation experiments (Fig 9)..."
if [ "$QUICK" -eq 0 ]; then
    bash scripts/run_ablation.sh 2>&1 | grep -E "^=|OK|FAIL|Saved" | tail -10
else
    echo "  Skipped in quick mode"
fi
echo ""

# Step 5: Block-size and reorder experiments (Fig 9)
echo "[Step 5] Running block-size bench + reorder compare (Fig 9)..."
if [ "$QUICK" -eq 0 ]; then
    bash scripts/run_blocksize_bench_paper37.sh 2>&1 | tail -5
    bash scripts/run_reorder_compare.sh 2>&1 | tail -5
else
    echo "  Skipped in quick mode"
fi
echo ""

# Step 6: Tau sensitivity sweep + E2E breakdown (Fig 6)
echo "[Step 6] Running tau sweep + E2E breakdown (Fig 6)..."
if [ "$QUICK" -eq 0 ]; then
    bash scripts/run_tau_sweep.sh both 2>&1 | tail -5
    bash scripts/run_e2e_breakdown.sh 2>&1 | tail -5
else
    echo "  Skipped in quick mode"
fi
echo ""

# Step 7: Generate figures
echo "[Step 7] Generating figures..."
export BTC_CSV_DIR="$RESULT_DIR/csv"
bash scripts/regenerate_all_figures.sh 2>&1 | grep -E "^\[|Done|Error"
echo ""

echo "============================================"
if [ "${VERIFY_RC:-0}" -ne 0 ] || [ "${BENCH_RC:-0}" -ne 0 ]; then
    echo "  Reproduction INCOMPLETE — core verification or benchmark reported failures"
    echo "  (verification rc=${VERIFY_RC:-0}, benchmark rc=${BENCH_RC:-0}); see errors/summary above"
    echo "  Results: $RESULT_DIR/   Figures: results/figures/"
    echo "============================================"
    exit 1
fi
echo "  Reproduction complete!"
echo "  Results: $RESULT_DIR/"
echo "  Figures: results/figures/"
echo "============================================"
