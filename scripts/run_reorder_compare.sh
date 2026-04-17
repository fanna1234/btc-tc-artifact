#!/bin/bash
# Compare BTC_REORDER=0 vs BTC_REORDER=8 (GPU HashOrder) on 42 datasets
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${PROJECT_ROOT}/build/apps/btc_tc_lite"
DATA="${PROJECT_ROOT}/data"
OUTDIR="${PROJECT_ROOT}/results/reorder_compare"
mkdir -p "$OUTDIR"

EXCLUDE="Hamrle3|p2p-Gnutella06|adder_dcop_30|circuit_3|patents_main"

CSV_M0="$OUTDIR/mode0_no_reorder.csv"
CSV_M8="$OUTDIR/mode8_gpu_hashorder.csv"

echo "Dataset,Triangles,Kernel_ms,Preprocess_ms,Blocks,E2E_ms" > "$CSV_M0"
echo "Dataset,Triangles,Kernel_ms,Preprocess_ms,Reorder_ms,Blocks,E2E_ms" > "$CSV_M8"

run_one() {
    local mtx="$1"
    local mode="$2"
    local name
    name=$(basename "$mtx" .mtx)

    local out
    out=$(BTC_REORDER="$mode" BTC_REORDER_NO_SKIP=1 "$BIN" -i "$mtx" 2>&1) || true

    local triangles kernel_ms preprocess_ms e2e_ms blocks reorder_ms
    triangles=$(echo "$out" | grep -oP 'Triangles \(GPU\): \K\d+' || echo "-1")
    kernel_ms=$(echo "$out" | grep -oP 'Compute \(Kernel\): \K[\d.]+' || echo "nan")
    preprocess_ms=$(echo "$out" | grep -oP '\[Preprocessing\] time: \K[\d.]+' || echo "nan")
    e2e_ms=$(echo "$out" | grep -oP '\[Total Time \(Preprocess\+Convert\+Compute\)\] time: \K[\d.]+' || echo "nan")
    blocks=$(echo "$out" | grep -oP 'num_blocks=\K\d+' | tail -1 || echo "0")
    reorder_ms=$(echo "$out" | grep -oP 'HashOrder-GPU.*?(\d+\.\d+)\s*ms' | grep -oP '[\d.]+\s*ms' | head -1 | grep -oP '[\d.]+' || echo "0")

    if [ "$mode" = "0" ]; then
        echo "$name,$triangles,$kernel_ms,$preprocess_ms,$blocks,$e2e_ms"
    else
        echo "$name,$triangles,$kernel_ms,$preprocess_ms,$reorder_ms,$blocks,$e2e_ms"
    fi
}

echo "=== Running mode 0 (no reorder) ==="
for mtx in "$DATA"/*.mtx; do
    name=$(basename "$mtx" .mtx)
    echo "$name" | grep -qE "^($EXCLUDE)$" && continue
    echo -n "  $name ... "
    result=$(run_one "$mtx" 0)
    echo "$result" >> "$CSV_M0"
    echo "done"
done

echo ""
echo "=== Running mode 8 (GPU HashOrder) ==="
for mtx in "$DATA"/*.mtx; do
    name=$(basename "$mtx" .mtx)
    echo "$name" | grep -qE "^($EXCLUDE)$" && continue
    echo -n "  $name ... "
    result=$(run_one "$mtx" 8)
    echo "$result" >> "$CSV_M8"
    echo "done"
done

echo ""
echo "Results saved to:"
echo "  $CSV_M0"
echo "  $CSV_M8"
