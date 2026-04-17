#!/bin/bash
# Run τ threshold sensitivity sweep on representative datasets
# Usage: bash scripts/run_tau_sweep.sh [128|32|both]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
BIN="${PROJECT_ROOT}/build/apps/tau_sweep"
OUT_DIR="${PROJECT_ROOT}/results/tau_sweep"

mkdir -p "$OUT_DIR"

MODE="${1:-both}"

# Representative datasets (8 graphs from the 36-graph paper suite):
#   Small: wiki-Vote (8K V, social), g7jac140sc (41K V, scientific)
#   Medium: consph (83K V, FEM), pwtk (218K V, FEM), cant (62K V, FEM)
#   Large: F1 (344K V, FEM), eu-2005 (863K V, web), Ga41As41H72 (268K V, scientific)
datasets=(
    "wiki-Vote.mtx"
    "g7jac140sc.mtx"
    "consph.mtx"
    "cant.mtx"
    "pwtk.mtx"
    "F1.mtx"
    "eu-2005.mtx"
    "Ga41As41H72.mtx"
)

run_sweep() {
    local block_type=$1
    local outfile="$OUT_DIR/tau_sweep_${block_type}.csv"

    echo "=== τ sweep for 16x${block_type} ==="
    # Write header once
    echo "dataset,block_type,tau,kernel_ms,triangles,correct" > "$outfile"

    for dataset in "${datasets[@]}"; do
        if [ ! -f "$DATA_DIR/$dataset" ]; then
            echo "SKIP: $DATA_DIR/$dataset not found"
            continue
        fi
        echo "  Running: $dataset (16x${block_type})"
        # Skip header and non-CSV lines
        $BIN "$DATA_DIR/$dataset" "$block_type" 2>/dev/null | grep -E "^[a-zA-Z0-9_.-]+,(128|32)," >> "$outfile"
    done

    # Also produce a clean version (no stale data)
    local cleanfile="$OUT_DIR/tau_sweep_${block_type}_clean.csv"
    cp "$outfile" "$cleanfile"
    echo "  Output: $outfile"
}

if [ "$MODE" = "128" ] || [ "$MODE" = "both" ]; then
    run_sweep 128
fi

if [ "$MODE" = "32" ] || [ "$MODE" = "both" ]; then
    run_sweep 32
fi

echo "=== Done ==="
