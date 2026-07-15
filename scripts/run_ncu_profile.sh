#!/bin/bash
# Collect Nsight Compute profiling data for the microarch profile figure (Fig 10).
#
# Usage:
#   bash scripts/run_ncu_profile.sh                    # all 8 datasets, methods btc128+tot
#   bash scripts/run_ncu_profile.sh cant consph         # specific datasets
#
# This re-collects the BTC-TC and ToT rows only (the two methods implemented in
# ncu_profile_metrics.sh). The paper's Fig 10 also shows TRUST and Polak; the
# full four-method figure regenerates from the bundled results/ncu/ data via
# scripts/figures/generate_microarch_profile_figure.py.
#
# Prerequisites:
#   - ncu (Nsight Compute CLI) in PATH
#   - Built BTC-TC + ToT (bash scripts/build_all.sh)
#   - Datasets downloaded (bash scripts/download_datasets.sh)
#
# Output: results/ncu/<tag>_<method>_<dataset>.raw.csv
# These CSVs are consumed by scripts/figures/generate_microarch_profile_figure.py

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not found. Install NVIDIA Nsight Compute or add it to PATH."
    echo "  Typical location: /usr/local/cuda/bin/ncu"
    exit 1
fi

# Auto-detect GPU tag from nvidia-smi (SIGPIPE-safe: capture the full output
# first, then take line 1 via parameter expansion -- piping straight into
# `head -1` makes nvidia-smi SIGPIPE on a multi-GPU host, and under
# `set -e -o pipefail` that silently kills the script before profiling).
_gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
TAG=$(printf '%s' "${_gpu_names%%$'\n'*}" | tr ' ' '_' | tr -d '()' | tr '[:upper:]' '[:lower:]')
[ -z "$TAG" ] && TAG="gpu"

OUT_DIR="results/ncu"
mkdir -p "$OUT_DIR"

# Default: 8 representative datasets used in the paper (Fig 10)
DEFAULT_DATASETS="cant consph pwtk F1 eu-2005 Ga41As41H72 g7jac140sc Si41Ge41H72"

if [ $# -gt 0 ]; then
    DATASETS="$*"
else
    DATASETS="$DEFAULT_DATASETS"
fi

METHODS="btc128 tot"

echo "=== NCU Profiling for Fig 10 ==="
echo "Tag: $TAG"
echo "Datasets: $DATASETS"
echo "Methods: $METHODS"
echo "Output: $OUT_DIR/"
echo ""

for ds in $DATASETS; do
    for method in $METHODS; do
        mtx="data/${ds}.mtx"
        if [ ! -f "$mtx" ]; then
            echo "SKIP $method/$ds: $mtx not found"
            continue
        fi
        echo -n "$method on $ds ... "
        if bash scripts/ncu_profile_metrics.sh \
            --method "$method" \
            --dataset "$mtx" \
            --tag "$TAG" \
            --out-dir "$OUT_DIR" \
            2>/dev/null; then
            echo "OK"
        else
            echo "FAIL (ncu may need root/sudo or --target-processes all)"
        fi
    done
done

echo ""
echo "=== Done ==="
echo "To regenerate Fig 10:"
echo "  python3 scripts/figures/generate_microarch_profile_figure.py"
echo ""
echo "Note: the plotter auto-discovers any <tag>_<method>_<dataset>*.raw.csv, so"
echo "  freshly collected files (tag '${TAG}') are picked up with no renaming needed."
