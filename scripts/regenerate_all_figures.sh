#!/bin/bash
# Regenerate ALL paper figures from result CSVs.
# Usage: bash scripts/regenerate_all_figures.sh
# Requires: pip install -r requirements.txt (matplotlib, pandas, numpy)

set -e
cd "$(dirname "$0")/.."
SCRIPTS=scripts/figures

mkdir -p results/figures

FIG_FAIL=0
# Run one figure script; do NOT pipe into tail (that would return tail's status and
# hide a figure-generation failure under set -e).
gen() {  # gen "<label>" <script.py>
    echo "$1"
    if python3 "$2" > /tmp/btc_fig.log 2>&1; then
        tail -1 /tmp/btc_fig.log
    else
        tail -5 /tmp/btc_fig.log; echo "  ERROR: $2 failed"; FIG_FAIL=1
    fi
}

echo "=== Regenerating all paper figures ==="
gen "[1/6] teaser (Fig 1)..."             "$SCRIPTS/generate_teaser_figure.py"
gen "[2/6] per_dataset_lines (Fig 7)..."  "$SCRIPTS/generate_per_dataset_lines.py"
gen "[3/6] cross_device (Fig 8)..."       "$SCRIPTS/generate_cross_device_box_figure.py"
gen "[4/6] ablation (Fig 9)..."           "$SCRIPTS/generate_ablation_figure.py"
gen "[5/6] tau_e2e_combined (Fig 6)..."   "$SCRIPTS/generate_tau_e2e_combined_figure.py"
gen "[6/6] microarch_profile (Fig 10)..." "$SCRIPTS/generate_microarch_profile_figure.py"

echo ""
if [ "$FIG_FAIL" -ne 0 ]; then
    echo "=== Done WITH ERRORS — one or more figures failed (see above) ==="
    exit 1
fi
echo "=== Done. Figures in results/figures/ ==="
