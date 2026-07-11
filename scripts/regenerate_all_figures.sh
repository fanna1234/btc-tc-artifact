#!/bin/bash
# Regenerate ALL paper figures from result CSVs.
# Usage: bash scripts/regenerate_all_figures.sh
# Requires: pip install -r requirements.txt (matplotlib, pandas, numpy)

set -e
cd "$(dirname "$0")/.."
SCRIPTS=scripts/figures

mkdir -p results/figures

echo "=== Regenerating all paper figures ==="

echo "[1/6] teaser (Fig 1)..."
python3 $SCRIPTS/generate_teaser_figure.py 2>&1 | tail -1

echo "[2/6] per_dataset_lines (Fig 7)..."
python3 $SCRIPTS/generate_per_dataset_lines.py 2>&1 | tail -1

echo "[3/6] cross_device (Fig 8)..."
python3 $SCRIPTS/generate_cross_device_box_figure.py 2>&1 | tail -1

echo "[4/6] ablation (Fig 9)..."
python3 $SCRIPTS/generate_ablation_figure.py 2>&1 | tail -1

echo "[5/6] tau_e2e_combined (Fig 6)..."
python3 $SCRIPTS/generate_tau_e2e_combined_figure.py 2>&1 | tail -1

echo "[6/6] microarch_profile (Fig 10)..."
python3 $SCRIPTS/generate_microarch_profile_figure.py 2>&1 | tail -1

echo ""
echo "=== Done. Figures in results/figures/ ==="
