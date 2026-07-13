#!/bin/bash
# BTC-TC SC26 — one-command full reproduction from scratch.
#
# Usage:
#   bash scripts/run_all.sh           # full pipeline, ~1-1.5 h
#   bash scripts/run_all.sh --quick   # core claims only, ~20 min
#   bash scripts/run_all.sh --smoke   # build + correctness only, ~5 min
#
# Chains: OS-deps check -> pip -> datasets -> build -> smoke -> reproduce.
# Exits on any step failure with a clear message.

set -uo pipefail
cd "$(dirname "$0")/.."

MODE="full"
case "${1:-}" in
    --quick)  MODE="quick" ;;
    --smoke)  MODE="smoke" ;;
    -h|--help)
        sed -n '2,10p' "$0"
        exit 0
        ;;
    "") ;;
    *) echo "Unknown arg: $1 (try --quick / --smoke / --help)" >&2; exit 2 ;;
esac

ts(){ date "+[%F %T]"; }

echo "=========================================="
echo "  BTC-TC SC26 One-Command Reproduction"
echo "  Mode: $MODE"
echo "=========================================="

# ---------- Step 0: OS deps ----------
echo
echo "$(ts) [0/5] Checking OS dependencies..."
MISSING=()
for pkg_cmd in \
    "libnuma-dev:numa.h:/usr/include/numa.h" \
    "libboost-all-dev:boost:/usr/include/boost/version.hpp" \
    "bc:bc:$(command -v bc 2>/dev/null || echo '')" \
    "libopenmpi-dev:mpicc:$(command -v mpicc 2>/dev/null || echo '')"; do
    IFS=: read -r pkg probe path <<< "$pkg_cmd"
    if [ -z "$path" ] || [ ! -e "$path" ]; then
        MISSING+=("$pkg")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  [!] Missing OS packages: ${MISSING[*]}"
    echo "  [!] Install with:"
    echo "      sudo apt-get update && sudo apt-get install -y ${MISSING[*]}"
    echo "  [!] (libnuma-dev, libboost-all-dev, bc are required; libopenmpi-dev is optional — TRUST baseline only)"
    if [ -t 0 ]; then
        read -r -p "  Continue without them? (y/N) " ans
        [[ "$ans" =~ ^[Yy]$ ]] || { echo "  Aborted."; exit 1; }
    else
        echo "  Non-interactive shell: install the packages above, then re-run. Aborting."
        exit 1
    fi
else
    echo "  OK (libnuma-dev + libboost-all-dev + bc + libopenmpi-dev present)"
fi

# ---------- Step 1: Python deps ----------
echo
echo "$(ts) [1/5] Installing Python dependencies..."
if python3 -c "import matplotlib, pandas, numpy" >/dev/null 2>&1; then
    echo "  OK (already installed)"
else
    if python3 -m pip install -q -r requirements.txt >/dev/null 2>&1; then
        echo "  OK (installed via pip)"
    elif python3 -m pip install -q --break-system-packages -r requirements.txt >/dev/null 2>&1; then
        echo "  OK (installed via pip --break-system-packages, PEP 668 bypass)"
    else
        echo "  [!] pip install failed. Try manually:"
        echo "      pip install --break-system-packages -r requirements.txt"
        echo "  or use a virtualenv:"
        echo "      python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
fi

# ---------- Step 2: Datasets ----------
echo
echo "$(ts) [2/5] Downloading datasets (~3.2 GB on disk, skippable if present)..."
NEEDED=$(awk 'NF && $1 !~ /^#/ {print $1}' data/paper_datasets.txt 2>/dev/null | wc -l)
HAVE=$(ls data/*.mtx 2>/dev/null | wc -l)
echo "  need=$NEEDED, have=$HAVE"
if [ "$HAVE" -lt "$NEEDED" ]; then
    bash scripts/download_datasets.sh || { echo "  [!] dataset download failed"; exit 1; }
else
    echo "  OK (all datasets already present)"
fi

# ---------- Step 3: Build ----------
echo
echo "$(ts) [3/5] Building BTC-TC + baselines (~3-5 min)..."
echo "  This compiles GraphBLAS from source."
bash scripts/build_all.sh 2>&1 | tail -20
if [ ! -x "./build/apps/btc_tc_lite" ]; then
    echo "  [!] build failed: btc_tc_lite missing"
    exit 1
fi
echo "  OK (main binaries built)"

# ---------- Step 4: Smoke ----------
echo
echo "$(ts) [4/5] Smoke test (<1 min)..."
bash scripts/smoke_test.sh || { echo "  [!] smoke test FAILED — stop here and investigate"; exit 1; }

if [ "$MODE" = "smoke" ]; then
    echo
    echo "$(ts) Smoke-only mode requested. Stopping here."
    echo "  To continue with full reproduction:  bash scripts/reproduce_paper.sh"
    echo "  Or quick mode:                       bash scripts/reproduce_paper.sh --quick"
    exit 0
fi

# ---------- Step 5: Full or Quick reproduction ----------
echo
if [ "$MODE" = "quick" ]; then
    echo "$(ts) [5/5] Quick reproduction (BTC + ToT + TRUST, ~15 min)..."
    bash scripts/reproduce_paper.sh --quick || { echo "  [!] reproduction reported failures (see above) — NOT complete"; exit 1; }
else
    echo "$(ts) [5/5] Full reproduction (all 15 methods x 36 datasets, ~1-1.5 h)..."
    bash scripts/reproduce_paper.sh || { echo "  [!] reproduction reported failures (see above) — NOT complete"; exit 1; }
fi

echo
echo "=========================================="
echo "$(ts)  Reproduction complete."
echo "  Results CSV:  results-reproduce/csv/"
echo "  Figures:      results/figures/"
echo "  See CLAIMS.md for claim-by-claim verification."
echo "=========================================="
