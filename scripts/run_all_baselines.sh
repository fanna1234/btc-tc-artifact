#!/bin/bash
# run_all_baselines.sh - Run ALL 13 baselines on 36 paper datasets
# Outputs individual CSV per method into $OUTDIR
# Usage: bash scripts/run_all_baselines.sh [output_dir]

set -uo pipefail
NRUNS=5
OUTDIR="${1:-results/pro6000/csv}"
mkdir -p "$OUTDIR"

# Paths
BTC="./build/apps/btc_tc_lite"
TOT="./build/baselines/ToT-TPDS25/apps/tot"
POLAK="./baselines/TC-Compare/approach/polak/polak"
TRUST_BIN="./baselines/TC-Compare/approach/TRUST/trust"
TRICORE="./baselines/TC-Compare/approach/tricore/tricore"
FOX="./baselines/TC-Compare/approach/Fox/fox"
GREEN="./baselines/TC-Compare/approach/Green/green"
GROUPTC="./baselines/TC-Compare/approach/GroupTC/grouptc"
BISSON="./baselines/TC-Compare/approach/Bisson/bisson"
HU="./baselines/TC-Compare/approach/Hu/hu"
HINDEX="./baselines/TC-Compare/approach/H-INDEX/hindex"
LAGRAPH_CPU="./baselines/LAGraph/build/experimental/benchmark/tcc_demo"
LAGRAPH_GPU="./baselines/LAGraph/build-gpu/experimental/benchmark/tcc_demo"

# Preprocessing
MTX2CSR="./baselines/TC-Compare/preprocessing/cpu_preprocessing/XXX2CSR/MTX2CSR"
CSR2POLAK="./baselines/TC-Compare/preprocessing/cpu_preprocessing/CSR2XXX/CSR2PolakEdgeList"
CSR2HU="./baselines/TC-Compare/preprocessing/cpu_preprocessing/CSR2XXX/CSR2HuEdgeList"
CSR2TRUST_SRC="./baselines/TC-Compare/preprocessing/cpu_preprocessing/CSR2XXX/CSR2TrustCSR.cpp"

# Build CSR2TrustCSR if needed
CSR2TRUST="/tmp/CSR2TrustCSR"
if [ ! -x "$CSR2TRUST" ] && [ -f "$CSR2TRUST_SRC" ]; then
    g++ -O3 -std=c++17 -I./baselines/TC-Compare/preprocessing/cpu_preprocessing/common -o "$CSR2TRUST" "$CSR2TRUST_SRC" 2>/dev/null && echo "Built CSR2TrustCSR"
fi

TMPDIR="/tmp/bench_data_$(hostname)"
mkdir -p "$TMPDIR"

# Paper 36 datasets
DATASETS=(
    shyy41 spaceStation_13 bcsstk23 bcsstm13 g7jac020
    lpl1 net50 msc04515 tandem_vtx delaunay_n17
    ex9 mac_econ_fwd500 torso2 wiki-Vote bcsstk24
    mc2depi dawson5 struct3 g7jac140sc nemeth16
    webbase-1M pli Freescale1 web-NotreDame cage14
    pcrystk03 pkustk06 bcsstk30 cant consph
    pdb1HYS pwtk F1 eu-2005 Si41Ge41H72 Ga41As41H72
)

# Skip BTC and ToT if already done
SKIP_BTC=0; SKIP_TOT=0
[ -f "$OUTDIR/BTC_Lite.csv" ] && [ $(wc -l < "$OUTDIR/BTC_Lite.csv") -gt 35 ] && SKIP_BTC=1 && echo "Skipping BTC (already done)"
[ -f "$OUTDIR/ToT.csv" ] && [ $(wc -l < "$OUTDIR/ToT.csv") -gt 35 ] && SKIP_TOT=1 && echo "Skipping ToT (already done)"

# Helper: run TC-Compare baseline
# Usage: run_tccompare METHOD_NAME BINARY PREPROC_TYPE
run_tccompare() {
    local method=$1 binary=$2 preproc=$3
    local csv="$OUTDIR/${method}.csv"

    if [ ! -x "$binary" ]; then
        echo "SKIP $method (binary not found: $binary)"
        return
    fi

    echo "Dataset,Status,Triangles,Kernel_ms,E2E_after_clean_ms" > "$csv"

    local total=${#DATASETS[@]} idx=0
    for name in "${DATASETS[@]}"; do
        idx=$((idx+1))
        local mtx=""
        for p in "data/${name}.mtx"; do [ -f "$p" ] && mtx="$p" && break; done

        if [ -z "$mtx" ]; then
            echo "$name,MISSING,0,0,0" >> "$csv"
            continue
        fi

        echo -n "  [$idx/$total] $name ... "

        # Preprocess: MTX -> CSR
        local csr_dir="$TMPDIR/${name}_csr"
        mkdir -p "$csr_dir"
        "$MTX2CSR" "$mtx" "$csr_dir/" > /dev/null 2>&1 || true

        if [ ! -s "$csr_dir/adjacent.bin" ]; then
            echo "PREPROC FAIL"
            echo "$name,FAIL,0,0,0" >> "$csv"
            continue
        fi

        local input_file="" kernel_ms="" tri="" status="OK"

        case "$preproc" in
            polak)
                local pdir="$TMPDIR/${name}_polak"; mkdir -p "$pdir"
                "$CSR2POLAK" "$csr_dir/" "$pdir/edges.bin" > /dev/null 2>&1 || true
                input_file="$pdir/edges.bin"
                if [ -f "$input_file" ]; then
                    raw=$("$binary" "$input_file" 0 $NRUNS 2>&1)
                    kernel_ms=$(echo "$raw" | grep "avg kernel use" | grep -oP '[0-9]+\.[0-9]+' | head -1)
                    [ -n "$kernel_ms" ] && kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
                    tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' || echo "$raw" | grep -oP '[0-9]+\s*triangles' | grep -oP '^[0-9]+')
                fi
                ;;
            trust)
                local tdir="$TMPDIR/${name}_trust"; mkdir -p "$tdir"
                if [ -x "$CSR2TRUST" ]; then
                    "$CSR2TRUST" "$csr_dir/" "$tdir/" > /dev/null 2>&1 || true
                fi
                if [ -f "$tdir/adjacent.bin" ]; then
                    raw=$("$binary" "$tdir/" 1 $NRUNS 2>&1)
                    kernel_ms=$(echo "$raw" | grep "avg kernel use" | grep -oP '[0-9]+\.[0-9]+' | head -1)
                    [ -n "$kernel_ms" ] && kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
                    tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' || echo "0")
                fi
                ;;
            hu)
                # Fox, Bisson, Hu, Tricore: CSR -> Hu edge list, then -f edges.bin
                local hdir="$TMPDIR/${name}_hu"; mkdir -p "$hdir"
                "$CSR2HU" "$csr_dir/" "$hdir/" > /dev/null 2>&1 || true
                if [ -f "$hdir/edges.bin" ]; then
                    raw=$("$binary" -f "$hdir/edges.bin" 0 $NRUNS 2>&1)
                    kernel_ms=$(echo "$raw" | grep "avg kernel use" | grep -oP '[0-9]+\.[0-9]+' | head -1)
                    [ -n "$kernel_ms" ] && kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
                    tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' || echo "0")
                fi
                ;;
            green)
                # Green: CSR dir with special params: dir/ 0 256 128 8 1
                raw=$("$binary" "$csr_dir/" 0 256 128 8 1 2>&1)
                kernel_ms=$(echo "$raw" | grep "avg kernel use" | grep -oP '[0-9]+\.[0-9]+' | head -1)
                [ -n "$kernel_ms" ] && kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
                tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' || echo "0")
                ;;
            hindex)
                # HIndex: CSR dir with 10 params: dir/ 1 256 64 64 1 1 0 NRUNS
                raw=$("$binary" "$csr_dir/" 1 256 64 64 1 1 0 $NRUNS 2>&1)
                kernel_ms=$(echo "$raw" | grep "avg kernel use" | grep -oP '[0-9]+\.[0-9]+' | head -1)
                [ -n "$kernel_ms" ] && kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
                tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' || echo "0")
                ;;
            grouptc)
                # GroupTC: CSR dir directly: dir/ 0 NRUNS
                raw=$("$binary" "$csr_dir/" 0 $NRUNS 2>&1)
                kernel_ms=$(echo "$raw" | grep "avg kernel use" | grep -oP '[0-9]+\.[0-9]+' | head -1)
                [ -n "$kernel_ms" ] && kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
                tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' || echo "0")
                ;;
        esac

        if [ -n "$kernel_ms" ]; then
            echo "$name,$status,${tri:-0},$kernel_ms,0" >> "$csv"
            echo "${kernel_ms}ms"
        else
            echo "$name,FAIL,0,0,0" >> "$csv"
            echo "FAIL"
        fi
    done
    echo "  -> $csv"
}

echo "=== Running all baselines on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Output: $OUTDIR"
echo ""

# Run TC-Compare baselines
echo "--- Polak ---"
run_tccompare Polak "$POLAK" polak

echo "--- TRUST ---"
run_tccompare TRUST "$TRUST_BIN" trust

echo "--- GroupTC ---"
run_tccompare GroupTC "$GROUPTC" grouptc

echo "--- Bisson ---"
run_tccompare Bisson "$BISSON" hu

echo "--- Green ---"
run_tccompare Green "$GREEN" green

echo "--- Tricore ---"
run_tccompare Tricore "$TRICORE" hu

echo "--- Hu ---"
run_tccompare Hu "$HU" hu

echo "--- HIndex ---"
run_tccompare HIndex "$HINDEX" hindex

echo "--- Fox ---"
run_tccompare Fox "$FOX" hu

# LAGraph CPU
echo "--- LAGraph-CPU ---"
if [ -x "$LAGRAPH_CPU" ]; then
    csv="$OUTDIR/LAGraph.csv"
    echo "Dataset,Status,Triangles,Kernel_ms,E2E_after_clean_ms" > "$csv"
    idx=0
    for name in "${DATASETS[@]}"; do
        idx=$((idx+1))
        mtx="data/${name}.mtx"
        [ -f "$mtx" ] || { echo "$name,MISSING,0,0,0" >> "$csv"; continue; }
        echo -n "  [$idx/${#DATASETS[@]}] $name ... "
        raw=$("$LAGRAPH_CPU" "$mtx" 2>&1)
        kernel_ms=$(echo "$raw" | grep -oP 'nthreads.*time:\s*\K[0-9.]+' | head -1)
        tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' | head -1)
        if [ -n "$kernel_ms" ]; then
            # LAGraph reports in seconds, convert to ms
            kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
            echo "$name,OK,${tri:-0},$kernel_ms,0" >> "$csv"
            echo "${kernel_ms}ms"
        else
            echo "$name,FAIL,0,0,0" >> "$csv"
            echo "FAIL"
        fi
    done
fi

# LAGraph GPU
echo "--- LAGraph-GPU ---"
if [ -x "$LAGRAPH_GPU" ]; then
    csv="$OUTDIR/LAGraph-gpu.csv"
    echo "Dataset,Status,Triangles,Kernel_ms,E2E_after_clean_ms" > "$csv"
    idx=0
    for name in "${DATASETS[@]}"; do
        idx=$((idx+1))
        mtx="data/${name}.mtx"
        [ -f "$mtx" ] || { echo "$name,MISSING,0,0,0" >> "$csv"; continue; }
        echo -n "  [$idx/${#DATASETS[@]}] $name ... "
        raw=$("$LAGRAPH_GPU" "$mtx" 2>&1)
        kernel_ms=$(echo "$raw" | grep -oP 'nthreads.*time:\s*\K[0-9.]+' | head -1)
        tri=$(echo "$raw" | grep -oP 'triangles:\s*\K[0-9]+' | head -1)
        if [ -n "$kernel_ms" ]; then
            kernel_ms=$(echo "$kernel_ms" | awk '{printf "%.4f", $1*1000}')
            echo "$name,OK,${tri:-0},$kernel_ms,0" >> "$csv"
            echo "${kernel_ms}ms"
        else
            echo "$name,FAIL,0,0,0" >> "$csv"
            echo "FAIL"
        fi
    done
fi

echo ""
echo "=== ALL BASELINES DONE ==="
echo "Results in $OUTDIR/"
ls -la "$OUTDIR/"*.csv | awk '{print $NF, $5}'
