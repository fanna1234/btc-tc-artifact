#!/bin/bash
# Build BTC-TC and ALL baselines from source.
# Usage: bash scripts/build_all.sh
# Prerequisites: CUDA >= 12.1 (>= 12.8 for Blackwell sm_120), GCC >= 11 (g++-12 preferred
#   for the TC-Compare baselines; auto-detected below), CMake >= 3.22, Boost (libboost-all-dev,
#   for the rabbit_order vertex-reordering headers), MPI (for TRUST)
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

# Ensure nvcc is reachable even from a non-login shell (some images, e.g. Chameleon's
# CC-Ubuntu-CUDA, keep CUDA off the default PATH). Harmless if nvcc is already found.
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH=/usr/local/cuda/bin:$PATH
fi

echo "============================================"
echo "  BTC-TC Full Build"
echo "============================================"

# Step 1: Main project + ToT + cuSPARSE (CMake)
echo "[1/5] Building main project (BTC-TC + ToT + cuSPARSE)..."
mkdir -p build && cd build
cmake .. 2>&1 | tail -3
make -j$(nproc) 2>&1 | tail -3
cd "$ROOT"
echo "  Done: build/apps/btc_tc_lite, build/baselines/ToT-TPDS25/apps/tot"
echo ""

# Step 2: TC-Compare baselines (individual Makefiles)
echo "[2/5] Building TC-Compare baselines..."
TC_DIR="baselines/TC-Compare/approach"

# Tricore needs a CCCL patch for CUDA >= 13.2 (ADL ambiguity in block_load_to_shared.cuh)
TRICORE_DIR="$TC_DIR/tricore"
if [ -d "$TRICORE_DIR" ]; then
    CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
    CUDA_MINOR=$(nvcc --version 2>/dev/null | grep -oP 'release [0-9]+\.\K[0-9]+' | head -1)
    if [ "${CUDA_MAJOR:-0}" -ge 13 ] && [ "${CUDA_MINOR:-0}" -ge 2 ]; then
        echo "  Applying CCCL patch for Tricore (CUDA >= 13.2)..."
        CCCL_FIX="$TRICORE_DIR/cccl_fix/cub/block"
        mkdir -p "$CCCL_FIX"
        CCCL_INCLUDE=$(nvcc --print-search-dirs 2>/dev/null | grep 'target dir' | awk '{print $NF}' | head -1)
        [ -z "$CCCL_INCLUDE" ] && CCCL_INCLUDE="/usr/local/cuda/targets/x86_64-linux/include"
        SRC_HDR="$CCCL_INCLUDE/cccl/cub/block/block_load_to_shared.cuh"
        if [ -f "$SRC_HDR" ]; then
            cp "$SRC_HDR" "$CCCL_FIX/"
            sed -i 's/= data(smem_dst)/= ::cuda::std::data(smem_dst)/g; s/data(gmem_src)/::cuda::std::data(gmem_src)/g' "$CCCL_FIX/block_load_to_shared.cuh"
        fi
    fi
fi

# Pick a host C++ compiler nvcc accepts. The Makefiles default to g++-12; CUDA 12.x
# also accepts gcc 13, while CUDA >=13 requires gcc <=12. Prefer g++-12, else fall back
# to whatever is installed so a reviewer isn't forced to match an exact compiler.
BL_HOST_CXX=""
for _c in g++-12 g++-13 g++-11 g++; do
    command -v "$_c" >/dev/null 2>&1 && { BL_HOST_CXX="$_c"; break; }
done

for method_dir in "$TC_DIR"/*/; do
    name=$(basename "$method_dir")
    [ "$name" = "tricore" ] && continue  # build Tricore separately below
    if [ -f "$method_dir/Makefile" ]; then
        echo -n "  $name... "
        # Some methods (e.g. Green) write objects into bin/ofiles/ without creating it.
        mkdir -p "$method_dir/bin/ofiles" 2>/dev/null
        # Build from source against THIS host's toolkit (prebuilt binaries are no longer
        # shipped, so make always compiles fresh -> no stale CUDA-runtime mismatch).
        if make -C "$method_dir" HOST_CXX="$BL_HOST_CXX" -j$(nproc) >"/tmp/build_${name}.log" 2>&1; then
            echo "OK"
        else
            echo "FAIL (non-critical; see /tmp/build_${name}.log)"
            tail -3 "/tmp/build_${name}.log" | sed 's/^/      /'
        fi
    fi
done

# Build Tricore with optional CCCL fix
if [ -d "$TRICORE_DIR" ]; then
    echo -n "  tricore... "
    cd "$TRICORE_DIR"
    EXTRA_I=""
    [ -d "cccl_fix" ] && EXTRA_I="-I$(pwd)/cccl_fix"
    nvcc -O3 -std=c++17 $EXTRA_I -Iinclude -rdc=true -dc src/tricount_gpu.cu -o tricount_gpu.o 2>/dev/null && \
    nvcc -O3 -std=c++17 $EXTRA_I -Iinclude -dlink tricount_gpu.o -o gpu.o -lcudadevrt 2>/dev/null && \
    g++ -O3 -std=c++17 -fopenmp -Iinclude -c src/main.cpp -o main.o 2>/dev/null && \
    g++ -O3 -std=c++17 -fopenmp -Iinclude -c src/log.cpp -o log.o 2>/dev/null && \
    CUDA_LIB=$(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc))/../lib64 && \
    g++ -O3 -std=c++17 -fopenmp -Iinclude -o tricore main.o tricount_gpu.o log.o gpu.o -L"$CUDA_LIB" -lcudart -lcudadevrt 2>/dev/null && \
    echo "OK" || echo "FAIL (non-critical)"
    cd "$ROOT"
fi
echo ""

# Step 3: TRUST
echo "[3/5] Building TRUST..."
TRUST_DIR="baselines/TRUST"
if [ -d "$TRUST_DIR/Without-graph-partition" ] && [ -f "$TRUST_DIR/Without-graph-partition/Makefile" ]; then
    echo -n "  TRUST... "
    if make -C "$TRUST_DIR/Without-graph-partition" -j$(nproc) 2>/dev/null; then
        echo "OK"
    else
        echo "FAIL"
    fi
    # Build preprocessing tools
    for tool_dir in "$TRUST_DIR/Preprocess"; do
        if [ -f "$tool_dir/Makefile" ]; then
            make -C "$tool_dir" -j$(nproc) 2>/dev/null && echo "  Preprocess tools OK" || echo "  Preprocess tools FAIL"
        fi
    done
    # Restore exec bits on preprocess scripts. A no-op on normal git clones;
    # a fix when the artifact was extracted from a zip/tarball archive, since
    # such archives may not preserve the executable bit.
    chmod +x "$TRUST_DIR"/Preprocess/{compile.sh,fromDirectToUndirect,partition,preprocess} 2>/dev/null || true
else
    echo "  SKIP (not found)"
fi
echo ""

# Step 4a: SuiteSparse:GraphBLAS (required by LAGraph)
echo "[4/6] Building SuiteSparse:GraphBLAS..."
GRB_DIR="baselines/SuiteSparse-GraphBLAS-cuda"
GRB_LIB="$GRB_DIR/build/libgraphblas.so"
if [ -d "$GRB_DIR" ]; then
    if [ ! -f "$GRB_LIB" ]; then
        echo "  GraphBLAS (~1-2 minutes)..."
        mkdir -p "$GRB_DIR/build" && cd "$GRB_DIR/build"
        if cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3 && make -j$(nproc) 2>&1 | tail -5; then
            echo "  GraphBLAS: OK"
        else
            echo "  GraphBLAS: FAIL (LAGraph will be skipped)"
        fi
        cd "$ROOT"
    else
        echo "  GraphBLAS: already built"
    fi
else
    echo "  SKIP (source not found)"
fi
echo ""

# Step 4b: LAGraph (CPU + GPU)
echo "[5/6] Building LAGraph..."
LA_DIR="baselines/LAGraph"
if [ -d "$LA_DIR" ]; then
    # CPU build
    if [ ! -f "$LA_DIR/build/experimental/benchmark/tcc_demo" ]; then
        echo -n "  LAGraph-CPU... "
        mkdir -p "$LA_DIR/build" && cd "$LA_DIR/build"
        GRB_BUILD="$ROOT/baselines/SuiteSparse-GraphBLAS-cuda/build"
        GRB_INC="$ROOT/baselines/SuiteSparse-GraphBLAS-cuda/Include"
        if cmake .. -DGRAPHBLAS_ROOT="$ROOT/baselines/SuiteSparse-GraphBLAS-cuda" \
                    -DGRAPHBLAS_LIBRARY="$GRB_BUILD/libgraphblas.so" \
                    -DGRAPHBLAS_INCLUDE_DIR="$GRB_INC" \
                    2>/dev/null && make -j$(nproc) 2>/dev/null; then
            echo "OK"
        else
            echo "FAIL"
        fi
        cd "$ROOT"
    else
        echo "  LAGraph-CPU: already built"
    fi
    # GPU build (requires SuiteSparse:GraphBLAS with CUDA)
    if [ ! -f "$LA_DIR/build-gpu/experimental/benchmark/tcc_demo" ]; then
        echo -n "  LAGraph-GPU... "
        if [ -d "$ROOT/baselines/SuiteSparse-GraphBLAS-cuda" ]; then
            mkdir -p "$LA_DIR/build-gpu" && cd "$LA_DIR/build-gpu"
            GRB_BUILD="$ROOT/baselines/SuiteSparse-GraphBLAS-cuda/build"
            if cmake .. -DGRAPHBLAS_ROOT="$ROOT/baselines/SuiteSparse-GraphBLAS-cuda" \
                        -DGRAPHBLAS_LIBRARY="$GRB_BUILD/libgraphblas.so" \
                        -DGRAPHBLAS_INCLUDE_DIR="$ROOT/baselines/SuiteSparse-GraphBLAS-cuda/Include" \
                        2>/dev/null && make -j$(nproc) 2>/dev/null; then
                echo "OK"
            else
                echo "FAIL (non-critical)"
            fi
            cd "$ROOT"
        else
            echo "SKIP (no GraphBLAS-CUDA)"
        fi
    else
        echo "  LAGraph-GPU: already built"
    fi
else
    echo "  SKIP (not found)"
fi
echo ""

# Step 6: TC-Compare preprocessing tools
echo "[6/6] Building TC-Compare preprocessing tools..."
PREPROC_DIR="baselines/TC-Compare/preprocessing/cpu_preprocessing"
for tool in XXX2CSR CSR2XXX; do
    tool_dir="$PREPROC_DIR/$tool"
    if [ -f "$tool_dir/Makefile" ]; then
        echo -n "  $tool... "
        make -C "$tool_dir" -j$(nproc) 2>/dev/null && echo "OK" || echo "FAIL"
    fi
done
echo ""

# Summary
echo "============================================"
echo "  Build Summary"
echo "============================================"
check() { [ -x "$1" ] && echo "  OK  $2" || echo "  --  $2 (missing)"; }
check build/apps/btc_tc_lite "BTC-TC (Lite)"
check build/apps/btc_tc_adaptive_16x128 "BTC-TC (16x128)"
check build/apps/btc_tc_adaptive_16x32 "BTC-TC (16x32)"
check build/baselines/ToT-TPDS25/apps/tot "ToT"
check "$TRUST_DIR/Without-graph-partition/trianglecounting.bin" "TRUST"
check "$PREPROC_DIR/XXX2CSR/MTX2CSR" "MTX2CSR converter"
check "$PREPROC_DIR/CSR2XXX/CSR2HuEdgeList" "CSR2Hu converter"
check "$PREPROC_DIR/CSR2XXX/CSR2PolakEdgeList" "CSR2Polak converter"
for m in Fox Hu Green GroupTC Bisson H-INDEX polak; do
    bin=$(find "$TC_DIR/$m" -maxdepth 1 -type f -executable 2>/dev/null | head -1)
    check "${bin:-/nonexistent}" "$m"
done
check baselines/LAGraph/build/experimental/benchmark/tcc_demo "LAGraph-CPU"
check baselines/LAGraph/build-gpu/experimental/benchmark/tcc_demo "LAGraph-GPU"
echo ""
echo "Next: bash scripts/download_datasets.sh"
echo "Then: bash scripts/reproduce_paper.sh"
