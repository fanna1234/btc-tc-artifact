#!/bin/bash
# Ablation Study Script for BTC-TC
# Evaluates: (1) Sparse path effectiveness (2) Block size impact

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/results/ablation"
BUILD_DIR="${PROJECT_ROOT}/build/apps"

# Create output directories
mkdir -p "${OUTPUT_DIR}/csv"
mkdir -p "${OUTPUT_DIR}/logs"

# Define methods and binaries
declare -A METHODS=(
    ["V3_16x128_PureTC"]="${BUILD_DIR}/btc_tc_v3"
    ["V5_16x128_Hybrid"]="${BUILD_DIR}/btc_tc_v5_16x128"
    ["V3_16x32_PureTC"]="${BUILD_DIR}/btc_tc_v3_16x32"
    ["V6_16x32_Hybrid"]="${BUILD_DIR}/btc_tc_v6_16x32"
)

echo "=========================================="
echo "BTC-TC Ablation Study"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run each method
for method in "${!METHODS[@]}"; do
    binary="${METHODS[$method]}"
    csv_file="${OUTPUT_DIR}/csv/${method}.csv"

    echo "=========================================="
    echo "Running: ${method}"
    echo "Binary: ${binary}"
    echo "=========================================="

    if [ ! -f "${binary}" ]; then
        echo "ERROR: Binary not found: ${binary}"
        continue
    fi

    # Create CSV header
    echo "Dataset,Triangles,Preprocess_ms,Convert_ms,Kernel_ms,Total_ms" > "${csv_file}"

    # Run on all datasets
    for dataset in "${DATA_DIR}"/*.mtx; do
        dataset_name=$(basename "${dataset}" .mtx)
        log_file="${OUTPUT_DIR}/logs/${method}_${dataset_name}.log"

        echo -n "  ${dataset_name}... "

        # Run benchmark
        if timeout 300s "${binary}" -i "${dataset}" > "${log_file}" 2>&1; then
            # Parse output
            triangles=$(grep "Triangles (GPU):" "${log_file}" | awk '{print $NF}')
            preprocess=$(grep "Preprocessing" "${log_file}" | grep -oP '[\d.]+(?=\s*ms)' | head -1)
            convert=$(grep "Converting to BCSR" "${log_file}" | grep -oP '[\d.]+(?=\s*ms)')
            kernel=$(grep "Counting Triangles" "${log_file}" | grep -oP '[\d.]+(?=\s*ms)')

            # Calculate total
            total=$(echo "${preprocess} + ${convert} + ${kernel}" | bc -l)

            # Write to CSV
            echo "${dataset_name},${triangles},${preprocess},${convert},${kernel},${total}" >> "${csv_file}"

            echo "OK (Kernel: ${kernel}ms)"
        else
            echo "FAILED"
            echo "${dataset_name},0,NaN,NaN,NaN,NaN" >> "${csv_file}"
        fi
    done

    echo "Saved: ${csv_file}"
    echo ""
done

echo "=========================================="
echo "Ablation study completed!"
echo "Results: ${OUTPUT_DIR}/csv/"
echo "=========================================="
