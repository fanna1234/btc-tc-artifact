#pragma once

#include "tc_16x32_mma_v5.h"
#include "tc_16x32_mma_v6.h"
#include <iostream>

namespace btc {

// Adaptive strategy: Dispatch to V5 (BinSearch) or V6 (Direct Lookup) based on problem size.
// Heuristic: Small graphs suffering from memory latency due to low occupancy might prefer V5.
// Large graphs benefit significantly from removing the binary search (V6).
// Adjusted threshold: 16x32 blocks are smaller, so overhead of V6 (memory lookup) persists longer.
// Based on benchmarks, crossover is around 30k-50k blocks.
unsigned long long count_triangles_16x32_adaptive(
    BCSR_16x32_Device& d_bcsr,
    int threshold_blocks,
    float* kernel_ms
) {
    // Determine which kernel to use
    if (d_bcsr.num_blocks < threshold_blocks) {
        // Use V5 (Binary Search) for small workloads
        return count_triangles_16x32_v5(d_bcsr, kernel_ms);
    } else {
        // Use V6 (Direct Lookup) for large workloads (most cases)
        return count_triangles_16x32_v6(d_bcsr, kernel_ms);
    }
}

inline unsigned long long count_triangles_16x32_adaptive(BCSR_16x32_Device& d_bcsr, int threshold_blocks = 40000) {
    return count_triangles_16x32_adaptive(d_bcsr, threshold_blocks, nullptr);
}

} // namespace btc
