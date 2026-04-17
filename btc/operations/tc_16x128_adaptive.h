#pragma once

#include "tc_16x128_mma_v4.h"
#include "tc_16x128_mma_v5.h"
#include <iostream>

namespace btc {

// Adaptive strategy: Dispatch to V4 (BinSearch) or V5 (Direct Lookup) based on problem size.
// Heuristic: Small graphs suffering from memory latency due to low occupancy might prefer V4.
// Large graphs benefit significantly from removing the binary search (V5).
unsigned long long count_triangles_16x128_adaptive(
    BCSR_16x128_Device& d_bcsr,
    int threshold_blocks,
    float* kernel_ms
) {
    // Determine which kernel to use
    if (d_bcsr.num_blocks < threshold_blocks) {
        // Use V4 for small workloads
        return count_triangles_16x128_v4(d_bcsr, kernel_ms);
    } else {
        // Use V5 (formerly V6) for large workloads (most cases)
        return count_triangles_16x128_v5(d_bcsr, kernel_ms);
    }
}

inline unsigned long long count_triangles_16x128_adaptive(BCSR_16x128_Device& d_bcsr, int threshold_blocks = 2048) {
    return count_triangles_16x128_adaptive(d_bcsr, threshold_blocks, nullptr);
}

} // namespace btc
