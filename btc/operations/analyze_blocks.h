#pragma once

#include <btc/btc.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace btc {

// Custom kernel to counting block transitions for both 16x32 and 16x128 in one pass
__global__ void count_blocks_kernel(
    const int* __restrict__ rows,
    const int* __restrict__ cols,
    int num_entries,
    unsigned long long* __restrict__ count_32,
    unsigned long long* __restrict__ count_128)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    unsigned long long local_32 = 0;
    unsigned long long local_128 = 0;

    // Iterate over elements 0 to num_entries - 2
    // We compare i and i+1
    for (int i = idx; i < num_entries - 1; i += stride) {
        int r1 = rows[i];
        int c1 = cols[i];
        int r2 = rows[i + 1];
        int c2 = cols[i + 1];

        // 16x32 Logic: Row/16 or Col/32 changes
        int br1_32 = r1 >> 4;   // div 16
        int bc1_32 = c1 >> 5;   // div 32
        int br2_32 = r2 >> 4;
        int bc2_32 = c2 >> 5;

        // Optimization: Precompute row changed
        bool row_changed = (br1_32 != br2_32);

        if (row_changed || (bc1_32 != bc2_32)) {
            local_32++;
        }

        // 16x128 Logic: Row/16 changes OR Col/128 changes
        if (row_changed) {
            local_128++;
        } else {
            // Only check cols if row didn't change
            int bc1_128 = c1 >> 7;  // div 128
            int bc2_128 = c2 >> 7;
            if (bc1_128 != bc2_128) {
                local_128++;
            }
        }
    }

    // Warp Shuffle Reduction
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_32 += __shfl_down_sync(mask, local_32, offset);
        local_128 += __shfl_down_sync(mask, local_128, offset);
    }

    // Leader of each warp adds to global (reducing contention vs per-thread)
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(count_32, local_32);
        atomicAdd(count_128, local_128);
    }
}

/**
 * @brief Quickly estimates the number of blocks for 16x32 and 16x128 formats 
 *        from a sorted COO matrix.
 * 
 * @param coo The input COO matrix (must be sorted by row, then col).
 * @param num_blocks_32 Output for 16x32 block count.
 * @param num_blocks_128 Output for 16x128 block count.
 */
inline void analyze_block_structure(
    const CooMatrix<int, float, device_memory>& coo,
    size_t& num_blocks_32,
    size_t& num_blocks_128)
{
    if (coo.num_entries == 0) {
        num_blocks_32 = 0;
        num_blocks_128 = 0;
        return;
    }

    // Allocate device memory for counters
    unsigned long long* d_counts;
    cudaMalloc(&d_counts, 2 * sizeof(unsigned long long));
    cudaMemset(d_counts, 0, 2 * sizeof(unsigned long long));

    int blockSize = 256;
    int numBlocks = (coo.num_entries + blockSize - 1) / blockSize;
    if (numBlocks > 1024) numBlocks = 1024; // Limit grid size
    if (numBlocks == 0) numBlocks = 1;

    count_blocks_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(coo.row_indices.data()),
        thrust::raw_pointer_cast(coo.column_indices.data()),
        (int)coo.num_entries,
        d_counts,
        d_counts + 1
    );

    unsigned long long h_counts[2];
    cudaMemcpy(h_counts, d_counts, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_counts);

    // Result is transitions + 1 (first block)
    num_blocks_32 = (size_t)(h_counts[0] + 1);
    num_blocks_128 = (size_t)(h_counts[1] + 1);
}

} // namespace btc
