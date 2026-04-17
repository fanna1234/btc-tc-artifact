#pragma once

#include "bcsr_16x128.h"
#include <cuda_runtime.h>

namespace btc {

// V5 kernel (formerly V6): Optimized version with no binary search (O(1) row lookup)
__global__ void kernel_16x128_mma_v5_no_binsearch(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    const uint32_t* __restrict__ row_indices,
    uint32_t num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int WARP_SIZE = 32;
    constexpr int COLS_U32 = 4;      // 128 columns / 32 bits = 4 uint32 per row
    constexpr int SIZE_U32 = 64;     // 16 rows × 4 uint32 = 64 uint32 per block

    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id_global >= (int)num_sample_blocks) return;

    uint32_t sample_idx = (uint32_t)warp_id_global;

    // Direct row lookup (No Binary Search)
    int I = (int)row_indices[sample_idx];

    uint32_t J_L = indices[sample_idx];
    const uint32_t* S_ptr = blocks + (size_t)sample_idx * SIZE_U32;
    uint32_t start_I = indptr[I];
    uint32_t end_I = indptr[I + 1];

    // Step 1: Calculate sparsity using redux.sync
    // Each thread reads multiple uint32 values and counts bits
    int local_nnz = 0;
    for (int i = lane_id; i < SIZE_U32; i += WARP_SIZE) {
        local_nnz += __popc(S_ptr[i]);
    }

    // Warp-level reduction using redux.sync (sm_80+)
    int total_nnz;
    asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(total_nnz) : "r"(local_nnz));

    // Step 2: Decide execution path based on sparsity
    // Threshold: 512 out of 2048 bits = 25% density (optimal after testing)
    bool use_dense = (total_nnz > 512);

    uint32_t total_sum = 0;

    if (use_dense) {
        // ========== Dense Path: Use Bit Tensor Core MMA ==========
        int groupID = lane_id >> 2;           // 0-7
        int threadID_in_group = lane_id & 3;  // 0-3

        // Iterate over 8 J_out blocks (128 columns = 8 × 16-column chunks)
        for (int iter = 0; iter < 8; iter++) {
            int J_out = (int)J_L * 8 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            int32_t c_accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            // Two-pointer merge to find matching K blocks
            int ptr_A = start_I, ptr_B = j_start;
            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A];
                uint32_t K_B = indices[ptr_B];

                if (K_A < K_B) {
                    ptr_A++;
                } else if (K_A > K_B) {
                    ptr_B++;
                } else {
                    // Matching K block found
                    const uint32_t* A_blk = blocks + (size_t)ptr_A * SIZE_U32;
                    const uint32_t* B_blk = blocks + (size_t)ptr_B * SIZE_U32;

                    uint32_t a_frag[2];
                    a_frag[0] = A_blk[groupID * COLS_U32 + threadID_in_group];
                    a_frag[1] = A_blk[(groupID + 8) * COLS_U32 + threadID_in_group];

                    uint32_t b_frag[2];
                    b_frag[0] = B_blk[groupID * COLS_U32 + threadID_in_group];
                    b_frag[1] = B_blk[(groupID + 8) * COLS_U32 + threadID_in_group];

                    // MMA computation
                    asm volatile(
                        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                        : "=r"(c_accum[0]), "=r"(c_accum[1]), "=r"(c_accum[2]), "=r"(c_accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(b_frag[0]),
                          "r"(c_accum[0]), "r"(c_accum[1]), "r"(c_accum[2]), "r"(c_accum[3]));

                    asm volatile(
                        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                        : "=r"(c_accum[4]), "=r"(c_accum[5]), "=r"(c_accum[6]), "=r"(c_accum[7])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(b_frag[1]),
                          "r"(c_accum[4]), "r"(c_accum[5]), "r"(c_accum[6]), "r"(c_accum[7]));

                    ptr_A++;
                    ptr_B++;
                }
            }

            // Apply mask and accumulate
            int rows[8] = {groupID, groupID, groupID + 8, groupID + 8,
                           groupID, groupID, groupID + 8, groupID + 8};
            int cols[8] = {threadID_in_group * 2, threadID_in_group * 2 + 1,
                           threadID_in_group * 2, threadID_in_group * 2 + 1,
                           threadID_in_group * 2 + 8, threadID_in_group * 2 + 9,
                           threadID_in_group * 2 + 8, threadID_in_group * 2 + 9};

            #pragma unroll
            for (int f = 0; f < 8; f++) {
                int row = rows[f];
                int col = cols[f];
                if (row >= 16 || col >= 16) continue;
                if (I * 16 + row >= n) continue;

                int bit_col = iter * 16 + col;
                int u32_idx = bit_col / 32;
                int bit_idx = bit_col % 32;
                if ((S_ptr[row * COLS_U32 + u32_idx] >> bit_idx) & 1) {
                    total_sum += c_accum[f];
                }
            }
        }
    } else {
        // ========== Sparse Path: Use CUDA Core + __ffs traversal ==========
        int my_row = lane_id / 2;        // 0-15 (each of 16 rows)
        int my_col_half = lane_id & 1;   // 0 or 1 (left or right half)

        // Iterate over 8 J_out blocks
        for (int iter = 0; iter < 8; iter++) {
            int J_out = (int)J_L * 8 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            // Get the mask for this thread's row and column range
            // Each thread handles 8 columns (16 columns per iter / 2 halves)
            int col_offset = iter * 16 + my_col_half * 8;
            int u32_idx = col_offset / 32;
            int bit_offset = col_offset % 32;

            uint32_t row_data = S_ptr[my_row * COLS_U32 + u32_idx];
            uint32_t my_mask = (row_data >> bit_offset) & 0xFF;

            if (my_mask == 0) continue;  // Skip if no work for this thread

            // Two-pointer merge
            int ptr_A = start_I, ptr_B = j_start;
            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A];
                uint32_t K_B = indices[ptr_B];

                if (K_A < K_B) {
                    ptr_A++;
                    continue;
                }
                if (K_A > K_B) {
                    ptr_B++;
                    continue;
                }

                // Matching K block found
                const uint32_t* A_blk = blocks + (size_t)ptr_A * SIZE_U32;
                const uint32_t* B_blk = blocks + (size_t)ptr_B * SIZE_U32;

                // Use __ffs to traverse only set bits in the mask
                uint32_t mask = my_mask;
                while (mask != 0) {
                    int c = __ffs(mask) - 1;  // Find first set bit (0-7)
                    int col = my_col_half * 8 + c;  // 0-15, row index in block B

                    // Compute popcount across all 4 uint32 values in the row
                    uint32_t sum = 0;
                    #pragma unroll
                    for (int u = 0; u < COLS_U32; u++) {
                        uint32_t A_val = A_blk[my_row * COLS_U32 + u];
                        uint32_t B_val = B_blk[col * COLS_U32 + u];
                        sum += __popc(A_val & B_val);
                    }
                    total_sum += sum;
                    mask &= (mask - 1);  // Clear the lowest set bit
                }

                ptr_A++;
                ptr_B++;
            }
        }
    }

    // Warp-level reduction using redux.sync
    uint32_t reduced_sum;
    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(reduced_sum) : "r"(total_sum));

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)reduced_sum);
    }
}

unsigned long long count_triangles_16x128_v5(BCSR_16x128_Device& d_bcsr, float* kernel_ms = nullptr) {
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Kernel-only timing (excludes D2H memcpy and any host-side work).
    cudaEvent_t k_start, k_end;
    if (kernel_ms) {
        cudaEventCreate(&k_start);
        cudaEventCreate(&k_end);
        cudaEventRecord(k_start);
    }

    kernel_16x128_mma_v5_no_binsearch<<<num_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks,
        d_bcsr.indptr, d_bcsr.indices, d_bcsr.blocks,
        d_bcsr.row_indices,
        d_bcsr.num_blocks,
        d_bcsr.result
    );

    if (kernel_ms) {
        cudaEventRecord(k_end);
    }

    unsigned long long h_result = 0;
    cudaMemcpy(&h_result, d_bcsr.result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    if (kernel_ms) {
        cudaEventSynchronize(k_end);
        cudaEventElapsedTime(kernel_ms, k_start, k_end);
        cudaEventDestroy(k_start);
        cudaEventDestroy(k_end);
    }

    return h_result;
}

} // namespace btc
