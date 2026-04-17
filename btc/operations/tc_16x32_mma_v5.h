#pragma once

#include "bcsr_16x32.h"
#include <cuda_runtime.h>

namespace btc {

// V5 kernel: True adaptive hybrid - dynamically choose MMA or CUDA Core based on sparsity
__global__ void kernel_16x32_mma_v5_adaptive(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    uint32_t num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int WARP_SIZE = 32;
    constexpr int SIZE_U32 = 16;

    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id_global >= (int)num_sample_blocks) return;

    uint32_t sample_idx = (uint32_t)warp_id_global;

    // Binary search to find row block I
    int I;
    {
        uint32_t lo = 0, hi = n_row_blocks;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (indptr[mid + 1] <= sample_idx) lo = mid + 1;
            else hi = mid;
        }
        I = lo;
    }

    uint32_t J_sample = indices[sample_idx];
    const uint32_t* S_ptr = blocks + (size_t)sample_idx * SIZE_U32;
    uint32_t start_I = indptr[I];
    uint32_t end_I = indptr[I + 1];

    // Step 1: Sparsity calculation using redux.sync (hardware-optimized)
    // Count each of the 16 rows once (one lane per row).
    int local_nnz = 0;
    if (lane_id < 16) {
        local_nnz = __popc(S_ptr[lane_id]);
    }

    // Warp-level reduction using single redux.sync instruction (sm_80+)
    int total_nnz;
    asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(total_nnz) : "r"(local_nnz));

    // Step 2: Decide execution path based on accurate sparsity
    // Threshold: 64 out of 512 bits = 12.5% density (tuned for 16x32).
    bool use_dense = (total_nnz > 64);

    uint32_t total_sum = 0;

    if (use_dense) {
        // ========== Dense Path: Use Bit Tensor Core MMA ==========
        int groupID = lane_id >> 2;           // 0-7
        int threadID_in_group = lane_id & 3;  // 0-3

        for (int iter = 0; iter < 2; iter++) {
            int J_out = (int)J_sample * 2 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            int32_t c_accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            // Two-pointer merge to collect matching K blocks
            int ptr_A = start_I, ptr_B = j_start;
            int match_A[4], match_B[4];
            int match_count = 0;

            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A];
                uint32_t K_B = indices[ptr_B];

                if (K_A < K_B) {
                    ptr_A++;
                } else if (K_A > K_B) {
                    ptr_B++;
                } else {
                    match_A[match_count] = ptr_A;
                    match_B[match_count] = ptr_B;
                    match_count++;

                    if (match_count == 4) {
                        int my_ptr_A = match_A[threadID_in_group];
                        int my_ptr_B = match_B[threadID_in_group];

                        const uint32_t* A_blk = blocks + (size_t)my_ptr_A * SIZE_U32;
                        const uint32_t* B_blk = blocks + (size_t)my_ptr_B * SIZE_U32;

                        uint32_t a_frag[2], b_frag[2];
                        a_frag[0] = A_blk[groupID];
                        a_frag[1] = A_blk[groupID + 8];
                        b_frag[0] = B_blk[groupID];
                        b_frag[1] = B_blk[groupID + 8];

                        // Dual MMA for full 16×16 coverage
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

                        match_count = 0;
                    }

                    ptr_A++;
                    ptr_B++;
                }
            }

            // Handle remaining matches (< 4)
            if (match_count > 0) {
                uint32_t a_frag[2] = {0, 0}, b_frag[2] = {0, 0};

                if (threadID_in_group < match_count) {
                    int my_ptr_A = match_A[threadID_in_group];
                    int my_ptr_B = match_B[threadID_in_group];

                    const uint32_t* A_blk = blocks + (size_t)my_ptr_A * SIZE_U32;
                    const uint32_t* B_blk = blocks + (size_t)my_ptr_B * SIZE_U32;

                    a_frag[0] = A_blk[groupID];
                    a_frag[1] = A_blk[groupID + 8];
                    b_frag[0] = B_blk[groupID];
                    b_frag[1] = B_blk[groupID + 8];
                }

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
                if ((S_ptr[row] >> bit_col) & 1) {
                    total_sum += c_accum[f];
                }
            }
        }
    } else {
        // ========== Sparse Path: Use CUDA Core + __ffs traversal ==========
        int my_row = lane_id / 2;        // 0-15 (each of 16 rows)
        int my_col_half = lane_id & 1;   // 0 or 1 (left or right half)

        for (int iter = 0; iter < 2; iter++) {
            int J_out = (int)J_sample * 2 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            // Get the mask for this thread's row and column range
            uint32_t S_row = S_ptr[my_row];
            int col_offset = iter * 16 + my_col_half * 8;
            uint32_t my_mask = (S_row >> col_offset) & 0xFF;

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
                uint32_t A_row = (blocks + (size_t)ptr_A * SIZE_U32)[my_row];
                const uint32_t* B_blk = blocks + (size_t)ptr_B * SIZE_U32;

                // Use __ffs to traverse only set bits in the mask
                uint32_t mask = my_mask;
                while (mask != 0) {
                    int c = __ffs(mask) - 1;  // Find first set bit (0-7)
                    int col = my_col_half * 8 + c;
                    total_sum += __popc(A_row & B_blk[col]);
                    mask &= (mask - 1);  // Clear the lowest set bit
                }

                ptr_A++;
                ptr_B++;
            }
        }
    }

    // Warp-level reduction using redux.sync (32-bit is sufficient for per-warp triangle count)
    uint32_t reduced_sum;
    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(reduced_sum) : "r"(total_sum));

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)reduced_sum);
    }
}

unsigned long long count_triangles_16x32_v5(BCSR_16x32_Device& d_bcsr, float* kernel_ms = nullptr) {
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

    kernel_16x32_mma_v5_adaptive<<<num_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks,
        d_bcsr.indptr, d_bcsr.indices, d_bcsr.blocks,
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
