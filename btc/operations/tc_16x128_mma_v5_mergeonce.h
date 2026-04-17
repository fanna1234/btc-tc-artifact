#pragma once

#include "bcsr_16x128.h"
#include <cuda_runtime.h>

namespace btc {

// V5_MO: Merge-Once + A_val register cache
// Opt A: Row I scanned once (not 8x) — saves 7m index reads
// Opt C: A_blk row cached in registers across __ffs iterations
__global__ void kernel_16x128_mma_v5_mergeonce(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    const uint32_t* __restrict__ row_indices,
    uint32_t num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int WARP_SIZE = 32;
    constexpr int COLS_U32 = 4;
    constexpr int SIZE_U32 = 64;

    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id_global >= (int)num_sample_blocks) return;

    uint32_t sample_idx = (uint32_t)warp_id_global;

    int I = (int)row_indices[sample_idx];
    uint32_t J_L = indices[sample_idx];
    const uint32_t* S_ptr = blocks + (size_t)sample_idx * SIZE_U32;
    uint32_t start_I = indptr[I];
    uint32_t end_I = indptr[I + 1];

    // Sparsity check via redux.sync
    int local_nnz = 0;
    for (int i = lane_id; i < SIZE_U32; i += WARP_SIZE) {
        local_nnz += __popc(S_ptr[i]);
    }
    int total_nnz;
    asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(total_nnz) : "r"(local_nnz));

    bool use_dense = (total_nnz > 512);

    uint32_t total_sum = 0;

    if (use_dense) {
        // ========== Dense Path: Bit Tensor Core MMA (unchanged from V5) ==========
        int groupID = lane_id >> 2;
        int threadID_in_group = lane_id & 3;

        for (int iter = 0; iter < 8; iter++) {
            int J_out = (int)J_L * 8 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            int32_t c_accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            int ptr_A = start_I, ptr_B = j_start;
            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A];
                uint32_t K_B = indices[ptr_B];

                if (K_A < K_B) { ptr_A++; }
                else if (K_A > K_B) { ptr_B++; }
                else {
                    const uint32_t* A_blk = blocks + (size_t)ptr_A * SIZE_U32;
                    const uint32_t* B_blk = blocks + (size_t)ptr_B * SIZE_U32;

                    uint32_t a_frag[2];
                    a_frag[0] = A_blk[groupID * COLS_U32 + threadID_in_group];
                    a_frag[1] = A_blk[(groupID + 8) * COLS_U32 + threadID_in_group];

                    uint32_t b_frag[2];
                    b_frag[0] = B_blk[groupID * COLS_U32 + threadID_in_group];
                    b_frag[1] = B_blk[(groupID + 8) * COLS_U32 + threadID_in_group];

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
        // ========== Sparse Path: Merge-Once + A_val Cache ==========
        int my_row = lane_id / 2;
        int my_col_half = lane_id & 1;

        // Precompute all 8 masks from S block
        uint32_t s_row[COLS_U32];
        #pragma unroll
        for (int u = 0; u < COLS_U32; u++)
            s_row[u] = S_ptr[my_row * COLS_U32 + u];

        uint32_t masks[8];
        #pragma unroll
        for (int iter = 0; iter < 8; iter++) {
            int col_offset = iter * 16 + my_col_half * 8;
            masks[iter] = (s_row[col_offset / 32] >> (col_offset % 32)) & 0xFF;
        }

        // Preload 8 J_out pointers (warp-uniform)
        // Invalid J_outs get j_ptr=j_end=0, so loops and checks are no-ops.
        int j_ptr[8], j_end_arr[8];
        #pragma unroll
        for (int iter = 0; iter < 8; iter++) {
            int J_out = (int)J_L * 8 + iter;
            if (J_out < n_row_blocks) {
                j_ptr[iter] = (int)indptr[J_out];
                j_end_arr[iter] = (int)indptr[J_out + 1];
            } else {
                j_ptr[iter] = 0;
                j_end_arr[iter] = 0;
            }
        }

        // Single pass through row I's blocks (instead of 8 separate merges)
        for (int ptr_A = (int)start_I; ptr_A < (int)end_I; ptr_A++) {
            uint32_t K_A = indices[ptr_A];
            const uint32_t* A_blk = blocks + (size_t)ptr_A * SIZE_U32;

            // Opt C: Cache A row in registers (loaded once per K, reused across all iters)
            uint32_t a_cache[COLS_U32];
            #pragma unroll
            for (int u = 0; u < COLS_U32; u++)
                a_cache[u] = A_blk[my_row * COLS_U32 + u];

            // Check all 8 J_out rows for this K
            for (int iter = 0; iter < 8; iter++) {
                // Advance j_ptr[iter] to catch up with K_A (warp-uniform)
                while (j_ptr[iter] < j_end_arr[iter] &&
                       indices[j_ptr[iter]] < K_A)
                    j_ptr[iter]++;

                // Match check (warp-uniform) + mask check (per-thread)
                if (j_ptr[iter] < j_end_arr[iter] &&
                    indices[j_ptr[iter]] == K_A &&
                    masks[iter] != 0) {
                    const uint32_t* B_blk = blocks + (size_t)j_ptr[iter] * SIZE_U32;

                    // __ffs traversal with cached A values
                    uint32_t mask = masks[iter];
                    while (mask != 0) {
                        int c = __ffs(mask) - 1;
                        int col = my_col_half * 8 + c;
                        uint32_t sum = 0;
                        #pragma unroll
                        for (int u = 0; u < COLS_U32; u++)
                            sum += __popc(a_cache[u] & B_blk[col * COLS_U32 + u]);
                        total_sum += sum;
                        mask &= (mask - 1);
                    }
                }
            }
        }
    }

    // Warp-level reduction
    uint32_t reduced_sum;
    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(reduced_sum) : "r"(total_sum));

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)reduced_sum);
    }
}

unsigned long long count_triangles_16x128_v5_mergeonce(BCSR_16x128_Device& d_bcsr, float* kernel_ms = nullptr) {
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    cudaEvent_t k_start, k_end;
    if (kernel_ms) {
        cudaEventCreate(&k_start);
        cudaEventCreate(&k_end);
        cudaEventRecord(k_start);
    }

    kernel_16x128_mma_v5_mergeonce<<<num_blocks, threads_per_block>>>(
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
