#pragma once

#include "bcsr_16x32.h"
#include <cuda_runtime.h>

namespace btc {

// V6-shfl: Same as V6 but replaces redux.sync with __shfl_down_sync + broadcast
// For ablation study: measuring redux.sync vs shuffle reduction overhead
__global__ void kernel_16x32_mma_v6_shfl(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    const uint32_t* __restrict__ row_indices,
    uint32_t num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int WARP_SIZE = 32;
    constexpr int SIZE_U32 = 16;

    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id_global >= (int)num_sample_blocks) return;

    uint32_t sample_idx = (uint32_t)warp_id_global;

    int I = (int)row_indices[sample_idx];

    uint32_t J_sample = indices[sample_idx];
    const uint32_t* S_ptr = blocks + (size_t)sample_idx * SIZE_U32;
    uint32_t start_I = indptr[I];
    uint32_t end_I = indptr[I + 1];

    // Step 1: Sparsity calculation using SHUFFLE (ablation: replacing redux.sync)
    int local_nnz = 0;
    if (lane_id < 16) {
        local_nnz = __popc(S_ptr[lane_id]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_nnz += __shfl_down_sync(0xFFFFFFFF, local_nnz, offset);
    }
    int total_nnz = __shfl_sync(0xFFFFFFFF, local_nnz, 0);

    bool use_dense = (total_nnz > 64);

    uint32_t total_sum = 0;

    if (use_dense) {
        int groupID = lane_id >> 2;
        int threadID_in_group = lane_id & 3;

        for (int iter = 0; iter < 2; iter++) {
            int J_out = (int)J_sample * 2 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            int32_t c_accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

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
        int my_row = lane_id / 2;
        int my_col_half = lane_id & 1;

        for (int iter = 0; iter < 2; iter++) {
            int J_out = (int)J_sample * 2 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            uint32_t S_row = S_ptr[my_row];
            int col_offset = iter * 16 + my_col_half * 8;
            uint32_t my_mask = (S_row >> col_offset) & 0xFF;

            if (my_mask == 0) continue;

            int ptr_A = start_I, ptr_B = j_start;
            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A];
                uint32_t K_B = indices[ptr_B];
                if (K_A < K_B) { ptr_A++; continue; }
                if (K_A > K_B) { ptr_B++; continue; }

                uint32_t A_row = (blocks + (size_t)ptr_A * SIZE_U32)[my_row];
                const uint32_t* B_blk = blocks + (size_t)ptr_B * SIZE_U32;

                uint32_t mask = my_mask;
                while (mask != 0) {
                    int c = __ffs(mask) - 1;
                    int col = my_col_half * 8 + c;
                    total_sum += __popc(A_row & B_blk[col]);
                    mask &= (mask - 1);
                }
                ptr_A++;
                ptr_B++;
            }
        }
    }

    // Final reduction using SHUFFLE (ablation: replacing redux.sync)
    uint32_t reduced_sum = total_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        reduced_sum += __shfl_down_sync(0xFFFFFFFF, reduced_sum, offset);
    }

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)reduced_sum);
    }
}

unsigned long long count_triangles_16x32_v6_shfl(BCSR_16x32_Device& d_bcsr, float* kernel_ms = nullptr) {
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

    kernel_16x32_mma_v6_shfl<<<num_blocks, threads_per_block>>>(
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
