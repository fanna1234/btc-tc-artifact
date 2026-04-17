#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "operations/bcsr_16x128.h"

namespace btc {

// ============================================================================
// Kernel (v2): 16x128 + m16n8k128 (双指针，减少在 J_out 上的二分查找)
// ============================================================================
__global__ void kernel_16x128_mma_twopointer(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
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

    uint32_t I;
    {
        uint32_t lo = 0, hi = n_row_blocks;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (indptr[mid + 1] <= sample_idx) lo = mid + 1;
            else hi = mid;
        }
        I = lo;
    }

    uint32_t J_L = indices[sample_idx];
    const uint32_t* sample_block_ptr = blocks + (size_t)sample_idx * SIZE_U32;
    uint32_t start_I = indptr[I];
    uint32_t end_I   = indptr[I + 1];

    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id & 3;

    uint32_t col_activity[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int r = lane_id; r < 16; r += WARP_SIZE) {
        uint32_t row0 = sample_block_ptr[(size_t)r * COLS_U32 + 0];
        uint32_t row1 = sample_block_ptr[(size_t)r * COLS_U32 + 1];
        uint32_t row2 = sample_block_ptr[(size_t)r * COLS_U32 + 2];
        uint32_t row3 = sample_block_ptr[(size_t)r * COLS_U32 + 3];

        col_activity[0] |= (row0 & 0x0000FFFF);
        col_activity[1] |= (row0 >> 16);
        col_activity[2] |= (row1 & 0x0000FFFF);
        col_activity[3] |= (row1 >> 16);
        col_activity[4] |= (row2 & 0x0000FFFF);
        col_activity[5] |= (row2 >> 16);
        col_activity[6] |= (row3 & 0x0000FFFF);
        col_activity[7] |= (row3 >> 16);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            col_activity[i] |= __shfl_down_sync(0xFFFFFFFF, col_activity[i], offset);
        }
        col_activity[i] = __shfl_sync(0xFFFFFFFF, col_activity[i], 0);
    }

    int j_start[8], j_end[8], j_ptr[8];
    #pragma unroll
    for (int iter = 0; iter < 8; iter++) {
        int J_out = (int)J_L * 8 + iter;
        if (J_out < n_row_blocks && col_activity[iter] != 0) {
            j_start[iter] = (int)indptr[J_out];
            j_end[iter]   = (int)indptr[J_out + 1];
            j_ptr[iter]   = j_start[iter];
        } else {
            j_start[iter] = j_end[iter] = j_ptr[iter] = 0;
        }
    }

    long long total_sum = 0;

    for (uint32_t ptr_I = start_I; ptr_I < end_I; ptr_I++) {
        uint32_t K = indices[ptr_I];
        const uint32_t* L_IK_ptr = blocks + (size_t)ptr_I * SIZE_U32;

        #pragma unroll
        for (int iter = 0; iter < 8; iter++) {
            if (j_start[iter] == j_end[iter]) continue;

            while (j_ptr[iter] < j_end[iter] && indices[j_ptr[iter]] < K) {
                j_ptr[iter]++;
            }

            if (j_ptr[iter] < j_end[iter] && indices[j_ptr[iter]] == K) {
                int idx_J = j_ptr[iter];
                const uint32_t* L_JK_ptr = blocks + (size_t)idx_J * SIZE_U32;

                uint32_t a_frag[2];
                a_frag[0] = L_IK_ptr[groupID * COLS_U32 + threadID_in_group];
                a_frag[1] = L_IK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

                uint32_t b_frag[2];
                b_frag[0] = L_JK_ptr[groupID * COLS_U32 + threadID_in_group];
                b_frag[1] = L_JK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

                int32_t c_frag[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                asm volatile(
                    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                    : "=r"(c_frag[0]), "=r"(c_frag[1]), "=r"(c_frag[2]), "=r"(c_frag[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(b_frag[0]),
                      "r"(0), "r"(0), "r"(0), "r"(0)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                    : "=r"(c_frag[4]), "=r"(c_frag[5]), "=r"(c_frag[6]), "=r"(c_frag[7])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(b_frag[1]),
                      "r"(0), "r"(0), "r"(0), "r"(0)
                );

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
                    if ((int)I * 16 + row >= n) continue;

                    int bit_col = iter * 16 + col;
                    int u32_idx = bit_col / 32;
                    int bit_idx = bit_col % 32;
                    uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];

                    if ((mask_word >> bit_idx) & 1) {
                        total_sum += c_frag[f];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xFFFFFFFF, total_sum, offset);
    }

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)total_sum);
    }
}

// ============================================================================
// 封装调用函数 (v2)
// ============================================================================
inline unsigned long long count_triangles_16x128_twopointer(BCSR_16x128_Device& d_bcsr) {
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_cuda_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    kernel_16x128_mma_twopointer<<<num_cuda_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks, d_bcsr.indptr, d_bcsr.indices,
        d_bcsr.blocks, d_bcsr.num_blocks, d_bcsr.result);

    cudaDeviceSynchronize();
    return d_bcsr.get_result();
}

}  // namespace btc

