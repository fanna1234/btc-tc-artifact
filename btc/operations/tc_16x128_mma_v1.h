#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "operations/bcsr_16x128.h"

namespace btc {

// ============================================================================
// Kernel (v1): 16x128 + m16n8k128 (二分查找)
// ============================================================================
__global__ void kernel_16x128_mma(
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

    long long warp_id_global = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id_global >= num_sample_blocks) return;

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
    int threadID_in_group = lane_id % 4;

    long long total_sum = 0;

    for (int iter = 0; iter < 8; iter++) {
        int J_out = (int)J_L * 8 + iter;
        if (J_out >= n_row_blocks) continue;

        int32_t c_frag[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        for (uint32_t ptr_I = start_I; ptr_I < end_I; ptr_I++) {
            uint32_t K = indices[ptr_I];
            const uint32_t* L_IK_ptr = blocks + (size_t)ptr_I * SIZE_U32;

            int idx_J = -1;
            {
                uint32_t left = indptr[J_out];
                uint32_t right = indptr[J_out + 1];
                while (left < right) {
                    uint32_t mid = left + (right - left) / 2;
                    uint32_t mid_col = indices[mid];
                    if (mid_col == K) { idx_J = (int)mid; break; }
                    if (mid_col < K) left = mid + 1;
                    else right = mid;
                }
            }
            if (idx_J < 0) continue;

            const uint32_t* L_JK_ptr = blocks + (size_t)idx_J * SIZE_U32;

            uint32_t a_frag[2];
            a_frag[0] = L_IK_ptr[groupID * COLS_U32 + threadID_in_group];
            a_frag[1] = L_IK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

            uint32_t b_frag[2];
            b_frag[0] = L_JK_ptr[groupID * COLS_U32 + threadID_in_group];
            b_frag[1] = L_JK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[0]), "=r"(c_frag[1]), "=r"(c_frag[2]), "=r"(c_frag[3])
                : "r"(a_frag[0]), "r"(a_frag[1]),
                  "r"(b_frag[0]),
                  "r"(c_frag[0]), "r"(c_frag[1]), "r"(c_frag[2]), "r"(c_frag[3])
            );

            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[4]), "=r"(c_frag[5]), "=r"(c_frag[6]), "=r"(c_frag[7])
                : "r"(a_frag[0]), "r"(a_frag[1]),
                  "r"(b_frag[1]),
                  "r"(c_frag[4]), "r"(c_frag[5]), "r"(c_frag[6]), "r"(c_frag[7])
            );
        }

        int rows[8] = {groupID, groupID, groupID + 8, groupID + 8,
                       groupID, groupID, groupID + 8, groupID + 8};
        int cols[8] = {threadID_in_group * 2, threadID_in_group * 2 + 1,
                       threadID_in_group * 2, threadID_in_group * 2 + 1,
                       threadID_in_group * 2 + 8, threadID_in_group * 2 + 9,
                       threadID_in_group * 2 + 8, threadID_in_group * 2 + 9};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int row = rows[i];
            int col = cols[i];

            if (row >= 16 || col >= 16) continue;
            if ((int)I * 16 + row >= n) continue;

            int bit_col = iter * 16 + col;
            int u32_idx = bit_col / 32;
            int bit_idx = bit_col % 32;
            uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];

            if ((mask_word >> bit_idx) & 1) {
                total_sum += c_frag[i];
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
// 封装调用函数 (v1)
// ============================================================================
inline unsigned long long count_triangles_16x128(BCSR_16x128_Device& d_bcsr) {
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_cuda_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    kernel_16x128_mma<<<num_cuda_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks, d_bcsr.indptr, d_bcsr.indices,
        d_bcsr.blocks, d_bcsr.num_blocks, d_bcsr.result);

    cudaDeviceSynchronize();
    return d_bcsr.get_result();
}

}  // namespace btc

