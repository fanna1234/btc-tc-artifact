#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "operations/bcsr_16x128.h"

namespace btc {

// ============================================================================
// BTC-v3: 改进的双指针三角形计数
//
// 核心改进：
//   - 使用更高效的双指针归并查找K交集
//   - 减少冗余的指针移动
//   - 保持v2的正确性，每个K匹配立即计算
// ============================================================================

__global__ void kernel_16x128_mma_v3_improved(
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

    // 二分查找行块 I
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

    uint32_t J_L = indices[sample_idx];
    const uint32_t* sample_block_ptr = blocks + (size_t)sample_idx * SIZE_U32;
    uint32_t start_I = indptr[I];
    uint32_t end_I = indptr[I + 1];

    int groupID = lane_id >> 2;
    int threadID_in_group = lane_id & 3;

    // 预计算列活跃度
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

    // Scheme 3: J_out外层循环，逐个处理以减少寄存器压力
    long long total_sum = 0;

    // 外层循环：遍历8个J_out
    #pragma unroll
    for (int iter = 0; iter < 8; iter++) {
        int J_out = (int)J_L * 8 + iter;

        // 跳过空的或越界的J_out
        if (J_out >= n_row_blocks || col_activity[iter] == 0) continue;

        // 当前J_out的指针范围
        int j_start = (int)indptr[J_out];
        int j_end = (int)indptr[J_out + 1];
        int j_ptr = j_start;

        // 当前J_out的累加器（只需8个寄存器）
        int32_t c_accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        // 遍历A[I,*]的每个K块
        for (uint32_t ptr_I = start_I; ptr_I < end_I; ptr_I++) {
            uint32_t K = indices[ptr_I];
            const uint32_t* L_IK_ptr = blocks + (size_t)ptr_I * SIZE_U32;

            // 双指针前进到K
            while (j_ptr < j_end && indices[j_ptr] < K) {
                j_ptr++;
            }

            // 找到匹配的K，累加到当前J_out的累加器
            if (j_ptr < j_end && indices[j_ptr] == K) {
                int idx_J = j_ptr;
                const uint32_t* L_JK_ptr = blocks + (size_t)idx_J * SIZE_U32;

                uint32_t a_frag[2];
                a_frag[0] = L_IK_ptr[groupID * COLS_U32 + threadID_in_group];
                a_frag[1] = L_IK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

                uint32_t b_frag[2];
                b_frag[0] = L_JK_ptr[groupID * COLS_U32 + threadID_in_group];
                b_frag[1] = L_JK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

                // 直接累加到当前累加器
                asm volatile(
                    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                    : "=r"(c_accum[0]), "=r"(c_accum[1]), "=r"(c_accum[2]), "=r"(c_accum[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(b_frag[0]),
                      "r"(c_accum[0]), "r"(c_accum[1]), "r"(c_accum[2]), "r"(c_accum[3]));

                asm volatile(
                    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                    : "=r"(c_accum[4]), "=r"(c_accum[5]), "=r"(c_accum[6]), "=r"(c_accum[7])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(b_frag[1]),
                      "r"(c_accum[4]), "r"(c_accum[5]), "r"(c_accum[6]), "r"(c_accum[7]));
            }
        }

        // 当前J_out处理完成，应用mask并累加
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
            uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];

            if ((mask_word >> bit_idx) & 1) {
                total_sum += c_accum[f];
            }
        }
    } // 结束J_out循环

    // Warp归约
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xFFFFFFFF, total_sum, offset);
    }

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)total_sum);
    }
}

// 主函数
inline unsigned long long count_triangles_16x128_v3(BCSR_16x128_Device& d_bcsr)
{
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_cuda_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    kernel_16x128_mma_v3_improved<<<num_cuda_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks, d_bcsr.indptr, d_bcsr.indices,
        d_bcsr.blocks, d_bcsr.num_blocks, d_bcsr.result);

    cudaDeviceSynchronize();
    return d_bcsr.get_result();
}

} // namespace btc
