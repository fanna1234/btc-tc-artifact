#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "operations/bcsr_16x32.h"

namespace btc {

// ============================================================================
// BTC-v3: 16×32 BCSR + Discrete K Combination
//
// 核心思想：
//   - 使用16×32的块粒度，提高稀疏性利用率
//   - 收集4个匹配的K块，组合成128位后执行MMA
//   - 减少块内零元素，提高计算效率
// ============================================================================

__global__ void kernel_16x32_mma_v3_discrete_k(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    uint32_t num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int WARP_SIZE = 32;
    constexpr int SIZE_U32 = 16; // 16×32块：16个uint32

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

    uint32_t J_sample = indices[sample_idx];
    const uint32_t* S_ptr = blocks + (size_t)sample_idx * SIZE_U32;
    uint32_t start_I = indptr[I];
    uint32_t end_I = indptr[I + 1];

    int groupID = lane_id >> 2;           // 0-7
    int threadID_in_group = lane_id & 3;  // 0-3

    long long total_sum = 0;

    // 遍历2个J_out（16×32块，每个sample block拆分成2个）
    for (int iter = 0; iter < 2; iter++) {
        int J_out = (int)J_sample * 2 + iter;
        if (J_out >= n_row_blocks) continue;

        int j_start = (int)indptr[J_out];
        int j_end = (int)indptr[J_out + 1];

        int32_t c_accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        // 双指针归并，收集匹配的K块（简单连续匹配策略）
        int ptr_A = start_I, ptr_B = j_start;

        // 交集缓冲区：存储匹配的(ptr_A, ptr_B)对
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
                // K_A == K_B，找到一个交集
                match_A[match_count] = ptr_A;
                match_B[match_count] = ptr_B;
                match_count++;

                // 凑够4个，执行一次MMA
                if (match_count == 4) {
                    // 每个线程加载自己负责的槽位
                    int my_ptr_A = match_A[threadID_in_group];
                    int my_ptr_B = match_B[threadID_in_group];

                    const uint32_t* A_blk = blocks + (size_t)my_ptr_A * SIZE_U32;
                    const uint32_t* B_blk = blocks + (size_t)my_ptr_B * SIZE_U32;

                    uint32_t a_frag[2], b_frag[2];
                    a_frag[0] = A_blk[groupID];
                    a_frag[1] = A_blk[groupID + 8];
                    b_frag[0] = B_blk[groupID];
                    b_frag[1] = B_blk[groupID + 8];

                    // MMA累加
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

                    match_count = 0;
                }

                ptr_A++;
                ptr_B++;
            }
        }

        // 处理剩余的K块（不足4个）
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

        // 应用mask并累加
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
inline unsigned long long count_triangles_16x32_v3(BCSR_16x32_Device& d_bcsr)
{
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_cuda_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    kernel_16x32_mma_v3_discrete_k<<<num_cuda_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks, d_bcsr.indptr, d_bcsr.indices,
        d_bcsr.blocks, d_bcsr.num_blocks, d_bcsr.result);

    cudaDeviceSynchronize();
    return d_bcsr.get_result();
}

} // namespace btc
