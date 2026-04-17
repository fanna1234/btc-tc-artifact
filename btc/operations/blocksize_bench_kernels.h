#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>

namespace btc::bench {

constexpr int kWarpSize = 32;

// 8x128 + m8n8k128 (cc >= 8.0), two-pointer matching.
__global__ void kernel_8x128_mma_twopointer(
    int n, int n_row_blocks,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    int num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int BLOCK_ROWS = 8;
    constexpr int BLOCK_COLS = 128;
    constexpr int COLS_U32 = BLOCK_COLS / 32;       // 4
    constexpr int SIZE_U32 = BLOCK_ROWS * COLS_U32; // 32

    const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (warp_id_global >= num_sample_blocks) return;

    const int sample_idx = warp_id_global;

    int I;
    {
        int lo = 0, hi = n_row_blocks;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (indptr[mid + 1] <= sample_idx) lo = mid + 1;
            else hi = mid;
        }
        I = lo;
    }

    const int J_L = indices[sample_idx];
    const uint32_t* sample_block_ptr = blocks + static_cast<size_t>(sample_idx) * SIZE_U32;
    const int start_I = indptr[I];
    const int end_I = indptr[I + 1];

    const int groupID = lane_id >> 2;          // 0..7
    const int threadID_in_group = lane_id & 3; // 0..3

    int j_end[16];
    int j_ptr[16];
    #pragma unroll
    for (int iter = 0; iter < 16; iter++) {
        const int J_out = J_L * 16 + iter;
        if (J_out < n_row_blocks) {
            const int s = indptr[J_out];
            const int e = indptr[J_out + 1];
            j_ptr[iter] = s;
            j_end[iter] = e;
        } else {
            j_ptr[iter] = 0;
            j_end[iter] = 0;
        }
    }

    long long total_sum = 0;

    for (int iter = 0; iter < 16; iter++) {
        const int J_out = J_L * 16 + iter;
        if (J_out >= n_row_blocks) continue;

        int32_t c_frag[2] = {0, 0};

        for (int ptr_I = start_I; ptr_I < end_I; ptr_I++) {
            const int K = indices[ptr_I];
            const uint32_t* L_IK_ptr = blocks + static_cast<size_t>(ptr_I) * SIZE_U32;

            int& jp = j_ptr[iter];
            const int je = j_end[iter];
            while (jp < je && indices[jp] < K) {
                jp++;
            }
            if (jp >= je || indices[jp] != K) continue;
            const int idx_J = jp;

            const uint32_t* L_JK_ptr = blocks + static_cast<size_t>(idx_J) * SIZE_U32;

            const uint32_t a_frag = L_IK_ptr[lane_id];
            const uint32_t b_frag = L_JK_ptr[lane_id];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            asm volatile(
                "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=r"(c_frag[0]), "=r"(c_frag[1])
                : "r"(a_frag), "r"(b_frag),
                  "r"(c_frag[0]), "r"(c_frag[1])
            );
#endif
        }

        const int row = groupID;
        const int col0 = threadID_in_group * 2;
        const int col1 = threadID_in_group * 2 + 1;

        if (I * BLOCK_ROWS + row < n) {
            if (col0 < 8) {
                const int bit_col = iter * 8 + col0;
                const int u32_idx = bit_col / 32;
                const int bit_idx = bit_col & 31;
                const uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];
                if ((mask_word >> bit_idx) & 1) total_sum += c_frag[0];
            }
            if (col1 < 8) {
                const int bit_col = iter * 8 + col1;
                const int u32_idx = bit_col / 32;
                const int bit_idx = bit_col & 31;
                const uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];
                if ((mask_word >> bit_idx) & 1) total_sum += c_frag[1];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xFFFFFFFF, total_sum, offset);
    }
    if (lane_id == 0) {
        atomicAdd(result, static_cast<unsigned long long>(total_sum));
    }
}

// 16x128 + m16n8k128 (cc >= 8.0), two-pointer matching.
__global__ void kernel_16x128_mma_twopointer(
    int n, int n_row_blocks,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    int num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int BLOCK_ROWS = 16;
    constexpr int BLOCK_COLS = 128;
    constexpr int COLS_U32 = BLOCK_COLS / 32;       // 4
    constexpr int SIZE_U32 = BLOCK_ROWS * COLS_U32; // 64

    const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (warp_id_global >= num_sample_blocks) return;

    const int sample_idx = warp_id_global;

    int I;
    {
        int lo = 0, hi = n_row_blocks;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (indptr[mid + 1] <= sample_idx) lo = mid + 1;
            else hi = mid;
        }
        I = lo;
    }

    const int J_L = indices[sample_idx];
    const uint32_t* sample_block_ptr = blocks + static_cast<size_t>(sample_idx) * SIZE_U32;
    const int start_I = indptr[I];
    const int end_I = indptr[I + 1];

    const int groupID = lane_id >> 2;          // 0..7
    const int threadID_in_group = lane_id & 3; // 0..3

    int j_end[8];
    int j_ptr[8];
    #pragma unroll
    for (int iter = 0; iter < 8; iter++) {
        const int J_out = J_L * 8 + iter;
        if (J_out < n_row_blocks) {
            const int s = indptr[J_out];
            const int e = indptr[J_out + 1];
            j_ptr[iter] = s;
            j_end[iter] = e;
        } else {
            j_ptr[iter] = 0;
            j_end[iter] = 0;
        }
    }

    long long total_sum = 0;

    for (int iter = 0; iter < 8; iter++) {
        const int J_out = J_L * 8 + iter;
        if (J_out >= n_row_blocks) continue;

        int32_t c_frag[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        for (int ptr_I = start_I; ptr_I < end_I; ptr_I++) {
            const int K = indices[ptr_I];
            const uint32_t* L_IK_ptr = blocks + static_cast<size_t>(ptr_I) * SIZE_U32;

            int& jp = j_ptr[iter];
            const int je = j_end[iter];
            while (jp < je && indices[jp] < K) {
                jp++;
            }
            if (jp >= je || indices[jp] != K) continue;
            const int idx_J = jp;

            const uint32_t* L_JK_ptr = blocks + static_cast<size_t>(idx_J) * SIZE_U32;

            uint32_t a_frag[2];
            a_frag[0] = L_IK_ptr[groupID * COLS_U32 + threadID_in_group];
            a_frag[1] = L_IK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

            uint32_t b_frag[2];
            b_frag[0] = L_JK_ptr[groupID * COLS_U32 + threadID_in_group];
            b_frag[1] = L_JK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
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
#endif
        }

        const int rows[8] = {groupID, groupID, groupID + 8, groupID + 8,
                             groupID, groupID, groupID + 8, groupID + 8};
        const int cols[8] = {threadID_in_group * 2, threadID_in_group * 2 + 1,
                             threadID_in_group * 2, threadID_in_group * 2 + 1,
                             threadID_in_group * 2 + 8, threadID_in_group * 2 + 9,
                             threadID_in_group * 2 + 8, threadID_in_group * 2 + 9};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int row = rows[i];
            const int col = cols[i];
            if (row >= 16 || col >= 16) continue;
            if (I * BLOCK_ROWS + row >= n) continue;

            const int bit_col = iter * 16 + col;
            const int u32_idx = bit_col / 32;
            const int bit_idx = bit_col & 31;
            const uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];
            if ((mask_word >> bit_idx) & 1) total_sum += c_frag[i];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xFFFFFFFF, total_sum, offset);
    }
    if (lane_id == 0) {
        atomicAdd(result, static_cast<unsigned long long>(total_sum));
    }
}

// 16x256 + m16n8k256 (cc >= 8.0), two-pointer matching.
__global__ void kernel_16x256_mma_twopointer(
    int n, int n_row_blocks,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    int num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int BLOCK_ROWS = 16;
    constexpr int BLOCK_COLS = 256;
    constexpr int COLS_U32 = BLOCK_COLS / 32;       // 8
    constexpr int SIZE_U32 = BLOCK_ROWS * COLS_U32; // 128

    const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (warp_id_global >= num_sample_blocks) return;

    const int sample_idx = warp_id_global;

    int I;
    {
        int lo = 0, hi = n_row_blocks;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (indptr[mid + 1] <= sample_idx) lo = mid + 1;
            else hi = mid;
        }
        I = lo;
    }

    const int J_L = indices[sample_idx];
    const uint32_t* sample_block_ptr = blocks + static_cast<size_t>(sample_idx) * SIZE_U32;
    const int start_I = indptr[I];
    const int end_I = indptr[I + 1];

    const int groupID = lane_id >> 2;          // 0..7
    const int threadID_in_group = lane_id & 3; // 0..3

    int j_end[16];
    int j_ptr[16];
    #pragma unroll
    for (int iter = 0; iter < 16; iter++) {
        const int J_out = J_L * 16 + iter;
        if (J_out < n_row_blocks) {
            const int s = indptr[J_out];
            const int e = indptr[J_out + 1];
            j_ptr[iter] = s;
            j_end[iter] = e;
        } else {
            j_ptr[iter] = 0;
            j_end[iter] = 0;
        }
    }

    long long total_sum = 0;

    for (int iter = 0; iter < 16; iter++) {
        const int J_out = J_L * 16 + iter;
        if (J_out >= n_row_blocks) continue;

        int32_t c_frag[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        for (int ptr_I = start_I; ptr_I < end_I; ptr_I++) {
            const int K = indices[ptr_I];
            const uint32_t* L_IK_ptr = blocks + static_cast<size_t>(ptr_I) * SIZE_U32;

            int& jp = j_ptr[iter];
            const int je = j_end[iter];
            while (jp < je && indices[jp] < K) {
                jp++;
            }
            if (jp >= je || indices[jp] != K) continue;
            const int idx_J = jp;

            const uint32_t* L_JK_ptr = blocks + static_cast<size_t>(idx_J) * SIZE_U32;

            uint32_t a_frag[4];
            a_frag[0] = L_IK_ptr[groupID * COLS_U32 + threadID_in_group];
            a_frag[1] = L_IK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];
            a_frag[2] = L_IK_ptr[groupID * COLS_U32 + threadID_in_group + 4];
            a_frag[3] = L_IK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group + 4];

            uint32_t b_frag[4];
            b_frag[0] = L_JK_ptr[groupID * COLS_U32 + threadID_in_group];
            b_frag[1] = L_JK_ptr[groupID * COLS_U32 + threadID_in_group + 4];
            b_frag[2] = L_JK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];
            b_frag[3] = L_JK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group + 4];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            asm volatile(
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=r"(c_frag[0]), "=r"(c_frag[1]), "=r"(c_frag[2]), "=r"(c_frag[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1]),
                  "r"(c_frag[0]), "r"(c_frag[1]), "r"(c_frag[2]), "r"(c_frag[3])
            );

            asm volatile(
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=r"(c_frag[4]), "=r"(c_frag[5]), "=r"(c_frag[6]), "=r"(c_frag[7])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[2]), "r"(b_frag[3]),
                  "r"(c_frag[4]), "r"(c_frag[5]), "r"(c_frag[6]), "r"(c_frag[7])
            );
#endif
        }

        const int rows[8] = {groupID, groupID, groupID + 8, groupID + 8,
                             groupID, groupID, groupID + 8, groupID + 8};
        const int cols[8] = {threadID_in_group * 2, threadID_in_group * 2 + 1,
                             threadID_in_group * 2, threadID_in_group * 2 + 1,
                             threadID_in_group * 2 + 8, threadID_in_group * 2 + 9,
                             threadID_in_group * 2 + 8, threadID_in_group * 2 + 9};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int row = rows[i];
            const int col = cols[i];
            if (row >= 16 || col >= 16) continue;
            if (I * BLOCK_ROWS + row >= n) continue;

            const int bit_col = iter * 16 + col;
            const int u32_idx = bit_col / 32;
            const int bit_idx = bit_col & 31;
            const uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];
            if ((mask_word >> bit_idx) & 1) total_sum += c_frag[i];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xFFFFFFFF, total_sum, offset);
    }
    if (lane_id == 0) {
        atomicAdd(result, static_cast<unsigned long long>(total_sum));
    }
}

// 32x128 + m16n8k128 (cc >= 8.0), 8x MMA per K, two-pointer matching.
__global__ void kernel_32x128_mma_twopointer(
    int n, int n_row_blocks,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    int num_sample_blocks,
    unsigned long long* __restrict__ result)
{
    constexpr int BLOCK_ROWS = 32;
    constexpr int BLOCK_COLS = 128;
    constexpr int COLS_U32 = BLOCK_COLS / 32;       // 4
    constexpr int SIZE_U32 = BLOCK_ROWS * COLS_U32; // 128

    const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (warp_id_global >= num_sample_blocks) return;

    const int sample_idx = warp_id_global;

    int I;
    {
        int lo = 0, hi = n_row_blocks;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (indptr[mid + 1] <= sample_idx) lo = mid + 1;
            else hi = mid;
        }
        I = lo;
    }

    const int J_L = indices[sample_idx];
    const uint32_t* sample_block_ptr = blocks + static_cast<size_t>(sample_idx) * SIZE_U32;
    const int start_I = indptr[I];
    const int end_I = indptr[I + 1];

    const int groupID = lane_id >> 2;          // 0..7
    const int threadID_in_group = lane_id & 3; // 0..3

    int j_end[4];
    int j_ptr[4];
    #pragma unroll
    for (int iter = 0; iter < 4; iter++) {
        const int J_out = J_L * 4 + iter;
        if (J_out < n_row_blocks) {
            const int s = indptr[J_out];
            const int e = indptr[J_out + 1];
            j_ptr[iter] = s;
            j_end[iter] = e;
        } else {
            j_ptr[iter] = 0;
            j_end[iter] = 0;
        }
    }

    long long total_sum = 0;

    for (int iter = 0; iter < 4; iter++) {
        const int J_out = J_L * 4 + iter;
        if (J_out >= n_row_blocks) continue;

        int32_t c_frag[32] = {0};

        for (int ptr_I = start_I; ptr_I < end_I; ptr_I++) {
            const int K = indices[ptr_I];
            const uint32_t* L_IK_ptr = blocks + static_cast<size_t>(ptr_I) * SIZE_U32;

            int& jp = j_ptr[iter];
            const int je = j_end[iter];
            while (jp < je && indices[jp] < K) {
                jp++;
            }
            if (jp >= je || indices[jp] != K) continue;
            const int idx_J = jp;

            const uint32_t* L_JK_ptr = blocks + static_cast<size_t>(idx_J) * SIZE_U32;

            uint32_t a_top[2], a_bot[2];
            a_top[0] = L_IK_ptr[groupID * COLS_U32 + threadID_in_group];
            a_top[1] = L_IK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];
            a_bot[0] = L_IK_ptr[(groupID + 16) * COLS_U32 + threadID_in_group];
            a_bot[1] = L_IK_ptr[(groupID + 24) * COLS_U32 + threadID_in_group];

            const uint32_t b_0 = L_JK_ptr[groupID * COLS_U32 + threadID_in_group];
            const uint32_t b_1 = L_JK_ptr[(groupID + 8) * COLS_U32 + threadID_in_group];
            const uint32_t b_2 = L_JK_ptr[(groupID + 16) * COLS_U32 + threadID_in_group];
            const uint32_t b_3 = L_JK_ptr[(groupID + 24) * COLS_U32 + threadID_in_group];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[0]), "=r"(c_frag[1]), "=r"(c_frag[2]), "=r"(c_frag[3])
                : "r"(a_top[0]), "r"(a_top[1]), "r"(b_0),
                  "r"(c_frag[0]), "r"(c_frag[1]), "r"(c_frag[2]), "r"(c_frag[3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[4]), "=r"(c_frag[5]), "=r"(c_frag[6]), "=r"(c_frag[7])
                : "r"(a_top[0]), "r"(a_top[1]), "r"(b_1),
                  "r"(c_frag[4]), "r"(c_frag[5]), "r"(c_frag[6]), "r"(c_frag[7])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[8]), "=r"(c_frag[9]), "=r"(c_frag[10]), "=r"(c_frag[11])
                : "r"(a_top[0]), "r"(a_top[1]), "r"(b_2),
                  "r"(c_frag[8]), "r"(c_frag[9]), "r"(c_frag[10]), "r"(c_frag[11])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[12]), "=r"(c_frag[13]), "=r"(c_frag[14]), "=r"(c_frag[15])
                : "r"(a_top[0]), "r"(a_top[1]), "r"(b_3),
                  "r"(c_frag[12]), "r"(c_frag[13]), "r"(c_frag[14]), "r"(c_frag[15])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[16]), "=r"(c_frag[17]), "=r"(c_frag[18]), "=r"(c_frag[19])
                : "r"(a_bot[0]), "r"(a_bot[1]), "r"(b_0),
                  "r"(c_frag[16]), "r"(c_frag[17]), "r"(c_frag[18]), "r"(c_frag[19])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[20]), "=r"(c_frag[21]), "=r"(c_frag[22]), "=r"(c_frag[23])
                : "r"(a_bot[0]), "r"(a_bot[1]), "r"(b_1),
                  "r"(c_frag[20]), "r"(c_frag[21]), "r"(c_frag[22]), "r"(c_frag[23])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[24]), "=r"(c_frag[25]), "=r"(c_frag[26]), "=r"(c_frag[27])
                : "r"(a_bot[0]), "r"(a_bot[1]), "r"(b_2),
                  "r"(c_frag[24]), "r"(c_frag[25]), "r"(c_frag[26]), "r"(c_frag[27])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=r"(c_frag[28]), "=r"(c_frag[29]), "=r"(c_frag[30]), "=r"(c_frag[31])
                : "r"(a_bot[0]), "r"(a_bot[1]), "r"(b_3),
                  "r"(c_frag[28]), "r"(c_frag[29]), "r"(c_frag[30]), "r"(c_frag[31])
            );
#endif
        }

        #pragma unroll
        for (int block = 0; block < 8; block++) {
            const int row_base = (block >= 4) ? 16 : 0;
            const int col_base = (block & 3) * 8;

            const int local_rows[4] = {groupID, groupID, groupID + 8, groupID + 8};
            const int local_cols[4] = {threadID_in_group * 2, threadID_in_group * 2 + 1,
                                       threadID_in_group * 2, threadID_in_group * 2 + 1};

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = row_base + local_rows[i];
                const int col = col_base + local_cols[i];
                if (row >= 32 || col >= 32) continue;
                if (I * BLOCK_ROWS + row >= n) continue;

                const int bit_col = iter * 32 + col;
                const int u32_idx = bit_col / 32;
                const int bit_idx = bit_col & 31;
                const uint32_t mask_word = sample_block_ptr[row * COLS_U32 + u32_idx];
                if ((mask_word >> bit_idx) & 1) total_sum += c_frag[block * 4 + i];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xFFFFFFFF, total_sum, offset);
    }
    if (lane_id == 0) {
        atomicAdd(result, static_cast<unsigned long long>(total_sum));
    }
}

}  // namespace btc::bench
