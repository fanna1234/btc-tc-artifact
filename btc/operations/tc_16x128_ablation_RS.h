#pragma once
#include "bcsr_16x128.h"
#include <cuda_runtime.h>

namespace btc {

// 16x128 RS: Redux for sparsity, Shuffle for final
__global__ void kernel_16x128_RS(int n, int n_row_blocks, const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices, const uint32_t* __restrict__ blocks,
    const uint32_t* __restrict__ row_indices, uint32_t num_sample_blocks, unsigned long long* __restrict__ result)
{
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id_global >= (int)num_sample_blocks) return;
    uint32_t sample_idx = (uint32_t)warp_id_global;
    int I = (int)row_indices[sample_idx];
    uint32_t J_L = indices[sample_idx];
    const uint32_t* S_ptr = blocks + (size_t)sample_idx * 64;
    uint32_t start_I = indptr[I], end_I = indptr[I + 1];

    // Sparsity: REDUX
    int local_nnz = 0;
    for (int i = lane_id; i < 64; i += 32) local_nnz += __popc(S_ptr[i]);
    int total_nnz;
    asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(total_nnz) : "r"(local_nnz));

    bool use_dense = (total_nnz > 512);
    uint32_t total_sum = 0;

    if (use_dense) {
        int groupID = lane_id >> 2, threadID_in_group = lane_id & 3;
        for (int iter = 0; iter < 8; iter++) {
            int J_out = (int)J_L * 8 + iter;
            if (J_out >= n_row_blocks) continue;
            int j_start = (int)indptr[J_out], j_end = (int)indptr[J_out + 1];
            int32_t c_accum[8] = {0};
            int ptr_A = start_I, ptr_B = j_start;
            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A], K_B = indices[ptr_B];
                if (K_A < K_B) ptr_A++;
                else if (K_A > K_B) ptr_B++;
                else {
                    const uint32_t* A_blk = blocks + (size_t)ptr_A * 64;
                    const uint32_t* B_blk = blocks + (size_t)ptr_B * 64;
                    uint32_t a_frag[2] = {A_blk[groupID * 4 + threadID_in_group], A_blk[(groupID + 8) * 4 + threadID_in_group]};
                    uint32_t b_frag[2] = {B_blk[groupID * 4 + threadID_in_group], B_blk[(groupID + 8) * 4 + threadID_in_group]};
                    asm volatile("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                        : "=r"(c_accum[0]), "=r"(c_accum[1]), "=r"(c_accum[2]), "=r"(c_accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(b_frag[0]), "r"(c_accum[0]), "r"(c_accum[1]), "r"(c_accum[2]), "r"(c_accum[3]));
                    asm volatile("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                        : "=r"(c_accum[4]), "=r"(c_accum[5]), "=r"(c_accum[6]), "=r"(c_accum[7])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(b_frag[1]), "r"(c_accum[4]), "r"(c_accum[5]), "r"(c_accum[6]), "r"(c_accum[7]));
                    ptr_A++; ptr_B++;
                }
            }
            int rows[8] = {groupID, groupID, groupID + 8, groupID + 8, groupID, groupID, groupID + 8, groupID + 8};
            int cols[8] = {threadID_in_group * 2, threadID_in_group * 2 + 1, threadID_in_group * 2, threadID_in_group * 2 + 1,
                           threadID_in_group * 2 + 8, threadID_in_group * 2 + 9, threadID_in_group * 2 + 8, threadID_in_group * 2 + 9};
            #pragma unroll
            for (int f = 0; f < 8; f++) {
                if (rows[f] < 16 && cols[f] < 16 && I * 16 + rows[f] < n) {
                    int bit_col = iter * 16 + cols[f];
                    if ((S_ptr[rows[f] * 4 + bit_col / 32] >> (bit_col % 32)) & 1) total_sum += c_accum[f];
                }
            }
        }
    } else {
        int my_row = lane_id / 2, my_col_half = lane_id & 1;
        for (int iter = 0; iter < 8; iter++) {
            int J_out = (int)J_L * 8 + iter;
            if (J_out >= n_row_blocks) continue;
            int col_offset = iter * 16 + my_col_half * 8;
            uint32_t my_mask = (S_ptr[my_row * 4 + col_offset / 32] >> (col_offset % 32)) & 0xFF;
            if (my_mask == 0) continue;
            int ptr_A = start_I, ptr_B = (int)indptr[J_out], j_end = (int)indptr[J_out + 1];
            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A], K_B = indices[ptr_B];
                if (K_A < K_B) { ptr_A++; continue; }
                if (K_A > K_B) { ptr_B++; continue; }
                const uint32_t* A_blk = blocks + (size_t)ptr_A * 64;
                const uint32_t* B_blk = blocks + (size_t)ptr_B * 64;
                uint32_t mask = my_mask;
                while (mask) {
                    int c = __ffs(mask) - 1;
                    uint32_t sum = 0;
                    #pragma unroll
                    for (int u = 0; u < 4; u++)
                        sum += __popc(A_blk[my_row * 4 + u] & B_blk[(my_col_half * 8 + c) * 4 + u]);
                    total_sum += sum;
                    mask &= (mask - 1);
                }
                ptr_A++; ptr_B++;
            }
        }
    }

    // Final: SHUFFLE
    uint32_t reduced_sum = total_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        reduced_sum += __shfl_down_sync(0xFFFFFFFF, reduced_sum, offset);
    if (lane_id == 0) atomicAdd(result, (unsigned long long)reduced_sum);
}

unsigned long long count_triangles_16x128_RS(btc::BCSR_16x128_Device& d_bcsr, float* kernel_ms = nullptr) {
    d_bcsr.reset_result();
    int threads = 128, blocks = (d_bcsr.num_blocks + 3) / 4;
    cudaEvent_t start, end;
    if (kernel_ms) { cudaEventCreate(&start); cudaEventCreate(&end); cudaEventRecord(start); }
    kernel_16x128_RS<<<blocks, threads>>>(d_bcsr.n, d_bcsr.n_row_blocks, d_bcsr.indptr, d_bcsr.indices, d_bcsr.blocks, d_bcsr.row_indices, d_bcsr.num_blocks, d_bcsr.result);
    if (kernel_ms) { cudaEventRecord(end); }
    unsigned long long h_result = 0;
    cudaMemcpy(&h_result, d_bcsr.result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (kernel_ms) { cudaEventSynchronize(end); cudaEventElapsedTime(kernel_ms, start, end); cudaEventDestroy(start); cudaEventDestroy(end); }
    return h_result;
}

} // namespace btc
