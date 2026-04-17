// tau_sweep.cu - Sweep τ threshold for sensitivity analysis (W1)
// For each dataset, run the kernel with multiple τ values and report timing + correctness.
// Usage: tau_sweep <input.mtx> [block_type]
//   block_type: 128 (default) or 32

#include <btc/btc.h>
#include <btc/operations/bcsr_16x128.h>
#include <btc/operations/bcsr_16x32.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace btc {

// ============================================================================
// 16x128 kernel with parameterized τ
// ============================================================================
__global__ void kernel_16x128_tau_sweep(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    const uint32_t* __restrict__ row_indices,
    uint32_t num_sample_blocks,
    int tau,  // <-- parameterized threshold
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

    // Sparsity check
    int local_nnz = 0;
    for (int i = lane_id; i < SIZE_U32; i += WARP_SIZE) {
        local_nnz += __popc(S_ptr[i]);
    }
    int total_nnz;
    asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(total_nnz) : "r"(local_nnz));

    // Use parameterized τ instead of hardcoded 512
    bool use_dense = (total_nnz > tau);

    uint32_t total_sum = 0;

    if (use_dense) {
        // Dense Path: Bit Tensor Core MMA
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
        // Sparse Path: CUDA Core + __ffs
        int my_row = lane_id / 2;
        int my_col_half = lane_id & 1;

        for (int iter = 0; iter < 8; iter++) {
            int J_out = (int)J_L * 8 + iter;
            if (J_out >= n_row_blocks) continue;

            int j_start = (int)indptr[J_out];
            int j_end = (int)indptr[J_out + 1];

            int col_offset = iter * 16 + my_col_half * 8;
            int u32_idx = col_offset / 32;
            int bit_offset = col_offset % 32;

            uint32_t row_data = S_ptr[my_row * COLS_U32 + u32_idx];
            uint32_t my_mask = (row_data >> bit_offset) & 0xFF;

            if (my_mask == 0) continue;

            int ptr_A = start_I, ptr_B = j_start;
            while (ptr_A < (int)end_I && ptr_B < j_end) {
                uint32_t K_A = indices[ptr_A];
                uint32_t K_B = indices[ptr_B];

                if (K_A < K_B) { ptr_A++; continue; }
                if (K_A > K_B) { ptr_B++; continue; }

                const uint32_t* A_blk = blocks + (size_t)ptr_A * SIZE_U32;
                const uint32_t* B_blk = blocks + (size_t)ptr_B * SIZE_U32;

                uint32_t mask = my_mask;
                while (mask != 0) {
                    int c = __ffs(mask) - 1;
                    int col = my_col_half * 8 + c;
                    uint32_t sum = 0;
                    #pragma unroll
                    for (int u = 0; u < COLS_U32; u++) {
                        uint32_t A_val = A_blk[my_row * COLS_U32 + u];
                        uint32_t B_val = B_blk[col * COLS_U32 + u];
                        sum += __popc(A_val & B_val);
                    }
                    total_sum += sum;
                    mask &= (mask - 1);
                }
                ptr_A++;
                ptr_B++;
            }
        }
    }

    uint32_t reduced_sum;
    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(reduced_sum) : "r"(total_sum));

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)reduced_sum);
    }
}

// ============================================================================
// 16x32 kernel with parameterized τ
// ============================================================================
__global__ void kernel_16x32_tau_sweep(
    int n, int n_row_blocks,
    const uint32_t* __restrict__ indptr,
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ blocks,
    const uint32_t* __restrict__ row_indices,
    uint32_t num_sample_blocks,
    int tau,  // <-- parameterized threshold
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

    // Sparsity check (shuffle for 16x32)
    int local_nnz = 0;
    if (lane_id < 16) {
        local_nnz = __popc(S_ptr[lane_id]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_nnz += __shfl_down_sync(0xFFFFFFFF, local_nnz, offset);
    int total_nnz = __shfl_sync(0xFFFFFFFF, local_nnz, 0);

    // Use parameterized τ instead of hardcoded 64
    bool use_dense = (total_nnz > tau);

    uint32_t total_sum = 0;

    if (use_dense) {
        // Dense Path: Bit Tensor Core MMA (V6 style with batched matches)
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

                if (K_A < K_B) { ptr_A++; }
                else if (K_A > K_B) { ptr_B++; }
                else {
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

            // Handle remaining matches
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
        // Sparse Path: CUDA Core + __ffs
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

    uint32_t reduced_sum;
    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;"
                 : "=r"(reduced_sum) : "r"(total_sum));

    if (lane_id == 0) {
        atomicAdd(result, (unsigned long long)reduced_sum);
    }
}

} // namespace btc

// ============================================================================
// Host-side sweep logic
// ============================================================================

struct SweepResult {
    int tau;
    float kernel_ms;
    unsigned long long count;
};

SweepResult run_128_with_tau(btc::BCSR_16x128_Device& d_bcsr, int tau) {
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    cudaEvent_t k_start, k_end;
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_end);

    // Warmup
    btc::kernel_16x128_tau_sweep<<<num_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks,
        d_bcsr.indptr, d_bcsr.indices, d_bcsr.blocks,
        d_bcsr.row_indices, d_bcsr.num_blocks, tau, d_bcsr.result);
    cudaDeviceSynchronize();
    d_bcsr.reset_result();

    // Timed run
    cudaEventRecord(k_start);
    btc::kernel_16x128_tau_sweep<<<num_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks,
        d_bcsr.indptr, d_bcsr.indices, d_bcsr.blocks,
        d_bcsr.row_indices, d_bcsr.num_blocks, tau, d_bcsr.result);
    cudaEventRecord(k_end);

    unsigned long long h_result = 0;
    cudaMemcpy(&h_result, d_bcsr.result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(k_end);

    float ms = 0;
    cudaEventElapsedTime(&ms, k_start, k_end);
    cudaEventDestroy(k_start);
    cudaEventDestroy(k_end);

    return {tau, ms, h_result};
}

SweepResult run_32_with_tau(btc::BCSR_16x32_Device& d_bcsr, int tau) {
    d_bcsr.reset_result();

    int WARP_SIZE = 32;
    int WARPS_PER_BLOCK = 4;
    int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    int num_blocks = (d_bcsr.num_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    cudaEvent_t k_start, k_end;
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_end);

    // Warmup
    btc::kernel_16x32_tau_sweep<<<num_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks,
        d_bcsr.indptr, d_bcsr.indices, d_bcsr.blocks,
        d_bcsr.row_indices, d_bcsr.num_blocks, tau, d_bcsr.result);
    cudaDeviceSynchronize();
    d_bcsr.reset_result();

    // Timed run
    cudaEventRecord(k_start);
    btc::kernel_16x32_tau_sweep<<<num_blocks, threads_per_block>>>(
        d_bcsr.n, d_bcsr.n_row_blocks,
        d_bcsr.indptr, d_bcsr.indices, d_bcsr.blocks,
        d_bcsr.row_indices, d_bcsr.num_blocks, tau, d_bcsr.result);
    cudaEventRecord(k_end);

    unsigned long long h_result = 0;
    cudaMemcpy(&h_result, d_bcsr.result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(k_end);

    float ms = 0;
    cudaEventElapsedTime(&ms, k_start, k_end);
    cudaEventDestroy(k_start);
    cudaEventDestroy(k_end);

    return {tau, ms, h_result};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <input.mtx> [128|32]\n", argv[0]);
        return 1;
    }

    int block_type = 128;
    if (argc >= 3) {
        block_type = std::atoi(argv[2]);
    }

    // Extract dataset name
    std::string dataset = argv[1];
    size_t pos = dataset.find_last_of("/\\");
    if (pos != std::string::npos) dataset = dataset.substr(pos + 1);
    pos = dataset.find(".mtx");
    if (pos != std::string::npos) dataset = dataset.substr(0, pos);

    // Read and preprocess
    btc::CsrMatrix<int, float, btc::device_memory> A_csr;
    btc::CooMatrix<int, float, btc::device_memory> A_coo;
    btc::read_matrix_file(A_csr, argv[1]);
    btc::convert_csr_to_coo(A_coo, A_csr);
    A_csr.free();
    btc::preprocess_for_triangle_counting(A_coo);

    if (block_type == 128) {
        btc::BCSR_16x128_Device d_bcsr;
        btc::convert_coo_to_bcsr_16x128_gpu(d_bcsr, A_coo);

        // τ sweep values for 16×128 (max 2048 bits)
        // 0 = all-dense (pure MMA), 2048 = all-sparse (pure CUDA core)
        int taus[] = {0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048};
        int n_taus = sizeof(taus) / sizeof(taus[0]);

        // CSV header
        std::printf("dataset,block_type,tau,kernel_ms,triangles,correct\n");

        // Get reference count with default τ=512
        auto ref = run_128_with_tau(d_bcsr, 512);

        for (int i = 0; i < n_taus; i++) {
            auto r = run_128_with_tau(d_bcsr, taus[i]);
            bool correct = (r.count == ref.count);
            std::printf("%s,128,%d,%.6f,%llu,%s\n",
                        dataset.c_str(), r.tau, r.kernel_ms, r.count,
                        correct ? "true" : "false");
        }

        d_bcsr.free();
    } else {
        btc::BCSR_16x32_Device d_bcsr;
        btc::convert_coo_to_bcsr_16x32_gpu(d_bcsr, A_coo);

        // τ sweep values for 16×32 (max 512 bits)
        // 0 = all-dense, 512 = all-sparse
        int taus[] = {0, 8, 16, 32, 48, 64, 96, 128, 256, 512};
        int n_taus = sizeof(taus) / sizeof(taus[0]);

        std::printf("dataset,block_type,tau,kernel_ms,triangles,correct\n");

        auto ref = run_32_with_tau(d_bcsr, 64);

        for (int i = 0; i < n_taus; i++) {
            auto r = run_32_with_tau(d_bcsr, taus[i]);
            bool correct = (r.count == ref.count);
            std::printf("%s,32,%d,%.6f,%llu,%s\n",
                        dataset.c_str(), r.tau, r.kernel_ms, r.count,
                        correct ? "true" : "false");
        }

        d_bcsr.free();
    }

    A_coo.free();
    return 0;
}
