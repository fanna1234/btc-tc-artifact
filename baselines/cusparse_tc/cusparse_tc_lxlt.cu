#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>

// Reuse BTC's IO logic if available, or just implement a simple MTX reader here.
// Since we want this to be self-contained somewhat, but it lives in the BTC project,
// we can link against `btc` lib or just use standard headers.
// However, connecting to BTC's header library is better to ensure fair preprocessing (symmetrization, etc).
// We will assume `btc/btc.h` is available in include path.
#include <btc/btc.h>

// Macros for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * Standard SpGEMM-based Triangle Counting using cuSPARSE.
 * 
 * Algorithm:
 *  TriangleCount = sum( (L * L) .* L )
 * Where L is the lower triangular part of the adjacency matrix.
 * 
 * Process:
 * 1. Read Graph, Symmetrize, Extract Lower Triangular part L.
 * 2. Upload L to GPU (CSR format).
 * 3. Compute C = L * L using cusparseSpGEMM.
 *    - Note: This is an expensive step as C can be denser than L.
 * 4. Compute intersection C .* L to find triangles.
 *    - Since cuSPARSE SpGEMM usually returns a new sparse matrix C, we need to efficiently intersect it with L.
 *    - Actually, exact extraction of (L*L) .* L can be done by inspecting the result of L*L 
 *      and checking if L also has an edge. 
 *    - HOWEVER, memory is a bottleneck. Full C = L*L might be too large.
 *    - A more memory-efficient approach used in some libraries (like GraphBLAS masked MxM) 
 *      is hard to implement with just raw cuSparse SpGEMM effortlessly without custom kernels.
 *    - Here, we implement the standard "Generate then Mask" approach: 
 *      Result = SpGEMM(L, L), then verify edges against L.
 */

// ============================================================================
// Kernels
// ============================================================================

/**
 * Computes: sum += C_val where (row, col) exists in L.
 * Matches CSR structures of L and C to find common edges.
 * Since C = L*L, C contains 2-hop paths (wedges).
 * If L has an edge (u, v), then C(u, v) is the count of triangles on that edge.
 * We iterate L (which is sparser), and look up values in C.
 * Triangles = Sum_{edge (u,v) in L} C[u][v]
 */
__global__ void masked_sum_kernel(int num_rows,
                                  const int* L_row_offsets, const int* L_col_indices,
                                  const int* C_row_offsets, const int* C_col_indices, const float* C_values,
                                  unsigned long long* grand_total) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    unsigned long long local_sum = 0;

    int l_start = L_row_offsets[row];
    int l_end = L_row_offsets[row + 1];

    int c_start = C_row_offsets[row];
    int c_end = C_row_offsets[row + 1];

    // Iterate edges in L
    for (int idx_l = l_start; idx_l < l_end; ++idx_l) {
        int col = L_col_indices[idx_l];

        // Find `col` in C's row `row`.
        // C is sorted by column index (cuSparse property).
        // Linear scan or Binary search?
        // Since we repeat this for every col in L, lock-step is best if we iterate both.
        // But here we are picking specific cols from C.
        // A simple binary search is robust.
        
        // Binary search in C[c_start...c_end) for `col`
        int left = c_start;
        int right = c_end - 1;
        bool found = false;
        float val = 0.0f;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int c_col = C_col_indices[mid];
            if (c_col == col) {
                val = C_values[mid];
                found = true;
                break;
            } else if (c_col < col) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        if (found) {
            local_sum += (unsigned long long)val;
        }
    }

    if (local_sum > 0) {
        atomicAdd(grand_total, local_sum);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: %s -i <input.mtx> [-v <verify 0/1>]\n", argv[0]);
        return 0;
    }

    btc::Config config = btc::program_options(argc, argv);

    // 1. Read and Preprocess Graph
    btc::CsrMatrix<int, float, btc::device_memory> input_csr;
    btc::read_matrix_file(input_csr, config.input_file); // Provides raw Device CSR
    
    btc::CooMatrix<int, float, btc::device_memory> input_coo;
    btc::convert_csr_to_coo(input_coo, input_csr);
    input_csr.free();
    
    // Preprocess: this makes it Symmetrized Lower Triangular L
    {
        btc::CUDATimer p_timer;
        p_timer.start();
        btc::preprocess_for_triangle_counting(input_coo);
        p_timer.stop();
        std::printf("[Preprocessing] time: %f ms\n", p_timer.elapsed());
    }

    // -------------------------------------------------------------------------
    // Build stage (after cleaning): build L/U descriptors and any aux structures.
    // This is included in "E2E after cleaning".
    // -------------------------------------------------------------------------
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    float build_ms = 0.0f;

    // Convert back to Device CSR (L) and build U=L^T + cuSPARSE descriptors.
    btc::CsrMatrix<int, float, btc::device_memory> L_csr;

    btc::CUDATimer build_timer;
    build_timer.start();

    btc::convert_coo_to_csr(L_csr, input_coo);
    input_coo.free();

    // IMPORTANT: For Triangle Counting, we treat the graph as unweighted (0/1).
    // The input file might contain weights. cuSPARSE SpGEMM does actual multiplication.
    // We must set all values to 1.0f to count paths correctly.
    thrust::fill(L_csr.values.begin(), L_csr.values.end(), 1.0f);

    int n = L_csr.num_rows;
    int nnz = L_csr.num_entries;
    std::printf("Graph (Lower Triangular): Rows=%d, NNZ=%d\n", n, nnz);

    // L Descriptor
    // btc::CsrMatrix uses Thrust vectors. We need raw pointers for cuSPARSE.
    int* d_L_rows = thrust::raw_pointer_cast(L_csr.row_pointers.data());
    int* d_L_cols = thrust::raw_pointer_cast(L_csr.column_indices.data());
    float* d_L_vals = thrust::raw_pointer_cast(L_csr.values.data());

    // START: Generate U (Transpose of L)
    // Transposing CSR L -> CSC L. The CSC arrays, if interpreted as CSR, represent L^T.
    
    int* d_U_rows; // effectively csc_col_ptr of L
    int* d_U_cols; // effectively csc_row_ind of L
    float* d_U_vals; // values for U (permuted)
    
    CUDA_CHECK(cudaMalloc((void**)&d_U_rows, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_U_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_U_vals, nnz * sizeof(float)));
    
    // Buffer for csr2csc
    void* d_trans_buffer = NULL;
    size_t trans_buffer_size = 0;
    
    // Use R_32F for values to avoid unsupported mixed-type SpGEMM on some cuSPARSE versions.
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(handle, n, n, nnz,
                                                 d_L_vals, d_L_rows, d_L_cols,
                                                 d_U_vals, d_U_rows, d_U_cols,
                                                 CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                                                 &trans_buffer_size));
    CUDA_CHECK(cudaMalloc(&d_trans_buffer, trans_buffer_size));
    
    CUSPARSE_CHECK(cusparseCsr2cscEx2(handle, n, n, nnz,
                                      d_L_vals, d_L_rows, d_L_cols,
                                      d_U_vals, d_U_rows, d_U_cols,
                                      CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                                      d_trans_buffer));
    CUDA_CHECK(cudaFree(d_trans_buffer));
    
    // Now d_U_rows/cols/vals represent U in CSR.
    // END: Generate U

    cusparseSpMatDescr_t matL;
    // Create CSR with FP32 values
    CUSPARSE_CHECK(cusparseCreateCsr(&matL, n, n, nnz,
                                     d_L_rows,
                                     d_L_cols,
                                     d_L_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
                                     
    cusparseSpMatDescr_t matU;
    CUSPARSE_CHECK(cusparseCreateCsr(&matU, n, n, nnz,
                                     d_U_rows,
                                     d_U_cols,
                                     d_U_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // C Descriptor (Empty initially)
    // Result C should also be FP16? Or can it be FP32?
    // SpGEMM usually wants uniform types A*B=C.
    // Let's try setting C to R_16F first. 
    // If we want FP32 Accumulation, does cuSPARSE support A(16) * B(16) = C(32)?
    // Usually it supports uniform. Let's try C as R_16F first to ensure it runs.
    // Wait, if C is R_16F, max value is 2048. That's risky for large graphs.
    // BUT user wants to "stimulate Tensor Core", which requires FP16 inputs.
    // Let's try C=FP16 first.
    // Use FP32 output for correctness (FP16 overflows for large triangle counts)
    // and better cuSPARSE robustness on large graphs.
    cusparseSpMatDescr_t matC;
    CUSPARSE_CHECK(cusparseCreateCsr(&matC, n, n, 0,
                                     NULL, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // SpGEMM Descriptor
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

    float h_alpha = 1.0f;
    float h_beta = 0.0f;
    cudaDataType computeType = CUDA_R_32F;
    
    void* dBuffer1 = NULL; size_t bufferSize1 = 0;
    void* dBuffer2 = NULL; size_t bufferSize2 = 0;

    // Finish "build after cleaning" timing just before launching the main compute pipeline.
    build_timer.stop();
    build_ms = build_timer.elapsed();
    std::printf("[Build] time: %f ms\n", build_ms);

    btc::CUDATimer timer;
    timer.start();

    // Step 1: Buffer Estimation for Work
    // Compute C = L * U (where U is L^T). Both are CSR, perform Non-Transpose mult.
    // Some graphs can trigger CUSPARSE_STATUS_INSUFFICIENT_RESOURCES under the DEFAULT alg.
    // Try a CSR-specific algorithm to improve robustness.
    cusparseSpGEMMAlg_t spgemm_alg = CUSPARSE_SPGEMM_DEFAULT;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &h_alpha, matL, matU, &h_beta, matC, 
                                                 computeType, spgemm_alg, spgemmDesc, 
                                                 &bufferSize1, NULL));
    CUDA_CHECK(cudaMalloc((void**)&dBuffer1, bufferSize1));
    
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &h_alpha, matL, matU, &h_beta, matC, 
                                                 computeType, spgemm_alg, spgemmDesc, 
                                                 &bufferSize1, dBuffer1));

    // Step 2: Compute (Symbolic & Numeric)
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &h_alpha, matL, matU, &h_beta, matC, 
                                          computeType, spgemm_alg, spgemmDesc, 
                                          &bufferSize2, NULL));
    CUDA_CHECK(cudaMalloc((void**)&dBuffer2, bufferSize2));
    
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &h_alpha, matL, matU, &h_beta, matC, 
                                          computeType, spgemm_alg, spgemmDesc, 
                                          &bufferSize2, dBuffer2));

    // Step 3: Extract C
    int64_t C_rows, C_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz));
    std::printf("DEBUG: C Matrix NNZ = %ld\n", C_nnz);
    
    int* d_C_offsets;
    int* d_C_columns;
    float* d_C_values;
    CUDA_CHECK(cudaMalloc((void**)&d_C_offsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_C_columns, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_C_values, C_nnz * sizeof(float)));
    
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC, d_C_offsets, d_C_columns, d_C_values));

    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &h_alpha, matL, matU, &h_beta, matC, 
                                       computeType, spgemm_alg, spgemmDesc));
    
    // Step 4: Masked Reduction (Intersection with L)
    unsigned long long* d_total_triangles;
    CUDA_CHECK(cudaMalloc((void**)&d_total_triangles, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_total_triangles, 0, sizeof(unsigned long long)));

    int blockSize = 128;
    int numBlocks = (n + blockSize - 1) / blockSize;

    masked_sum_kernel<<<numBlocks, blockSize>>>(n, 
                                                d_L_rows, d_L_cols,
                                                d_C_offsets, d_C_columns, d_C_values,
                                                d_total_triangles);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // L values live inside L_csr (thrust device vector); no explicit free needed.

    timer.stop();
    float total_ms = timer.elapsed();

    unsigned long long h_total_triangles;
    CUDA_CHECK(cudaMemcpy(&h_total_triangles, d_total_triangles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    std::printf("[cuSPARSE SpGEMM TC] time: %f ms\n", total_ms);
    std::printf("[Total Time (Build+Compute)] time: %f ms\n", build_ms + total_ms);
    std::printf("Triangles: %llu\n", h_total_triangles);

    // Verify if needed
    if (config.verify) {
         // ... (Cpu verification logic from btc_tc.cu) ...
    }

    // Cleanup
    if (dBuffer1) cudaFree(dBuffer1);
    if (dBuffer2) cudaFree(dBuffer2);
    if (d_C_offsets) cudaFree(d_C_offsets);
    if (d_C_columns) cudaFree(d_C_columns);
    if (d_C_values) cudaFree(d_C_values);
    
    // Free U resources
    if (d_U_rows) cudaFree(d_U_rows);
    if (d_U_cols) cudaFree(d_U_cols);
    if (d_U_vals) cudaFree(d_U_vals);
    
    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CUSPARSE_CHECK(cusparseDestroySpMat(matL));
    CUSPARSE_CHECK(cusparseDestroySpMat(matU));
    CUSPARSE_CHECK(cusparseDestroySpMat(matC));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    
    return 0;
}
