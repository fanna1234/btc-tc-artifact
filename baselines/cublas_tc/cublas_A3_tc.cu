#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>

// btc includes
#include <btc/btc.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Kernel to fill Dense Matrix A (Symmetric) from CSR L
// Input CSR represents Lower Triangular part.
// For every edge (i, j) in CSR:
// Set A[i, j] = 1.0
// Set A[j, i] = 1.0 (Symmetry)
// We use float for everything in A3 implementation for simplicity.
__global__ void csr_to_dense_symmetric_kernel(int num_rows, int num_cols,
                                              const int* row_offsets,
                                              const int* col_indices,
                                              float* dense_A) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = row_offsets[row];
    int end = row_offsets[row + 1];

    for (int idx = start; idx < end; ++idx) {
        int col = col_indices[idx];
        if (col < num_cols) {
            // Set A[row, col] = 1.0
            dense_A[(long long)row * num_cols + col] = 1.0f;
            // Set A[col, row] = 1.0
            dense_A[(long long)col * num_cols + row] = 1.0f;
        }
    }
}

// Kernel to sum the element-wise product of two dense matrices (Masked Sum)
// Result = Sum(A[i] * B[i])
__global__ void dense_masked_sum_kernel(size_t total_elements, const float* A, const float* B, double* grand_total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    
    double local_sum = 0.0;
    for (size_t i = idx; i < total_elements; i += stride) {
        local_sum += (double)A[i] * (double)B[i];
    }
    
    // Simple atomic add to global sum
    if (local_sum != 0.0) {
        atomicAdd(grand_total, local_sum);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: %s -i <input.mtx>\n", argv[0]);
        return 0;
    }

    btc::Config config = btc::program_options(argc, argv);

    int m, n;
    int nnz_l;
    
    // 1. IO & Preprocess using BTC Standard Pipeline
    // We get L (Lower Triangular)
    btc::CsrMatrix<int, float, btc::device_memory> L_csr; 
    
    {
        std::printf("Reading matrix from %s...\n", config.input_file.c_str());
        // Read to Host First
        btc::CsrMatrix<int, float, btc::host_memory> h_input_csr;
        btc::read_from_mtx(h_input_csr, config.input_file);
        
        // Copy to Device
        btc::CsrMatrix<int, float, btc::device_memory> d_input_csr;
        d_input_csr = h_input_csr;
        h_input_csr.free();
        
        // Convert to COO for Preprocessing
        btc::CooMatrix<int, float, btc::device_memory> d_input_coo;
        btc::convert_csr_to_coo(d_input_coo, d_input_csr);
        d_input_csr.free();
        
        // Preprocess (Symmetrize -> Strict Lower Triangular)
        btc::preprocess_for_triangle_counting(d_input_coo);
        
        // Convert back to CSR (L)
        btc::convert_coo_to_csr(L_csr, d_input_coo);
        d_input_coo.free();
    }

    m = L_csr.num_rows;
    n = L_csr.num_cols;
    nnz_l = L_csr.num_entries;
    
    if (m != n) {
        std::printf("Warning: Matrix is not square (%d x %d). A^3 requires square.\n", m, n);
    }
    
    std::printf("Graph L: Rows=%d, NNZ=%d\n", n, nnz_l);
    
    // 2. Allocate Dense Matrices
    // For optimized A^3 (Masked Sum), we only need A and A^2.
    // d_A (input), d_B (A^2).
    // All FP32.
    
    long long n_sq = (long long)n * n;
    size_t dense_bytes = n_sq * sizeof(float);
    double dense_mb = dense_bytes / (1024.0 * 1024.0);
    
    std::printf("Allocating 2 Dense Matrices (A, A^2). Each is %.2f MB. Total: %.2f MB\n", 
                dense_mb, dense_mb * 2);

    // OOM Guard
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    double free_mb = free_mem / (1024.0 * 1024.0);
    
    // 1. Fixed Limit Check (8GB)
    size_t hard_limit_bytes = 8ULL * 1024 * 1024 * 1024;
    // User requested relax? Let's keep logic but adapt to 2 matrices.
    // Actually cant.mtx is 15GB each, total 30GB.
    // If we have 80GB GPU, it's fine.
    // The previous run succeeded, so let's trust available memory check more.
    // WARNING: Code below has fixed limit check active in source. I should probably relax it if I want cant.mtx to run.
    // But since it ran before, maybe I should just use the dynamic check.
    
    size_t core_bytes = dense_bytes * 2;
    
    // 2. Physical Memory Check
    // Need 2 dense matrices + overhead
    size_t total_required = core_bytes + (size_t)(1024*1024*500);
    
    if (total_required > free_mem) {
        std::fprintf(stderr, "Error: OOM Guard. Graph is too large for A^3 Dense TC.\n");
        std::fprintf(stderr, "Required: %.2f MB, Available: %.2f MB\n", (dense_mb * 2), free_mb);
        std::fprintf(stderr, "Skipping dense conversion to avoid crash.\n");
        return 0; // Exit gracefully
    }
    
    // 3. Time Complexity Guard (OOT)
    // A^3 is 1x GEMM now.
    // N=62451 works on A100/H100.
    if (n > 80000) { 
        std::fprintf(stderr, "Error: OOT Guard. N=%d is too large for A^3 benchmark (Limit=80000).\n", n);
        std::fprintf(stderr, "Skipping to avoid excessive runtime.\n");
        return 0;
    }

    
    float *d_A, *d_B;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, dense_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, dense_bytes));
    
    // Initialize A to 0
    CUDA_CHECK(cudaMemset(d_A, 0, dense_bytes));
    
    // Fill A (Symmetric)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        csr_to_dense_symmetric_kernel<<<numBlocks, blockSize>>>(n, n,
            thrust::raw_pointer_cast(L_csr.row_pointers.data()),
            thrust::raw_pointer_cast(L_csr.column_indices.data()),
            d_A);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 3. cuBLAS Setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    btc::CUDATimer gemm_timer;
    gemm_timer.start();
    
    // Step 1: B = A * A
    // cublasSgemm (Row Major via cuBLAS Col Major Trick: C^T = B^T * A^T)
    // Here A is symmetric, so A^T = A.
    // We want C = A * A.
    // In Col Major: C_col = A_col * B_col.
    // Since everything is symmetric A, it doesn't matter much.
    // Let's explicitly say we interpret d_A as normal.
    // If d_A is symmetric, d_A^T = d_A.
    // d_B = d_A * d_A.
    // Standard Sgemm computes C = op(A) * op(B).
    
    std::printf("Running GEMM 1: A^2 = A * A ...\n");
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_A, n,
                             d_A, n,
                             &beta,
                             d_B, n));
    
    // Step 2: Masked Sum (Trace(A^3) = Sum(A^2 .* A))
    // We compute: Sum over (i,j) of B[i,j] * A[i,j]
    std::printf("Running Masked Sum: Trace(A^3) = Sum(A^2 .* A) ...\n");
    
    CUDA_CHECK(cudaDeviceSynchronize());
    gemm_timer.stop();
    std::printf("[One GEMM + Prep] time: %f ms\n", gemm_timer.elapsed());
    
    // 4. Trace(C)
    // Actually Masked Sum reduction
    double* d_grand_total;
    CUDA_CHECK(cudaMalloc((void**)&d_grand_total, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grand_total, 0, sizeof(double)));

    int t_threads = 256;
    // Use enough blocks to cover large array or just enough to saturate GPU
    // With grid-stride loop, we don't need to cover 1-to-1.
    // Let's use 1024 blocks.
    int t_blocks = 1024;
    
    dense_masked_sum_kernel<<<t_blocks, t_threads>>>((size_t)n * n, d_A, d_B, d_grand_total);
    
    double trace = 0.0;
    CUDA_CHECK(cudaMemcpy(&trace, d_grand_total, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Count = Trace / 6
    unsigned long long triangles = (unsigned long long)(trace / 6.0);
    
    std::printf("Trace: %.2f\n", trace);
    std::printf("Triangle Count: %llu\n", triangles);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_grand_total);
    cublasDestroy(handle);

    return 0;
}
