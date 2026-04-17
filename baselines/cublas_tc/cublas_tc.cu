#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// Kernel to convert CSR to Dense (Row Major)
// Note: cuBLAS uses Column Major. If we populate Row Major, it looks like Transpose to cuBLAS.
// But we established that (L^T * L^T) = (L*L)^T, so result is Transposed, so Row Major matches.
// We fill generic dense matrix.
__global__ void csr_to_dense_kernel(int num_rows, int num_cols,
                                    const int* row_offsets, const int* col_indices, const float* values,
                                    half* dense_matrix) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int start = row_offsets[row];
        int end = row_offsets[row + 1];
        for (int i = start; i < end; ++i) {
            int col = col_indices[i];
            // Row Major Index: row * num_cols + col
            // Only set if within bounds (it should be for adjacency)
            if (col < num_cols) {
                // We ignore weights and set to 1.0
                dense_matrix[(long long)row * num_cols + col] = __float2half(1.0f);
            }
        }
    }
}

// Kernel to mask result: Sum C[i, j] where L[i, j] exists
// We reuse the CSR structure of L to visit only relevant entries in C.
__global__ void dense_masked_sum_kernel(int num_rows, int num_cols,
                                        const int* row_offsets, const int* col_indices,
                                        const float* dense_C,
                                        unsigned long long* grand_total) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    unsigned long long local_sum = 0;
    int start = row_offsets[row];
    int end = row_offsets[row + 1];

    for (int i = start; i < end; ++i) {
        int col = col_indices[i];
        if (col < num_cols) {
            // Read dense C at (row, col)
            // Assuming Dense C is in Row Major (which matches the logic derived)
            float val = dense_C[(long long)row * num_cols + col];
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

    // 1. IO & Preprocess using BTC
    // We strictly use FP16 for the Matrix Multiplication as requested.
    // However, for reading the MTX, we use BTC's utilities which load into standard types first.
    
    // Allocate host memory for Dense matrices
    // Using pinned memory for faster transfer if possible, but standard new is fine for now.
    // Matrix size: N x N. 
    // M = N, K = N, N = N.
    // Since we don't know N yet, we read first.

    int m, n, k;
    int nnz;
    
    // 1. IO & Preprocess using BTC Standard Pipeline
    // To ensure we match the triangle counting standard (Symmetrize + Lower Triangular),
    // we use BTC's preprocessing pipeline on Device.
    
    btc::CsrMatrix<int, float, btc::device_memory> L_csr; 
    
    {
        btc::CUDATimer io_timer;
        io_timer.start();
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
        
        io_timer.stop();
        std::printf("[IO & Preprocess] time: %f ms\n", io_timer.elapsed());
    }

    m = L_csr.num_rows;
    n = L_csr.num_cols;
    nnz = L_csr.num_entries;
    k = n;
    
    if (m != n) {
        std::printf("Warning: Matrix is not square (%d x %d). L^2 requires square or compatible dimensions.\n", m, n);
    }

    std::printf("Graph (Lower Triangular): Rows=%d, NNZ=%d\n", n, nnz);
    
    size_t dense_elements = (size_t)n * n;
    
    // Check memory requirements
    long long n_sq = (long long)n * n;
    size_t dense_bytes_half = n_sq * sizeof(half);
    size_t dense_bytes_float = n_sq * sizeof(float);
    double dense_mb = (dense_bytes_half + dense_bytes_float) / (1024.0 * 1024.0);
    std::printf("Dense Matrix Size: %lld elements, %.2f MB (L half + C float)\n", n_sq, dense_mb);

    // OOM & OOT Guard
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    double free_mb = free_mem / (1024.0 * 1024.0);
    
    // 1. Fixed Limit Check (8GB)
    size_t hard_limit_bytes = 8ULL * 1024 * 1024 * 1024;
    size_t core_bytes = dense_bytes_half + dense_bytes_float;
    
    if (core_bytes > hard_limit_bytes) {
        std::fprintf(stderr, "Error: Graph requires %.2f GB, which exceeds the fixed limit of 8.00 GB.\n", dense_mb / 1024.0);
        std::fprintf(stderr, "Skipping to avoid OOM or huge delays.\n");
        return 0;
    }

    // 2. Physical Memory Check
    // Safety Margin: Leave 500MB
    size_t required_bytes = core_bytes + (size_t)(1024*1024*500); 
    
    if (required_bytes > free_mem) {
        std::fprintf(stderr, "Error: OOM Guard. Graph is too large for current GPU memory.\n");
        std::fprintf(stderr, "Required: %.2f MB, Available: %.2f MB\n", dense_mb, free_mb);
        std::fprintf(stderr, "Skipping dense conversion to avoid crash.\n");
        return 0; // Exit gracefully
    }
    
    // 3. Time Complexity Guard (OOT)
    // Dense GEMM is O(N^3).
    // N=40,000 is a reasonable cutoff for benchmark utility.
    // Beyond this, use Sparse algorithms.
    if (n > 40000) { 
        std::fprintf(stderr, "Error: OOT Guard. N=%d is too large for dense benchmark logic limits (Limit=40000).\n", n);
        std::fprintf(stderr, "Skipping to avoid excessive runtime or inefficiency.\n");
        return 0;
    }

    
    // Allocate Dense L and C
    // L is half (FP16), C is float (FP32)
    half *d_L_dense;
    float *d_C_dense;
    
    CUDA_CHECK(cudaMalloc((void**)&d_L_dense, dense_bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&d_C_dense, dense_bytes_float));

    // Initialize Dense L to 0
    CUDA_CHECK(cudaMemset(d_L_dense, 0, dense_bytes_half));
    CUDA_CHECK(cudaMemset(d_C_dense, 0, dense_bytes_float));

    // Fill d_L_dense from L_csr using Kernel
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        csr_to_dense_kernel<<<numBlocks, blockSize>>>(n, n, 
                                                      thrust::raw_pointer_cast(L_csr.row_pointers.data()),
                                                      thrust::raw_pointer_cast(L_csr.column_indices.data()),
                                                      thrust::raw_pointer_cast(L_csr.values.data()),
                                                      d_L_dense);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 3. cuBLAS Setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Enable Tensor Cores
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0f;
    float beta = 0.0f;

    // 4. Run C = L * L
    // Note: We switch to cublasGemmEx for Mixed Precision (Half Inputs, Float Output)
    
    std::printf("Running cuBLAS Dense FP16-FP32 GEMM...\n");
    btc::CUDATimer gemm_timer;
    gemm_timer.start();
    
    // cublasGemmEx
    // Op(A') = L^T (OP_N of d_L_dense)
    // Op(B') = L^T (OP_N of d_L_dense) 
    // Result C' = L^T * L^T.
    // Stored C_col = C'. Row-Major view of C_col = (L^T * L^T)^T = L * L.
    
    cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         n, n, n,
                                         &alpha,
                                         d_L_dense, CUDA_R_16F, n,
                                         d_L_dense, CUDA_R_16F, n,
                                         &beta,
                                         d_C_dense, CUDA_R_32F, n,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    
    cudaDeviceSynchronize();
    gemm_timer.stop();
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS failed with error code " << status << std::endl;
        return 1;
    }
    std::printf("[Dense GEMM] time: %f ms\n", gemm_timer.elapsed());

    // 5. Mask and Sum
    // We compute Sum(C_ij) where L_ij=1.
    // C is in d_C_dense. 
    // Logic: C is Row-Major L * L.
    // We have L's sparsity pattern in L_csr.
    // These indices (i, j) correspond to L_ij.
    // So we just read d_C_dense[i*N + j].
    
    unsigned long long* d_total;
    cudaMalloc(&d_total, sizeof(unsigned long long));
    cudaMemset(d_total, 0, sizeof(unsigned long long));
    
    std::printf("Masking with L (NNZ=%d)...\n", nnz);

    btc::CUDATimer mask_timer;
    mask_timer.start();
    
    int threads = 128;
    int blocks = (m + threads - 1) / threads;
    
    dense_masked_sum_kernel<<<blocks, threads>>>(m, n,
                                                 thrust::raw_pointer_cast(L_csr.row_pointers.data()),
                                                 thrust::raw_pointer_cast(L_csr.column_indices.data()),
                                                 d_C_dense,
                                                 d_total);
    cudaDeviceSynchronize();
    mask_timer.stop();
    std::printf("[Mask & Sum] time: %f ms\n", mask_timer.elapsed());
    
    unsigned long long h_total;
    cudaMemcpy(&h_total, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    std::printf("Triangle Count: %llu\n", h_total);

    // Cleanup
    cudaFree(d_L_dense);
    cudaFree(d_C_dense);
    cudaFree(d_total);
    cublasDestroy(handle);

    // L_csr destructor will handle its own memory


    return 0;
}

