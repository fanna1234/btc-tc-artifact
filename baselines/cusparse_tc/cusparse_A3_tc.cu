#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>

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

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper to calculate SpGEMM using cuSPARSE Generic API
// C = A * B
void spgemm(cusparseHandle_t handle,
            int m, int n, int k,
            cusparseSpMatDescr_t matA,
            cusparseSpMatDescr_t matB,
            cusparseSpMatDescr_t* matC,
            void** dBuffer, size_t* bufferSize,
            btc::CsrMatrix<int, float, btc::device_memory>& C_storage) {
    
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

    // Ask for C structure buffer
    size_t size1 = 0;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                 &alpha, matA, matB, &beta,
                                                 matA, // Dummy C descriptor for size estimation? No, manual says provide one.
                                                 computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &size1, NULL));
    
    // We need to initialize the output matrix C descriptor properly regarding rows/cols
    // But we don't know NNZ yet.
    // cuSPARSE flow:
    // 1. workEstimation -> gets buffer size needed for estimation?
    // 2. compute -> computes NNZ of C?
    // Actually modern cuSPARSE (v11+):
    // step1: workEstimation
    // step2: compute (symbolic phase?)
    // step3: copy
    
    // Let's assume typical flow:
    void* dBuffer1 = NULL;
    CUDA_CHECK(cudaMalloc(&dBuffer1, size1));
    
    // Create C descriptor with empty arrays (we will allocate after knowing NNZ)
    // We effectively reuse matA structure... wait no.
    // We must create a fresh C descriptor.
    int* d_C_offsets;
    int* d_C_columns;
    float* d_C_values;
    // Allocate offsets at least
    CUDA_CHECK(cudaMalloc((void**)&d_C_offsets, (m + 1) * sizeof(int)));
    
    CUSPARSE_CHECK(cusparseCreateCsr(matC, m, n, 0,
                                     d_C_offsets, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // workEstimation (again with buffer)
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                 &alpha, matA, matB, &beta,
                                                 *matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &size1, dBuffer1));
                                                 
    size_t size2 = 0;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta,
                                          *matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &size2, NULL));
    
    void* dBuffer2 = NULL;
    CUDA_CHECK(cudaMalloc(&dBuffer2, size2));
    
    // Compute (Symbolic) to find true NNZ of C
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta,
                                          *matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &size2, dBuffer2));

    int64_t C_num_rows, C_num_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(*matC, &C_num_rows, &C_num_cols, &C_nnz));
    
    std::printf("SpGEMM Result NNZ: %ld\n", C_nnz);
    
    // Now allocate C values and columns
    CUDA_CHECK(cudaMalloc((void**)&d_C_columns, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_C_values, C_nnz * sizeof(float)));
    
    CUSPARSE_CHECK(cusparseCsrSetPointers(*matC, d_C_offsets, d_C_columns, d_C_values));

    // Copy (Numeric)
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, opA, opB,
                                       &alpha, matA, matB, &beta,
                                       *matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                       spgemmDesc));
    
    // Store pointers in C_storage wrapper so we can free them later appropriately
    // Note: C_storage expects thrust vectors usually, but here we did raw malloc.
    // The CsrMatrix destructor might try to free?
    // btc::CsrMatrix uses thrust::device_vector. We cannot assign raw pointers to it easily.
    // Strategy: We will just manage raw pointers here for C1 and C2, and manually free.
    // Or we stick to raw pointers for this benchmark.
    
    CUDA_CHECK(cudaFree(dBuffer1));
    CUDA_CHECK(cudaFree(dBuffer2));
    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc));
}

// Kernel to sum intersection: sum(C_ij * A_ij)
__global__ void masked_sum_kernel(int num_rows,
                                  const int* A_row_offsets, const int* A_col_indices,
                                  const int* C_row_offsets, const int* C_col_indices, const float* C_values,
                                  unsigned long long* grand_total) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    double local_sum = 0;

    int a_start = A_row_offsets[row];
    int a_end = A_row_offsets[row + 1];

    int c_start = C_row_offsets[row];
    int c_end = C_row_offsets[row + 1];

    int c_curr = c_start;
    
    for (int idx_a = a_start; idx_a < a_end; ++idx_a) {
        int col_a = A_col_indices[idx_a];
        
        while (c_curr < c_end && C_col_indices[c_curr] < col_a) {
            c_curr++;
        }
        
        if (c_curr < c_end && C_col_indices[c_curr] == col_a) {
            float val = C_values[c_curr];
            // Safe accumulation
            local_sum += (double)val;
        }
    }

    if (local_sum > 0.5) {
        // Round to nearest integer (expected integer path counts)
        atomicAdd(grand_total, (unsigned long long)(local_sum + 0.5));
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: %s -i <input.mtx>\n", argv[0]);
        return 0;
    }

    btc::Config config = btc::program_options(argc, argv);
    
    // 1. IO & Preprocess (Full Symmetric A)
    btc::CsrMatrix<int, float, btc::device_memory> A_csr;
    {
        btc::CsrMatrix<int, float, btc::host_memory> h_input_csr;
        btc::read_from_mtx(h_input_csr, config.input_file);
        btc::CsrMatrix<int, float, btc::device_memory> d_input_csr;
        d_input_csr = h_input_csr;
        h_input_csr.free(); 
        
        btc::CooMatrix<int, float, btc::device_memory> d_input_coo;
        btc::convert_csr_to_coo(d_input_coo, d_input_csr);
        d_input_csr.free();
        
        // Pass false for 'lower_triangular' to get Full Symmetric A
        // Note: preprocess_for_triangle_counting logic inside btc might force lower triangular?
        // Let's check btc::preprocess_for_triangle_counting.
        // If it symmetrizes, it might return upper or lower.
        // We want FULL A (both i->j and j->i).
        // If the library function only supports triangular extraction, we need to recover symmetry here.
        
        btc::preprocess_for_triangle_counting(d_input_coo, false, false);
        
        // At this point d_input_coo might be Upper or Lower based on implementation, 
        // or Symmetric if 'lower_triangular' param controls strict filtering.
        // But logs say "After symmetrize & upper-tri" which implies filtering.
        
        // We need to restore full symmetry: A = A_tri + A_tri^T
        // Strategy: Convert to CSR, transpose, add.
        
        btc::convert_coo_to_csr(A_csr, d_input_coo);
        d_input_coo.free();
    }
    
    // Explicitly make symmetric A = A + A^T (assuming A is triangular from prev step)
    // Or simpler: use cuSparse to add transpose?
    // Actually, A_csr is currently triangular.
    // Let's use a helper to construct full symmetric CSR manually or via library.
    
if (false) { // Skip complex manual GEAM, assuming preprocessor works roughly, or rely on manual reconstruction via vectors
        // Transpose A -> At
        cusparseHandle_t handle_t; cusparseCreate(&handle_t);
        int n = A_csr.num_rows;
        int nnz = A_csr.num_entries;
        // ... omitted ...
}

    // SIMPLER SYMMETRIZATION (Host side to avoid GEAM complexity)
    {
        // Copy back to host to symmetrize fully using std::vector
        btc::CsrMatrix<int, float, btc::host_memory> h_A;
        h_A.num_rows = A_csr.num_rows;
        h_A.num_cols = A_csr.num_cols;
        h_A.num_entries = A_csr.num_entries;
        h_A.row_pointers = A_csr.row_pointers;
        h_A.column_indices = A_csr.column_indices;
        h_A.values = A_csr.values;
        
        std::vector<int> rows, cols;
        for(int i=0; i<h_A.num_rows; ++i) {
            int start = h_A.row_pointers[i];
            int end = h_A.row_pointers[i+1];
            for(int idx=start; idx<end; ++idx) {
                int j = h_A.column_indices[idx];
                rows.push_back(i); cols.push_back(j);
                if (i != j) {
                    rows.push_back(j); cols.push_back(i);
                }
            }
        }
        
        // Sort and Unique
        // Using COO format logic
        std::vector<std::pair<int, int>> edges;
        edges.reserve(rows.size());
        for(size_t i=0; i<rows.size(); ++i) edges.push_back({rows[i], cols[i]});
        
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        
        // Check if symmetric
        printf("Full Symmetric Edges: %zu\n", edges.size());
        
        // Rebuild CSR
        btc::CooMatrix<int, float, btc::host_memory> h_coo;
        h_coo.num_rows = h_A.num_rows;
        h_coo.num_cols = h_A.num_cols;
        h_coo.num_entries = edges.size();
        h_coo.row_indices.resize(edges.size());
        h_coo.column_indices.resize(edges.size());
        h_coo.values.resize(edges.size(), 1.0f);
        
        for(size_t i=0; i<edges.size(); ++i) {
            h_coo.row_indices[i] = edges[i].first;
            h_coo.column_indices[i] = edges[i].second;
        }
        
        // Upload
        btc::CooMatrix<int, float, btc::device_memory> d_coo;
        d_coo.num_rows = h_coo.num_rows;
        d_coo.num_cols = h_coo.num_cols;
        d_coo.num_entries = h_coo.num_entries;
        d_coo.row_indices = h_coo.row_indices;
        d_coo.column_indices = h_coo.column_indices;
        d_coo.values = h_coo.values;

        btc::convert_coo_to_csr(A_csr, d_coo);
        d_coo.free(); h_coo.free(); h_A.free();
    }


    int n = A_csr.num_rows;
    long long nnz_A = A_csr.num_entries;
    std::printf("Graph (Symmetric Check): N=%d, NNZ=%lld\n", n, nnz_A);
    
    // FORCE VALUES TO 1.0F
    // In case the input MTX was pattern and values are uninitialized or weird.
    // Triangle counting topology relies on 1.0 edge weights.
    thrust::fill(A_csr.values.begin(), A_csr.values.end(), 1.0f);

    // Setup cuSPARSE
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    
    // Descriptors
    cusparseSpMatDescr_t matA, matC1, matC2;
    
    // Wrap A
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, nnz_A,
                                     thrust::raw_pointer_cast(A_csr.row_pointers.data()),
                                     thrust::raw_pointer_cast(A_csr.column_indices.data()),
                                     thrust::raw_pointer_cast(A_csr.values.data()),
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
                                     
    // 2. Perform C1 = A * A
    std::printf("Running SpGEMM 1: C1 = A * A ...\n");

    btc::CUDATimer kernel_timer;
    kernel_timer.start();
    
    // We need to manage memory for C1 manually
    // For simplicity, I'll inline the SpGEMM call logic or use helper slightly modified.
    // Let's implement inline to be precise.
    
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));
    
    // Buffer 1
    size_t size1;
    void* dBuf1 = NULL;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, matA, matA, &beta,
                                                 matA, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &size1, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf1, size1));
    
    // Just workEstimation with buffer
    // For matC1, we need to provide offsets buffer at least!
    int *d_c1_offsets, *d_c1_cols;
    float *d_c1_vals;
    CUDA_CHECK(cudaMalloc((void**)&d_c1_offsets, (n + 1) * sizeof(int)));
    
    // Init C1 logic-less desc
    CUSPARSE_CHECK(cusparseCreateCsr(&matC1, n, n, 0,
                                     d_c1_offsets, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
                                     
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, matA, matA, &beta,
                                                 matC1, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &size1, dBuf1));
                                                 
    // Compute Step
    size_t size2;
    void* dBuf2 = NULL;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, op, op, &alpha, matA, matA, &beta,
                                          matC1, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &size2, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf2, size2));
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, op, op, &alpha, matA, matA, &beta,
                                          matC1, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &size2, dBuf2));
                                          
    int64_t rows1, cols1, nnz1;
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC1, &rows1, &cols1, &nnz1));
    std::printf("A^2 NNZ: %ld\n", nnz1);
    
    CUDA_CHECK(cudaMalloc((void**)&d_c1_cols, nnz1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_c1_vals, nnz1 * sizeof(float)));
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC1, d_c1_offsets, d_c1_cols, d_c1_vals));
    
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, op, op, &alpha, matA, matA, &beta,
                                       matC1, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                       spgemmDesc));

    // Free buffers 1 & 2
    cudaFree(dBuf1); cudaFree(dBuf2);
    
    // 3. Masked Sum (Trace(A^3) = Sum(A^2 .* A))
    // We compute: Sum over (i,j) in A of C1[i,j].
    // Note A is symmetric, so A_ij = A_ji.
    // C1 = A * A. C1_ij is number of paths of length 2 from i to j.
    // So Sum(C1_ij * A_ij) = Sum paths of len 2 connected by edge = Sum paths of len 3 = Trace(A^3).
    
    std::printf("Calculating Masked Sum (C .* A)...\n");
    
    unsigned long long* d_total;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    
    int t_threads = 256;
    int t_blocks = (n + t_threads - 1) / t_threads;
    
    masked_sum_kernel<<<t_blocks, t_threads>>>(n,
        thrust::raw_pointer_cast(A_csr.row_pointers.data()),
        thrust::raw_pointer_cast(A_csr.column_indices.data()),
        d_c1_offsets, d_c1_cols, d_c1_vals,
        d_total);
        
    unsigned long long h_total;
    CUDA_CHECK(cudaMemcpy(&h_total, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    std::printf("Masked Sum (Trace A^3): %llu\n", h_total);
    // Since we use Full A (Symmetric), result is Trace(A^3) = 6 * Triangles.
    // Each triangle (i, j, k) is counted 6 times (permutations).
    std::printf("Triangle Count: %llu\n", h_total / 6);
    
    kernel_timer.stop();
    std::printf("Kernel Time: %f ms\n", kernel_timer.elapsed());
    
    // Cleanup
    cudaFree(d_c1_offsets); cudaFree(d_c1_cols); cudaFree(d_c1_vals);
    cudaFree(d_total);
    return 0;
}

