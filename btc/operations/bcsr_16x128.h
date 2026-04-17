#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cstdint>

namespace btc {

// ============================================================================
// 16x128 BCSR 格式定义
// ============================================================================
struct BCSR_16x128_Device {
    uint32_t* indptr = nullptr;
    uint32_t* indices = nullptr;
    uint32_t* blocks = nullptr;
    uint32_t* row_indices = nullptr;
    unsigned long long* result = nullptr;
    int n;
    int n_row_blocks;
    uint32_t num_blocks;

    void free() {
        if (indptr) cudaFree(indptr);
        if (indices) cudaFree(indices);
        if (blocks) cudaFree(blocks);
        if (row_indices) cudaFree(row_indices);
        if (result) cudaFree(result);
        indptr = indices = blocks = row_indices = nullptr;
        result = nullptr;
    }

    void reset_result() { cudaMemset(result, 0, sizeof(unsigned long long)); }

    unsigned long long get_result() {
        unsigned long long r;
        cudaMemcpy(&r, result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        return r;
    }
};

// ============================================================================
// GPU 转换 Functors
// ============================================================================
struct LocateTile_16x128 {
    int num_block_cols;
    LocateTile_16x128(int nbc) : num_block_cols(nbc) {}
    __host__ __device__ long long operator()(const thrust::tuple<int, int>& t) const {
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        return (long long)(r / 16) * num_block_cols + (c / 128);
    }
};

struct CalcPosInTile_16x128 {
    __host__ __device__ int operator()(const thrust::tuple<int, int>& t) const {
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        return (r % 16) * 128 + (c % 128);
    }
};

struct FillBlocks {
    uint32_t* blocks;
    FillBlocks(uint32_t* b) : blocks(b) {}
    __host__ __device__ void operator()(const thrust::tuple<int, int>& t) const {
#ifdef __CUDA_ARCH__
        int block_idx = thrust::get<0>(t); // 0..num_blocks-1
        int pos       = thrust::get<1>(t); // 0..2047

        int u32_idx = pos / 32;
        int bit_idx = pos % 32;

        atomicOr(&blocks[(size_t)block_idx * 64 + u32_idx], (1u << bit_idx));
#endif
    }
};

struct ExtractColBlock {
    int num_block_cols;
    ExtractColBlock(int nbc) : num_block_cols(nbc) {}
    __host__ __device__ int operator()(long long tile_idx) const { return (int)(tile_idx % num_block_cols); }
};

struct ExtractRowBlock {
    int num_block_cols;
    ExtractRowBlock(int nbc) : num_block_cols(nbc) {}
    __host__ __device__ int operator()(long long tile_idx) const { return (int)(tile_idx / num_block_cols); }
};

// Phase 2 functor: fill block bits directly from COO edges via binary search
struct FillBlocksDirect_16x128 {
    uint32_t* indptr;
    uint32_t* indices;
    uint32_t* blocks;
    FillBlocksDirect_16x128(uint32_t* ip, uint32_t* idx, uint32_t* b)
        : indptr(ip), indices(idx), blocks(b) {}
    __host__ __device__ void operator()(const thrust::tuple<int, int>& t) const {
#ifdef __CUDA_ARCH__
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        int row_block = r / 16;
        uint32_t col_block = (uint32_t)(c / 128);

        // Binary search for col_block in indices[indptr[row_block]..indptr[row_block+1])
        uint32_t lo = indptr[row_block];
        uint32_t hi = indptr[row_block + 1];
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (indices[mid] < col_block) lo = mid + 1;
            else hi = mid;
        }

        int pos = (r % 16) * 128 + (c % 128);
        int u32_idx = pos / 32;
        int bit_idx = pos % 32;
        atomicOr(&blocks[(size_t)lo * 64 + u32_idx], (1u << bit_idx));
#endif
    }
};

// ============================================================================
// 转换函数：从 COO 格式转为 BCSR 16x128 (GPU 版本, 两阶段低内存)
// ============================================================================
template<typename CooMatrixType>
void convert_coo_to_bcsr_16x128_gpu(BCSR_16x128_Device& bcsr, const CooMatrixType& coo) {
    bcsr.n = coo.num_rows;
    int BLOCK_ROWS = 16;
    int BLOCK_COLS = 128;
    int num_block_cols = (bcsr.n + BLOCK_COLS - 1) / BLOCK_COLS;
    bcsr.n_row_blocks = (bcsr.n + BLOCK_ROWS - 1) / BLOCK_ROWS;

    int num_blocks;

    // ========== Phase 1: Compute BCSR structure ==========
    // Allocates tile_indices only (no pos_in_tile, no copy for unique).
    // All intermediates freed at scope exit before Phase 2 block allocation.
    {
        thrust::device_vector<long long> tile_indices(coo.num_entries);

        auto zip_coords =
            thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

        thrust::transform(zip_coords, zip_coords + coo.num_entries, tile_indices.begin(), LocateTile_16x128(num_block_cols));

        // Sort keys only (no sort_by_key — saves N*4B for pos_in_tile)
        thrust::sort(tile_indices.begin(), tile_indices.end());

        // Unique in-place (no copy — saves N*8B)
        auto end_unique = thrust::unique(tile_indices.begin(), tile_indices.end());
        num_blocks = (int)(end_unique - tile_indices.begin());

        bcsr.num_blocks = (uint32_t)num_blocks;

        cudaError_t err;
        err = cudaMalloc(&bcsr.indptr, (bcsr.n_row_blocks + 1) * sizeof(uint32_t));
        if (err != cudaSuccess) { printf("Malloc indptr failed: %s\n", cudaGetErrorString(err)); return; }

        err = cudaMalloc(&bcsr.indices, num_blocks * sizeof(uint32_t));
        if (err != cudaSuccess) { printf("Malloc indices failed: %s\n", cudaGetErrorString(err)); return; }

        err = cudaMalloc(&bcsr.row_indices, num_blocks * sizeof(uint32_t));
        if (err != cudaSuccess) { printf("Malloc row_indices failed: %s\n", cudaGetErrorString(err)); return; }

        err = cudaMalloc(&bcsr.result, sizeof(unsigned long long));
        if (err != cudaSuccess) { printf("Malloc result failed: %s\n", cudaGetErrorString(err)); return; }

        // Extract col/row block indices from unique tile_indices
        // (only read first num_blocks elements — the valid unique range)
        thrust::transform(tile_indices.begin(),
                          tile_indices.begin() + num_blocks,
                          thrust::device_pointer_cast(bcsr.indices),
                          ExtractColBlock(num_block_cols));

        thrust::device_vector<int> row_block_ids(num_blocks);
        thrust::transform(tile_indices.begin(),
                          tile_indices.begin() + num_blocks,
                          row_block_ids.begin(),
                          ExtractRowBlock(num_block_cols));

        // tile_indices no longer needed — free immediately to reclaim N*8B
        thrust::device_vector<long long>().swap(tile_indices);

        cudaMemcpy(bcsr.row_indices, thrust::raw_pointer_cast(row_block_ids.data()),
                   num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        // Build indptr via reduce_by_key + scatter + exclusive_scan
        thrust::device_vector<int> unique_rows(num_blocks);
        thrust::device_vector<int> counts_per_existing_row(num_blocks);

        auto end_row_reduce = thrust::reduce_by_key(row_block_ids.begin(),
                                                   row_block_ids.end(),
                                                   thrust::make_constant_iterator(1),
                                                   unique_rows.begin(),
                                                   counts_per_existing_row.begin());

        int num_nonempty_rows = (int)(end_row_reduce.first - unique_rows.begin());

        thrust::device_vector<int> row_counts(bcsr.n_row_blocks, 0);
        thrust::scatter(counts_per_existing_row.begin(),
                        counts_per_existing_row.begin() + num_nonempty_rows,
                        unique_rows.begin(),
                        row_counts.begin());

        thrust::device_ptr<uint32_t> dev_indptr(bcsr.indptr);
        thrust::exclusive_scan(row_counts.begin(), row_counts.end(), dev_indptr, 0);

        int total_blocks = num_blocks;
        cudaMemcpy(bcsr.indptr + bcsr.n_row_blocks, &total_blocks, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    // Phase 1 done — all temporaries freed, only BCSR structure remains on GPU

    // ========== Phase 2: Allocate blocks and fill bits from COO ==========
    size_t blocks_size = (size_t)num_blocks * 64 * sizeof(uint32_t);
    printf("[BCSR 16x128] num_blocks=%d, blocks=%.1f MB\n", num_blocks, blocks_size / (1024.0 * 1024.0));

    cudaError_t err = cudaMalloc(&bcsr.blocks, blocks_size);
    if (err != cudaSuccess) {
        printf("Malloc blocks failed: %s (need %.1f GB)\n", cudaGetErrorString(err), blocks_size / (1024.0*1024.0*1024.0));
        return;
    }
    cudaMemset(bcsr.blocks, 0, blocks_size);

    // Fill bits directly from COO edges — each edge does binary search in indptr/indices
    auto zip_coords =
        thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

    thrust::for_each(zip_coords, zip_coords + coo.num_entries,
                     FillBlocksDirect_16x128(bcsr.indptr, bcsr.indices, bcsr.blocks));
}

}  // namespace btc
