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
// 16x32 BCSR 辅助函数
// ============================================================================

struct LocateTile_16x32 {
    int num_block_cols;
    LocateTile_16x32(int nbc) : num_block_cols(nbc) {}
    __host__ __device__ long long operator()(const thrust::tuple<int, int>& t) const {
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        return (long long)(r / 16) * num_block_cols + (c / 32);
    }
};

struct CalcPosInTile_16x32 {
    __host__ __device__ int operator()(const thrust::tuple<int, int>& t) const {
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        return (r % 16) * 32 + (c % 32);
    }
};

struct FillBlocks_16x32 {
    uint32_t* blocks;
    FillBlocks_16x32(uint32_t* b) : blocks(b) {}
    __host__ __device__ void operator()(const thrust::tuple<int, int>& t) const {
#ifdef __CUDA_ARCH__
        int block_idx = thrust::get<0>(t);
        int pos       = thrust::get<1>(t); // 0..511 (16*32-1)

        int u32_idx = pos / 32;
        int bit_idx = pos % 32;

        atomicOr(&blocks[(size_t)block_idx * 16 + u32_idx], (1u << bit_idx));
#endif
    }
};

struct ExtractColBlock_16x32 {
    int num_block_cols;
    ExtractColBlock_16x32(int nbc) : num_block_cols(nbc) {}
    __host__ __device__ int operator()(long long tile_idx) const {
        return (int)(tile_idx % num_block_cols);
    }
};

struct ExtractRowBlock_16x32 {
    int num_block_cols;
    ExtractRowBlock_16x32(int nbc) : num_block_cols(nbc) {}
    __host__ __device__ int operator()(long long tile_idx) const {
        return (int)(tile_idx / num_block_cols);
    }
};

// Phase 2 functor: fill block bits directly from COO edges via binary search
struct FillBlocksDirect_16x32 {
    uint32_t* indptr;
    uint32_t* indices;
    uint32_t* blocks;
    FillBlocksDirect_16x32(uint32_t* ip, uint32_t* idx, uint32_t* b)
        : indptr(ip), indices(idx), blocks(b) {}
    __host__ __device__ void operator()(const thrust::tuple<int, int>& t) const {
#ifdef __CUDA_ARCH__
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        int row_block = r / 16;
        uint32_t col_block = (uint32_t)(c / 32);

        // Binary search for col_block in indices[indptr[row_block]..indptr[row_block+1])
        uint32_t lo = indptr[row_block];
        uint32_t hi = indptr[row_block + 1];
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (indices[mid] < col_block) lo = mid + 1;
            else hi = mid;
        }

        int pos = (r % 16) * 32 + (c % 32);
        int u32_idx = pos / 32;
        int bit_idx = pos % 32;
        atomicOr(&blocks[(size_t)lo * 16 + u32_idx], (1u << bit_idx));
#endif
    }
};

// ============================================================================
// 16x32 BCSR 格式定义
// ============================================================================
struct BCSR_16x32_Device {
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

    void reset_result() {
        cudaMemset(result, 0, sizeof(unsigned long long));
    }

    unsigned long long get_result() {
        unsigned long long h_result;
        cudaMemcpy(&h_result, result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        return h_result;
    }
};

// ============================================================================
// 转换函数：从 COO 格式转为 BCSR 16x32 (GPU 版本, 两阶段低内存)
// ============================================================================
template <typename CooMatrixType>
void convert_coo_to_bcsr_16x32_gpu(BCSR_16x32_Device& bcsr, const CooMatrixType& coo) {
    bcsr.n = coo.num_rows;
    int BLOCK_ROWS = 16;
    int BLOCK_COLS = 32;
    int num_block_cols = (bcsr.n + BLOCK_COLS - 1) / BLOCK_COLS;
    bcsr.n_row_blocks = (bcsr.n + BLOCK_ROWS - 1) / BLOCK_ROWS;

    int num_blocks;

    // ========== Phase 1: Compute BCSR structure ==========
    // Only tile_indices allocated; freed at scope exit before Phase 2.
    {
        fprintf(stderr, "[BCSR32] Phase1: alloc tile_indices (%zu × 8B = %.1f GB)\n",
                (size_t)coo.num_entries, (size_t)coo.num_entries * 8.0 / 1e9);
        thrust::device_vector<long long> tile_indices(coo.num_entries);

        auto zip_coords =
            thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

        thrust::transform(zip_coords, zip_coords + coo.num_entries, tile_indices.begin(), LocateTile_16x32(num_block_cols));
        cudaDeviceSynchronize();
        fprintf(stderr, "[BCSR32] Phase1: transform done\n");

        thrust::sort(tile_indices.begin(), tile_indices.end());
        cudaDeviceSynchronize();
        fprintf(stderr, "[BCSR32] Phase1: sort done\n");

        auto end_unique = thrust::unique(tile_indices.begin(), tile_indices.end());
        num_blocks = (int)(end_unique - tile_indices.begin());
        fprintf(stderr, "[BCSR32] Phase1: unique done, num_blocks=%d\n", num_blocks);

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

        thrust::transform(tile_indices.begin(),
                          tile_indices.begin() + num_blocks,
                          thrust::device_pointer_cast(bcsr.indices),
                          ExtractColBlock_16x32(num_block_cols));

        thrust::device_vector<int> row_block_ids(num_blocks);
        thrust::transform(tile_indices.begin(),
                          tile_indices.begin() + num_blocks,
                          row_block_ids.begin(),
                          ExtractRowBlock_16x32(num_block_cols));

        // Free tile_indices immediately to reclaim N*8B
        thrust::device_vector<long long>().swap(tile_indices);
        fprintf(stderr, "[BCSR32] Phase1: tile_indices freed\n");

        cudaMemcpy(bcsr.row_indices, thrust::raw_pointer_cast(row_block_ids.data()),
                   num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        // Build indptr
        thrust::device_vector<uint32_t> indptr_temp(bcsr.n_row_blocks + 1, 0);
        thrust::for_each(row_block_ids.begin(), row_block_ids.end(),
                         [indptr_raw = thrust::raw_pointer_cast(indptr_temp.data())] __device__(int rb) {
                             atomicAdd(&indptr_raw[rb + 1], 1u);
                         });

        thrust::inclusive_scan(indptr_temp.begin(), indptr_temp.end(), indptr_temp.begin());
        cudaMemcpy(bcsr.indptr, thrust::raw_pointer_cast(indptr_temp.data()),
                   (bcsr.n_row_blocks + 1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        fprintf(stderr, "[BCSR32] Phase1: indptr built, done\n");
    }
    // Phase 1 done — all temporaries freed

    // ========== Phase 2: Allocate blocks and fill bits from COO ==========
    size_t blocks_size = (size_t)num_blocks * 16 * sizeof(uint32_t);
    printf("[BCSR 16x32] num_blocks=%d, blocks=%.1f MB\n", num_blocks, blocks_size / (1024.0 * 1024.0));

    cudaError_t err = cudaMalloc(&bcsr.blocks, blocks_size);
    if (err != cudaSuccess) {
        printf("Malloc blocks failed: %s (need %.1f GB)\n", cudaGetErrorString(err), blocks_size / (1024.0*1024.0*1024.0));
        return;
    }
    cudaMemset(bcsr.blocks, 0, blocks_size);
    fprintf(stderr, "[BCSR32] Phase2: blocks allocated and zeroed\n");

    // Fill bits directly from COO edges — each edge does binary search in indptr/indices
    auto zip_coords =
        thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

    thrust::for_each(zip_coords, zip_coords + coo.num_entries,
                     FillBlocksDirect_16x32(bcsr.indptr, bcsr.indices, bcsr.blocks));
    cudaDeviceSynchronize();
    {
        cudaError_t sync_err = cudaGetLastError();
        if (sync_err != cudaSuccess)
            fprintf(stderr, "[BCSR32] Phase2: FillBlocksDirect FAILED: %s\n", cudaGetErrorString(sync_err));
        else
            fprintf(stderr, "[BCSR32] Phase2: FillBlocksDirect done OK\n");
    }
}

} // namespace btc
