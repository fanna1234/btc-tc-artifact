#pragma once

#include <btc/common/macros.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace btc::bench {

template<int BLOCK_ROWS, int BLOCK_COLS>
struct BCSRHost {
    static_assert(BLOCK_COLS % 32 == 0);
    static constexpr int COLS_U32 = BLOCK_COLS / 32;
    static constexpr int SIZE_U32 = BLOCK_ROWS * COLS_U32;

    int n = 0;
    int n_row_blocks = 0;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<uint32_t> blocks;  // row-major, SIZE_U32 per block

    struct BlockEntry {
        int col = 0;
        uint32_t data[SIZE_U32];
    };

    void build_from_edges(int num_nodes, const std::vector<std::pair<int, int>>& edges_lower)
    {
        n = num_nodes;
        n_row_blocks = (n + BLOCK_ROWS - 1) / BLOCK_ROWS;

        std::vector<std::vector<BlockEntry>> temp(n_row_blocks);

        for (const auto& e : edges_lower) {
            int u = e.first;
            int v = e.second;
            if (u == v) continue;
            if (u < v) std::swap(u, v);  // ensure lower (u > v)

            const int I = u / BLOCK_ROWS;
            const int J = v / BLOCK_COLS;
            const int r = u % BLOCK_ROWS;
            const int c = v % BLOCK_COLS;
            const int c_u32 = c / 32;
            const int c_bit = c % 32;

            bool found = false;
            for (auto& entry : temp[I]) {
                if (entry.col == J) {
                    entry.data[r * COLS_U32 + c_u32] |= (1u << c_bit);
                    found = true;
                    break;
                }
            }
            if (!found) {
                BlockEntry entry;
                entry.col = J;
                std::memset(entry.data, 0, sizeof(entry.data));
                entry.data[r * COLS_U32 + c_u32] = (1u << c_bit);
                temp[I].push_back(entry);
            }
        }

        indptr.assign(n_row_blocks + 1, 0);
        indices.clear();
        blocks.clear();

        for (int I = 0; I < n_row_blocks; I++) {
            std::sort(temp[I].begin(), temp[I].end(), [](const BlockEntry& a, const BlockEntry& b) {
                return a.col < b.col;
            });

            for (const auto& entry : temp[I]) {
                indices.push_back(entry.col);
                for (int i = 0; i < SIZE_U32; i++) {
                    blocks.push_back(entry.data[i]);
                }
            }

            indptr[I + 1] = static_cast<int>(indices.size());
        }
    }

    int num_blocks() const { return static_cast<int>(indices.size()); }
};

template<int SIZE_U32>
struct BCSRDevice {
    int* indptr = nullptr;
    int* indices = nullptr;
    uint32_t* blocks = nullptr;
    unsigned long long* result = nullptr;
    int n = 0;
    int n_row_blocks = 0;
    int num_blocks = 0;

    template<int BLOCK_ROWS, int BLOCK_COLS>
    void allocate_and_copy(const BCSRHost<BLOCK_ROWS, BLOCK_COLS>& h)
    {
        n = h.n;
        n_row_blocks = h.n_row_blocks;
        num_blocks = h.num_blocks();

        CHECK_CUDA(cudaMalloc(&indptr, h.indptr.size() * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&indices, h.indices.size() * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&blocks, h.blocks.size() * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&result, sizeof(unsigned long long)));

        CHECK_CUDA(cudaMemcpy(indptr, h.indptr.data(), h.indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(indices, h.indices.data(), h.indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(blocks, h.blocks.data(), h.blocks.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    void reset_result() { CHECK_CUDA(cudaMemset(result, 0, sizeof(unsigned long long))); }

    unsigned long long get_result() const
    {
        unsigned long long r = 0;
        CHECK_CUDA(cudaMemcpy(&r, result, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        return r;
    }

    void free()
    {
        if (indptr) cudaFree(indptr);
        if (indices) cudaFree(indices);
        if (blocks) cudaFree(blocks);
        if (result) cudaFree(result);
        indptr = nullptr;
        indices = nullptr;
        blocks = nullptr;
        result = nullptr;
    }

    size_t bytes_total() const
    {
        return (static_cast<size_t>(n_row_blocks) + 1) * sizeof(int)
            + static_cast<size_t>(num_blocks) * sizeof(int)
            + static_cast<size_t>(num_blocks) * static_cast<size_t>(SIZE_U32) * sizeof(uint32_t)
            + sizeof(unsigned long long);
    }
};

}  // namespace btc::bench

