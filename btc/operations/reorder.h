#pragma once

// ============================================================================
// reorder.h — Graph reordering for BCSR triangle counting
//
// Extracted from convert.h. Contains:
// - Shared functors (OrientLowerTriangular, OrientUpperTriangular, ApplyPerm)
// - CPU reorder wrappers (BFS, Rabbit, Gorder, HashOrder, RCM)
// - GPU-native HashOrder (mode 8)
// ============================================================================

#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <random>

#include "reorder_rabbit.h"
#include "reorder_gorder.h"
#include "reorder_hashorder.h"
#include "reorder_rcm.h"

namespace btc {

// ============================================================================
// Shared functors: edge orientation
// Used by both symmetrization (convert.h) and reorder functions
// ============================================================================
struct OrientLowerTriangular {
    __host__ __device__
    thrust::tuple<int, int> operator()(const thrust::tuple<int, int>& t) const {
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        return (r >= c) ? t : thrust::make_tuple(c, r);
    }
};

struct OrientUpperTriangular {
    __host__ __device__
    thrust::tuple<int, int> operator()(const thrust::tuple<int, int>& t) const {
        int r = thrust::get<0>(t);
        int c = thrust::get<1>(t);
        return (r <= c) ? t : thrust::make_tuple(c, r);
    }
};

// ============================================================================
// GPU functor: apply permutation perm[old_id] → new_id
// ============================================================================
struct ApplyPerm {
    int* perm;
    ApplyPerm(int* p) : perm(p) {}
    __host__ __device__ int operator()(int id) const {
#ifdef __CUDA_ARCH__
        return perm[id];
#else
        return id;
#endif
    }
};

// ============================================================================
// BFS 图重排序 — 在 CPU 上构建全对称 CSR 后 BFS，让同社区节点获得连续 ID
// 对稀疏社交图(如 Friendster)可将 BCSR block 数降低一个数量级
// ============================================================================
template<typename CooMatrix>
void reorder_bfs_cpu(CooMatrix& coo) {
    int N = coo.num_rows;
    size_t E = coo.num_entries;

    // 仅对大图启用（小图 block 不会碎片化）
    if (N < 1000000) {
        fprintf(stderr, "[Reorder] Skipped (N=%d < 1M)\n", N);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] BFS ordering: %d nodes, %zu edges\n", N, E);

    // Step 1: COO → host
    std::vector<int> h_rows(E), h_cols(E);
    thrust::copy(coo.row_indices.begin(), coo.row_indices.end(), h_rows.begin());
    thrust::copy(coo.column_indices.begin(), coo.column_indices.end(), h_cols.begin());

    // Step 2: 构建全对称 CSR（从下三角边推导）
    std::vector<int> degree(N, 0);
    for (size_t i = 0; i < E; ++i) {
        degree[h_rows[i]]++;
        degree[h_cols[i]]++;
    }

    std::vector<size_t> row_ptr(N + 1, 0);
    for (int i = 0; i < N; ++i) row_ptr[i + 1] = row_ptr[i] + degree[i];
    size_t full_E = row_ptr[N];

    fprintf(stderr, "[Reorder] Full symmetric edges: %zu (%.1f GB)\n",
            full_E, full_E * 4.0 / 1e9);

    std::vector<int> col_idx(full_E);
    std::vector<size_t> offset(row_ptr.begin(), row_ptr.end());
    for (size_t i = 0; i < E; ++i) {
        int r = h_rows[i], c = h_cols[i];
        col_idx[offset[r]++] = c;
        col_idx[offset[c]++] = r;
    }
    h_rows.clear(); h_rows.shrink_to_fit();
    h_cols.clear(); h_cols.shrink_to_fit();
    offset.clear(); offset.shrink_to_fit();

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] CSR built (%.1f s)\n",
            std::chrono::duration<double>(t1 - t0).count());

    // Step 3: BFS from highest-degree node
    int start = (int)(std::max_element(degree.begin(), degree.end()) - degree.begin());
    degree.clear(); degree.shrink_to_fit();

    std::vector<int> perm(N, -1);
    std::vector<int> bfs_queue;
    bfs_queue.reserve(N);

    perm[start] = 0;
    bfs_queue.push_back(start);
    int new_id = 1;
    size_t front = 0;

    while (new_id < N) {
        while (front < bfs_queue.size()) {
            int u = bfs_queue[front++];
            for (size_t j = row_ptr[u]; j < row_ptr[u + 1]; ++j) {
                int v = col_idx[j];
                if (perm[v] == -1) {
                    perm[v] = new_id++;
                    bfs_queue.push_back(v);
                }
            }
        }
        // 处理非连通分量
        if (new_id < N) {
            for (int i = 0; i < N; ++i) {
                if (perm[i] == -1) {
                    perm[i] = new_id++;
                    bfs_queue.push_back(i);
                    break;
                }
            }
        }
    }
    col_idx.clear(); col_idx.shrink_to_fit();
    row_ptr.clear(); row_ptr.shrink_to_fit();
    bfs_queue.clear(); bfs_queue.shrink_to_fit();

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] BFS done (%.1f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    // Step 4: 将 perm 复制到 GPU 并应用置换
    thrust::device_vector<int> d_perm(perm.begin(), perm.end());
    perm.clear(); perm.shrink_to_fit();

    int* perm_ptr = thrust::raw_pointer_cast(d_perm.data());
    thrust::transform(coo.row_indices.begin(), coo.row_indices.end(),
                      coo.row_indices.begin(), ApplyPerm(perm_ptr));
    thrust::transform(coo.column_indices.begin(), coo.column_indices.end(),
                      coo.column_indices.begin(), ApplyPerm(perm_ptr));

    // 重新定向为下三角（置换后 row/col 大小关系改变）
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.end(), coo.column_indices.end()));
    thrust::transform(zip_begin, zip_end, zip_begin, OrientLowerTriangular());

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Complete (total %.1f s)\n",
            std::chrono::duration<double>(t3 - t0).count());
}

// ============================================================================
// Rabbit Order 图重排序 — 基于社区检测的高质量重排 (IPDPS'16)
// 对稀疏社交图可显著提高 BCSR block 密度
// ============================================================================
template<typename CooMatrix>
void reorder_rabbit_order_cpu(CooMatrix& coo) {
    int N = coo.num_rows;
    size_t E = coo.num_entries;

    if (N < 1000000) {
        fprintf(stderr, "[Reorder] Skipped Rabbit Order (N=%d < 1M)\n", N);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Rabbit Order: %d nodes, %zu edges\n", N, E);

    // Step 1: Copy COO to host
    std::vector<int> h_rows(E), h_cols(E);
    thrust::copy(coo.row_indices.begin(), coo.row_indices.end(), h_rows.begin());
    thrust::copy(coo.column_indices.begin(), coo.column_indices.end(), h_cols.begin());

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] D2H copy done (%.1f s)\n",
            std::chrono::duration<double>(t1 - t0).count());

    // Step 2: Compute permutation via Rabbit Order
    std::vector<int> perm = compute_rabbit_order_perm(N, E, h_rows.data(), h_cols.data());
    h_rows.clear(); h_rows.shrink_to_fit();
    h_cols.clear(); h_cols.shrink_to_fit();

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Rabbit Order done (%.1f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    // Step 3: Apply permutation on GPU
    thrust::device_vector<int> d_perm(perm.begin(), perm.end());
    perm.clear(); perm.shrink_to_fit();

    int* perm_ptr = thrust::raw_pointer_cast(d_perm.data());
    thrust::transform(coo.row_indices.begin(), coo.row_indices.end(),
                      coo.row_indices.begin(), ApplyPerm(perm_ptr));
    thrust::transform(coo.column_indices.begin(), coo.column_indices.end(),
                      coo.column_indices.begin(), ApplyPerm(perm_ptr));

    // Re-orient to lower triangular after permutation
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.end(), coo.column_indices.end()));
    thrust::transform(zip_begin, zip_end, zip_begin, OrientLowerTriangular());

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Complete (total %.1f s)\n",
            std::chrono::duration<double>(t3 - t0).count());
}

// ============================================================================
// Gorder 图重排序 — 滑动窗口贪心 (SIGMOD'16)
// 缓存行级别局部性优化
// ============================================================================
template<typename CooMatrix>
void reorder_gorder_cpu(CooMatrix& coo, int window_size = 5) {
    int N = coo.num_rows;
    size_t E = coo.num_entries;

    if (N < 1000000) {
        fprintf(stderr, "[Reorder] Skipped Gorder (N=%d < 1M)\n", N);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Gorder: %d nodes, %zu edges, window=%d\n", N, E, window_size);

    // Step 1: Copy COO to host
    std::vector<int> h_rows(E), h_cols(E);
    thrust::copy(coo.row_indices.begin(), coo.row_indices.end(), h_rows.begin());
    thrust::copy(coo.column_indices.begin(), coo.column_indices.end(), h_cols.begin());

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] D2H copy done (%.1f s)\n",
            std::chrono::duration<double>(t1 - t0).count());

    // Step 2: Compute permutation via Gorder
    std::vector<int> perm = compute_gorder_perm(N, E, h_rows.data(), h_cols.data(), window_size);
    h_rows.clear(); h_rows.shrink_to_fit();
    h_cols.clear(); h_cols.shrink_to_fit();

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Gorder done (%.1f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    // Step 3: Apply permutation on GPU
    thrust::device_vector<int> d_perm(perm.begin(), perm.end());
    perm.clear(); perm.shrink_to_fit();

    int* perm_ptr = thrust::raw_pointer_cast(d_perm.data());
    thrust::transform(coo.row_indices.begin(), coo.row_indices.end(),
                      coo.row_indices.begin(), ApplyPerm(perm_ptr));
    thrust::transform(coo.column_indices.begin(), coo.column_indices.end(),
                      coo.column_indices.begin(), ApplyPerm(perm_ptr));

    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.end(), coo.column_indices.end()));
    thrust::transform(zip_begin, zip_end, zip_begin, OrientLowerTriangular());

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Complete (total %.1f s)\n",
            std::chrono::duration<double>(t3 - t0).count());
}

// ============================================================================
// HashOrder 图重排序 — MinHash on multi-hop neighborhoods (ICLR'24)
// ~592x faster than Gorder with comparable or higher quality
// ============================================================================
template<typename CooMatrix>
void reorder_hashorder_cpu(CooMatrix& coo, int hops = 1, int num_hashes = 16) {
    int N = coo.num_rows;
    size_t E = coo.num_entries;

    if (N < 1000000) {
        fprintf(stderr, "[Reorder] Skipped HashOrder (N=%d < 1M)\n", N);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] HashOrder: %d nodes, %zu edges, hops=%d, hashes=%d\n",
            N, E, hops, num_hashes);

    // Step 1: Copy COO to host
    std::vector<int> h_rows(E), h_cols(E);
    thrust::copy(coo.row_indices.begin(), coo.row_indices.end(), h_rows.begin());
    thrust::copy(coo.column_indices.begin(), coo.column_indices.end(), h_cols.begin());

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] D2H copy done (%.1f s)\n",
            std::chrono::duration<double>(t1 - t0).count());

    // Step 2: Compute permutation via HashOrder
    std::vector<int> perm = compute_hashorder_perm(N, E, h_rows.data(), h_cols.data(), hops, num_hashes);
    h_rows.clear(); h_rows.shrink_to_fit();
    h_cols.clear(); h_cols.shrink_to_fit();

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] HashOrder done (%.1f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    // Step 3: Apply permutation on GPU
    thrust::device_vector<int> d_perm(perm.begin(), perm.end());
    perm.clear(); perm.shrink_to_fit();

    int* perm_ptr = thrust::raw_pointer_cast(d_perm.data());
    thrust::transform(coo.row_indices.begin(), coo.row_indices.end(),
                      coo.row_indices.begin(), ApplyPerm(perm_ptr));
    thrust::transform(coo.column_indices.begin(), coo.column_indices.end(),
                      coo.column_indices.begin(), ApplyPerm(perm_ptr));

    // Re-orient to lower triangular after permutation
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.end(), coo.column_indices.end()));
    thrust::transform(zip_begin, zip_end, zip_begin, OrientLowerTriangular());

    auto t3_ho = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Complete (total %.1f s)\n",
            std::chrono::duration<double>(t3_ho - t0).count());
}

// ============================================================================
// RCM 图重排序 — Reverse Cuthill-McKee bandwidth reduction
// 将非零元素集中到对角线附近，对 BCSR 三角计数极为有利
// ============================================================================
template<typename CooMatrix>
void reorder_rcm_cpu(CooMatrix& coo) {
    int N = coo.num_rows;
    size_t E = coo.num_entries;

    if (N < 1000000) {
        fprintf(stderr, "[Reorder] Skipped RCM (N=%d < 1M)\n", N);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] RCM: %d nodes, %zu edges\n", N, E);

    std::vector<int> h_rows(E), h_cols(E);
    thrust::copy(coo.row_indices.begin(), coo.row_indices.end(), h_rows.begin());
    thrust::copy(coo.column_indices.begin(), coo.column_indices.end(), h_cols.begin());

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<int> perm = compute_rcm_perm(N, E, h_rows.data(), h_cols.data());
    h_rows.clear(); h_rows.shrink_to_fit();
    h_cols.clear(); h_cols.shrink_to_fit();

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] RCM done (%.1f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    thrust::device_vector<int> d_perm(perm.begin(), perm.end());
    perm.clear(); perm.shrink_to_fit();

    int* perm_ptr = thrust::raw_pointer_cast(d_perm.data());
    thrust::transform(coo.row_indices.begin(), coo.row_indices.end(),
                      coo.row_indices.begin(), ApplyPerm(perm_ptr));
    thrust::transform(coo.column_indices.begin(), coo.column_indices.end(),
                      coo.column_indices.begin(), ApplyPerm(perm_ptr));

    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.end(), coo.column_indices.end()));
    thrust::transform(zip_begin, zip_end, zip_begin, OrientLowerTriangular());

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Reorder] Complete (total %.1f s)\n",
            std::chrono::duration<double>(t3 - t0).count());
}

// ============================================================================
// GPU-native HashOrder — 全 GPU 实现，无 D2H/H2D 传输
// GPU CSR 构建 → GPU MinHash → GPU Radix Sort → GPU Apply
// ============================================================================

// Universal hash: h(v) = (a*v + b) mod P
struct HashInitFunctor {
    uint32_t a, b;
    __host__ __device__
    uint32_t operator()(int v) const {
        constexpr uint64_t P = 4294967291ULL;  // largest prime < 2^32
        return (uint32_t)(((uint64_t)a * (unsigned)v + b) % P);
    }
};

// 1-hop MinHash: h_out[v] = min(h_in[v], min_{u ∈ N(v)} h_in[u])
struct MinHashHopFunctor {
    const int* row_ptr;
    const int* col_idx;
    const uint32_t* h_in;
    uint32_t* h_out;
    __host__ __device__
    void operator()(int v) const {
        uint32_t min_val = h_in[v];
        for (int k = row_ptr[v]; k < row_ptr[v + 1]; k++) {
            uint32_t nv = h_in[col_idx[k]];
            if (nv < min_val) min_val = nv;
        }
        h_out[v] = min_val;
    }
};

template<typename CooMatrix>
void reorder_hashorder_gpu(CooMatrix& coo, int num_hashes = 16, int num_hops = 1) {
    int N = coo.num_rows;
    size_t E = coo.num_entries;

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder-GPU] N=%d, E=%zu, hashes=%d, hops=%d\n",
            N, E, num_hashes, num_hops);

    // ================================================================
    // Step 1: Build symmetric CSR on GPU
    // ================================================================
    size_t sym_E = E * 2;
    thrust::device_vector<int> csr_cols(sym_E);
    thrust::device_vector<int> row_ptr(N + 1);
    {
        thrust::device_vector<int> sym_rows(sym_E);
        // Duplicate edges: (r,c) + (c,r)
        thrust::copy(coo.row_indices.begin(), coo.row_indices.end(), sym_rows.begin());
        thrust::copy(coo.column_indices.begin(), coo.column_indices.end(), sym_rows.begin() + E);
        thrust::copy(coo.column_indices.begin(), coo.column_indices.end(), csr_cols.begin());
        thrust::copy(coo.row_indices.begin(), coo.row_indices.end(), csr_cols.begin() + E);

        thrust::sort_by_key(sym_rows.begin(), sym_rows.end(), csr_cols.begin());

        thrust::lower_bound(thrust::device,
                            sym_rows.begin(), sym_rows.end(),
                            thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>(N + 1),
                            row_ptr.begin());
        // sym_rows freed at scope exit
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder-GPU] CSR built (%.3f s)\n",
            std::chrono::duration<double>(t1 - t0).count());

    const int* rp_ptr = thrust::raw_pointer_cast(row_ptr.data());
    const int* ci_ptr = thrust::raw_pointer_cast(csr_cols.data());

    // ================================================================
    // Step 2: MinHash init + 1-hop, per hash function
    // h_all[j * N + v] stores final hash value for vertex v, function j
    // ================================================================
    thrust::device_vector<uint32_t> h_all((size_t)num_hashes * N);

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(1, 4294967290u);
    thrust::device_vector<uint32_t> h_cur(N), h_next(N);

    for (int j = 0; j < num_hashes; j++) {
        // Init: h(v) = (a*v + b) % P
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(N),
            h_cur.begin(),
            HashInitFunctor{dist(rng), dist(rng)});

        // Multi-hop message passing (ping-pong between h_cur / h_next)
        MinHashHopFunctor hop_fn;
        hop_fn.row_ptr = rp_ptr;
        hop_fn.col_idx = ci_ptr;
        for (int hop = 0; hop < num_hops; hop++) {
            if (hop % 2 == 0) {
                hop_fn.h_in  = thrust::raw_pointer_cast(h_cur.data());
                hop_fn.h_out = thrust::raw_pointer_cast(h_next.data());
            } else {
                hop_fn.h_in  = thrust::raw_pointer_cast(h_next.data());
                hop_fn.h_out = thrust::raw_pointer_cast(h_cur.data());
            }
            thrust::for_each(
                thrust::device,
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(N),
                hop_fn);
        }

        // Result in h_next (odd hops) or h_cur (even hops)
        auto& result = (num_hops % 2 == 1) ? h_next : h_cur;
        thrust::copy(result.begin(), result.end(),
                     h_all.begin() + (size_t)j * N);
    }

    // Free CSR + temp hash buffers
    h_cur.resize(0); h_cur.shrink_to_fit();
    h_next.resize(0); h_next.shrink_to_fit();
    csr_cols.resize(0); csr_cols.shrink_to_fit();
    row_ptr.resize(0); row_ptr.shrink_to_fit();

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder-GPU] MinHash done (%.3f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    // ================================================================
    // Step 3: Lexicographic sort by hash signature
    // Multi-pass stable radix sort (least significant hash first)
    // ================================================================
    thrust::device_vector<int> order(N);
    thrust::sequence(order.begin(), order.end());

    thrust::device_vector<uint32_t> key(N);
    for (int j = num_hashes - 1; j >= 0; j--) {
        // Gather: key[i] = h_all[j*N + order[i]]
        thrust::gather(order.begin(), order.end(),
                       h_all.begin() + (size_t)j * N,
                       key.begin());
        // Stable radix sort (uint32_t keys)
        thrust::stable_sort_by_key(key.begin(), key.end(), order.begin());
    }
    h_all.resize(0); h_all.shrink_to_fit();
    key.resize(0); key.shrink_to_fit();

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder-GPU] Sort done (%.3f s)\n",
            std::chrono::duration<double>(t3 - t2).count());

    // ================================================================
    // Step 4: order → perm, apply to COO
    // ================================================================
    thrust::device_vector<int> perm(N);
    // perm[order[i]] = i
    thrust::scatter(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        order.begin(),
        perm.begin());
    order.resize(0); order.shrink_to_fit();

    int* perm_ptr = thrust::raw_pointer_cast(perm.data());
    thrust::transform(coo.row_indices.begin(), coo.row_indices.end(),
                      coo.row_indices.begin(), ApplyPerm(perm_ptr));
    thrust::transform(coo.column_indices.begin(), coo.column_indices.end(),
                      coo.column_indices.begin(), ApplyPerm(perm_ptr));

    // Re-orient to lower triangular
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(coo.row_indices.end(), coo.column_indices.end()));
    thrust::transform(zip_begin, zip_end, zip_begin, OrientLowerTriangular());

    auto t4 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder-GPU] Total (%.3f s)\n",
            std::chrono::duration<double>(t4 - t0).count());
}

}  // namespace btc
