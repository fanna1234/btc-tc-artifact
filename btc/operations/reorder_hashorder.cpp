#include "reorder_hashorder.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <parallel/algorithm>
#include <random>
#include <vector>

// HashOrder: probabilistic graph reordering via MinHash (ICLR'24)
// Algorithm: multi-hop MinHash → sort by hash signature → local degree tiebreak
// O(hops × E × num_hashes) time, trivially parallelizable

namespace btc {
namespace hashorder_impl {

struct SymCSR {
    int n;
    std::vector<size_t> cd;
    std::vector<int> adj;

    SymCSR(int N, size_t E, const int* rows, const int* cols) : n(N), cd(N + 1, 0) {
        for (size_t i = 0; i < E; i++) {
            cd[rows[i] + 1]++;
            cd[cols[i] + 1]++;
        }
        for (int i = 1; i <= n; i++) cd[i] += cd[i - 1];

        adj.resize(cd[n]);
        std::vector<size_t> offset(cd.begin(), cd.end());
        for (size_t i = 0; i < E; i++) {
            int r = rows[i], c = cols[i];
            adj[offset[r]++] = c;
            adj[offset[c]++] = r;
        }
    }
};

}  // namespace hashorder_impl

std::vector<int> compute_hashorder_perm(int N, size_t E,
                                        const int* rows, const int* cols,
                                        int hops, int num_hashes) {
    using namespace hashorder_impl;

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder] Building CSR: N=%d, E=%zu\n", N, E);

    SymCSR g(N, E, rows, cols);

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder] CSR built (%.1f s), hops=%d, hashes=%d\n",
            std::chrono::duration<double>(t1 - t0).count(), hops, num_hashes);

    // Step 1: Initialize random permutations as hash functions
    // pi_j[v] = random permutation of [0, N) for hash function j
    // We use uint32_t hash values to save memory (N < 2^32)
    std::vector<std::vector<uint32_t>> h_cur(num_hashes, std::vector<uint32_t>(N));

    {
        std::mt19937 rng(42);
        std::vector<uint32_t> perm(N);
        for (int j = 0; j < num_hashes; j++) {
            std::iota(perm.begin(), perm.end(), 0u);
            std::shuffle(perm.begin(), perm.end(), rng);
            #pragma omp parallel for
            for (int v = 0; v < N; v++)
                h_cur[j][v] = perm[v];
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder] Init done (%.1f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    // Step 2: Multi-hop message passing: h[v] = min over neighbors of h[u]
    for (int hop = 0; hop < hops; hop++) {
        auto hop_start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < num_hashes; j++) {
            std::vector<uint32_t> h_next(N);
            #pragma omp parallel for schedule(dynamic, 4096)
            for (int v = 0; v < N; v++) {
                uint32_t min_val = h_cur[j][v];  // include self
                for (size_t k = g.cd[v]; k < g.cd[v + 1]; k++) {
                    uint32_t nv = h_cur[j][g.adj[k]];
                    if (nv < min_val) min_val = nv;
                }
                h_next[v] = min_val;
            }
            h_cur[j] = std::move(h_next);
        }

        auto hop_end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "[HashOrder] Hop %d done (%.1f s)\n", hop + 1,
                std::chrono::duration<double>(hop_end - hop_start).count());
    }

    // Step 3: Sort vertices by concatenated hash signature
    // For sorting, we use the hash values as a composite key
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);

    // Sort by hash signature (lexicographic on h[0], h[1], ..., h[num_hashes-1])
    // Then by degree (descending) as tiebreaker for same-bucket vertices
    __gnu_parallel::sort(order.begin(), order.end(), [&](int a, int b) {
        for (int j = 0; j < num_hashes; j++) {
            if (h_cur[j][a] != h_cur[j][b])
                return h_cur[j][a] < h_cur[j][b];
        }
        // Same hash bucket: sort by degree descending
        size_t deg_a = g.cd[a + 1] - g.cd[a];
        size_t deg_b = g.cd[b + 1] - g.cd[b];
        return deg_a > deg_b;
    });

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[HashOrder] Sort done (%.1f s), total=%.1f s\n",
            std::chrono::duration<double>(t3 - t2).count(),
            std::chrono::duration<double>(t3 - t0).count());

    // Step 4: Convert order → rank (perm[old_id] = new_id)
    std::vector<int> perm(N);
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        perm[order[i]] = i;

    return perm;
}

}  // namespace btc
