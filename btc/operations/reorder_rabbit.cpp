#include "reorder_rabbit.h"

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <vector>

// Rabbit Order: header-only community-based graph reordering (IPDPS'16)
// https://github.com/araij/rabbit_order
#include "../external/rabbit_order.hpp"

namespace btc {

std::vector<int> compute_rabbit_order_perm(int N, size_t E,
                                           const int* rows, const int* cols) {
    using rabbit_order::vint;
    using rabbit_order::edge;

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RabbitOrder] Building adjacency list: N=%d, E=%zu, threads=%d\n",
            N, E, omp_get_max_threads());

    // Step 1: Count degrees for pre-allocation
    std::vector<size_t> deg(N, 0);
    #pragma omp parallel for
    for (size_t i = 0; i < E; ++i) {
        #pragma omp atomic
        deg[rows[i]]++;
        #pragma omp atomic
        deg[cols[i]]++;
    }

    // Step 2: Build symmetric adjacency list (lower-tri → both directions)
    std::vector<std::vector<edge>> adj(N);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int v = 0; v < N; ++v)
        adj[v].reserve(deg[v]);
    deg.clear();
    deg.shrink_to_fit();

    // Sequential fill (vectors not thread-safe for push_back to same element)
    for (size_t i = 0; i < E; ++i) {
        int r = rows[i], c = cols[i];
        adj[r].push_back({static_cast<vint>(c), 1.0f});
        adj[c].push_back({static_cast<vint>(r), 1.0f});
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RabbitOrder] Adj list built (%.1f s)\n",
            std::chrono::duration<double>(t1 - t0).count());

    // Step 3: Run Rabbit Order aggregation + permutation
    fprintf(stderr, "[RabbitOrder] Running community detection & reordering...\n");
    auto g = rabbit_order::aggregate(std::move(adj));

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RabbitOrder] Aggregation done (%.1f s), computing permutation...\n",
            std::chrono::duration<double>(t2 - t1).count());

    auto p = rabbit_order::compute_perm(g);

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RabbitOrder] Permutation done (%.1f s), total=%.1f s\n",
            std::chrono::duration<double>(t3 - t2).count(),
            std::chrono::duration<double>(t3 - t0).count());

    // Step 4: Convert vint (uint32_t) perm to int
    std::vector<int> result(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        result[i] = static_cast<int>(p[i]);

    return result;
}

}  // namespace btc
