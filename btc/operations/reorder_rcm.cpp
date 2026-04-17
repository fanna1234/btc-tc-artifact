#include "reorder_rcm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

// Reverse Cuthill-McKee (RCM) for bandwidth reduction.
// Algorithm:
//   1. Build symmetric CSR
//   2. Find pseudo-peripheral starting node (George-Liu algorithm)
//   3. BFS with degree-ascending neighbor ordering (Cuthill-McKee)
//   4. Reverse the order (RCM)
// Result: nonzeros concentrate near diagonal → fewer BCSR blocks

namespace btc {
namespace rcm_impl {

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

        // Sort neighbors by degree ascending (Cuthill-McKee enhancement)
        // We'll sort after building since we need all degrees first
    }

    int degree(int u) const { return (int)(cd[u + 1] - cd[u]); }
};

// George-Liu algorithm: find a pseudo-peripheral node
// Start from a low-degree node, BFS to find the farthest node, repeat.
static int find_pseudo_peripheral(const SymCSR& g) {
    // Start from a minimum-degree non-isolated node
    int start = -1;
    int min_deg = g.n + 1;
    for (int i = 0; i < g.n; i++) {
        int d = g.degree(i);
        if (d > 0 && d < min_deg) {
            min_deg = d;
            start = i;
        }
    }
    if (start == -1) return 0;  // all isolated

    // Iterate BFS to find pseudo-peripheral node
    std::vector<int> dist(g.n, -1);
    std::vector<int> queue;
    queue.reserve(g.n);

    for (int iter = 0; iter < 5; iter++) {
        // BFS from start
        std::fill(dist.begin(), dist.end(), -1);
        queue.clear();
        dist[start] = 0;
        queue.push_back(start);
        size_t front = 0;
        int farthest = start;
        int max_dist = 0;

        while (front < queue.size()) {
            int u = queue[front++];
            for (size_t j = g.cd[u]; j < g.cd[u + 1]; j++) {
                int v = g.adj[j];
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                    if (dist[v] > max_dist) {
                        max_dist = dist[v];
                        farthest = v;
                    }
                }
            }
        }

        if (farthest == start) break;  // converged

        // Among nodes at max distance, pick one with minimum degree
        int best = farthest;
        int best_deg = g.degree(farthest);
        for (size_t i = front; i-- > 0; ) {
            int u = queue[i];
            if (dist[u] < max_dist) break;
            if (g.degree(u) < best_deg) {
                best = u;
                best_deg = g.degree(u);
            }
        }
        start = best;
    }

    return start;
}

}  // namespace rcm_impl

std::vector<int> compute_rcm_perm(int N, size_t E,
                                   const int* rows, const int* cols) {
    using namespace rcm_impl;

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RCM] Building CSR: N=%d, E=%zu\n", N, E);

    SymCSR g(N, E, rows, cols);

    // Sort each node's neighbors by degree ascending (Cuthill-McKee key step)
    for (int u = 0; u < N; u++) {
        std::sort(g.adj.begin() + g.cd[u], g.adj.begin() + g.cd[u + 1],
                  [&](int a, int b) { return g.degree(a) < g.degree(b); });
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RCM] CSR built + sorted (%.1f s)\n",
            std::chrono::duration<double>(t1 - t0).count());

    // Find pseudo-peripheral starting node
    int start = find_pseudo_peripheral(g);
    fprintf(stderr, "[RCM] Pseudo-peripheral node: %d (degree=%d)\n",
            start, g.degree(start));

    // Cuthill-McKee BFS: within each level, neighbors already sorted by degree
    std::vector<int> order;
    order.reserve(N);
    std::vector<bool> visited(N, false);

    order.push_back(start);
    visited[start] = true;
    size_t front = 0;

    while (front < order.size()) {
        int u = order[front++];
        // Neighbors are pre-sorted by degree ascending
        for (size_t j = g.cd[u]; j < g.cd[u + 1]; j++) {
            int v = g.adj[j];
            if (!visited[v]) {
                visited[v] = true;
                order.push_back(v);
            }
        }
    }

    // Handle disconnected components
    if ((int)order.size() < N) {
        // For remaining components, find pseudo-peripheral within each
        for (int i = 0; i < N; i++) {
            if (!visited[i]) {
                // BFS from this node for its component
                order.push_back(i);
                visited[i] = true;
                size_t comp_front = order.size() - 1;
                while (comp_front < order.size()) {
                    int u = order[comp_front++];
                    for (size_t j = g.cd[u]; j < g.cd[u + 1]; j++) {
                        int v = g.adj[j];
                        if (!visited[v]) {
                            visited[v] = true;
                            order.push_back(v);
                        }
                    }
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RCM] BFS done (%.1f s)\n",
            std::chrono::duration<double>(t2 - t1).count());

    // REVERSE the order (the "R" in RCM)
    std::reverse(order.begin(), order.end());

    // Convert order → rank (perm[old_id] = new_id)
    std::vector<int> perm(N);
    for (int i = 0; i < N; i++)
        perm[order[i]] = i;

    auto t3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[RCM] Complete (total %.1f s)\n",
            std::chrono::duration<double>(t3 - t0).count());

    return perm;
}

}  // namespace btc
