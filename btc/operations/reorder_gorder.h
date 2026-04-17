#pragma once

#include <cstddef>
#include <vector>

namespace btc {

// Compute Gorder permutation (SIGMOD'16: "Speedup Graph Processing by Graph Ordering")
// Uses sliding-window greedy: each step picks the vertex with most neighbors in window.
// Input:  N = number of nodes, E = number of edges, rows[E], cols[E] (lower-tri)
// Output: permutation array perm[old_id] = new_id, size N
// window_size: sliding window size (default=5 in paper; larger = better quality, slower)
std::vector<int> compute_gorder_perm(int N, size_t E,
                                     const int* rows, const int* cols,
                                     int window_size = 5);

}  // namespace btc
