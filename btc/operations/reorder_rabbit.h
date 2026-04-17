#pragma once

#include <cstddef>
#include <vector>

namespace btc {

// Compute Rabbit Order permutation from lower-triangular COO edges (host memory).
// Input:  N = number of nodes, E = number of edges, rows[E], cols[E] (lower-tri: rows[i] > cols[i])
// Output: permutation array perm[old_id] = new_id, size N
std::vector<int> compute_rabbit_order_perm(int N, size_t E,
                                           const int* rows, const int* cols);

}  // namespace btc
