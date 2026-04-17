#pragma once

#include <cstddef>
#include <vector>

namespace btc {

// Compute Reverse Cuthill-McKee permutation for bandwidth reduction.
// Concentrates nonzeros near the diagonal — ideal for BCSR triangle counting.
// Input:  N = number of nodes, E = number of edges, rows[E], cols[E] (lower-tri)
// Output: permutation array perm[old_id] = new_id, size N
std::vector<int> compute_rcm_perm(int N, size_t E,
                                   const int* rows, const int* cols);

}  // namespace btc
