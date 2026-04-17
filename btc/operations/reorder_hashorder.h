#pragma once

#include <cstddef>
#include <vector>

namespace btc {

// Compute HashOrder permutation (ICLR'24: probabilistic graph reordering)
// Uses MinHash on multi-hop neighborhoods to cluster frequently co-accessed nodes.
// ~592x faster than Gorder with comparable or higher quality.
// Input:  N = number of nodes, E = number of edges, rows[E], cols[E] (lower-tri)
// Output: permutation array perm[old_id] = new_id, size N
// hops: number of message-passing hops (default=1, best for BCSR block reduction)
// num_hashes: number of MinHash functions (default=16)
std::vector<int> compute_hashorder_perm(int N, size_t E,
                                        const int* rows, const int* cols,
                                        int hops = 1, int num_hashes = 16);

}  // namespace btc
