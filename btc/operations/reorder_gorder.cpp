#include "reorder_gorder.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// Self-contained Gorder implementation (SIGMOD'16)
// Based on: Hao Wei et al., "Speedup Graph Processing by Graph Ordering"
// Adapted from: lecfab/rescience-gorder (MIT License)

namespace btc {
namespace gorder_impl {

typedef unsigned long ul;
static constexpr int INFTY = INT_MAX / 2;

// ============================================================================
// UnitHeap: O(1) amortized priority queue for integer keys
// ============================================================================
struct ListElement {
    int key;
    ul prev, next;
};

struct HeadEnd {
    ul first, second;
};

struct UnitHeap {
    std::vector<int> update;
    std::vector<ListElement> LL;
    std::vector<HeadEnd> Header;
    size_t heapsize = 0;
    ul top;
    ul huge;
    ul none;

    UnitHeap(ul size) : none(size + 2), huge((ul)sqrt((double)size)) {
        LL.resize(size, {INFTY, none, none});
        update.resize(size, INFTY);
    }

    void InsertElement(ul index, int key) {
        LL[index].key = key;
        update[index] = -key;
        heapsize++;
    }

    void ReConstruct() {
        std::vector<ul> g(heapsize);
        std::iota(g.begin(), g.end(), 0);
        std::sort(g.begin(), g.end(), [&](ul a, ul b) {
            return LL[a].key > LL[b].key || (LL[a].key == LL[b].key && a < b);
        });

        top = g[0];
        int cur_key = LL[top].key;
        Header.resize(10 * cur_key + 2, {none, none});
        Header[cur_key].first = top;

        for (size_t i = 0; i < g.size(); i++) {
            ul v = g[i];
            LL[v].prev = (i > 0) ? g[i - 1] : none;
            LL[v].next = (i < g.size() - 1) ? g[i + 1] : none;
            int key = LL[v].key;
            if (key != cur_key) {
                Header[cur_key].second = g[i - 1];
                Header[key].first = g[i];
                cur_key = key;
            }
        }
        Header[cur_key].second = g.back();
    }

    void erase_key_element(ul index, ul next, ul prev) {
        int key = LL[index].key;
        if (Header[key].first == Header[key].second)
            Header[key].first = Header[key].second = none;
        else if (index == Header[key].first)
            Header[key].first = next;
        else if (index == Header[key].second)
            Header[key].second = prev;
    }

    void DecreaseTop() {
        ul next = LL[top].next;
        if (next == none) return;

        int key = LL[top].key;
        int leftover = update[top] / 2;
        int new_key = key + update[top] - leftover;
        if (new_key >= LL[next].key) return;
        update[top] = leftover;

        ul level_tail = Header[key].second;
        ul next_level = LL[level_tail].next;
        while (next_level != none && LL[next_level].key >= new_key) {
            level_tail = Header[LL[next_level].key].second;
            next_level = LL[level_tail].next;
        }

        LL[next].prev = none;
        LL[top].prev = level_tail;
        LL[top].next = next_level;
        LL[level_tail].next = top;
        if (next_level != none) LL[next_level].prev = top;

        erase_key_element(top, next, none);
        LL[top].key = new_key;
        Header[new_key].second = top;
        if (Header[new_key].first == none) Header[new_key].first = top;
        top = next;
    }

    ul ExtractMax() {
        ul tmptop;
        do {
            tmptop = top;
            if (update[top] < 0) DecreaseTop();
        } while (top != tmptop);
        DeleteElement(top);
        return tmptop;
    }

    void DeleteElement(ul index) {
        update[index] = INFTY;
        ul prev = LL[index].prev;
        ul next = LL[index].next;
        if (prev != none) LL[prev].next = next;
        if (next != none) LL[next].prev = prev;
        erase_key_element(index, next, prev);
        if (top == index) top = next;
        LL[index].prev = LL[index].next = none;
        heapsize--;
    }

    void lazyIncrement(ul index, int up) {
        if (update[index] == INFTY) return;
        if (update[index] == 0 && up > 0)
            IncrementKey(index);
        else
            update[index] += up;
    }

    void IncrementKey(ul index) {
        ul level_head = Header[LL[index].key].first;
        ul prev = LL[index].prev;
        ul next = LL[index].next;

        if (level_head != index) {
            LL[prev].next = next;
            if (next != none) LL[next].prev = prev;
            ul prev_level = LL[level_head].prev;
            LL[index].prev = prev_level;
            LL[index].next = level_head;
            LL[level_head].prev = index;
            if (prev_level != none) LL[prev_level].next = index;
        }

        erase_key_element(index, next, prev);
        int key = ++LL[index].key;
        Header[key].second = index;
        if (Header[key].first == none) {
            Header[key].first = index;
            if (key > LL[top].key) top = index;
        }

        if (key + 4 >= (int)Header.size())
            Header.resize(Header.size() * 2, {none, none});
    }
};

// ============================================================================
// Symmetric CSR graph
// ============================================================================
struct SymCSR {
    ul n;
    std::vector<ul> cd;   // cumulative degree, size n+1
    std::vector<ul> adj;  // neighbor IDs

    // Build from lower-triangular edges
    SymCSR(int N, size_t E, const int* rows, const int* cols) : n(N), cd(N + 1, 0) {
        // Count degrees
        for (size_t i = 0; i < E; i++) {
            cd[rows[i] + 1]++;
            cd[cols[i] + 1]++;
        }
        for (ul i = 1; i <= n; i++) cd[i] += cd[i - 1];

        adj.resize(cd[n]);
        std::vector<ul> offset(cd.begin(), cd.end());
        for (size_t i = 0; i < E; i++) {
            int r = rows[i], c = cols[i];
            adj[offset[r]++] = c;
            adj[offset[c]++] = r;
        }

        // Sort neighbors for each vertex
        for (ul u = 0; u < n; u++)
            std::sort(adj.begin() + cd[u], adj.begin() + cd[u + 1]);
    }

    ul degree(ul u) const { return cd[u + 1] - cd[u]; }
    const ul* neigh_beg(ul u) const { return &adj[cd[u]]; }
    const ul* neigh_end(ul u) const { return &adj[cd[u + 1]]; }
};

// ============================================================================
// Gorder core algorithm
// ============================================================================
static void move_window(const SymCSR& g, UnitHeap& heap,
                        ul new_node, ul old_node) {
    auto old_it = g.neigh_beg(old_node);
    auto new_it = g.neigh_beg(new_node);
    auto old_end = g.neigh_end(old_node);
    auto new_end = g.neigh_end(new_node);

    if (old_node == new_node)
        old_it = old_end;  // no old node to remove
    else {
        // Decrease children of old node
        if (g.degree(old_node) <= heap.huge)
            for (auto it = g.neigh_beg(old_node); it < old_end; ++it)
                heap.lazyIncrement(*it, -1);
    }

    // Find non-common parents between old and new node
    std::vector<ul> old_parents, new_parents;
    while (true) {
        int factor = -1;
        if (old_it >= old_end) {
            if (new_it >= new_end) break;
            factor = 1;
        } else if (new_it < new_end) {
            if (*new_it == *old_it) {
                old_it++;
                new_it++;
                continue;
            }
            if (*new_it < *old_it) factor = 1;
        }

        if (factor == -1) {
            if (g.degree(*old_it) <= heap.huge) old_parents.push_back(*old_it);
            old_it++;
        } else {
            if (g.degree(*new_it) <= heap.huge) new_parents.push_back(*new_it);
            new_it++;
        }
    }

    // Decrease parents and siblings of old node
    for (auto p : old_parents) {
        heap.lazyIncrement(p, -1);
        for (auto it = g.neigh_beg(p); it < g.neigh_end(p); ++it)
            if (*it != old_node) heap.lazyIncrement(*it, -1);
    }

    // Increase children of new node
    if (g.degree(new_node) <= heap.huge)
        for (auto it = g.neigh_beg(new_node); it < new_end; ++it)
            heap.lazyIncrement(*it, +1);

    // Increase parents and siblings of new node
    for (auto p : new_parents) {
        heap.lazyIncrement(p, +1);
        for (auto it = g.neigh_beg(p); it < g.neigh_end(p); ++it)
            if (*it != new_node) heap.lazyIncrement(*it, +1);
    }
}

static std::vector<ul> gorder_core(const SymCSR& g, ul window) {
    UnitHeap heap(g.n);

    std::vector<ul> isolates;
    for (ul u = 0; u < g.n; u++) {
        if (g.degree(u) == 0)
            isolates.push_back(u);
        else
            heap.InsertElement(u, (int)g.degree(u));
    }

    heap.ReConstruct();

    std::vector<ul> order;
    order.reserve(g.n);

    ul hub = heap.top;
    order.push_back(hub);
    heap.DeleteElement(hub);
    move_window(g, heap, hub, hub);

    ul progress_step = g.n / 20;
    if (progress_step == 0) progress_step = 1;

    while (heap.heapsize > 0) {
        ul new_node = heap.ExtractMax();
        order.push_back(new_node);

        ul old_node = new_node;
        if (order.size() > window)
            old_node = order[order.size() - window - 1];
        move_window(g, heap, new_node, old_node);

        if (order.size() % progress_step == 0)
            fprintf(stderr, "[Gorder] Progress: %zu/%lu (%.0f%%)\n",
                    order.size(), g.n, 100.0 * order.size() / g.n);
    }

    order.insert(order.end(), isolates.begin(), isolates.end());

    // Convert order → rank (perm[old_id] = new_id)
    std::vector<ul> rank(g.n);
    for (ul i = 0; i < order.size(); i++)
        rank[order[i]] = i;
    return rank;
}

}  // namespace gorder_impl

// ============================================================================
// Public interface
// ============================================================================
std::vector<int> compute_gorder_perm(int N, size_t E,
                                     const int* rows, const int* cols,
                                     int window_size) {
    using namespace gorder_impl;

    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Gorder] Building CSR: N=%d, E=%zu\n", N, E);

    SymCSR g(N, E, rows, cols);

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Gorder] CSR built (%.1f s), running Gorder (window=%d)...\n",
            std::chrono::duration<double>(t1 - t0).count(), window_size);

    auto rank = gorder_core(g, (ul)window_size);

    auto t2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[Gorder] Done (%.1f s), total=%.1f s\n",
            std::chrono::duration<double>(t2 - t1).count(),
            std::chrono::duration<double>(t2 - t0).count());

    std::vector<int> result(N);
    for (int i = 0; i < N; i++)
        result[i] = (int)rank[i];
    return result;
}

}  // namespace btc
