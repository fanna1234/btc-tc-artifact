#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace btc {

template<typename CsrMatrix>
size_t cpu_tc_intersection(const CsrMatrix& mat)
{
    size_t numTriangles = 0;
    const auto num_rows = mat.num_rows;
    const auto& row_ptr = mat.row_pointers;
    const auto& col_idx = mat.column_indices;

    // Using OpenMP for parallel execution
    #pragma omp parallel for reduction(+:numTriangles) schedule(dynamic, 256)
    for (int r = 0; r < num_rows; ++r) {
        int r_start = row_ptr[r];
        int r_end = row_ptr[r+1];
        
        // Iterate over neighbors u of r
        for (int i = r_start; i < r_end; ++i) {
            int u = col_idx[i];
            
            // Perform intersection of neighbor lists of r and u
            // N(r) = col_idx[r_start...r_end]
            // N(u) = col_idx[u_start...u_end]
            
            int u_start = row_ptr[u];
            int u_end = row_ptr[u+1];
            
            int p_r = r_start;
            int p_u = u_start;
            
            // Merge-path based intersection (requires sorted adjacency lists)
            while (p_r < r_end && p_u < u_end) {
                int v_r = col_idx[p_r];
                int v_u = col_idx[p_u];
                
                if (v_r < v_u) {
                    p_r++;
                } else if (v_r > v_u) {
                    p_u++;
                } else {
                    // Found common neighbor
                    numTriangles++;
                    p_r++;
                    p_u++;
                }
            }
        }
    }
    return numTriangles;
}

template<typename CsrMatrix>
size_t cpu_tc_intersection_lower_triangular(const CsrMatrix& mat) {
    size_t numTriangles = 0;
    const auto num_rows = mat.num_rows;
    const auto& row_ptr = mat.row_pointers;
    const auto& col_idx = mat.column_indices;

    // 假设输入已经是下三角矩阵 (row > col)
    #pragma omp parallel for reduction(+:numTriangles) schedule(dynamic, 256)
    for (int r = 0; r < num_rows; ++r) {
        int r_start = row_ptr[r];
        int r_end = row_ptr[r + 1];
        
        // 遍历r的所有邻居u (由于下三角，u < r)
        for (int i = r_start; i < r_end; ++i) {
            int u = col_idx[i];
            
            int u_start = row_ptr[u];
            int u_end = row_ptr[u + 1];
            
            int p_r = r_start;
            int p_u = u_start;
            
            // 交集计数：寻找同时是r和u邻居的节点v
            while (p_r < r_end && p_u < u_end) {
                int v_r = col_idx[p_r];
                int v_u = col_idx[p_u];
                
                if (v_r < v_u) {
                    p_r++;
                } else if (v_r > v_u) {
                    p_u++;
                } else {
                    // 找到公共邻居v，构成三角形 (r, u, v)
                    numTriangles++;
                    p_r++;
                    p_u++;
                }
            }
        }
    }
    return numTriangles;
}

}  // namespace btc