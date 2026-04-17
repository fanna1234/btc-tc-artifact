# Meta-Adaptive Strategy Results

We have implemented a **Meta-Adaptive** execution model that dynamically selects between `16x32` and `16x128` block sizes based on input graph characteristics.

## Heuristic Logic

1.  **Analysis Phase**: Rapidly count potential blocks ($N_{32}$ and $N_{128}$) using Thrust inner products on the COO structure.
2.  **Decision Tree**:
    *   **Huge Graph** ($N_{128} > 500,000$): Force **16x128**. (Maximize throughput for massive workloads).
    *   **Tiny Graph** ($N_{128} < 1,000$): Force **16x32**. (Minimize kernel launch/tail overheads).
    *   **Medium Graph**: Check Density Ratio $R = N_{32} / N_{128}$.
        *   If $R < 1.3$ (Sparse): Use **16x32** (Better load balancing for sparse/irregular patterns).
        *   Else (Dense): Use **16x128** (Better memory coalescing).

## Verification Results

| Dataset | $N_{32}$ Blocks | $N_{128}$ Blocks | Ratio | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **webbase-1M** | 587,597 | 513,796 | 1.14 | **16x128** (Huge Rule > 500k) |
| **cant.mtx** | 280,527 | 162,379 | 1.73 | **16x128** (Dense Ratio > 1.3) |
| **web-NotreDame** | 472,924 | 421,084 | 1.12 | **16x32** (Sparse Ratio < 1.3) |
| **road_usa** | 21.0M | 19.1M | 1.10 | **16x128** (Huge Rule) |

## Implementation Details

*   **Source**: `apps/btc_tc_meta_adaptive.cu`
*   **Analysis Kernel**: `btc/operations/analyze_blocks.h`
*   **Overhead**: The analysis phase takes minimal time (included in preprocessing) relative to the compute time improvement.

Run the adaptive solver:
```bash
./build/apps/btc_tc_meta_adaptive -i ../data/web-NotreDame.mtx
```
