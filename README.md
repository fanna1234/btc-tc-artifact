# BTC-TC: Exact GPU Triangle Counting with Hybrid Bit Tensor Cores and CUDA Cores

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SC26](https://img.shields.io/badge/SC26-accepted-2ea44f.svg)](https://sc26.supercomputing.org/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#citation)

Reproducibility artifact for the SC '26 paper:

> Kaifan Jia, Yongchun Jiang, Zhihao Ling, Minghui Zhang, Xuran Wang, Ran Bao,
> Haonan Zou, and Heng Zhang.
> *BTC-TC: Exact GPU Triangle Counting with Hybrid Bit Tensor Cores and CUDA Cores.*
> In Proceedings of the International Conference for High Performance Computing,
> Networking, Storage, and Analysis (SC '26), 2026.

**Authors** (✉ corresponding author):

| Author | Email |
|--------|-------|
| Kaifan Jia | jiakaifan23@mails.ucas.ac.cn |
| Yongchun Jiang | jiangyongchun24@mails.ucas.ac.cn |
| Zhihao Ling | lingzhihao24@mails.ucas.ac.cn |
| Minghui Zhang | zhangminghui241@mails.ucas.ac.cn |
| Xuran Wang | wangxuran25@mails.ucas.ac.cn |
| Ran Bao | baoran24@mails.ucas.ac.cn |
| Haonan Zou | zouhaonan23@mails.ucas.ac.cn |
| Heng Zhang ✉ | zhangheng17@iscas.ac.cn |

- **Code:** <https://github.com/fanna1234/btc-tc-artifact>
- **Archived:** Zenodo DOI — assigned at the 2026-08-25 artifact freeze (added here once minted)

BTC-TC accelerates exact triangle counting on GPUs through format-operator-dispatch
co-design around binary Tensor Cores (`m16n8k128.and.popc`). It evaluates
$L \odot (LL^T)$ using a hybrid masked operator that dispatches each mask block
by output density: dense blocks to Bit Tensor Cores, sparse blocks to a selective
CUDA-core path.

## Quick Start

**One command from scratch** — handles OS/Python deps, dataset download, build,
smoke test, and reproduction, in that order:

```bash
# Before the first run, install OS packages (needs sudo):
sudo apt-get install -y libnuma-dev libopenmpi-dev

bash scripts/run_all.sh           # full reproduction,  ~2.5-3 hours
bash scripts/run_all.sh --quick   # core claims only,   ~35 min
bash scripts/run_all.sh --smoke   # build + correctness, ~15 min
```

Total disk: ~8 GB (datasets 1.2 GB + build 5-6 GB). Total wall time on a single
modern NVIDIA GPU (sm_80+): roughly 30 min for smoke, 45 min for quick, 3 h for
full. `run_all.sh` is idempotent — rerunning resumes from the last failed step.

### Manual Step-by-Step (if `run_all.sh` has issues)

```bash
# 1. Python deps (add --break-system-packages on Ubuntu 24.04+ for PEP 668)
pip install -r requirements.txt

# 2. Download 36 benchmark datasets (~1.2 GB)
bash scripts/download_datasets.sh

# 3. Build BTC-TC + baselines (~30-45 min, compiles GraphBLAS from source)
bash scripts/build_all.sh

# 4. Smoke test (<1 min)
bash scripts/smoke_test.sh

# 5. Full reproduction (~2-3 hours) — or add --quick for ~30 min
bash scripts/reproduce_paper.sh
```

### Expected Output (smoke test)

```
=== BTC-TC Smoke Test ===
PASS  BTC-TC  lpl1: 97633 triangles (expected 97633)
PASS  BTC-TC  cant: 18370150 triangles (expected 18370150)
PASS  BTC-TC  bcsstk23: 29737 triangles (expected 29737)
PASS  ToT     lpl1: 97633 triangles (expected 97633)
PASS  ToT     cant: 18370150 triangles (expected 18370150)
=== Results: 5 passed, 0 failed, 0 skipped / 5 total ===
ALL PASSED
```

## Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA sm_80+ (Ampere/Hopper/Blackwell), >= 16 GB | A800 80GB, H100 80GB, PRO 6000 96GB |
| CUDA Toolkit | >= 12.2 | 12.2, 13.1, 13.2 |
| GCC | >= 11 | 13.3 |
| CMake | >= 3.18 | 3.28 |
| Python | >= 3.10 | 3.12 |
| libnuma-dev | required | (for vertex reordering) |
| MPI | optional | (only for TRUST baseline) |

## Claims and Verification

See **[CLAIMS.md](CLAIMS.md)** for a detailed mapping of each paper claim to the
exact command and expected output that verifies it. Key claims:

| Claim | Section | Verification |
|-------|---------|--------------|
| BTC-TC exact on all 36 datasets | 4.3 | `smoke_test.sh` (quick) or full benchmark |
| 1.92x GM kernel speedup vs ToT | 4.2 | `CLAIMS.md` "Kernel Speedup" |
| 8.0x GM E2E speedup vs ToT | 4.2 | `CLAIMS.md` "E2E Speedup" |
| Hybrid dispatch adds 1.45x | 4.6 | `run_ablation.sh` |
| Consistent across 3 GPU generations | 4.4 | Pre-computed CSVs for 3 devices |
| Threshold insensitive in [64, 2048] | 3.4 | `run_tau_sweep.sh` |

## Artifact Evaluation (SC26)

This artifact targets the SC26 reproducibility badges below. Each row lists the single
check a reviewer can run to grant it:

| Badge | How a reviewer verifies it |
|-------|----------------------------|
| **Artifacts Available** | Public repository, archived on Zenodo with a persistent DOI (see [Citation](#citation)). |
| **Artifacts Evaluated — Functional** | `bash scripts/run_all.sh --smoke` builds BTC-TC + baselines and passes all correctness checks (~15 min). |
| **Results Reproduced** | `bash scripts/run_all.sh --quick` reproduces the headline speedups (1.92× kernel, 8.0× E2E) in ~35 min. |

Every paper claim is mapped to its exact command and expected value in
**[CLAIMS.md](CLAIMS.md)**.

## Directory Structure

```
btc-tc-artifact/
├── btc/                  Core BTC-TC library (header-only C++/CUDA)
├── apps/                 Paper executables (11 targets)
│   ├── btc_tc_lite.cu        Main program (BTC-Lite: auto block-size)
│   ├── btc_tc_adaptive_*.cu  Fixed block-size variants
│   ├── btc_tc_v*.cu          Ablation variants (pure-MMA vs hybrid)
│   ├── tau_sweep.cu          Threshold sensitivity (Fig 6)
│   ├── hybrid_ablation.cu    Hybrid dispatch ablation (Fig 9a)
│   └── btc_blocksize_bench.cu MMA shape benchmark (Fig 9c)
├── baselines/            12 baseline implementations (source only)
├── scripts/
│   ├── reproduce_paper.sh    One-click reproduction pipeline
│   ├── build_all.sh          Build BTC-TC + all baselines
│   ├── download_datasets.sh  Download 36 benchmark datasets
│   ├── smoke_test.sh         Quick correctness verification
│   ├── run_*.sh              Individual experiment runners
│   └── figures/              Figure generation scripts
├── results/              Pre-computed results (CSV, for verification)
│   ├── pro6000/csv/          RTX PRO 6000 (Blackwell sm_120a)
│   ├── h100/csv/             H100 SXM (Hopper sm_90a)
│   ├── a800/csv/             A800 SXM (Ampere sm_80)
│   ├── ablation/csv/         Ablation study data
│   └── ...                   tau_sweep, e2e_breakdown, ncu, etc.
├── data/
│   └── paper_datasets.txt    36 dataset names
├── CLAIMS.md             Claim-to-command verification guide
├── CMakeLists.txt
├── requirements.txt
└── LICENSE
```

## Building

```bash
# Recommended: build everything (BTC-TC + all baselines)
bash scripts/build_all.sh

# Or manually:
mkdir -p build && cd build
cmake ..                    # Auto-detects GPU architecture
make -j$(nproc)
```

To build only BTC-TC (skip baselines):
```bash
cmake .. -DBTC_BUILD_BASELINES=OFF && make -j$(nproc)
```

## Reproduction Pipeline

`reproduce_paper.sh` runs the complete pipeline:

1. **Smoke test** — correctness verification on 3 small graphs
2. **Full benchmark** — all methods x 36 datasets (5 runs each)
3. **Claim verification** — automated check of key paper numbers
4. **Ablation experiments** — hybrid dispatch, block-size, MMA shape, reordering
5. **Sensitivity sweeps** — threshold (tau) and E2E breakdown
6. **Figure generation** — all paper figures from result CSVs

### Verifying Without Re-running

Pre-computed results from 3 GPU platforms are included in `results/`. To regenerate
figures directly from these results:
```bash
bash scripts/regenerate_all_figures.sh    # Output: results/figures/
```

To verify specific paper numbers without running benchmarks, see `CLAIMS.md` for
one-liner Python commands that compute speedups from the pre-computed CSVs.

## Datasets

36 graphs from [SuiteSparse Matrix Collection](https://sparse.tamu.edu), covering:
- Scientific computing meshes (FEM, molecular dynamics)
- Social networks (wiki-Vote, flickr)
- Web graphs (eu-2005, web-Google, web-NotreDame)
- Structural engineering (bcsstk series)

Total download: ~1.2 GB. List: `data/paper_datasets.txt`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `numa.h: No such file` | Install `libnuma-dev`: `apt install libnuma-dev` |
| `thrust::distance` error (ToT) | Already patched in this artifact for CUDA >= 13.2 |
| TRUST build fails | Ensure MPI is installed: `apt install libopenmpi-dev` |
| Some TC-Compare baselines fail | Non-critical; `build_all.sh` continues on failure |
| CMake picks wrong CUDA | Set: `cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` |

## Citation

If you use BTC-TC in your research, please cite:

```bibtex
@inproceedings{jia2026btctc,
  author    = {Jia, Kaifan and Jiang, Yongchun and Ling, Zhihao and Zhang, Minghui and
               Wang, Xuran and Bao, Ran and Zou, Haonan and Zhang, Heng},
  title     = {{BTC-TC}: Exact {GPU} Triangle Counting with Hybrid Bit Tensor Cores and {CUDA} Cores},
  booktitle = {Proceedings of the International Conference for High Performance Computing,
               Networking, Storage, and Analysis (SC '26)},
  year      = {2026},
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided; GitHub renders a
**"Cite this repository"** button from it on the project page.

## License

Released under the [MIT License](LICENSE).
