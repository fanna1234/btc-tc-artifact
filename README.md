# BTC-TC: Exact GPU Triangle Counting via Bit Tensor Core Co-Design

Artifact for SC26 submission **pap257**.

BTC-TC accelerates exact triangle counting on GPUs through format-operator-dispatch
co-design around binary Tensor Cores (`m16n8k128.and.popc`). It evaluates
$L \odot (LL^T)$ using a hybrid masked operator that dispatches each mask block
by output density: dense blocks to Bit Tensor Cores, sparse blocks to a selective
CUDA-core path.

## Quick Start

```bash
# 1. Install Python dependencies (for figure generation)
pip install -r requirements.txt

# 2. Download all 36 benchmark datasets (~1.2 GB)
bash scripts/download_datasets.sh

# 3. Build BTC-TC and all baselines
bash scripts/build_all.sh

# 4. Quick smoke test (<1 min)
bash scripts/smoke_test.sh

# 5. One-click full reproduction (~2-3 hours)
bash scripts/reproduce_paper.sh
```

Quick mode (core claims only, ~30 min):
```bash
bash scripts/reproduce_paper.sh --quick
```

## Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA sm_80+ (Ampere / Hopper / Blackwell), >= 16 GB VRAM |
| CUDA Toolkit | >= 12.2 |
| GCC | >= 11 |
| CMake | >= 3.18 |
| Python | >= 3.10 (matplotlib, pandas, numpy for figures) |
| MPI | OpenMPI or MPICH (for TRUST baseline only) |

## Directory Structure

```
btc-tc-artifact/
├── btc/                  Core BTC-TC library (header-only C++/CUDA)
├── apps/                 Paper executables (11 targets)
│   ├── btc_tc_lite.cu        Main program (BTC-Lite: auto block-size selection)
│   ├── btc_tc_adaptive_*.cu  Fixed block-size variants (16x128, 16x32)
│   ├── btc_tc_v*.cu          Ablation variants (pure-MMA vs hybrid)
│   ├── tau_sweep.cu           Threshold sensitivity (Fig 6)
│   ├── hybrid_ablation.cu     Hybrid dispatch ablation (Fig 9a)
│   └── btc_blocksize_bench.cu MMA shape benchmark (Fig 9c)
├── baselines/            12 baseline implementations (source only)
├── scripts/
│   ├── reproduce_paper.sh    One-click reproduction pipeline
│   ├── build_all.sh          Build BTC-TC + all baselines
│   ├── download_datasets.sh  Download 36 benchmark datasets
│   ├── smoke_test.sh         Quick correctness verification
│   ├── run_*.sh              Individual experiment runners
│   └── figures/              Figure generation scripts (7 figures)
├── results/              Pre-computed results (CSV) for verification
│   ├── pro6000/csv/          RTX PRO 6000 (Blackwell sm_120a)
│   ├── h100/csv/             H100 SXM (Hopper sm_90a)
│   ├── a800/csv/             A800 SXM (Ampere sm_80)
│   ├── ablation/csv/         Ablation study data
│   ├── tau_sweep/            Threshold sensitivity data
│   ├── e2e_breakdown/        End-to-end timing breakdown
│   ├── blocksize_bench/      MMA shape comparison
│   ├── ncu/                  Nsight Compute profiling exports
│   └── reorder_compare/      Vertex reordering comparison
├── data/
│   └── paper_datasets.txt    36 dataset names (download_datasets.sh fetches them)
├── CMakeLists.txt
├── requirements.txt
├── LICENSE
└── README.md
```

## Building

```bash
mkdir -p build && cd build
cmake ..                    # Auto-detects GPU architecture
make -j$(nproc)
```

To build only BTC-TC (skip baselines):
```bash
cmake .. -DBTC_BUILD_BASELINES=OFF
```

## Reproduction Pipeline

`reproduce_paper.sh` runs the complete pipeline:

1. **Smoke test** — correctness verification on 3 small graphs
2. **Full benchmark** — all methods x 36 datasets (5 runs each)
3. **Claim verification** — automated check of key paper numbers
4. **Ablation experiments** — hybrid dispatch, block-size, MMA shape, reordering
5. **Sensitivity sweeps** — threshold (tau) and E2E breakdown
6. **Figure generation** — all 7 paper figures from result CSVs

### Verifying Without Re-running

Pre-computed results are included in `results/`. To regenerate figures directly:
```bash
bash scripts/regenerate_all_figures.sh
```

Output: `results/figures/`

## Key Claims Verified by This Artifact

| Claim | Section | How to Verify |
|-------|---------|---------------|
| BTC-TC exact on all 36 datasets | 4.3 | `smoke_test.sh` + full benchmark |
| 1.92x GM kernel speedup vs ToT | 4.2 | `reproduce_paper.sh` Step 3 |
| 8.0x GM E2E speedup vs ToT | 4.2 | `reproduce_paper.sh` Step 3 |
| Hybrid dispatch adds 1.45x | 4.6 | `run_ablation.sh` |
| Consistent across 3 GPU generations | 4.4 | Run on different GPUs |

## Datasets

36 graphs from SuiteSparse Matrix Collection, covering:
- Scientific computing meshes (FEM, molecular dynamics)
- Social networks (wiki-Vote, com-Youtube, flickr)
- Web graphs (eu-2005, web-Google, web-NotreDame)
- Structural engineering (bcsstk series)

Total download size: ~1.2 GB. See `data/paper_datasets.txt` for the full list.
