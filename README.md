# BTC-TC: Exact GPU Triangle Counting with Hybrid Bit Tensor Cores and CUDA Cores

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SC26](https://img.shields.io/badge/SC26-accepted-2ea44f.svg)](https://sc26.supercomputing.org/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#citation)
[![Reproduced on Chameleon](https://img.shields.io/badge/reproduced-Chameleon%20A100%20%2B%20H100-2ea44f.svg)](#artifact-evaluation)

> **Reproducibility artifact for the SC '26 paper.**<br>
> A claim-indexed, end-to-end package for building, validating, and reproducing
> the reported results.

> Independently reproduced end-to-end on neutral **Chameleon Cloud** bare-metal
> nodes — A100 (`sm_80`, CHI@UC) and H100 (`sm_90`, CHI@TACC): **36/36 bit-exact**
> on both, kernel geomean 2.3–2.4× over ToT, no local GPU required. See
> [Artifact Evaluation](#artifact-evaluation).

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
- **Archived:** code-artifact Zenodo DOI is assigned at the 2026-08-25 freeze (added here once minted); the 36-dataset mirror is already at DOI 10.5281/zenodo.21306210

## Contents

- [TL;DR](#tldr)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Claims and Verification](#claims-and-verification)
- [Reproduce Each Figure & Table](#reproduce-each-figure--table)
- [Artifact Evaluation](#artifact-evaluation)
- [Repository Map](#repository-map)
- [Building](#building)
- [Reproduction Pipeline](#reproduction-pipeline)
- [Datasets](#datasets)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## TL;DR

**Exact triangle counting on GPUs, with no sampling or approximation.** BTC-TC
expresses triangle counting as the masked sparse product **L ⊙ (LLᵀ)** (the
lower-triangular adjacency times its transpose, masked back onto the edges). This
formulation reduces to set-intersection-and-count over neighbor bit-vectors. BTC-TC
executes the intersection on **binary Tensor Cores**: the `m16n8k128.and.popc`
instruction ANDs two 128-bit rows and popcounts the result, accumulating in int32 so
**every count is exact** — on all 36 benchmark graphs and across three GPU
generations.

BTC-TC exploits these exact, inexpensive 128-bit intersections to match neighbor
blocks **online inside the kernel** (sorted Bit-BSR + two-pointer), eliminating a
separate symbolic-preprocessing pass. The central contribution is a
**format–operator–dispatch co-design** for the highly non-uniform masked product
(per-block density ranges 0.1%–99%): dense blocks execute on Bit Tensor Cores,
sparse blocks use a selective CUDA-core path, and an O(1) policy selects the
per-graph tile size. Across 36 SuiteSparse graphs BTC-TC is **exact on every
dataset** and reaches a **1.92× geomean kernel** and **8.0× geomean end-to-end**
speedup over the prior state of the art, consistently across three GPU generations
(Ampere / Hopper / Blackwell; 108/108 exact).

## Quick Start

The driver below executes the complete workflow in order: system and Python
dependencies, dataset acquisition, compilation, smoke testing, and result
reproduction.

```bash
# Before the first run, install OS packages (needs sudo):
sudo apt-get install -y libnuma-dev libboost-all-dev libopenmpi-dev bc python3-pip g++-12

bash scripts/run_all.sh           # full reproduction,  ~1-1.5 h
bash scripts/run_all.sh --quick   # core claims only,   ~20 min
bash scripts/run_all.sh --smoke   # build + correctness, ~5 min
```

Total disk: ~9 GB (datasets 3.2 GB extracted + build 5-6 GB). Total wall time on a
single modern NVIDIA GPU (sm_80+): roughly 4 min for smoke, 20 min for quick, 1.5 h
for full. `run_all.sh` is idempotent — rerunning resumes from the last failed step.

### Manual Fallback

If `run_all.sh` encounters an environment-specific issue, execute the same stages
individually:

```bash
# 1. Python deps (add --break-system-packages on Ubuntu 24.04+ for PEP 668)
pip install -r requirements.txt

# 2. Download 36 benchmark datasets (~3.2 GB on disk)
bash scripts/download_datasets.sh

# 3. Build BTC-TC + baselines (~3-5 min, compiles GraphBLAS from source)
bash scripts/build_all.sh

# 4. Smoke test (<1 min)
bash scripts/smoke_test.sh

# 5. Full reproduction (~1-1.5 h) — or add --quick for ~15 min
bash scripts/reproduce_paper.sh
```

### Reference Smoke-Test Output

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

The following matrix distinguishes the minimum supported environment from the
configurations used to validate the artifact.

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA sm_80+ (Ampere/Hopper/Blackwell), >= 16 GB | A800 80GB, H100 80GB, PRO 6000 96GB |
| CUDA Toolkit | >= 12.2 (>= 12.8 for Blackwell sm_120) | 12.2, 13.1, 13.2 |
| NVIDIA driver | >= 525 (CUDA 12) / >= 580 (CUDA 13.x) | 535, 590, 595 |
| GCC | >= 11 | 13.3 |
| CMake | >= 3.22 | 3.28 |
| Python | >= 3.10 | 3.12 |
| libnuma-dev | required | (for vertex reordering) |
| Boost (libboost-all-dev) | required | (rabbit_order reordering headers) |
| bc | required | (timing arithmetic in run_ablation.sh) |
| MPI | optional | (only for TRUST baseline) |

## Claims and Verification

**[CLAIMS.md](CLAIMS.md)** provides the complete claim-to-evidence contract: each
paper claim is paired with the exact verification command and expected output. The
principal claims are summarized below.

| Claim | Section | Verification |
|-------|---------|--------------|
| BTC-TC exact on all 36 datasets | 4.3 | `smoke_test.sh` (quick) or full benchmark |
| 1.92x GM kernel speedup vs ToT | 4.2 | `CLAIMS.md` "Kernel Speedup" |
| 8.0x GM E2E speedup vs ToT | 4.2 | `CLAIMS.md` "E2E Speedup" |
| Hybrid dispatch adds 1.45x | 4.6 | `run_ablation.sh` |
| Block-size heuristic within 1.5x of optimal on all 36 | 3.4 | `CLAIMS.md` "Block-Size Heuristic" |
| Consistent across 3 GPU generations | 4.4 | Pre-computed CSVs for 3 devices |
| Default tau within 6% of optimal (8 graphs) | 3.4 | `run_tau_sweep.sh` |

## Reproduce Each Figure & Table

Every paper table and figure is reproduced by one command, mirroring the AE appendix.
For each element: **Run** the measurement, read the **Expected** result, **Plot** the
figure, and **Check** the number with a copy-paste one-liner. The checks read the
**bundled** reference CSVs and print the stated value (point the reads at
`results-reproduce/csv/` to check your own fresh run). Per-method CSV columns are
`Dataset, Status, Triangles, Kernel_ms, E2E_after_clean_ms` — **correctness is the
`Triangles` column vs. ground truth**, not `Status` (which is only process success).

### Table IV — correctness, kernel & E2E speedup (master table)

```bash
# Run (all 36 graphs, ~15 min): writes results-reproduce/csv/{BTC_Lite,ToT,TRUST}.csv
bash scripts/reproduce_paper.sh --quick
```

**Expected:** both methods complete (Run 36/36); exact triangle count vs. the CPU-exact
**LAGraph** — **BTC-TC 36/36, ToT 24/36**; kernel geomean **1.92×**, E2E geomean **8.0×** over ToT.

```bash
# Check exactness against the independent CPU baseline LAGraph:
python3 -c "import csv
G=lambda m:{r['Dataset']:r['Triangles'] for r in csv.DictReader(open('results/pro6000/csv/'+m+'.csv'))}
l=G('LAGraph')
for m in ('BTC_Lite','ToT'): print(m, sum(G(m)[d]==l[d] for d in l),'/',len(l))"
# -> BTC_Lite 36 / 36     ToT 24 / 36

# Check kernel geomean speedup (swap Kernel_ms -> E2E_after_clean_ms for the 8.0 E2E ratio):
python3 -c "import csv,math
K=lambda m:{r['Dataset']:float(r['Kernel_ms']) for r in csv.DictReader(open('results/pro6000/csv/'+m+'.csv'))}
b,t=K('BTC_Lite'),K('ToT'); d=[x for x in b if x in t]
print(round(math.exp(sum(math.log(t[x]/b[x]) for x in d)/len(d)),2))"
# -> 1.92
```

### Figure 1 — teaser scatter (all 13 methods)

```bash
python3 scripts/figures/generate_teaser_figure.py     # -> results/figures/ (teaser .pdf/.png)
```

**Expected:** BTC-TC sits alone at the fast + exact frontier. Uses bundled `results/pro6000/csv/` (no GPU).

### Figure 6 — dispatch threshold (τ) sensitivity

```bash
bash scripts/run_tau_sweep.sh both                            # ~39 s -> results/tau_sweep/tau_sweep_128_clean.csv
python3 scripts/figures/generate_tau_e2e_combined_figure.py   # -> results/figures/tau_e2e_combined.pdf
```

**Expected:** the default τ=512 (for 16×128; the 16×32 path uses τ=64) is within **6%** of the
per-dataset optimal on all 8 swept graphs; only τ=0 (pure MMA) is catastrophic.

### Figure 7 — per-dataset kernel time

```bash
python3 scripts/figures/generate_per_dataset_lines.py         # -> results/figures/fig_per_dataset_lines.pdf
```

**Expected:** BTC-TC leads on every dataset except 9 small graphs (both < 70 µs) where ToT wins;
kernel geomean **1.92×**, peak per-dataset **13.2×** (g7jac140sc). Data: the `reproduce_paper.sh --quick`
run above (BTC-TC, ToT, TRUST), or the bundled `results/pro6000/csv/`; the full 13-method ordering needs full mode.

### Figure 8 — cross-device ordering (3 GPU generations)

```bash
python3 scripts/figures/generate_cross_device_box_figure.py   # -> results/figures/fig_cross_device_all.pdf
```

**Expected:** BTC-TC holds the best kernel & post-clean E2E geomean on Ampere/Hopper/Blackwell.
Uses the bundled `results/{pro6000,h100,a800}/csv/` — **no GPU needed** to regenerate.

```bash
# Check: BTC-TC wins post-clean E2E on 36/36 datasets per device:
python3 -c "import csv
for dev in ('pro6000','h100','a800'):
    T=lambda m:{r['Dataset']:float(r['E2E_after_clean_ms']) for r in csv.DictReader(open('results/'+dev+'/csv/'+m+'.csv'))}
    b,t=T('BTC_Lite'),T('ToT'); d=[x for x in b if x in t]
    print(dev, sum(b[x]<t[x] for x in d),'/',len(d))"
# -> pro6000 36 / 36     h100 36 / 36     a800 36 / 36
```

### Figure 9 — design ablation (4 panels)

```bash
bash scripts/run_ablation.sh                            # ~3.2 min -> results/ablation/csv/
bash scripts/run_blocksize_bench_paper37.sh             # block-granularity / MMA-shape bench (panels b,c)
python3 scripts/figures/generate_ablation_figure.py     # -> results/figures/fig_ablation.pdf
```

**Expected:** (a) hybrid dispatch is **1.45×** (GM) over pure-TC (a CUDA-core-only path is 1.5×
slower); (b) BTC-Lite stays within **1.5×** of the per-dataset optimal on all 36 (worst 1.50×,
median 1.01×); (c) the 16×128 MMA shape wins on **35/36** over 8×128/16×256; (d) vertex reordering
helps 6 graphs, hurts 30.

```bash
# Check (a) hybrid-dispatch contribution, filtered to the 36 paper graphs:
python3 -c "import csv,math
P=set(l.split()[0] for l in open('data/paper_datasets.txt') if l.strip())
K=lambda f:{r['Dataset']:float(r['Kernel_ms']) for r in csv.DictReader(open('results/ablation/csv/'+f))}
p,h=K('V3_16x128_PureTC.csv'),K('V5_16x128_Hybrid.csv'); d=[x for x in p if x in h and x in P]
print(round(math.exp(sum(math.log(p[x]/h[x]) for x in d)/len(d)),2))"
# -> 1.45

# Check (b) heuristic within 1.5x of the per-dataset optimal on all 36:
python3 -c "import csv,statistics as st
K=lambda f:{r['Dataset']:float(r['Kernel_ms']) for r in csv.DictReader(open('results/pro6000/csv/'+f))}
lite,f128,f32=K('BTC_Lite.csv'),K('BTC_16x128_Adaptive.csv'),K('BTC_16x32_Adaptive.csv')
P=set(l.split()[0] for l in open('data/paper_datasets.txt') if l.strip())
r=sorted(lite[x]/min(f128[x],f32[x]) for x in lite if x in f128 and x in f32 and x in P)
print('within 1.5x: %d/%d ; worst %.3fx ; median %.3fx'%(sum(v<=1.5 for v in r),len(r),max(r),st.median(r)))"
# -> within 1.5x: 36/36 ; worst 1.497x ; median 1.013x
```

### Figure 10 — microarchitectural profile (optional, explanatory)

```bash
python3 scripts/figures/generate_microarch_profile_figure.py  # -> results/figures/microarch_profile.pdf
# To re-collect the BTC-TC/ToT rows yourself (needs ncu + admin counter access):
#   sudo bash scripts/run_ncu_profile.sh
```

**Expected:** ToT's warp stalls (Σ51) are **3.6×** BTC-TC's (Σ14); ToT saturates L2 at 84–92% of
peak vs. < 14% for BTC-TC; BTC-TC shows **2.3×** compute throughput and **2.9×** IPC. Bundled data:
`results/ncu/` (4 methods × 8 graphs, incl. Si41Ge41H72) — no counter access needed to regenerate the figure.

> All copy-paste checks also live, one per claim, in **[CLAIMS.md](CLAIMS.md)**.

## Artifact Evaluation

<a id="artifact-evaluation"></a>

This artifact targets the SC26 reproducibility badges below. Each badge is paired
with a direct reviewer-facing verification path:

| Badge | How a reviewer verifies it |
|-------|----------------------------|
| **Artifacts Available** | Public repository; a persistent Zenodo DOI for the code artifact is assigned at the 2026-08-25 freeze (the 36-dataset mirror is already archived at DOI 10.5281/zenodo.21306210). |
| **Artifacts Evaluated — Functional** | `bash scripts/run_all.sh --smoke` builds BTC-TC + baselines and passes all correctness checks (~5 min). |
| **Results Reproduced** | `bash scripts/run_all.sh --quick` reproduces the device-independent correctness invariant (BTC-TC **36/36** bit-exact, ToT **24/36**) and the kernel speedup over ToT for your GPU (≈1.9× on the paper's RTX PRO 6000; ≈2.3–2.4× on the Chameleon A100/H100 — see the AD/AE appendix for the per-device target) in ~20 min. |

Every paper claim is mapped to its exact command and expected value in
**[CLAIMS.md](CLAIMS.md)**.

## Repository Map

The repository separates the implementation, experiment drivers, reference data,
and verification documentation as follows.

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
│   └── ...                   tau_sweep, ncu, etc.
├── data/
│   └── paper_datasets.txt    36 dataset names
├── CLAIMS.md             Claim-to-command verification guide
├── CMakeLists.txt
├── requirements.txt
└── LICENSE
```

## Building

Use the complete build for paper-level reproduction and baseline comparisons.

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

`reproduce_paper.sh` executes the complete pipeline in the following order:

1. **Smoke test** — correctness verification on 3 small graphs
2. **Full benchmark** — all methods x 36 datasets (5 runs each)
3. **Claim verification** — automated check of key paper numbers
4. **Ablation experiments** — hybrid dispatch, block-size, MMA shape, reordering
5. **Sensitivity sweep** — dispatch threshold (tau)
6. **Figure generation** — all paper figures from result CSVs

### Verification Without Re-running

Pre-computed results from 3 GPU platforms are included in `results/`. To regenerate
figures directly from these results:
```bash
bash scripts/regenerate_all_figures.sh    # Output: results/figures/
```

To verify specific paper numbers without running benchmarks, see `CLAIMS.md` for
one-liner Python commands that compute speedups from the pre-computed CSVs.

## Datasets

36 graphs from [SuiteSparse Matrix Collection](https://sparse.tamu.edu), covering:

- Scientific computing meshes (FEM, molecular dynamics: cant, consph, Si41Ge41H72)
- Social / citation networks (wiki-Vote)
- Web graphs (eu-2005, web-NotreDame, webbase-1M)
- Structural engineering (bcsstk / pcrystk / pkustk series)

Datasets occupy ~3.2 GB on disk once extracted (the compressed downloads are smaller). List: `data/paper_datasets.txt`.

## Troubleshooting

The following remedies cover the most common dependency and toolchain failures.

| Problem | Solution |
|---------|----------|
| `numa.h: No such file` | Install `libnuma-dev`: `apt install libnuma-dev` |
| `boost/algorithm/... No such file` | Install Boost: `apt install libboost-all-dev` |
| `thrust::distance` error (ToT) | Already patched in this artifact for CUDA >= 13.2 |
| TRUST build fails | Ensure MPI is installed: `apt install libopenmpi-dev` |
| Some TC-Compare baselines fail | Non-critical; `build_all.sh` continues on failure |
| CMake picks wrong CUDA | Set: `cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` |

## Citation

Please cite the paper when using BTC-TC in research:

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
