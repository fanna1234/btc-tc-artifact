# Baselines

Third-party GPU/CPU triangle-counting implementations that BTC-TC is compared
against in the paper. Each subtree retains its **original upstream license**
(see the individual subdirectories and the repository root `LICENSE`).

| Directory | Method(s) | Kind |
|---|---|---|
| `ToT-TPDS25/` | ToT (Tensor-core triangle counting, TPDS'25) | GPU, Tensor Core |
| `TC-Compare/` | Green, Fox, Bisson, Hu, GroupTC, Tricore, H-Index, Polak | GPU/CPU harness (several methods) |
| `TRUST/` | TRUST | GPU |
| `LAGraph/` | LAGraph triangle count (CPU oracle for exactness checks) | CPU |
| `SuiteSparse-GraphBLAS-cuda/` | GraphBLAS backend required by LAGraph | dependency |
| `cusparse_tc/` | cuSPARSE-based triangle counting | GPU |
| `cublas_tc/` | cuBLAS-based triangle counting | GPU |

All baselines are compiled from source by `scripts/build_all.sh`; a few CPU
preprocessing / format-conversion helpers ship prebuilt for portability. See the
repository root `README.md` and `CLAIMS.md` for the full reproduction workflow.
