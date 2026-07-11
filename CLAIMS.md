# Paper Claims and Verification

This document maps each major paper claim to the specific command and expected output
that verifies it. All commands assume you are in the artifact root directory with
a completed build (`bash scripts/build_all.sh`).

## Correctness (Section 4.3, 4.5)

**Claim**: BTC-TC produces exact triangle counts on all 36 datasets across 3 GPU platforms.

```bash
# Quick verification (3 datasets, <1 min):
bash scripts/smoke_test.sh
# Expected: 3/3 PASS for BTC-TC, 2/2 PASS for ToT

# Full verification (36 datasets):
python3 scripts/bench_baselines.py --methods BTC_Lite --run-dir results-verify
# Expected: all 36 datasets show Status=OK, counts match ground truth
```

**Pre-computed evidence**: `results/pro6000/csv/BTC_Lite.csv` — all 36 rows have `Status=OK`.

---

## Kernel Speedup (Section 4.2, Table IV)

**Claim**: BTC-TC achieves 1.92x geometric-mean kernel speedup over ToT on PRO 6000.

```bash
python3 -c "
import pandas as pd, numpy as np
btc = pd.read_csv('results/pro6000/csv/BTC_Lite.csv')
tot = pd.read_csv('results/pro6000/csv/ToT.csv')
m = btc.merge(tot, on='Dataset', suffixes=('_b','_t'))
m = m[m['Status_b']=='OK']
gm = np.exp(np.mean(np.log(m['Kernel_ms_t']/m['Kernel_ms_b'])))
print(f'GM kernel speedup: {gm:.2f}x')
"
# Expected: ~1.92x
```

---

## End-to-End Speedup (Section 4.2, Table IV)

**Claim**: BTC-TC achieves 8.0x geometric-mean E2E speedup over ToT.

```bash
python3 -c "
import pandas as pd, numpy as np
btc = pd.read_csv('results/pro6000/csv/BTC_Lite.csv')
tot = pd.read_csv('results/pro6000/csv/ToT.csv')
m = btc.merge(tot, on='Dataset', suffixes=('_b','_t'))
m = m[m['Status_b']=='OK']
gm = np.exp(np.mean(np.log(m['E2E_after_clean_ms_t']/m['E2E_after_clean_ms_b'])))
print(f'GM E2E speedup: {gm:.1f}x')
"
# Expected: ~8.0x
```

---

## Hybrid Dispatch Contribution (Section 4.6, Fig 9a)

**Claim**: Hybrid dispatch adds 1.45x additional kernel speedup over pure-MMA.

```bash
# Reproduce:
bash scripts/run_ablation.sh

# Verify from pre-computed:
python3 -c "
import pandas as pd, numpy as np
pure = pd.read_csv('results/ablation/csv/V3_16x128_PureTC.csv')
hybrid = pd.read_csv('results/ablation/csv/V5_16x128_Hybrid.csv')
m = pure.merge(hybrid, on='Dataset', suffixes=('_p','_h'))
valid = m[(m['Kernel_ms_p']>0) & (m['Kernel_ms_h']>0)]
gm = np.exp(np.mean(np.log(valid['Kernel_ms_p']/valid['Kernel_ms_h'])))
print(f'Hybrid dispatch GM speedup: {gm:.2f}x')
"
# Expected: ~1.45x
```

---

## Cross-Device Consistency (Section 4.4, Fig 8)

**Claim**: BTC-TC is fastest E2E on all 36 datasets across 3 GPU generations.

```bash
python3 -c "
import pandas as pd
for dev, path in [('PRO6000','results/pro6000/csv'), ('H100','results/h100/csv'), ('A800','results/a800/csv')]:
    btc = pd.read_csv(f'{path}/BTC_Lite.csv')
    tot = pd.read_csv(f'{path}/ToT.csv')
    m = btc.merge(tot, on='Dataset', suffixes=('_b','_t'))
    wins = (m['E2E_after_clean_ms_b'] < m['E2E_after_clean_ms_t']).sum()
    print(f'{dev}: BTC-TC wins E2E on {wins}/{len(m)} datasets')
"
# Expected: 36/36 on all 3 devices
```

---

## Threshold Insensitivity (Section 3.4, Fig 6)

**Claim**: Kernel time is stable across tau in [64, 2048].

```bash
# Reproduce:
bash scripts/run_tau_sweep.sh both

# Pre-computed data: results/tau_sweep/tau_sweep_128_clean.csv
```

---

## MMA Shape Selection (Section 4.6, Fig 9c)

**Claim**: BTC-Lite's heuristic matches per-dataset optimal on 22/36 graphs.

```bash
# Reproduce:
bash scripts/run_blocksize_bench_paper37.sh

# Pre-computed: results/blocksize_bench/summary.txt
```

---

## Microarchitecture Profile (Section 4.2, Fig 10)

**Claim**: BTC-TC achieves 2.3x higher compute throughput than ToT on Nsight Compute metrics.

```bash
# Reproduce on your device (requires ncu in PATH):
bash scripts/run_ncu_profile.sh

# Or profile a single dataset:
bash scripts/ncu_profile_metrics.sh --method btc128 --dataset data/consph.mtx --tag mydevice

# Regenerate Fig 10:
python3 scripts/figures/generate_microarch_profile_figure.py
```

**Pre-computed data**: `results/ncu/pro6000_*.raw.csv` (32 files, 8 datasets x 4 methods).

**Note**: NCU requires root/sudo on some systems. If profiling fails, try:
`sudo ncu ...` or set `/proc/sys/kernel/perf_event_paranoid` to a lower value.

---

## Figure Reproduction

All 6 data-driven figures can be regenerated from pre-computed CSVs:

```bash
bash scripts/regenerate_all_figures.sh
# Output: results/figures/
```

| Figure | Script | Data Source |
|--------|--------|-------------|
| Fig 1 (teaser) | `generate_teaser_figure.py` | `results/pro6000/csv/` |
| Fig 7 (per-dataset) | `generate_per_dataset_lines.py` | `results/pro6000/csv/` |
| Fig 6 (tau sensitivity) | `generate_tau_e2e_combined_figure.py` | `results/tau_sweep/` |
| Fig 8 (cross-device) | `generate_cross_device_box_figure.py` | `results/{pro6000,h100,a800}/csv/` |
| Fig 9 (ablation 4-panel) | `generate_ablation_figure.py` | `results/ablation/csv/` + others |
| Fig 10 (microarch) | `generate_microarch_profile_figure.py` | `results/ncu/` |
