#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/results/blocksize_bench"
DATASET_LIST="${ROOT_DIR}/data/paper_datasets.txt"
BENCH_BIN="${ROOT_DIR}/build/apps/btc_blocksize_bench"

RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-2}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/csv"

if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "[ERROR] Missing bench binary: ${BENCH_BIN}" >&2
  echo "Build it with: cmake --build build -j" >&2
  exit 2
fi

if [[ ! -f "${DATASET_LIST}" ]]; then
  echo "[ERROR] Missing dataset list: ${DATASET_LIST}" >&2
  exit 2
fi

# Run one dataset per log file, so the job is restartable.
# To rerun a dataset, delete its log.
mapfile -t stems < <(grep -vE "^\s*#" "${DATASET_LIST}" | sed -e "s/#.*$//" -e "s/^\s\+//" -e "s/\s\+$//" | awk "NF")

echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Datasets: ${#stems[@]}"
echo "[INFO] Runs=${RUNS} Warmup=${WARMUP}"

echo "[INFO] GPU:"
(nvidia-smi -L || true) | sed -e "s/^/[INFO]   /"

echo "[INFO] Starting blocksize bench..."

i=0
for stem in "${stems[@]}"; do
  i=$((i+1))
  mtx="${ROOT_DIR}/data/${stem}.mtx"
  log="${OUT_DIR}/logs/${stem}.log"

  if [[ ! -f "${mtx}" ]]; then
    echo "[WARN] (${i}/${#stems[@]}) missing ${mtx}, skip" >&2
    continue
  fi

  if [[ -f "${log}" ]] && grep -q "^RESULT_CSV," "${log}"; then
    echo "[SKIP] (${i}/${#stems[@]}) ${stem} (log exists)"
    continue
  fi

  echo "[RUN ] (${i}/${#stems[@]}) ${stem}"
  set +e
  "${BENCH_BIN}" -i "${mtx}" --runs "${RUNS}" --warmup "${WARMUP}" --no-kernel-info >"${log}" 2>&1
  rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[FAIL] (${i}/${#stems[@]}) ${stem} (rc=${rc}); see ${log}" >&2
    continue
  fi

  if ! grep -q "^RESULT_CSV," "${log}"; then
    echo "[FAIL] (${i}/${#stems[@]}) ${stem} produced no RESULT_CSV lines; see ${log}" >&2
    continue
  fi

  echo "[OK  ] (${i}/${#stems[@]}) ${stem}"
done

echo "[INFO] Parsing logs -> CSV + summary"
python3 "${ROOT_DIR}/scripts/summarize_blocksize_bench_paper37.py" \
  --log-dir "${OUT_DIR}/logs" \
  --out-csv "${OUT_DIR}/csv/blocksize_bench_paper37.csv" \
  --out-summary "${OUT_DIR}/summary.txt"

echo "[DONE] ${OUT_DIR}"
