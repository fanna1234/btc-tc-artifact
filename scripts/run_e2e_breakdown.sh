#!/usr/bin/env bash
# Re-run BTC_Lite on the 42 paper datasets and capture detailed E2E timing breakdown.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${PROJECT_ROOT}/build/apps/btc_tc_lite"
DATA_DIR="${PROJECT_ROOT}/data"
OUT_DIR="${PROJECT_ROOT}/results/e2e_breakdown"
OUT_CSV="${OUT_DIR}/breakdown.csv"

DATASETS=(
    "lpl1.mtx"
    "net50.mtx"
    "msc04515.mtx"
    "tandem_vtx.mtx"
    "delaunay_n17.mtx"
    "ex9.mtx"
    "mac_econ_fwd500.mtx"
    "road_usa.mtx"
    "torso2.mtx"
    "soc-Slashdot0811.mtx"
    "wiki-Vote.mtx"
    "bcsstk24.mtx"
    "mc2depi.mtx"
    "dawson5.mtx"
    "struct3.mtx"
    "com-Youtube.mtx"
    "g7jac140sc.mtx"
    "nemeth16.mtx"
    "webbase-1M.mtx"
    "pli.mtx"
    "Freescale1.mtx"
    "web-NotreDame.mtx"
    "cage14.mtx"
    "pcrystk03.mtx"
    "pkustk06.mtx"
    "web-Google.mtx"
    "bcsstk30.mtx"
    "cant.mtx"
    "consph.mtx"
    "pdb1HYS.mtx"
    "pwtk.mtx"
    "higgs-twitter.mtx"
    "flickr.mtx"
    "F1.mtx"
    "eu-2005.mtx"
    "Si41Ge41H72.mtx"
    "Ga41As41H72.mtx"
    "p2p-Gnutella06.mtx"
    "patents_main.mtx"
    "shyy41.mtx"
    "spaceStation_13.mtx"
    "adder_dcop_30.mtx"
)

csv_value() {
    local value="${1:-}"
    if [ -n "$value" ]; then
        printf '%s' "$value"
    else
        printf 'NaN'
    fi
}

if [ ! -x "$BIN" ]; then
    echo "ERROR: Binary not found or not executable: $BIN" >&2
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Dataset,Preprocess_ms,Convert_ms,Kernel_ms,Post_ms,E2E_ms,Triangles" > "$OUT_CSV"

total=${#DATASETS[@]}
idx=0
for dataset in "${DATASETS[@]}"; do
    idx=$((idx + 1))
    dataset_path="${DATA_DIR}/${dataset}"
    dataset_name="${dataset%.mtx}"

    echo "[$idx/$total] Running ${dataset_name}"

    if [ ! -f "$dataset_path" ]; then
        echo "WARNING: Missing dataset: $dataset_path" >&2
        echo "${dataset_name},NaN,NaN,NaN,NaN,NaN,NaN" >> "$OUT_CSV"
        continue
    fi

    out=$("$BIN" -i "$dataset_path" 2>/dev/null || true)

    preprocess=$(printf '%s\n' "$out" | grep -m1 -E '^\[Preprocessing\] time:' | sed -nE 's/.*time: ([0-9.]+) ms/\1/p' || true)
    breakdown=$(printf '%s\n' "$out" | grep -m1 -E '^\[Time Breakdown\]' | sed -nE 's/.*Convert: ([0-9.]+) ms, Compute \(Kernel\): ([0-9.]+) ms, Post: ([0-9.]+) ms/\1,\2,\3/p' || true)
    e2e=$(printf '%s\n' "$out" | grep -m1 -E '^\[Total Time \(Convert\+Compute\)\] time:' | sed -nE 's/.*time: ([0-9.]+) ms/\1/p' || true)
    triangles=$(printf '%s\n' "$out" | grep -m1 -E '^Triangles \(GPU\):' | sed -nE 's/.*: ([0-9]+)/\1/p' || true)

    convert=""
    kernel=""
    post=""
    if [ -n "$breakdown" ]; then
        IFS=',' read -r convert kernel post <<< "$breakdown"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset_name" \
        "$(csv_value "$preprocess")" \
        "$(csv_value "$convert")" \
        "$(csv_value "$kernel")" \
        "$(csv_value "$post")" \
        "$(csv_value "$e2e")" \
        "$(csv_value "$triangles")" >> "$OUT_CSV"
done

echo "Saved breakdown CSV to: $OUT_CSV"
