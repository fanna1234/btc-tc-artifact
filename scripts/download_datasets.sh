#!/bin/bash
# Download all 36 benchmark datasets in Matrix Market format.
# All from SuiteSparse Matrix Collection (https://sparse.tamu.edu).
# Usage: bash scripts/download_datasets.sh
set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

# Correct Group/Name paths verified from .mtx file headers.
DATASETS=(
    "Shyy/shyy41"
    "VDOL/spaceStation_13"
    "HB/bcsstk23"
    "HB/bcsstm13"
    "Hollinger/g7jac020"
    "Andrianov/lpl1"
    "Andrianov/net50"
    "Boeing/msc04515"
    "Pothen/tandem_vtx"
    "DIMACS10/delaunay_n17"
    "FIDAP/ex9"
    "Williams/mac_econ_fwd500"
    "Norris/torso2"
    "SNAP/wiki-Vote"
    "HB/bcsstk24"
    "Williams/mc2depi"
    "GHS_indef/dawson5"
    "Rothberg/struct3"
    "Hollinger/g7jac140sc"
    "Nemeth/nemeth16"
    "Williams/webbase-1M"
    "Li/pli"
    "Freescale/Freescale1"
    "SNAP/web-NotreDame"
    "vanHeukelum/cage14"
    "Boeing/pcrystk03"
    "Chen/pkustk06"
    "HB/bcsstk30"
    "Williams/cant"
    "Williams/consph"
    "Williams/pdb1HYS"
    "Boeing/pwtk"
    "Koutsovasilis/F1"
    "LAW/eu-2005"
    "PARSEC/Si41Ge41H72"
    "PARSEC/Ga41As41H72"
)

echo "Downloading ${#DATASETS[@]} datasets to $DATA_DIR/"
echo ""

OK=0; SKIP=0; FAIL=0

for ds in "${DATASETS[@]}"; do
    name=$(basename "$ds")
    if [ -f "$DATA_DIR/${name}.mtx" ]; then
        echo "  [skip] ${name}.mtx exists"
        SKIP=$((SKIP + 1))
        continue
    fi
    echo -n "  [get]  ${name} from ${ds}... "
    url="https://sparse.tamu.edu/MM/${ds}.tar.gz"
    if wget -q "$url" -O "/tmp/${name}.tar.gz" 2>/dev/null && \
       tar -xzf "/tmp/${name}.tar.gz" -C /tmp/ && \
       cp /tmp/${name}/${name}.mtx "$DATA_DIR/" 2>/dev/null; then
        echo "OK"
        OK=$((OK + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
    fi
    rm -rf "/tmp/${name}" "/tmp/${name}.tar.gz"
done

echo ""
echo "Done: $OK downloaded, $SKIP skipped, $FAIL failed"
echo "Total .mtx files: $(ls "$DATA_DIR"/*.mtx 2>/dev/null | wc -l)"
