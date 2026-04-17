#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
BIN="${PROJECT_ROOT}/build/apps/hybrid_ablation"

echo "Dataset,128_v3_pureMMA,128_v4_hybrid,128_v5_hybrid_O1,32_v3_pureMMA,32_v4_pureMMA2,32_v5_hybrid,32_v6_hybrid_O1_SR"

datasets=(
    "lpl1.mtx" "net50.mtx" "msc04515.mtx" "tandem_vtx.mtx" "delaunay_n17.mtx"
    "ex9.mtx" "mac_econ_fwd500.mtx" "road_usa.mtx" "torso2.mtx" "soc-Slashdot0811.mtx"
    "wiki-Vote.mtx" "bcsstk24.mtx" "mc2depi.mtx" "dawson5.mtx" "struct3.mtx"
    "com-Youtube.mtx" "g7jac140sc.mtx" "nemeth16.mtx" "webbase-1M.mtx" "pli.mtx"
    "Freescale1.mtx" "web-NotreDame.mtx" "cage14.mtx" "pcrystk03.mtx" "pkustk06.mtx"
    "web-Google.mtx" "bcsstk30.mtx" "cant.mtx" "consph.mtx" "pdb1HYS.mtx"
    "pwtk.mtx" "higgs-twitter.mtx" "flickr.mtx" "F1.mtx" "eu-2005.mtx"
    "Si41Ge41H72.mtx" "Ga41As41H72.mtx"
)

for dataset in "${datasets[@]}"; do
    [ -f "$DATA_DIR/$dataset" ] && $BIN "$DATA_DIR/$dataset" 2>/dev/null
done
