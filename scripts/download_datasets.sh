#!/bin/bash
# Download all 36 benchmark datasets in Matrix Market format.
# All from the SuiteSparse Matrix Collection (https://sparse.tamu.edu).
#
# Usage:
#   bash scripts/download_datasets.sh
#
# Slow or restricted network? Fetch the pre-packaged mirror in one shot
# (a single archive of all 36 .mtx files, e.g. the frozen Zenodo record):
#   BTC_DATASETS_URL=<tarball-url> bash scripts/download_datasets.sh
#
# Robustness: each download has an idle timeout (no infinite hangs on a
# stalled connection), is retried, and is verified for integrity before it
# counts as done. Works with either wget or curl.
#
# Tunables (env): BTC_DL_TIMEOUT (idle seconds, default 60),
#                 BTC_DL_TRIES  (attempts per file, default 4).
set -uo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"

TIMEOUT="${BTC_DL_TIMEOUT:-60}"   # abort a download idle for this many seconds
TRIES="${BTC_DL_TRIES:-4}"        # attempts per file
TMP="$(mktemp -d "${TMPDIR:-/tmp}/btc_ds.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# --- pick a downloader: wget preferred, curl fallback (macOS ships only curl) ---
if command -v wget >/dev/null 2>&1; then
    DL=wget
elif command -v curl >/dev/null 2>&1; then
    DL=curl
else
    echo "ERROR: need 'wget' or 'curl' on PATH to download datasets." >&2
    exit 2
fi

# fetch <url> <out> : ONE attempt, follows redirects, aborts if the transfer
# stalls (no bytes) for $TIMEOUT seconds — this is what prevents the silent
# infinite hang seen with a bare 'wget -q' on a flaky link.
fetch() {
    local url="$1" out="$2"
    if [ "$DL" = wget ]; then
        wget --quiet --tries=1 --timeout="$TIMEOUT" "$url" -O "$out"
    else
        curl --silent --show-error --location \
             --connect-timeout 30 --speed-limit 1 --speed-time "$TIMEOUT" \
             "$url" -o "$out"
    fi
}

# get_targz <out> <url> : retry fetch until a *valid* gzip lands (or give up).
get_targz() {
    local out="$1" url="$2" attempt
    for attempt in $(seq 1 "$TRIES"); do
        rm -f "$out"
        if fetch "$url" "$out" 2>/dev/null && gzip -t "$out" 2>/dev/null; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# ---------------------------------------------------------------------------
# Mirror mode: one archive of all 36 .mtx files (for slow/restricted networks).
# ---------------------------------------------------------------------------
if [ -n "${BTC_DATASETS_URL:-}" ]; then
    echo "Mirror mode: fetching bundled dataset archive"
    echo "  $BTC_DATASETS_URL"
    if get_targz "$TMP/all.tar.gz" "$BTC_DATASETS_URL"; then
        tar -xzf "$TMP/all.tar.gz" -C "$TMP/"
        # copy every .mtx found in the archive into data/
        find "$TMP" -name '*.mtx' -exec cp -f {} "$DATA_DIR/" \;
        echo "Done (mirror): $(ls "$DATA_DIR"/*.mtx 2>/dev/null | wc -l) .mtx files in $DATA_DIR/"
        exit 0
    else
        echo "ERROR: could not fetch mirror archive from BTC_DATASETS_URL" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Default mode: fetch each dataset from SuiteSparse.
# Correct Group/Name paths verified from .mtx file headers.
# ---------------------------------------------------------------------------
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

echo "Downloading ${#DATASETS[@]} datasets to $DATA_DIR/ (downloader: $DL)"
echo ""

OK=0; SKIP=0; FAIL=0
FAILED_NAMES=()

for ds in "${DATASETS[@]}"; do
    name=$(basename "$ds")
    if [ -f "$DATA_DIR/${name}.mtx" ]; then
        echo "  [skip] ${name}.mtx exists"
        SKIP=$((SKIP + 1))
        continue
    fi
    echo -n "  [get]  ${name} from ${ds}... "
    url="https://sparse.tamu.edu/MM/${ds}.tar.gz"
    if get_targz "$TMP/${name}.tar.gz" "$url" \
       && tar -xzf "$TMP/${name}.tar.gz" -C "$TMP/" 2>/dev/null \
       && [ -s "$TMP/${name}/${name}.mtx" ] \
       && cp "$TMP/${name}/${name}.mtx" "$DATA_DIR/"; then
        echo "OK"
        OK=$((OK + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
        FAILED_NAMES+=("$name")
    fi
    rm -rf "$TMP/${name}" "$TMP/${name}.tar.gz"
done

echo ""
echo "Done: $OK downloaded, $SKIP skipped, $FAIL failed"
echo "Total .mtx files: $(ls "$DATA_DIR"/*.mtx 2>/dev/null | wc -l) / ${#DATASETS[@]}"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Failed (${FAIL}): ${FAILED_NAMES[*]}"
    echo "Re-run to retry only the missing ones, or if your network cannot reach"
    echo "sparse.tamu.edu, use the bundled mirror:"
    echo "  BTC_DATASETS_URL=<tarball-url> bash scripts/download_datasets.sh"
    exit 1
fi
