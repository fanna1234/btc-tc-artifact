#!/bin/bash
# Download all 36 benchmark datasets in Matrix Market format.
# All from the SuiteSparse Matrix Collection (https://sparse.tamu.edu).
#
# Usage:
#   bash scripts/download_datasets.sh
#
# Slow or restricted network (e.g. cannot reach sparse.tamu.edu)? Fetch the
# frozen Zenodo mirror instead (DOI 10.5281/zenodo.21306210) — the script
# downloads the split-part archive, reassembles it, verifies its SHA-256, and
# extracts all 36 .mtx in one shot:
#   BTC_DATASETS_MIRROR=1 bash scripts/download_datasets.sh
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

# get_part <out> <url> : retry until a non-empty file lands. Split parts are NOT
# individually valid gzip, so integrity is checked on the reassembled whole.
get_part() {
    local out="$1" url="$2" attempt
    for attempt in $(seq 1 "$TRIES"); do
        rm -f "$out"
        if fetch "$url" "$out" 2>/dev/null && [ -s "$out" ]; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# sha256_of <file> : print the file's SHA-256 (portable across Linux/macOS).
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# reassemble_parts <template-with-{PART}> <suffix-list> <out> : fetch each part
# in order and concatenate into <out>.
reassemble_parts() {
    local tmpl="$1" suffixes="$2" out="$3" s url
    : > "$out"
    for s in $suffixes; do
        url="${tmpl/\{PART\}/$s}"
        echo "  [part $s] fetching..."
        if get_part "$TMP/part.$s" "$url"; then
            cat "$TMP/part.$s" >> "$out"; rm -f "$TMP/part.$s"
        else
            echo "ERROR: failed to fetch part '$s' from $url" >&2; return 1
        fi
    done
    return 0
}

# ---------------------------------------------------------------------------
# Mirror modes (for slow/restricted networks): fetch one bundle instead of 36.
#
#   BTC_DATASETS_MIRROR=1        Zenodo split-part mirror, DOI 10.5281/zenodo.21306210
#                                (recommended; SHA-256 of the reassembled archive
#                                is verified automatically).
#   BTC_DATASETS_PARTS_URL=<t>   generic split parts; <t> is a URL template with a
#                                {PART} placeholder. Suffixes = BTC_DATASETS_PARTS
#                                (default "aa ab ac ad ae af ag ah").
#   BTC_DATASETS_URL=<url>       a single .tar.gz archive.
# ---------------------------------------------------------------------------
ZENODO_DOI="10.5281/zenodo.21306210"
ZENODO_PARTS_TMPL="https://zenodo.org/records/21306210/files/btc-tc-datasets.tar.gz.part-{PART}?download=1"
ZENODO_PART_SUFFIXES="aa ab ac ad ae af ag ah"
ZENODO_SHA256="7561713019a02173194eca44c6955cee0f860435f4efa9cc2f71b879cd8e2a5a"

place_from_targz() {   # <tarball> : extract and copy every .mtx into data/
    tar -xzf "$1" -C "$TMP/" || { echo "ERROR: could not extract archive" >&2; return 1; }
    find "$TMP" -name '*.mtx' -exec cp -f {} "$DATA_DIR/" \;
    echo "Done (mirror): $(ls "$DATA_DIR"/*.mtx 2>/dev/null | wc -l) .mtx files in $DATA_DIR/"
}

if [ -n "${BTC_DATASETS_MIRROR:-}" ] || [ -n "${BTC_DATASETS_PARTS_URL:-}" ]; then
    if [ -n "${BTC_DATASETS_MIRROR:-}" ]; then
        echo "Mirror mode: Zenodo split-part archive (DOI $ZENODO_DOI)"
        tmpl="$ZENODO_PARTS_TMPL"; suffixes="$ZENODO_PART_SUFFIXES"; want="$ZENODO_SHA256"
    else
        echo "Mirror mode: split parts from BTC_DATASETS_PARTS_URL"
        tmpl="$BTC_DATASETS_PARTS_URL"; suffixes="${BTC_DATASETS_PARTS:-aa ab ac ad ae af ag ah}"; want=""
    fi
    if ! reassemble_parts "$tmpl" "$suffixes" "$TMP/all.tar.gz"; then
        echo "ERROR: could not reassemble split-part mirror" >&2; exit 1
    fi
    if [ -n "$want" ]; then
        got=$(sha256_of "$TMP/all.tar.gz")
        if [ "$got" != "$want" ]; then
            echo "ERROR: reassembled archive SHA-256 mismatch" >&2
            echo "  expected $want" >&2
            echo "  got      $got" >&2
            exit 1
        fi
        echo "  SHA-256 verified OK"
    fi
    gzip -t "$TMP/all.tar.gz" 2>/dev/null || { echo "ERROR: reassembled archive is a corrupt gzip" >&2; exit 1; }
    place_from_targz "$TMP/all.tar.gz" && exit 0
    exit 1
fi

if [ -n "${BTC_DATASETS_URL:-}" ]; then
    echo "Mirror mode: single archive"
    echo "  $BTC_DATASETS_URL"
    if get_targz "$TMP/all.tar.gz" "$BTC_DATASETS_URL"; then
        place_from_targz "$TMP/all.tar.gz" && exit 0
        exit 1
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
