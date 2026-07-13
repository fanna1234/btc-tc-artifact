#!/bin/bash
# Quick smoke test: verify BTC-TC and ToT produce correct triangle counts.
# Usage: bash scripts/smoke_test.sh
# Expected runtime: <1 minute
set -uo pipefail
cd "$(dirname "$0")/.."

BTC="./build/apps/btc_tc_lite"
# ToT may be built by main cmake (build/baselines/...) or standalone (baselines/.../build/)
TOT="./build/baselines/ToT-TPDS25/apps/tot"
[ ! -x "$TOT" ] && TOT="./baselines/ToT-TPDS25/build/apps/tot"
PASS=0; FAIL=0; TOTAL=0

check() {
    local name="$1" bin="$2" dataset="$3" expected="$4"
    TOTAL=$((TOTAL + 1))
    if [ ! -x "$bin" ]; then
        echo "SKIP  $name: binary not found ($bin)"
        return
    fi
    if [ ! -f "$dataset" ]; then
        echo "SKIP  $name: dataset not found ($dataset)"
        return
    fi
    local output
    output=$("$bin" -i "$dataset" 2>&1)
    local count
    count=$(echo "$output" | grep -oP '(?i)triangles?\s*(\(GPU\))?\s*[:=]\s*\K[0-9]+' | head -1)
    if [ "$count" = "$expected" ]; then
        echo "PASS  $name: $count triangles (expected $expected)"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name: got '$count', expected $expected"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== BTC-TC Smoke Test ==="
echo ""

# Small graphs with known triangle counts
check "BTC-TC  lpl1"     "$BTC" "data/lpl1.mtx"     "97633"
check "BTC-TC  cant"     "$BTC" "data/cant.mtx"      "18370150"
check "BTC-TC  bcsstk23" "$BTC" "data/bcsstk23.mtx"  "29737"
check "ToT     lpl1"     "$TOT" "data/lpl1.mtx"      "97633"
check "ToT     cant"     "$TOT" "data/cant.mtx"      "18370150"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $((TOTAL - PASS - FAIL)) skipped / $TOTAL total ==="
# Require every test to have actually RUN and passed — otherwise a run where all
# tests were SKIPPED (missing binary/dataset) would print ALL PASSED and exit 0.
if [ "$PASS" -eq "$TOTAL" ]; then
    echo "ALL PASSED"
else
    echo "NOT ALL PASSED: $PASS/$TOTAL passed ($FAIL failed, $((TOTAL - PASS - FAIL)) skipped)"
    exit 1
fi
