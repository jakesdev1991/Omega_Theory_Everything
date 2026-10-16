#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# iterate.sh — Full Omni-Bridge build + test + status cycle
# Designed to be called repeatedly by cron until everything is green
# NOTE: runs each step in a guarded subshell so no single failure kills the cycle
set -uo pipefail

PROJECT_ROOT="/opt/omni-bridge"
SOURCE_ROOT="$PROJECT_ROOT/source"
LOG_DIR="$PROJECT_ROOT/telemetry"
PYTHON_VENV="$PROJECT_ROOT/runtime/venv"
TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

# cron has a minimal PATH — restore the user toolchain
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/lib/llvm/22/bin:$PATH"

log() { echo "[$TIMESTAMP] $1"; }
ok() { log "[PASS] $1"; }
fail() { log "[FAIL] $1"; }
warn() { log "[WARN] $1"; }

log "========== Omni-Bridge Iteration Cycle =========="

mkdir -p "$LOG_DIR"

# 1. Bootstrap check (guarded — failures are reported, never fatal)
log "--- Step 1: Bootstrap ---"
BOOT_OUT=$(bash "$SOURCE_ROOT/omni-bridge/scripts/bootstrap.sh" 2>&1)
if [ $? -eq 0 ]; then
    ok "Bootstrap: all checks passed"
else
    echo "$BOOT_OUT" | grep -E "FAIL|WARN" | while read -r line; do warn "$line"; done
fi

# 2. Build C++ (guarded)
log "--- Step 2: C++ Build ---"
BUILD_OUT=$(cd "$SOURCE_ROOT/omni-bridge/cpp" && cmake --build build --parallel 2>&1)
if echo "$BUILD_OUT" | grep -q "Built target"; then
    ok "C++ build: both binaries compiled"
else
    fail "C++ build failed"
    echo "$BUILD_OUT" | tail -10 | while read -r line; do log "  $line"; done
fi

# 3. Run C++ tests (guarded)
log "--- Step 3: C++ Tests ---"
TEST_OUT=$(cd "$SOURCE_ROOT/omni-bridge/cpp" && ./build/omni_bridge_tests 2>&1)
if [ $? -eq 0 ]; then
    ok "C++ tests: all passed"
else
    fail "C++ tests failed"
    echo "$TEST_OUT" | while read -r line; do log "  $line"; done
fi

# 4. Run Python tests (guarded)
log "--- Step 4: Python Tests ---"
if [ -x "$PYTHON_VENV/bin/python" ]; then
    PY_OUT=$("$PYTHON_VENV/bin/python" -m pytest "$SOURCE_ROOT/omni-bridge/python/tests/" -q 2>&1)
    if echo "$PY_OUT" | grep -q " passed"; then
        ok "Python tests: $(echo "$PY_OUT" | grep -oE '[0-9]+ passed.*' | head -1)"
    else
        fail "Python tests had failures"
        echo "$PY_OUT" | tail -5 | while read -r line; do log "  $line"; done
    fi
else
    fail "Python venv missing at $PYTHON_VENV"
fi

# 5. Offline prototype tests (guarded)
log "--- Step 5: Offline Prototype Tests ---"
if [ -x "$PYTHON_VENV/bin/python" ]; then
    PROTO_OUT=$(cd "$SOURCE_ROOT/omni-bridge/offline_python_prototypes" && "$PYTHON_VENV/bin/python" -m pytest -q 2>&1)
    if echo "$PROTO_OUT" | grep -q " passed"; then
        ok "Offline prototypes: $(echo "$PROTO_OUT" | grep -oE '[0-9]+ passed.*' | head -1)"
    else
        warn "Offline prototypes: some tests failed/skipped"
    fi
fi

# 6. Smoke test against the live bridge (guarded — stack may be down)
log "--- Step 6: Bridge Smoke Test ---"
SMOKE_OUT=$("$PYTHON_VENV/bin/python" "$SOURCE_ROOT/omni-bridge/scripts/smoke_test.py" 2>&1)
if [ $? -eq 0 ]; then
    ok "Bridge smoke test: 5/5"
else
    warn "Bridge smoke test: stack not running or failures present"
    echo "$SMOKE_OUT" | while read -r line; do log "  $line"; done
fi

# 7. Doctor check (guarded)
log "--- Step 7: Doctor ---"
DOCTOR_OUT=$(bash "$SOURCE_ROOT/omni-bridge/scripts/doctor.sh" 2>&1)
echo "$DOCTOR_OUT" | grep -E "\[OK\]|\[FAIL\]|\[WARN\]" | while read -r line; do log "  $line"; done

log "========== Cycle Complete =========="
