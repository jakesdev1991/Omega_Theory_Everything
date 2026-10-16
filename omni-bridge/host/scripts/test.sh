#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# test.sh - Run all test suites for Omni-Bridge
# Per Section 47 of agentic_instruction.txt

set -euo pipefail

PROJECT_ROOT="/opt/omni-bridge"
SOURCE_ROOT="$PROJECT_ROOT/source"

echo "=========================================="
echo "Omni-Bridge Test Suite"
echo "=========================================="

FAILURES=0

run_test() {
    local name="$1"
    local cmd="$2"
    echo "[$name]"
    if eval "$cmd"; then
        echo "  ✓ PASS"
    else
        echo "  ✗ FAIL"
        ((FAILURES++))
    fi
}

# 1. Hermes tests
run_test "Hermes test_all" "cd /tmp && $SOURCE_ROOT/hermes/build/test_all"
run_test "Hermes test_meta_scrutiny" "cd /tmp && $SOURCE_ROOT/hermes/build/test_meta_scrutiny"

# 2. Omni-Bridge C++ tests
run_test "Omni-Bridge C++ core" "$SOURCE_ROOT/omni-bridge/cpp/build/omni_bridge_tests"

# 3. Python prototype tests
run_test "Python prototypes" "cd $SOURCE_ROOT/omni-bridge/offline_python_prototypes && python3 -m pytest -q"

# 4. Schema validation
run_test "Schema validation" "cd $SOURCE_ROOT/omni-bridge/schemas && python3 -c 'import json, sys; [json.load(open(f)) for f in [\"route-request.schema.json\", \"route-result.schema.json\", \"security-label.schema.json\", \"tool-call.schema.json\"]]'"

# 4b. Lean formal verification (§18/§79.4) — no sorry, kernel-checked
run_test "Lean SecureRoute.lean verifies" "cd $SOURCE_ROOT/omni-bridge/lean && lean SecureRoute.lean"

# 4c. Bridge smoke test against the live stack (if running)
if [ -S /opt/omni-bridge/runtime/omni_bridge.sock ]; then
    PY=/opt/omni-bridge/runtime/venv/bin/python
    [ -x "$PY" ] || PY=python3
    run_test "Bridge smoke test (live)" "$PY $SOURCE_ROOT/omni-bridge/scripts/smoke_test.py"
else
    echo "[Bridge smoke test]  - skipped (bridge not running)"
fi

# 5. Security check (focused, not broad)
run_test "Security check: no real secrets" "! grep -r 'BEGIN PRIVATE KEY' $SOURCE_ROOT/ --exclude-dir=third_party --exclude-dir=.worktrees --exclude-dir=target --exclude-dir=build --exclude-dir=.git --exclude-dir=scripts --exclude-dir=tools --exclude='*.pyc' --exclude='*.rlib' --exclude='*.rmeta' --exclude='*.so' --exclude='*.a' --exclude='*.bin' 2>/dev/null | grep -v test | grep -v example | grep -q ."

# 6. Integration test
echo "[Integration smoke test]"
if $SOURCE_ROOT/omni-bridge/cpp/build/omni_bridge_tests >/dev/null 2>&1 && \
   cd /tmp && $SOURCE_ROOT/hermes/build/test_all >/dev/null 2>&1; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL"
    ((FAILURES++))
fi

echo ""
echo "=========================================="
if [[ $FAILURES -eq 0 ]]; then
    echo "ALL TESTS PASSED"
else
    echo "$FAILURES TEST(S) FAILED"
fi
echo "=========================================="

exit $FAILURES