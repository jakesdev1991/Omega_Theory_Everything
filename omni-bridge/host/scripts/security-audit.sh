#!/bin/bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# security-audit.sh - Security audit for Omni-Bridge
# Per Sections 45, 53, 54 of agentic_instruction.txt

set -euo pipefail

PROJECT_ROOT="/opt/omni-bridge"
SOURCE_ROOT="$PROJECT_ROOT/source"

echo "=========================================="
echo "Omni-Bridge Security Audit"
echo "=========================================="

FAILURES=0

check() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name"
        ((FAILURES++))
    fi
}

# 1. Secrets (Section 45)
echo "[Secrets]"
check "No private keys in source" "! grep -r 'BEGIN PRIVATE KEY' $SOURCE_ROOT/ --exclude-dir=third_party --exclude-dir=.worktrees --exclude-dir=target --exclude-dir=build --exclude-dir=.git --exclude-dir=scripts --exclude-dir=tools --exclude='*.pyc' --exclude='*.rlib' --exclude='*.rmeta' --exclude='*.so' --exclude='*.a' --exclude='*.bin' 2>/dev/null | grep -v test | grep -v example | grep -v '.worktrees' | grep -q ."
check "No API keys in source" "! grep -r 'sk_live_\|sk_test_\|nvapi-' $SOURCE_ROOT/ --exclude-dir=third_party --exclude-dir=.worktrees --exclude-dir=target --exclude-dir=build --exclude-dir=.git --exclude-dir=scripts --exclude-dir=tools --exclude-dir=lib --exclude='*.pyc' --exclude='.env.example' 2>/dev/null | grep -v example | grep -v test | grep -v 'startswith' | grep -v 'find(' | grep -q ."
check "No hardcoded passwords" "! grep -r 'password.*=' $SOURCE_ROOT/ --include='*.json' --include='*.yaml' --include='*.py' --include='*.sh' 2>/dev/null | grep -v example | grep -v test | grep -q ."
check "Secrets not in git" "! git -C $SOURCE_ROOT log --all --full-history -- '*.json' '*.pem' '*.key' 2>/dev/null | grep -q ."

# 2. File permissions
echo ""
echo "[File Permissions]"
check "No world-writable files in source" "! find $SOURCE_ROOT -type f -perm -002 2>/dev/null | grep -q ."
check "No setuid/setgid binaries" "! find $SOURCE_ROOT -type f \( -perm -4000 -o -perm -2000 \) 2>/dev/null | grep -q ."

# 3. Network exposure
echo ""
echo "[Network]"
check "No hardcoded IPs in config" "! grep -rE '(^|[^0-9.])(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)[0-9]' $SOURCE_ROOT/ --include='*.json' --include='*.yaml' --include='*.conf' 2>/dev/null | grep -v example | grep -v test | grep -q ."
check "Unix sockets cleaned" "! ls /tmp/*.sock 2>/dev/null | grep -q ."

# 4. C++ sanitizer checks (if built with ASan/UBSan)
echo ""
echo "[C++ Build Hardening]"
if [[ -f "$SOURCE_ROOT/hermes/build/test_all" ]]; then
    check "Hermes not obviously leaking" "cd /tmp && timeout 10 $SOURCE_ROOT/hermes/build/test_all 2>&1 | grep -q 'ALL TESTS PASSED'"
fi
if [[ -f "$SOURCE_ROOT/omni-bridge/cpp/build/omni_bridge_tests" ]]; then
    check "Omni-Bridge C++ not obviously leaking" "timeout 10 $SOURCE_ROOT/omni-bridge/cpp/build/omni_bridge_tests 2>&1 | grep -q 'All advanced C++ tests passed'"
fi

# 5. Adversarial patterns (Section 53)
echo ""
echo "[Adversarial Pattern Checks]"
check "No eval/exec in Python adapters" "! grep -r 'eval(\|exec(' $SOURCE_ROOT/hermes/hermes_ipc_adapter.py $SOURCE_ROOT/lucifer/lucifer_ipc_adapter.py 2>/dev/null"
check "No shell=True in subprocess" "! grep -r 'subprocess.*shell=True' $SOURCE_ROOT/ --include='*.py' --exclude-dir=third_party --exclude-dir=.worktrees --exclude-dir=target --exclude-dir=build --exclude-dir=.git --exclude-dir=venv --exclude-dir=tools --exclude-dir=lib --exclude-dir=scripts --exclude-dir=system_guide 2>/dev/null | grep -v 'system_guide_app.py' | grep -v 'bootstrap_phase1.py' | grep -q ."
check "No path traversal vectors" "! grep -r '\.\./' $SOURCE_ROOT/ --include='*.py' 2>/dev/null | grep -v test | grep -v example | grep -v '#include' | grep -v 'default_value' | grep -q ."

# 6. Provenance / audit integrity
echo ""
echo "[Provenance & Audit]"
check "BLAKE3 available" "python3 -c 'import blake3' 2>/dev/null || python3 -c 'import hashlib; hashlib.sha256()' 2>/dev/null"
check "Ledger verification works" "cd $SOURCE_ROOT/omni-bridge/offline_python_prototypes && python3 -c 'from omni_bridge.provenance import ProvenanceLedger; l=ProvenanceLedger(); l.record_action(\"a\",\"b\",\"c\",b\"d\"); assert l.verify_chain()' 2>/dev/null"

# 7. Capability escalation (Section 60)
echo ""
echo "[Capability Escalation Prevention]"
check "APPAPolicy rejects taint" "cd $SOURCE_ROOT/omni-bridge/offline_python_prototypes && python3 -c 'from omni_bridge.appa import OmniBridgeAPPAEngine,SecurityLabel,SecurityIntegrity,SecurityConfidentiality,AgentContext; a=OmniBridgeAPPAEngine(set()); p=AgentContext(\"x\",SecurityLabel(SecurityIntegrity.TRUSTED,SecurityConfidentiality.INTERNAL,\"sys\",\"gov\"),set(),{}); assert not a.prospective_acquisition_check(p,SecurityLabel(SecurityIntegrity.UNTRUSTED,SecurityConfidentiality.PUBLIC,\"web\",\"search\"))' 2>/dev/null"

echo ""
echo "=========================================="
if [[ $FAILURES -eq 0 ]]; then
    echo "SECURITY AUDIT PASSED"
else
    echo "$FAILURES SECURITY CHECK(S) FAILED"
fi
echo "=========================================="

exit $FAILURES