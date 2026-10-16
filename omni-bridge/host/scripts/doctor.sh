#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# doctor.sh — health check for the rebuilt control boundary
# Checks: storage, ledger chain integrity (§34), adapters, sockets, kill
# switches (§72), ToE grounding corpus (§71), config sanity.
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILURES=$((FAILURES+1)); }
FAILURES=0

PROJECT_ROOT="/opt/omni-bridge"
RUNTIME_DIR="$PROJECT_ROOT/runtime"
LEDGER="$PROJECT_ROOT/audit/bridge_ledger.jsonl"
VENV_PY="$PROJECT_ROOT/runtime/venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=python3

echo "Running Omni-Bridge Doctor..."

# Storage
if touch "$PROJECT_ROOT/.doctor_test" 2>/dev/null; then
    rm "$PROJECT_ROOT/.doctor_test"; ok "/opt/omni-bridge writable"
else
    fail "/opt/omni-bridge not writable"
fi
btrfs device stats "$PROJECT_ROOT" >/dev/null 2>&1 && ok "Btrfs device stats clean" \
    || warn "btrfs device stats unavailable (needs root)"

# Databases
[ -r "$PROJECT_ROOT/memory/omni_memory.db" ] && ok "Memory DB readable" || fail "Memory DB missing"
[ -r "$PROJECT_ROOT/audit/evidence.db" ] && ok "Python evidence DB readable" || warn "Python evidence DB missing"

# §34: control-boundary ledger chain verification (C++ JSONL chain).
# Chain-shape check always runs; hash re-derivation only when blake3 is
# importable — NEVER silently swap in blake2b (that would break the chain).
if [ -f "$LEDGER" ]; then
    if "$VENV_PY" - "$LEDGER" << 'PYEOF'
import sys
path = sys.argv[1]
try:
    import blake3
    def h(b): return blake3.blake3(b).hexdigest()
except ImportError:
    def h(b): return None
prev = ""
n = 0
with open(path) as f:
    for line in f:
        if not line.strip(): continue
        t1 = line.find('\t'); t2 = line.find('\t', t1+1)
        if t1 < 0 or t2 < 0: sys.exit(1)
        p, c, canon = line[:t1], line[t1+1:t2], line[t2+1:].rstrip('\n')
        if p != prev: sys.exit(1)
        if h is not None and h((p+canon).encode()) != c: sys.exit(1)
        prev = c; n += 1
print(f"{n} records")
PYEOF
    then
        ok "Bridge ledger chain verified"
    else
        fail "Bridge ledger chain BROKEN or malformed"
    fi
else
    warn "Bridge ledger not yet created (daemon may not have run)"
fi

# Sockets
for sock in omni_bridge hermes lucifer; do
    if [ -S "$RUNTIME_DIR/$sock.sock" ]; then ok "Socket $sock up"; else warn "Socket $sock missing"; fi
done

# Processes
for proc in omni_bridge_daemon hermes_ipc_adapter lucifer_ipc_adapter; do
    if pgrep -f "$proc" >/dev/null; then ok "Process $proc running"; else warn "Process $proc not running"; fi
done

# Kill switches (§72) — normalized: 1/true/yes/on all count as active
echo "Kill Switches (§72):"
for var in OMNI_DYNAMIC_EXPERTISE_OFF OMNI_LUCIFER_OFF OMNI_HERMES_RETRIEVAL_OFF \
           OMNI_GENERATION_OFF OMNI_LEARNING_OFF OMNI_AUDIT_READONLY; do
    val="${!var:-}"
    if [ "$val" = "1" ] || [ "$val" = "true" ] || [ "$val" = "yes" ] || [ "$val" = "on" ]; then
        warn "$var ACTIVE ($val) — degraded mode"
    else
        ok "$var inactive"
    fi
done

# ToE grounding corpus (§71)
LEAN_DIR="$HOME/Omega_Theory_Everything/lean_proofs"
if [ -f "$LEAN_DIR/kernel_audit_targets.json" ]; then
    n=$(grep -o '"' "$LEAN_DIR/kernel_audit_targets.json" | wc -l)
    ok "ToE Lean corpus present (kernel audit targets file exists)"
else
    warn "ToE Lean corpus missing at $LEAN_DIR — daemon runs ungrounded"
fi

# Config sanity: canonical socket paths, no /tmp sockets
for conf in "$PROJECT_ROOT/source/omni-bridge/config/production/bridge.yaml" \
            "$PROJECT_ROOT/source/omni-bridge/config/development/bridge.yaml"; do
    if [ -f "$conf" ]; then
        if grep -q "/tmp/.*\.sock" "$conf"; then
            fail "$conf still references /tmp sockets (PrivateTmp isolates them)"
        else
            ok "$(basename $(dirname $conf)) config: canonical socket paths"
        fi
    fi
done

df -h "$PROJECT_ROOT" | tail -1 | awk '{print "Disk: "$3" used / "$2" ("$5")"}'

if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}Doctor: no failures.${NC}"
else
    echo -e "${RED}Doctor: $FAILURES failure(s).${NC}"
fi
exit "$FAILURES"
