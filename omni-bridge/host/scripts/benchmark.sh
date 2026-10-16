#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# benchmark.sh — REAL performance measurement (§49/§50/§79.8)
# Measures the actual C++ core paths + live bridge round-trip, and writes
# populated percentile JSON — never empty templates.
set -euo pipefail

PROJECT_ROOT="/opt/omni-bridge"
SOURCE_ROOT="$PROJECT_ROOT/source"
RESULTS_DIR="$PROJECT_ROOT/benchmark-results"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS="$RESULTS_DIR/benchmark_$TIMESTAMP.json"
HOST=$(uname -n)
KERNEL=$(uname -r)
CPU=$(lscpu | grep 'Model name' | head -1 | sed 's/.*: *//')
MEM=$(free -h | awk '/Mem:/{print $2}')

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "Omni-Bridge Benchmark  ($TIMESTAMP)"
echo "============================================"

python3 - "$RESULTS" "$HOST" "$KERNEL" "$CPU" "$MEM" "$SOURCE_ROOT" << 'PYEOF'
import json, subprocess, socket, statistics, sys, time, os

results_path, host, kernel, cpu, mem, source_root = sys.argv[1:7]
out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "host": host, "kernel": kernel, "cpu": cpu, "memory": mem,
       "benchmarks": {}}

def percentiles(xs):
    xs = sorted(xs)
    def pct(p):
        return round(xs[min(len(xs)-1, int(len(xs)*p/100))], 3)
    return {"p50_ms": pct(50), "p90_ms": pct(90), "p95_ms": pct(95),
            "p99_ms": pct(99), "p999_ms": pct(99.9),
            "mean_ms": round(statistics.mean(xs), 3),
            "min_ms": round(xs[0], 3), "max_ms": round(xs[-1], 3),
            "samples": len(xs)}

# 1. C++ core unit-test binary latency (cold process spawn)
xs = []
binpath = f"{source_root}/omni-bridge/cpp/build/omni_bridge_tests"
for _ in range(10):
    t0 = time.perf_counter()
    subprocess.run([binpath], capture_output=True, check=False)
    xs.append((time.perf_counter()-t0)*1000)
out["benchmarks"]["cpp_core_process_ms"] = percentiles(xs)

# 2. Live bridge round-trip over the unix socket
SOCK = "/opt/omni-bridge/runtime/omni_bridge.sock"
req = json.dumps({"payload": "benchmark probe",
                  "security_label": {"integrity": "TRUSTED",
                                     "confidentiality": "INTERNAL",
                                     "origin": "bench", "purpose": "measure"}})
def once():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(SOCK)
    s.sendall(req.encode())
    data = b""
    try:
        while True:
            c = s.recv(65536)
            if not c: break
            data += c
    except socket.timeout:
        pass
    s.close()
    return data

if os.path.exists(SOCK):
    once()  # warmup
    xs = []
    for i in range(200):
        t0 = time.perf_counter()
        once()
        xs.append((time.perf_counter()-t0)*1000)
    out["benchmarks"]["bridge_ipc_roundtrip_ms"] = percentiles(xs)
else:
    out["benchmarks"]["bridge_ipc_roundtrip_ms"] = {"skipped": "bridge socket not up"}

# 3. Ledger throughput: 1000 appends to a temp ledger
sys.path.insert(0, f"{source_root}/omni-bridge/python")
xs = []
try:
    from omni_bridge.evidence_ledger import EvidenceLedger, LedgerRecord
    from omni_bridge.appa import SecurityLabel, IntegrityLevel, ConfidentialityLevel
    import tempfile
    for _ in range(5):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tf:
            led = EvidenceLedger(tf.name)
            t0 = time.perf_counter()
            for i in range(1000):
                led.record(LedgerRecord(
                    timestamp=time.time(), trace_id=f"t{i}", actor="bench",
                    route="hermes", model="n/a", skill="n/a", tool="n/a",
                    input_hash="0"*64, output_hash="0"*64,
                    security_labels=SecurityLabel(IntegrityLevel.TRUSTED,
                                                   ConfidentialityLevel.INTERNAL,
                                                   "b", "b"),
                    capabilities=[], budget_used=0.0, latency_ms=0.0,
                    verification_result="n/a", outcome="x", previous_hash=""))
            xs.append((time.perf_counter()-t0)*1000)
    out["benchmarks"]["python_ledger_1000_appends_ms"] = \
        {"mean_ms": round(statistics.mean(xs),3), "samples": len(xs),
         "per_record_us": round(statistics.mean(xs),3)}
except Exception as e:
    out["benchmarks"]["python_ledger_1000_appends_ms"] = {"error": str(e)[:120]}

# 4. APPA algebra throughput (C++ tests exercise it; measure Python parity)
xs = []
try:
    from omni_bridge.appa import AppaEngine, Capability, IntegrityLevel, ConfidentialityLevel
    eng = AppaEngine()
    S = {Capability(f"c{i}", IntegrityLevel.TRUSTED, ConfidentialityLevel.INTERNAL, False)
         for i in range(64)}
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(1000):
            eng.compute_effective_permissions(S, S, S, S, S, set())
        xs.append((time.perf_counter()-t0)*1000)
    out["benchmarks"]["appa_1000_checks_ms"] = {"mean_ms": round(statistics.mean(xs),3),
                                                "per_check_us": round(statistics.mean(xs)/1.0,3)}
except Exception as e:
    out["benchmarks"]["appa_1000_checks_ms"] = {"error": str(e)[:120]}

with open(results_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["benchmarks"], indent=2))
print(f"\nWrote {results_path}")
PYEOF
