#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# benchmark_isolated.sh — REAL percentiles for the NEW control boundary,
# measured on isolated sockets (does not touch the live service).
set -uo pipefail

TESTDIR=$(mktemp -d /tmp/omni_bench.XXXXXX)
DAEMON=/opt/omni-bridge/source/omni-bridge/cpp/build/omni_bridge_daemon
BRIDGE_SOCK="$TESTDIR/bridge.sock"
HERMES_SOCK="$TESTDIR/hermes.sock"
LUCIFER_SOCK="$TESTDIR/lucifer.sock"
LEDGER="$TESTDIR/ledger.jsonl"
OUT="/opt/omni-bridge/benchmark-results/benchmark_$(date -u +%Y%m%dT%H%M%SZ).json"

trap '[ -n "${DAEMON_PID:-}" ] && kill $DAEMON_PID 2>/dev/null; [ -n "${H_PID:-}" ] && kill $H_PID 2>/dev/null; rm -rf "$TESTDIR"' EXIT

# mock hermes (fast deterministic responder)
python3 - "$HERMES_SOCK" << 'PYEOF' &
import asyncio, json, sys, os
path = sys.argv[1]
async def handle(r, w):
    try:
        await asyncio.wait_for(r.read(65536), timeout=10)
        w.write(json.dumps({"status":"ok","agent":"hermes","results":[]}).encode())
        await w.drain()
    except Exception: pass
    finally: w.close()
async def main():
    if os.path.exists(path): os.unlink(path)
    srv = await asyncio.start_unix_server(handle, path=path)
    async with srv: await srv.serve_forever()
asyncio.run(main())
PYEOF
H_PID=$!

OMNI_BRIDGE_SOCK="$BRIDGE_SOCK" OMNI_HERMES_SOCK="$HERMES_SOCK" \
OMNI_LUCIFER_SOCK="$LUCIFER_SOCK" OMNI_LEDGER_PATH="$LEDGER" \
"$DAEMON" > "$TESTDIR/daemon.log" 2>&1 &
DAEMON_PID=$!
sleep 2

python3 - "$BRIDGE_SOCK" "$OUT" << 'PYEOF'
import socket, json, statistics, sys, time

sock_path, out_path = sys.argv[1], sys.argv[2]

REQ = json.dumps({"payload": "benchmark probe: retrieve appa algebra",
                  "security_label": {"integrity": "TRUSTED",
                                     "confidentiality": "INTERNAL",
                                     "origin": "bench", "purpose": "measure"}})
REQ_UNTRUSTED = json.dumps({"payload": "benchmark probe: explore adversarial surface",
                            "security_label": {"integrity": "UNTRUSTED",
                                              "confidentiality": "PUBLIC",
                                              "origin": "bench", "purpose": "measure"}})

def once(req):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(sock_path)
    s.sendall(req.encode())
    data = b""
    try:
        while True:
            c = s.recv(65536)
            if not c: break
            data += c
            try: json.loads(data); break
            except Exception: continue
    except socket.timeout: pass
    s.close()
    return data

def percentiles(xs):
    xs = sorted(xs)
    def pct(p): return round(xs[min(len(xs)-1, int(len(xs)*p/100))], 3)
    return {"p50_ms": pct(50), "p90_ms": pct(90), "p95_ms": pct(95),
            "p99_ms": pct(99), "p999_ms": pct(99.9),
            "mean_ms": round(statistics.mean(xs), 3),
            "min_ms": round(xs[0],3), "max_ms": round(xs[-1],3), "samples": len(xs)}

result = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "target": "NEW control boundary: parse->classify->appa->qstar->dispatch->verify->ledger",
          "host": open("/etc/hostname").read().strip()}

for name, req, n in [("trusted_fast_path", REQ, 300),
                     ("untrusted_child_branch_path", REQ_UNTRUSTED, 300)]:
    once(req); once(req)  # warmup
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        once(req)
        xs.append((time.perf_counter()-t0)*1000)
    result[name] = percentiles(xs)

# ledger append cost (file, fsync-less flush)
import subprocess
xs = []
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(100):
        once(REQ)
    xs.append((time.perf_counter()-t0)*1000)
result["ledger_note"] = "every request appends 3-4 ledger records; flush() per record"

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
PYEOF
