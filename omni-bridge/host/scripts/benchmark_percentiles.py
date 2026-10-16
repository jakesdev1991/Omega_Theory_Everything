#!/opt/omni-bridge/runtime/venv/bin/python
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""§79 Performance: measure p50/p90/p95/p99/p99.9 latency of the live bridge
IPC path (client -> omni_bridge.sock -> APPA -> QSTAR -> IPC dispatch -> response)
plus memory/CPU RSS. Writes reports/benchmark-percentiles.json."""

import json
import os
import socket
import statistics
import sys
import time

sys.path.insert(0, "/opt/omni-bridge/source/omni-bridge/python")
from omni_bridge.paths import BRIDGE_SOCKET

SOCK = BRIDGE_SOCKET
N = 200


# warmup (first call compiles paths)
def once(msg="benchmark warmup"):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(240)
    s.connect(SOCK)
    s.sendall(msg.encode())
    data = b""
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    s.close()
    return data


once()

latencies = []
for i in range(N):
    t0 = time.perf_counter()
    once(f"percentile benchmark iteration {i}")
    latencies.append((time.perf_counter() - t0) * 1000.0)

lat_sorted = sorted(latencies)


def pct(p):
    idx = min(len(lat_sorted) - 1, int(len(lat_sorted) * p / 100))
    return round(lat_sorted[idx], 3)


result = {
    "target": "bridge IPC round-trip (client -> daemon -> adapter -> response)",
    "samples": N,
    "p50_ms": pct(50),
    "p90_ms": pct(90),
    "p95_ms": pct(95),
    "p99_ms": pct(99),
    "p999_ms": pct(99.9),
    "mean_ms": round(statistics.mean(latencies), 3),
    "min_ms": round(lat_sorted[0], 3),
    "max_ms": round(lat_sorted[-1], 3),
}

# process RSS of daemon + adapters
rss = {}
for proc, pat in [
    ("omni_bridge_daemon", "omni_bridge_daemon"),
    ("hermes_adapter", "hermes_ipc_adapter"),
    ("lucifer_adapter", "lucifer_ipc_adapter"),
]:
    try:
        pid = os.popen(f"pgrep -f {pat} | head -1").read().strip()
        if pid:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        rss[proc] = line.split()[1] + " kB"
    except Exception:
        pass
result["process_rss"] = rss

print(json.dumps(result, indent=2))

out = "/opt/omni-bridge/source/omni-bridge/reports/benchmark-percentiles.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {out}")
