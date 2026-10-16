#!/opt/omni-bridge/runtime/venv/bin/python
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Live production benchmark: typed-envelope round-trips through the RUNNING
bridge daemon on its production socket (parse->classify->appa->qstar->
dispatch->hermes adapter->verify->persistent ledger append)."""

import json
import socket
import statistics
import sys
import time

sys.path.insert(0, "/opt/omni-bridge/source/omni-bridge/python")
from omni_bridge.paths import BRIDGE_SOCKET

SOCK = BRIDGE_SOCKET
N = 300

REQ_TRUSTED = json.dumps(
    {
        "payload": "benchmark probe: retrieve the appa permission algebra",
        "security_label": {
            "integrity": "TRUSTED",
            "confidentiality": "INTERNAL",
            "origin": "bench",
            "purpose": "measure",
        },
    }
)
REQ_UNTRUSTED = json.dumps(
    {
        "payload": "benchmark probe: explore the adversarial surface",
        "security_label": {
            "integrity": "UNTRUSTED",
            "confidentiality": "PUBLIC",
            "origin": "bench",
            "purpose": "measure",
        },
    }
)


def once(req):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(SOCK)
    s.sendall(req.encode())
    data = b""
    try:
        while True:
            c = s.recv(65536)
            if not c:
                break
            data += c
            try:
                json.loads(data)
                break
            except Exception:
                continue
    except socket.timeout:
        pass
    s.close()
    return data


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p / 100))], 3)


result: dict[str, object] = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "target": "PRODUCTION bridge: typed envelope through live daemon + adapters",
}

for name, req in [
    ("trusted_fast_path", REQ_TRUSTED),
    ("untrusted_child_branch_path", REQ_UNTRUSTED),
]:
    once(req)
    once(req)  # warmup
    xs = []
    for _ in range(N):
        t0 = time.perf_counter()
        resp = once(req)
        dt = (time.perf_counter() - t0) * 1000
        j = json.loads(resp)
        assert j["status"] in ("EXECUTED", "FAILED", "REJECTED"), j["status"]
        xs.append(dt)
    result[name] = {
        "p50_ms": pct(xs, 50),
        "p90_ms": pct(xs, 90),
        "p95_ms": pct(xs, 95),
        "p99_ms": pct(xs, 99),
        "p999_ms": pct(xs, 99.9),
        "mean_ms": round(statistics.mean(xs), 3),
        "min_ms": round(min(xs), 3),
        "max_ms": round(max(xs), 3),
        "samples": len(xs),
    }

out = "/opt/omni-bridge/benchmark-results/benchmark_live_{}.json".format(
    time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
)
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print(f"\nWrote {out}")
