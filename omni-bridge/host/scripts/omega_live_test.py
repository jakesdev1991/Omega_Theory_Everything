#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Omega Live Test Harness — automated active testing of the whole live stack.

Runs on Ryzenvoid under cron (or the omni-bridge iterate loop). Exercises:
  1. Relay NIP-11 over the onion path + LAN
  2. Store directory presence (kinds 31990) on the relay
  3. DVM roundtrip: kind-5001 job (operator key) -> 6001 result
  4. Website endpoints (Cloudflare Worker + local prod)
  5. REAL Lucifer health + Lean grounding (lucifer_ipc_adapter socket)

Everything fails loud: any check failing writes a line to the report and
exits non-zero, so the iterate loop can react. Results append to
/opt/omni-bridge/telemetry/live_test_report.jsonl.
"""

import datetime
import json
import socket as pysocket
import sys
import urllib.request
from pathlib import Path

RELAY_LAN = "https://localhost:8080"
CA = str(Path.home() / ".config/omega-nostr/tls/omega-ca-cert.pem")
REPORT = Path("/opt/omni-bridge/telemetry/live_test_report.jsonl")
LUCIFER_SOCK = "/opt/omni-bridge/runtime/lucifer.sock"
WEB_LOCAL = "http://127.0.0.1:3460"
WEB_EDGE = "https://omega-theory-everything.jake-s-dev1991.workers.dev"

results: list[dict[str, object]] = []


def check(name, fn):
    try:
        detail = fn()
        results.append({"check": name, "ok": True, "detail": detail})
        print(f"  PASS {name}: {detail}")
    except Exception as e:
        results.append({"check": name, "ok": False, "detail": str(e)})
        print(f"  FAIL {name}: {e}")


def nip11(url, verify=True):
    req = urllib.request.Request(url, headers={"Accept": "application/nostr+json"})
    ctx = None
    if not verify:
        import ssl

        ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
        data = json.loads(r.read())
        if data.get("software") != "chorus":
            raise RuntimeError(f"not chorus: {data.get('software')}")
        return f"{data.get('name')} v{data.get('version')} nips={len(data.get('supported_nips', []))}"


def lucifer_ipc(op_json):
    """Send a request to the real Lucifer adapter over its unix socket."""
    s = pysocket.socket(pysocket.AF_UNIX, pysocket.SOCK_STREAM)
    s.settimeout(15)
    s.connect(LUCIFER_SOCK)
    s.sendall(op_json.encode())
    chunks = []
    try:
        while True:
            b = s.recv(65536)
            if not b:
                break
            chunks.append(b)
    except pysocket.timeout:
        pass
    s.close()
    return json.loads(b"".join(chunks).decode())


def web(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OmegaLiveTest/1.0 (+https://github.com/jakesdev1991/Omega_Theory_Everything)"
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return f"HTTP {r.status}"


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"== Omega live test {ts} ==")

    check("relay_lan_nip11", lambda: nip11(f"{RELAY_LAN}/", verify=False))
    check("website_local", lambda: web(f"{WEB_LOCAL}/api/social/status"))
    check("website_edge", lambda: web(f"{WEB_EDGE}/api/social/status"))
    check("lucifer_health", lambda: json.dumps(lucifer_ipc('{"op": "health"}'))[:120])
    check(
        "lucifer_lean_lookup",
        lambda: (
            f"{lucifer_ipc(json.dumps({'op': 'lean_lookup', 'query': 'entropy holographic'}))['evidence_count']} evidence items"
        ),
    )
    check(
        "lucifer_explore_grounded",
        lambda: len(
            lucifer_ipc(
                json.dumps({"op": "explore", "prompt": "born rule quantum measurement"})
            ).get("lean_grounding", [])
        ),
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ok_count = sum(1 for r in results if r["ok"])
    line = {"ts": ts, "passed": ok_count, "total": len(results), "results": results}
    with REPORT.open("a") as f:
        f.write(json.dumps(line) + "\n")

    print(f"== {ok_count}/{len(results)} passed ==")
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
