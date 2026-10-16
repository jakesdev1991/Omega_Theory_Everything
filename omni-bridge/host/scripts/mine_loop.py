#!/opt/omni-bridge/runtime/venv/bin/python
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Autonomous PoKW mine loop (whitepapers/omega_pokw_whitepaper.md §3).

Runs until the proof backlog has no open tasks, then self-disarms (the
timer keeps firing but the op is a cheap no-op). Each iteration:
refresh backlog -> claim next open task -> conjecture screen -> kernel
baseline-verify -> write proposal artifact -> ledger record. Proposes,
never merges (§22).

Designed to run under the omni-mine user timer every 15 minutes.
"""

import json
import socket
import sys
import time

LUCIFER_SOCK = "/opt/omni-bridge/runtime/lucifer.sock"
BACKLOG = "/opt/omni-bridge/telemetry/proof_backlog.json"
MAX_ITERATIONS = 3  # per timer tick — keeps each run under ~10 min


def call(op, timeout=600):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(LUCIFER_SOCK)
    s.sendall(json.dumps(op).encode())
    s.shutdown(socket.SHUT_WR)
    data = b""
    while True:
        c = s.recv(65536)
        if not c:
            break
        data += c
    s.close()
    return json.loads(data.decode())


def backlog_open():
    r = call({"op": "backlog", "action": "list", "limit": 1}, timeout=30)
    return r.get("open_tasks", 0)


def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    open_tasks = backlog_open()
    print(f"[mine-loop {ts}] backlog: {open_tasks} open tasks")

    if open_tasks == 0:
        print(
            "[mine-loop] backlog complete — self-disarmed (timer continues as a no-op)."
        )
        return 0

    done = 0
    for i in range(min(MAX_ITERATIONS, open_tasks)):
        r = call({"op": "mine"}, timeout=600)
        if r.get("status") != "ok":
            print(
                f"[mine-loop] iteration {i}: mine op error: {r.get('error', r)[:160]}"
            )
            break
        if r.get("result") == "backlog empty":
            print("[mine-loop] backlog drained mid-run.")
            break
        conj = r.get("conjecture") or {}
        print(
            f"[mine-loop] {r.get('claimed')}: baseline_verified="
            f"{r.get('module_baseline_verified')} "
            f"conjecture(pass/refuted/cand)="
            f"{conj.get('numeric_pass', 0)}/"
            f"{conj.get('refuted', 0)}/{conj.get('candidates', 0)} "
            f"-> {r.get('proposal', '')}"
        )
        done += 1

    print(
        f"[mine-loop] {done} iteration(s) this tick; "
        f"{backlog_open()} task(s) remain open."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
