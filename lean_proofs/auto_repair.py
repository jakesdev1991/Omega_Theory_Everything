#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""
OmegaProtocol historical repair record — 2026-09-23 auto_repair run.

PRESERVED FOR PROVENANCE (2026-10-16). This file is the surviving output of
the 2026-09-23 07:03 auto_repair run (original: lean_proofs/auto_repair.py,
salvaged from the local-draft-2026-09-23-0703 branch, commit 3dcfa10).

Historical context:
  The auto_repair script mechanically replaced `:= trivial` stub corollaries
  in the Vol files with definitional-unfolding proofs. That run is what
  allowed the kernel-check pipeline to reach zero `trivial` stubs. Its
  outputs were then superseded by the P0/P1 claim-tightening passes
  (PROOF_AUDIT.md), which RETIRED the resulting `*_from_omega` definitional
  restatements and `*_Stmt` vacuous unification propositions as misleading,
  replacing them with honestly-scoped models and the `bridge_*` naming
  convention.

Why this file survives as a record rather than as a tool:
  * Its function is fully covered — and exceeded — by the current audit
    toolchain: audit_vacuity.py (eight honesty metrics incl. tautology
    detection), audit_axioms.py, audit_kernel.py, and the
    statement-audit/parity harnesses. The regex-based repair it performed
    targeted a stub pattern that no longer exists in the corpus.
  * Re-running it against the current corpus would be a no-op (it greps for
    ':= trivial' stubs, of which zero remain), but keeping it as an active
    tool invites confusion with the audit harnesses and regression of the
    retired-theorem cleanup it would 'undo'.

Original run output (from the preserved repair_report.txt of that branch):
  60 .lean files scanned; OmegaUnifiedFoundation/OmegaAxioms CLEAN;
  Vol corollaries repaired from trivial stubs to definitional proofs.

See also: PROOF_AUDIT.md (the audit that superseded this approach),
          audit_vacuity.py (the living replacement).
"""

import datetime
import glob
import json
import os
import re
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(DIR, "repair_log.jsonl")


def log(msg):
    ts = datetime.datetime.now().isoformat()
    with open(LOG, "a") as f:
        f.write(json.dumps({"ts": ts, "msg": msg}) + "\n")
    print(f"[{ts}] {msg}")


def count_trivial(filepath):
    """Count trivial stubs in a file (historical metric; corpus now at zero)."""
    with open(filepath) as f:
        content = f.read()
    by_trivial = len(re.findall(r":=\s*by\s*trivial", content))
    plain_trivial = len(re.findall(r":=\s*trivial(?!\s*by)", content))
    return by_trivial, plain_trivial


def report_status():
    """Read-only status: count any residual trivial stubs.

    This function intentionally performs NO modification. The original
    repair_vol_corollaries/repair_omegaprotocol write paths are NOT
    preserved: their target pattern (per-corollary ':= trivial' stubs)
    no longer exists in this corpus, and the P0/P1 audits deliberately
    retired the definitional restatements the repair produced. Re-introducing
    them would regress PROOF_AUDIT.md's retired-theorem cleanup.

    Exit codes: 0 = clean (no stubs), 1 = stubs found (investigate; do
    not auto-repair — use the audit harnesses and a human-reviewed fix).
    """
    total = 0
    files = sorted(glob.glob(os.path.join(DIR, "*.lean")))
    log(f"status check: {len(files)} .lean files")
    for fp in files:
        by_t, plain_t = count_trivial(fp)
        if by_t or plain_t:
            log(f"  STUBS PRESENT {os.path.basename(fp)}: by={by_t} plain={plain_t}")
            total += by_t + plain_t
    log(f"total trivial stubs: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(report_status())
