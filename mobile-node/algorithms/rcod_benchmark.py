#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""RCOD Noise Recovery — allowlisted algorithm for the Lucifer mobile node.

Toy stand-in for rcod/benchmark_noise_recovery.py: reports the honest verdict
for the spec thresholds at a requested noise level. Bounded, offline.
"""

import json
import sys


def main() -> int:
    try:
        params = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        print(json.dumps({"status": "error", "reason": "params are not valid JSON"}))
        return 1

    noise_pct = float(params.get("noisePct", 20))
    recovered = max(0.0, 1.0 - (noise_pct / 100.0) ** 1.5)
    verdict = "pass" if recovered >= 0.9 else "fail"

    print(
        json.dumps(
            {
                "status": "success",
                "algorithm": "rcod-benchmark",
                "noisePct": noise_pct,
                "recovery": round(recovered, 4),
                "verdict": verdict,
                "note": "Spec-as-written thresholds remain honest: see rcod/RESULTS.md.",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
