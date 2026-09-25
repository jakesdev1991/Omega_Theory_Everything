#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
"""Lean Axiom Audit — allowlisted algorithm for the Lucifer mobile node.

Counts `sorry` occurrences and `axiom` declarations in a named lean_proofs
module (or the whole directory when module is "*"). Pure filesystem reads
inside the repository; prints a JSON summary.
"""

import json
import os
import sys


def main() -> int:
    try:
        params = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        print(json.dumps({"status": "error", "reason": "params are not valid JSON"}))
        return 1

    repo = os.environ.get(
        "OMEGA_REPO",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    lean_dir = os.path.join(repo, "lean_proofs")
    module = str(params.get("module", "*"))

    if not os.path.isdir(lean_dir):
        print(
            json.dumps(
                {"status": "error", "reason": f"lean_proofs not found at {lean_dir}"}
            )
        )
        return 1

    names = [f"{module}.lean"] if module != "*" else sorted(os.listdir(lean_dir))
    report = {}
    for name in names:
        path = os.path.join(lean_dir, name)
        if not name.endswith(".lean") or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        report[name] = {
            "sorry": text.count("sorry"),
            "axiom": sum(
                1 for line in text.splitlines() if line.strip().startswith("axiom ")
            ),
        }

    print(
        json.dumps(
            {
                "status": "success",
                "algorithm": "lean-audit",
                "module": module,
                "report": report,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
