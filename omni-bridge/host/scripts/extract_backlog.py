#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Extract the PoKW proof backlog from PROOF_AUDIT.md's per-module 'Next:'
entries into a machine-readable JSON task queue (PoKW whitepaper §4).

Read-only toward the repo; output: /opt/omni-bridge/telemetry/proof_backlog.json
Classes per the PoKW paper:
  A  open proof task (from 'Next:' entries)
  B  registered conjecture (from conjecture_pilot transforms — added by the
     mine loop, not this extractor)
"""

import json
import re
import sys
from pathlib import Path

AUDIT = Path.home() / "Omega_Theory_Everything" / "lean_proofs" / "PROOF_AUDIT.md"
OUT = Path("/opt/omni-bridge/telemetry/proof_backlog.json")

ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|", re.M)
NEXT = re.compile(r"\bNext:\s*(.+?)(?:\.\s*$|$)", re.I)


def main():
    text = AUDIT.read_text()
    # preserve claim state across regenerations (the mine loop refreshes
    # the backlog each iteration — claims must survive the refresh)
    prior = {}
    if OUT.exists():
        try:
            for t in json.loads(OUT.read_text()).get("tasks", []):
                if t.get("status") in ("claimed", "done"):
                    prior[(t.get("module"), t.get("task"))] = t
        except Exception:
            pass

    tasks = []
    seen = set()
    # module rows in the two inventory tables: | `File.lean` | ... Next: ... |
    for m in re.finditer(r"^\|\s*`([\w.]+\.lean)`\s*\|([^|]+)\|", text, re.M):
        module, body = m.group(1), m.group(2)
        for nm in NEXT.finditer(body):
            desc = nm.group(1).strip().rstrip(".")
            if not desc or len(desc) < 8:
                continue
            key = (module, desc)
            if key in seen:
                continue
            seen.add(key)
            task = {
                "task_id": f"A-{module.split('.')[0]}-{len(tasks):03d}",
                "class": "A",
                "module": module,
                "task": desc,
                "source": "PROOF_AUDIT.md",
                "status": "open",
            }
            old = prior.get(key)
            if old:
                task["status"] = old["status"]
                for k in ("claimed_at", "claimed_by"):
                    if k in old:
                        task[k] = old[k]
            tasks.append(task)
    # header Next-work items (sections with 'Next:' in prose)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_from": str(AUDIT),
        "repo_head_note": "regenerate after each audit pass",
        "classes": {
            "A": "open proof task",
            "B": "registered conjecture",
            "C": "counterexample/retirement",
            "D": "model upgrade",
            "E": "preregistered benchmark (no monetary weight)",
        },
        "tasks": tasks,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"[backlog] {len(tasks)} Class-A tasks written to {OUT}")
    for t in tasks[:8]:
        print(f"  {t['task_id']}  {t['module']}: {t['task'][:70]}")
    if len(tasks) > 8:
        print(f"  ... +{len(tasks) - 8} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
