#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0
"""Audit vacuous or misleading declarations in the Lean proof corpus.

Complements `audit_axioms.py` (which counts declaration-level `axiom`s) by
tracking three *honesty* metrics:

1. misleading_trivial
   Theorems whose conclusion is one of the degenerate-model tautologies of
   `OmegaAxioms.lean` (`d R R = 0`, `vonNeumannEntropy R >= 0`, ...) while
   NOT following the honest `bridge_*` naming convention. Such a declaration
   dresses a structural consistency check up as a physical law, which is
   exactly what the `bridge_*` convention exists to prevent.

2. unit_stubs
   Existence "proofs" of the form `Nonempty X := ⟨()⟩` - placeholders that
   assert nothing beyond the inhabitation of `Unit`.

3. unit_types
   `def X : Type := Unit` inside the volume files (Vol*.lean). These are the
   tell-tale of an un-formalized volume. Model primitives in
   `OmegaAxioms.lean` are deliberate and are exempt.

All three metrics are ratcheted in CI: they may only go down.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Conclusions that are tautologies of the concrete Q-region model.
TRIVIAL_CONCLUSION = [
    re.compile(r":\s*d\s+(\S+)\s+\1\s*=\s*0"),
    re.compile(r":\s*d\s+\S+\s+\S+\s*≥\s*0"),
    re.compile(r":\s*d\s+\S+\s+\S+\s*≤\s*d\s+\S+\s+\S+\s*\+\s*d\s+\S+\s+\S+"),
    re.compile(r":\s*vonNeumannEntropy\s+\S+\s*≥\s*0"),
    re.compile(r":\s*vonNeumannEntropy\s+\S+\s*≤\s*Real\.pi"),
    re.compile(r":\s*mutualInformation\s+\S+\s+\S+\s*≥\s*0"),
    re.compile(r":\s*Φ\s+\S+\s+\S+\s*=\s*Φ"),
    re.compile(r":\s*Φ\s+\S+\s+\S+\s*≥\s*0"),
]

UNIT_STUB = re.compile(r"Nonempty\s+\S+\s*:=\s*⟨\(\)⟩")
UNIT_TYPE = re.compile(r"^\s*def\s+\S+\s*:\s*Type\s*:=\s*Unit\b", re.M)
THEOREM_HEAD = re.compile(r"^\s*(?:noncomputable\s+)?theorem\s+([A-Za-z0-9_']+)")

# Bundles that are documented as honest aggregates of structural facts.
ALLOWED_NAMES = {
    "omega_protocol_structural_bundle",
    "omega_theory_structural_consistency",
}


def theorem_declarations(text: str) -> list[tuple[str, str]]:
    """Return (name, full_statement) for each theorem, joining up to `:=`."""
    out: list[tuple[str, str]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = THEOREM_HEAD.match(line)
        if not m:
            continue
        stmt = line
        j = i
        while ":=" not in stmt and j < min(i + 12, len(lines) - 1):
            j += 1
            stmt += " " + lines[j].strip()
        out.append((m.group(1), stmt))
    return out


def audit(root: Path) -> tuple[list[str], list[str], list[str]]:
    misleading: list[str] = []
    stubs: list[str] = []
    unit_types: list[str] = []
    for path in sorted(root.glob("*.lean")):
        text = path.read_text(encoding="utf-8")
        for name, stmt in theorem_declarations(text):
            if name.startswith("bridge_") or name in ALLOWED_NAMES:
                continue
            if any(rx.search(stmt) for rx in TRIVIAL_CONCLUSION):
                misleading.append(f"{path.name}: {name}")
        for n, line in enumerate(text.splitlines(), 1):
            if UNIT_STUB.search(line):
                stubs.append(f"{path.name}:{n}: {line.strip()[:90]}")
        if path.name.startswith("Vol"):
            for m in UNIT_TYPE.finditer(text):
                unit_types.append(f"{path.name}: {m.group(0).strip()}")
    return misleading, stubs, unit_types


def report(title: str, items: list[str], maximum: int | None) -> int:
    print(f"{title}: {len(items)}")
    for item in items:
        print(f"  {item}")
    if maximum is not None and len(items) > maximum:
        print(f"FAIL: {len(items)} > --max {maximum}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-misleading", type=int, default=None)
    parser.add_argument("--max-unit-stubs", type=int, default=None)
    parser.add_argument("--max-unit-types", type=int, default=None)
    args = parser.parse_args()

    misleading, stubs, unit_types = audit(args.root)
    rc = 0
    rc |= report("misleading trivially-stated theorems", misleading, args.max_misleading)
    rc |= report("Nonempty-unit stubs", stubs, args.max_unit_stubs)
    rc |= report("Unit-typed volume definitions", unit_types, args.max_unit_types)
    if rc == 0:
        print("vacuity audit: OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
