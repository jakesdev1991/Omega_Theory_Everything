#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Audit vacuous or misleading declarations in the Lean proof corpus.

Complements `audit_axioms.py` (which counts declaration-level `axiom`s) by
tracking three *honesty* metrics plus a per-export legacy-alias baseline:

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

The numeric metrics may only go down. Parameterized Unit definitions are included.
The optional alias baseline inventories known *_Stmt scope debt by declaration,
so new legacy-style exports cannot reuse another declaration's quota. None of
these lexical checks establish satisfiability of hypotheses or semantic scope.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lean_source import lean_sources, mask_comments_and_strings

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
UNIT_TYPE = re.compile(
    r"^\s*(?:noncomputable\s+)?(?:def|abbrev)\s+\S+"
    r"(?:\s*(?:\([^)]*\)|\{[^}]*\}|\[[^\]]*\]))*"
    r"\s*:\s*Type(?:\s+\d+)?\s*:=\s*Unit\b",
    re.M,
)
THEOREM_HEAD = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable)\s+)*"
    r"(?:theorem|lemma)\s+([^\s(:{]+)"
)

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
        while ":=" not in stmt and j < len(lines) - 1:
            j += 1
            stmt += " " + lines[j].strip()
        out.append((m.group(1), stmt))
    return out


def audit(root: Path) -> tuple[list[str], list[str], list[str]]:
    misleading: list[str] = []
    stubs: list[str] = []
    unit_types: list[str] = []
    for path in lean_sources(root):
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        for name, stmt in theorem_declarations(text):
            if name.startswith("bridge_") or name in ALLOWED_NAMES:
                continue
            if any(rx.search(stmt) for rx in TRIVIAL_CONCLUSION):
                misleading.append(f"{path.relative_to(root)}: {name}")
        for n, line in enumerate(text.splitlines(), 1):
            if UNIT_STUB.search(line):
                stubs.append(f"{path.relative_to(root)}:{n}: {line.strip()[:90]}")
        if path.name.startswith("Vol"):
            for m in UNIT_TYPE.finditer(text):
                unit_types.append(f"{path.relative_to(root)}: {m.group(0).strip()}")
    return misleading, stubs, unit_types


def statement_aliases(root: Path) -> list[str]:
    """Inventory theorem conclusions hidden behind legacy *_Stmt names.

    This is a lexical scope-debt detector, NOT a determination that every
    proposition alias is vacuous. Reviewed legacy exports remain visible in a
    per-declaration baseline; new aliases cannot silently reuse a numeric quota.
    """
    findings: list[str] = []
    pattern = re.compile(r":\s*([\w.]+_Stmt)\s*:=", re.UNICODE)
    for path in lean_sources(root):
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        for name, statement in theorem_declarations(text):
            match = pattern.search(statement)
            if match and not name.startswith("bridge_"):
                findings.append(f"{path.relative_to(root)}: {name} -> {match.group(1)}")
    return findings


def alias_baseline_errors(root: Path, baseline: Path) -> tuple[list[str], list[str]]:
    """New debt fails, and removed debt must be deleted from the baseline."""
    expected = set(json.loads(baseline.read_text(encoding="utf-8")))
    actual = set(statement_aliases(root))
    return sorted(actual - expected), sorted(expected - actual)


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
    parser.add_argument(
        "--alias-baseline",
        type=Path,
        default=None,
        help="Exact reviewed legacy *_Stmt exports; not a proof certificate",
    )
    args = parser.parse_args()

    misleading, stubs, unit_types = audit(args.root)
    rc = 0
    rc |= report(
        "misleading trivially-stated theorems", misleading, args.max_misleading
    )
    rc |= report("Nonempty-unit stubs", stubs, args.max_unit_stubs)
    rc |= report("Unit-typed volume definitions", unit_types, args.max_unit_types)
    if args.alias_baseline is not None:
        new, stale = alias_baseline_errors(args.root, args.alias_baseline)
        print(
            f"Legacy statement aliases (scope debt): {len(statement_aliases(args.root))}"
        )
        for item in new:
            print(f"FAIL: unreviewed statement alias: {item}")
        for item in stale:
            print(f"FAIL: remove retired baseline entry: {item}")
        rc |= int(bool(new or stale))
    if rc == 0:
        print("vacuity audit: OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
