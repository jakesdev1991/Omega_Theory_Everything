#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Audit vacuous or misleading declarations in the Lean proof corpus.

Complements `audit_axioms.py` (which counts declaration-level `axiom`s) by
tracking four *honesty* metrics:

1. misleading_trivial
   Theorems whose conclusion is one of the degenerate-model tautologies of
   `OmegaAxioms.lean` (`d R R = 0`, `vonNeumannEntropy R >= 0`, ...) while
   NOT following the honest `bridge_*` naming convention. Such a declaration
   dresses a structural consistency check up as a physical law, which is
   exactly what the `bridge_*` convention exists to prevent.
   Statements that merely *reference* a `*_Stmt` proposition alias are
   resolved to the alias body first, so the indirection cannot be used to
   evade this check (cf. ADVERSARIAL_AUDIT.md finding C1).

2. unit_stubs
   Existence "proofs" of the form `Nonempty X := ⟨()⟩` - placeholders that
   assert nothing beyond the inhabitation of `Unit`.

3. unit_types
   `def X : Type := Unit` inside the volume files (Vol*.lean). These are the
   tell-tale of an un-formalized volume. Model primitives in
   `OmegaAxioms.lean` are deliberate and are exempt.

4. trivial_proofs
   Proofs that consist of exactly `trivial` (on the `:= by` line or a
   following line). Such a proof always proves a definitional `True`, so
   it must be written `exact True.intro` (making the definitional
   unfolding explicit) or replaced by a real proof. This closes the hole
   where a single-line lexical gate could not see `:= by` newline
   `trivial`.

All four metrics are ratcheted in CI: they may only go down.
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
    re.compile(r":\s*mutualInformation\s+\S+\s+\S+\s*≤\s*maxMutualInformation"),
    re.compile(r":\s*Φ\s+\S+\s+\S+\s*=\s*Φ"),
    re.compile(r":\s*Φ\s+\S+\s+\S+\s*≥\s*0"),
    re.compile(r":\s*forwardFlux\s+\S+\s+\S+\s*=\s*Φ"),
    re.compile(r":\s*informationalImpedance\s+\S+\s+\S+\s*≥\s*0"),
]


def _bare(rx: re.Pattern) -> re.Pattern:
    """Drop the leading `:` anchor: alias bodies restate the conclusion
    without the `theorem ... :` prefix (e.g. after a `∀ ... ,`)."""
    prefix = r":\s*"
    assert rx.pattern.startswith(prefix), rx.pattern
    return re.compile(rx.pattern[len(prefix) :])


BARE_TRIVIAL = [_bare(rx) for rx in TRIVIAL_CONCLUSION]

# Single-line proposition aliases (`def Foo_Stmt : Prop := <body>`), the
# indirection mechanism behind finding C1. Multi-line aliases are not
# resolved (a known limitation, documented so it cannot be exploited
# silently: any `*_Stmt` reference that fails to resolve is reported).
STMT_DEF = re.compile(
    r"^\s*(?:noncomputable\s+)?def\s+([A-Za-z0-9_']+_Stmt)\s*:\s*Prop\s*:=\s*(.+)$",
    re.M,
)
STMT_REF = re.compile(r":\s*([A-Za-z0-9_']+_Stmt)\b")

# Files defining the model lemmas themselves: alias bodies that live here
# are the honest API, not dressed-up restatements.
MODEL_FILES = {"OmegaAxioms.lean"}

UNIT_STUB = re.compile(r"Nonempty\s+\S+\s*:=\s*⟨\(\)⟩")
UNIT_TYPE = re.compile(r"^\s*def\s+\S+\s*:\s*Type\s*:=\s*Unit\b", re.M)
THEOREM_HEAD = re.compile(r"^\s*(?:noncomputable\s+)?theorem\s+([A-Za-z0-9_']+)")
# `:= by trivial` with `trivial` as the whole proof (same line or a following
# line, modulo whitespace and a trailing comment). `trivial` either closes the
# goal or fails, so anything this matches is a trivial-only proof or dead code.
TRIVIAL_PROOF = re.compile(r":=\s*by\s*trivial[ \t]*(?:--[^\n]*)?$", re.M)

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


def audit(root: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    misleading: list[str] = []
    stubs: list[str] = []
    unit_types: list[str] = []
    trivial_proofs: list[str] = []
    for path in sorted(root.glob("*.lean")):
        text = path.read_text(encoding="utf-8")
        aliases = {m.group(1): m.group(2) for m in STMT_DEF.finditer(text)}
        for name, stmt in theorem_declarations(text):
            if name.startswith("bridge_") or name in ALLOWED_NAMES:
                continue
            if any(rx.search(stmt) for rx in TRIVIAL_CONCLUSION):
                misleading.append(f"{path.name}: {name}")
                continue
            if path.name in MODEL_FILES:
                continue
            ref = STMT_REF.search(stmt)
            if ref and ref.group(1) in aliases:
                body = aliases[ref.group(1)]
                if any(rx.search(body) for rx in BARE_TRIVIAL):
                    misleading.append(f"{path.name}: {name} (via {ref.group(1)})")
            elif ref:
                misleading.append(f"{path.name}: {name} (unresolved {ref.group(1)})")
        for n, line in enumerate(text.splitlines(), 1):
            if UNIT_STUB.search(line):
                stubs.append(f"{path.name}:{n}: {line.strip()[:90]}")
        if path.name.startswith("Vol"):
            for m in UNIT_TYPE.finditer(text):
                unit_types.append(f"{path.name}: {m.group(0).strip()}")
        for m in TRIVIAL_PROOF.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            trivial_proofs.append(f"{path.name}:{lineno}")
    return misleading, stubs, unit_types, trivial_proofs


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
    parser.add_argument("--max-trivial-proofs", type=int, default=None)
    args = parser.parse_args()

    misleading, stubs, unit_types, trivial_proofs = audit(args.root)
    rc = 0
    rc |= report(
        "misleading trivially-stated theorems", misleading, args.max_misleading
    )
    rc |= report("Nonempty-unit stubs", stubs, args.max_unit_stubs)
    rc |= report("Unit-typed volume definitions", unit_types, args.max_unit_types)
    rc |= report("trivial-only proofs", trivial_proofs, args.max_trivial_proofs)
    if rc == 0:
        print("vacuity audit: OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
