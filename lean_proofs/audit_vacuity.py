#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Audit vacuous or misleading declarations in the Lean proof corpus.

Complements `audit_axioms.py` (which counts declaration-level `axiom`s) by
tracking eight *honesty* metrics plus a per-export legacy-alias baseline:

1. misleading_trivial
   Theorems whose conclusion is one of the degenerate-model tautologies of
   `OmegaAxioms.lean` (`d R R = 0`, `vonNeumannEntropy R >= 0`, ...) while
   NOT following the honest `bridge_*` naming convention. Such a declaration
   dresses a structural consistency check up as a physical law, which is
   exactly what the `bridge_*` convention exists to prevent. Conclusions are
   checked both verbatim and after stripping leading `∀`-binders and
   `→`-hypotheses, so `theorem x : ∀ R, d R R = 0` cannot hide behind its
   binder list (a blind spot of the previous revision).

2. misleading_axiom_names
   No theorem, lemma, definition, or structure field/assignment may be
   *named* `*_axiom` / `axiom_*`: the corpus contains zero Lean `axiom`
   primitives, so a declaration carrying "axiom" in its name misrepresents
   either its proof status or its role. Structure law-fields use the
   `law_*` prefix instead; model laws carried as data are fields, not
   axioms.

3. unit_stubs
   Existence "proofs" of the form `Nonempty X := ⟨()⟩` - placeholders that
   assert nothing beyond the inhabitation of `Unit`.

4. unit_types
   `def X : Type := Unit` inside the volume files (Vol*.lean). These are the
   tell-tale of an un-formalized volume. Model primitives in
   `OmegaAxioms.lean` are deliberate and are exempt.

5. trivial_proofs
   Proof scripts that consist of exactly `trivial` prove nothing and are
   rejected outright.

6. decorative_hypotheses
   Theorem hypotheses that neither appear elsewhere in the statement nor in
   the proof script (proofs closed by hypothesis-consuming automation are
   skipped). A binder that constrains nothing is a domain restriction in
   costume: it suggests a generality the statement does not have.

7. trivially_true_hypotheses
   Hypotheses whose *type* is a closed truth (`0 ≤ 0`, `n = n`, `True`).
   These are decorative by construction and are rejected outright.

8. identity_proofs
   Proofs of the exact shape `intro h; exact h` / `exact h` where `h` is one
   of the theorem's own hypotheses: the conclusion restates a hypothesis
   (P→P), so the declaration contributes no mathematical content even when
   the two sides differ syntactically (definitional collapses included).

The numeric metrics may only go down. Parameterized Unit definitions are included.
The optional alias baseline inventories known *_Stmt scope debt by declaration,
so new legacy-style exports cannot reuse another declaration's quota. None of
these lexical checks establish satisfiability of hypotheses or semantic scope;
they close specific review blind spots (binder-hidden conclusions, decorative
hypotheses, P→P pass-throughs) that a name-based audit alone cannot see.
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
# Declaration names (after a declaration keyword) and bare structure
# field/assignment lines whose identifier contains "axiom".
DECL_NAME = re.compile(
    r"\b(?:theorem|lemma|def|abbrev|instance|structure|inductive|class)\s+"
    r"([A-Za-z_][A-Za-z0-9_']*)"
)
FIELD_LINE = re.compile(r"^[\t ]*([A-Za-z_][A-Za-z0-9_'.]*)[\t ]*:(?:=|\t| )", re.M)

# Proof scripts whose tactic may consume a hypothesis without naming it.
HYPOTHESIS_CONSUMING = re.compile(
    r"\b(simp_all|omega|linarith|nlinarith|aesop|tauto|bound|gcongr|group"
    r"|field_simp|positivity|norm_num)\b"
)


def _word(name: str) -> str:
    """Identifier-boundary pattern that treats primes as part of the name."""
    return r"(?<![\w'])" + re.escape(name) + r"(?![\w'])"


PROP_LIKE = re.compile(r"(<|>|≤|≥|=|≠|∈|→|↔|∀|∃|\bTrue\b|\bFalse\b)")
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_'₀-₉]*"
# `exact h`, `intro h; exact h`, or several names introduced at once.
IDENTITY_PROOF = re.compile(
    r"^(?:intro\s+((?:" + IDENTIFIER + r")(?:\s+" + IDENTIFIER + r")*)\s*;\s*)?"
    r"exact\s+(" + IDENTIFIER + r")$"
)
# Term-mode pass-through: the whole proof body is a single binder name.
TERM_IDENTITY = re.compile(r"^(" + IDENTIFIER + r")$")

# Bundles that are documented as honest aggregates of structural facts.
ALLOWED_NAMES = {
    "omega_protocol_structural_bundle",
    "omega_theory_structural_consistency",
}


def _depth_delta(char: str) -> int:
    if char in "([{⟨":
        return 1
    if char in ")]}⟩":
        return -1
    return 0


def strip_forall(statement: str) -> str:
    """Drop leading `∀ ... ,` binder groups (grouping aware, not a parser)."""
    text = statement.strip()
    while text.startswith("∀"):
        rest = text[1:]
        depth = 0
        cut = None
        for index, char in enumerate(rest):
            depth += _depth_delta(char)
            if char == "," and depth == 0:
                cut = index
                break
        if cut is None:
            break
        text = rest[cut + 1 :].strip()
    return text


def top_level_split(statement: str, separator: str) -> list[str]:
    """Split on `separator` occurrences at bracket depth zero."""
    parts: list[str] = []
    current = ""
    depth = 0
    index = 0
    while index < len(statement):
        char = statement[index]
        depth += _depth_delta(char)
        if depth == 0 and statement.startswith(separator, index):
            parts.append(current)
            current = ""
            index += len(separator)
            continue
        current += char
        index += 1
    parts.append(current)
    return parts


def final_conclusion(statement: str) -> str:
    """The conclusion of a theorem statement.

    Strips the declaration head, leading `∀`-binders and every top-level
    `→`-separated hypothesis, so a conclusion shaped like one of the
    degenerate-model tautologies is found even when it is quantified.
    """
    _, body = split_signature(statement)
    return top_level_split(strip_forall(body), "→")[-1].strip()


def theorem_blocks(text: str) -> list[tuple[str, str, str]]:
    """Return (name, statement, proof body) for each theorem/lemma.

    The statement runs up to `:=`; the body is the remainder of that line
    plus following lines that are indented (Lean top-level commands start at
    column zero). This is a lexical heuristic, not a Lean parser.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, str, str]] = []
    index = 0
    while index < len(lines):
        match = THEOREM_HEAD.match(lines[index])
        if not match:
            index += 1
            continue
        statement = lines[index]
        end = index
        while ":=" not in statement and end < len(lines) - 1:
            end += 1
            statement += " " + lines[end].strip()
        body_lines: list[str] = []
        if ":=" in statement:
            body_lines.append(statement.split(":=", 1)[1])
            cursor = end + 1
            while cursor < len(lines) and (
                lines[cursor].strip() == "" or lines[cursor][:1] in (" ", "\t")
            ):
                body_lines.append(lines[cursor])
                cursor += 1
            index = max(cursor, index + 1)
        else:
            index = end + 1
        blocks.append((match.group(1), statement, "\n".join(body_lines)))
    return blocks


def split_signature(statement: str) -> tuple[str, str]:
    """Split a theorem statement into (signature, body) at the first
    top-level `:` that is not part of a binder or `:=`."""
    depth = 0
    index = 0
    while index < len(statement):
        char = statement[index]
        depth += _depth_delta(char)
        if depth == 0 and char == ":" and not statement.startswith(":=", index):
            return statement[:index], statement[index + 1 :]
        index += 1
    return statement, ""


def hypothesis_binders(statement: str) -> list[tuple[str, str]]:
    """Return (name, type) for each proposition-like `(name : type)` binder.

    Only the signature and the leading hypothesis chain are scanned: a
    parenthesised term ascription in a conclusion (`f : ℕ → ℕ → Bool`) is
    not a hypothesis. Handles `{name : type}` and shared binders `(a b : T)`;
    non-Prop-like types (data parameters) are skipped because an unused data
    parameter is what parametricity looks like, not a defect.
    """
    signature, body = split_signature(statement)
    chain = top_level_split(strip_forall(body), "→")
    hypothesis_part = "→".join(chain[:-1])
    return find_binders(signature) + find_binders(hypothesis_part)


def find_binders(text: str, require_prop: bool = True) -> list[tuple[str, str]]:
    """Return (name, type) for `(name : type)` binders in text.

    With `require_prop` (default) only proposition-like types are returned."""
    found: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if (
            char in "({"
            and index + 1 < len(text)
            and (text[index + 1].isalpha() or text[index + 1] == "_")
        ):
            depth = 0
            close = index
            while close < len(text):
                depth += _depth_delta(text[close])
                if depth == 0:
                    break
                close += 1
            inner = text[index + 1 : close]
            cut = None
            inner_depth = 0
            for offset, inner_char in enumerate(inner):
                inner_depth += _depth_delta(inner_char)
                if inner_char == ":" and inner_depth == 0:
                    cut = offset
                    break
            if cut is not None:
                names = inner[:cut].strip()
                type_text = inner[cut + 1 :].strip()
                if re.fullmatch(IDENTIFIER + r"(?:\s+" + IDENTIFIER + r")*", names):
                    if not require_prop or PROP_LIKE.search(type_text):
                        found.append((names.split()[0], type_text))
            index = close + 1
        else:
            index += 1
    return found


def all_binder_names(statement: str) -> set[str]:
    """Every name introduced by a `(name : type)` / `{name : type}` binder.

    Unlike `hypothesis_binders`, no property-like filter is applied: this is
    used to recognise proofs that hand a bound name straight back.
    """
    return {name for name, _ in find_binders(statement, require_prop=False)}


def _drop_type_ascription(text: str) -> str:
    """Erase `(e : T)` ascriptions for numeral-only comparisons."""
    return re.sub(r"\(\s*([^()]*?)\s*:\s*(?:ℝ|ℕ|ℤ|ℚ|ℂ)\s*\)", r"(\1)", text)


def trivially_true_prop(type_text: str) -> bool:
    """True for closed truths usable as a hypothesis without content."""
    text = re.sub(r"\s+", " ", _drop_type_ascription(type_text)).strip()
    if text in {"True", "1 = 1", "0 = 0", "0 ≤ 0", "0 < 1", "1 ≠ 0", "0 ≠ 1"}:
        return True
    numeral = re.fullmatch(
        r"\(?\s*(?P<ln>\d+)\s*(?:/\s*(?P<ld>\d+))?\s*\)?\s*"
        r"(?P<op>≤|<|≥|>|=|≠)\s*"
        r"\(?\s*(?P<rn>\d+)\s*(?:/\s*(?P<rd>\d+))?\s*\)?",
        text,
    )
    if numeral:
        left = int(numeral.group("ln")) / int(numeral.group("ld") or 1)
        right = int(numeral.group("rn")) / int(numeral.group("rd") or 1)
        return {
            "≤": left <= right,
            "<": left < right,
            "≥": left >= right,
            ">": left > right,
            "=": left == right,
            "≠": left != right,
        }[numeral.group("op")]
    reflexive = re.fullmatch(r"(\S+)\s*(=|≤|≥)\s*(\S+)", text)
    if reflexive and reflexive.group(1) == reflexive.group(3):
        return True
    return False


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
            rel = path.relative_to(root)
            if any(rx.search(stmt) for rx in TRIVIAL_CONCLUSION):
                misleading.append(f"{rel}: {name}")
                continue
            hidden = ": " + final_conclusion(stmt)
            if any(rx.search(hidden) for rx in TRIVIAL_CONCLUSION):
                misleading.append(f"{rel}: {name} (quantified conclusion)")
        for n, line in enumerate(text.splitlines(), 1):
            if UNIT_STUB.search(line):
                stubs.append(f"{path.relative_to(root)}:{n}: {line.strip()[:90]}")
        if path.name.startswith("Vol"):
            for m in UNIT_TYPE.finditer(text):
                unit_types.append(f"{path.relative_to(root)}: {m.group(0).strip()}")
    return misleading, stubs, unit_types


def decorative_hypotheses(root: Path) -> list[str]:
    """Hypotheses that constrain nothing: unused in statement and proof alike."""
    findings: list[str] = []
    for path in lean_sources(root):
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        for name, statement, body in theorem_blocks(text):
            if not body.strip() or HYPOTHESIS_CONSUMING.search(body):
                continue
            for hyp_name, hyp_type in hypothesis_binders(statement):
                pattern = re.compile(_word(hyp_name))
                if len(pattern.findall(statement)) > 1:
                    continue
                if pattern.search(body):
                    continue
                flat = re.sub(r"\s+", " ", hyp_type)[:70]
                findings.append(
                    f"{path.relative_to(root)}: {name} [{hyp_name} : {flat}]"
                )
    return findings


def trivially_true_hypotheses(root: Path) -> list[str]:
    """Hypotheses whose type is a closed truth carry no information."""
    findings: list[str] = []
    for path in lean_sources(root):
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        for name, statement, _ in theorem_blocks(text):
            for hyp_name, hyp_type in hypothesis_binders(statement):
                if trivially_true_prop(hyp_type):
                    flat = re.sub(r"\s+", " ", hyp_type)[:70]
                    findings.append(
                        f"{path.relative_to(root)}: {name} [{hyp_name} : {flat}]"
                    )
    return findings


def identity_proofs(root: Path) -> list[str]:
    """Proofs that hand a hypothesis back as the conclusion (P→P).

    Covers tactic-mode bodies (`:= by exact h`, `:= by intro h; exact h`) and
    term-mode bodies (`:= h`). A bare binder name is only accepted when it
    really is a binder of the same declaration, so term references to other
    declarations (`:= S.tr_one`) are not flagged.
    """
    findings: list[str] = []
    for path in lean_sources(root):
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        for name, statement, body in theorem_blocks(text):
            flat = re.sub(r"\s+", " ", body.replace("\n", " ; ")).strip(" ;")
            flat = re.sub(r"^by\s*;?\s*", "", flat)
            match = IDENTITY_PROOF.fullmatch(flat)
            if match and match.group(2) in set(
                (match.group(1) or "").split()
            ) | all_binder_names(statement):
                findings.append(f"{path.relative_to(root)}: {name}")
            elif TERM_IDENTITY.fullmatch(flat) and flat in all_binder_names(statement):
                findings.append(f"{path.relative_to(root)}: {name} (term-mode)")
    return findings


def axiom_named_declarations(root: Path) -> list[str]:
    """Flag declaration names and structure fields/assignments containing
    `axiom`. The corpus has zero Lean `axiom` primitives; such names
    misrepresent kernel-checked theorems or data fields as postulates.
    Comments and strings are masked, so prose mentions do not match."""
    findings: list[str] = []
    for path in lean_sources(root):
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        hits: set[str] = set()
        for match in DECL_NAME.finditer(text):
            if "axiom" in match.group(1):
                hits.add(match.group(1))
        for match in FIELD_LINE.finditer(text):
            if "axiom" in match.group(1):
                hits.add(match.group(1))
        for name in sorted(hits):
            findings.append(f"{path.relative_to(root)}: {name}")
    return findings


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


def trivial_proofs(root: Path) -> list[str]:
    """Reject trivial-only proof scripts, including multiline ones, lexically."""
    pattern = re.compile(r":=\s*by\s*trivial[ \t]*$", re.M)
    findings = []
    for path in lean_sources(root):
        text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            findings.append(f"{path.relative_to(root)}:{line}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-misleading", type=int, default=None)
    parser.add_argument("--max-axiom-names", type=int, default=None)
    parser.add_argument("--max-unit-stubs", type=int, default=None)
    parser.add_argument("--max-unit-types", type=int, default=None)
    parser.add_argument("--max-decorative-hypotheses", type=int, default=None)
    parser.add_argument("--max-trivial-hypotheses", type=int, default=None)
    parser.add_argument("--max-identity-proofs", type=int, default=None)
    parser.add_argument(
        "--alias-baseline",
        type=Path,
        default=None,
        help="Exact reviewed legacy *_Stmt exports; not a proof certificate",
    )
    parser.add_argument("--max-trivial-proofs", type=int, default=None)
    args = parser.parse_args()

    misleading, stubs, unit_types = audit(args.root)
    rc = 0
    rc |= report(
        "misleading trivially-stated theorems", misleading, args.max_misleading
    )
    rc |= report(
        "axiom-named declarations (zero real axioms exist)",
        axiom_named_declarations(args.root),
        args.max_axiom_names,
    )
    rc |= report("Nonempty-unit stubs", stubs, args.max_unit_stubs)
    rc |= report("Unit-typed volume definitions", unit_types, args.max_unit_types)
    rc |= report(
        "trivial-only proofs", trivial_proofs(args.root), args.max_trivial_proofs
    )
    rc |= report(
        "decorative hypotheses (unused in statement and proof)",
        decorative_hypotheses(args.root),
        args.max_decorative_hypotheses,
    )
    rc |= report(
        "trivially-true hypotheses",
        trivially_true_hypotheses(args.root),
        args.max_trivial_hypotheses,
    )
    rc |= report(
        "identity proofs (conclusion restates a hypothesis)",
        identity_proofs(args.root),
        args.max_identity_proofs,
    )
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
