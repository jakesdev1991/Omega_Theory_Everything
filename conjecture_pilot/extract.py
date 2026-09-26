# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Lexical extraction of theorem signatures and simple real-valued defs.

This is deliberately *not* a Lean parser. It recovers ``theorem name binders :
conclusion`` for the plain declaration style used in lean_proofs/, and reports
anything it cannot parse with a reason, so that coverage is a measured number
rather than an assumption (pre-registration §6, "unparsed" column).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .schema import Binder, Statement

_LEAN_PROOFS = Path(__file__).resolve().parents[1] / "lean_proofs"
if str(_LEAN_PROOFS) not in sys.path:
    sys.path.insert(0, str(_LEAN_PROOFS))

from lean_source import mask_comments_and_strings  # noqa: E402

OPEN = {"(": ")", "{": "}", "[": "]", "⦃": "⦄"}
CLOSE = {v: k for k, v in OPEN.items()}

_DECL = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\][ \t]*)?(?:private[ \t]+|protected[ \t]+)?"
    r"(theorem|lemma)[ \t]+([^\s(){}\[\]⦃⦄:]+)",
    re.M,
)
_DEF = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\][ \t]*)?(?:private[ \t]+|protected[ \t]+)?"
    r"(?:noncomputable[ \t]+)?def[ \t]+([^\s(){}\[\]⦃⦄:]+)",
    re.M,
)
_NAMESPACE = re.compile(r"^[ \t]*(namespace|end|section)\b[ \t]*([^\s]*)", re.M)

# Symbols whose presence in a binder type marks it as a proposition. ``→`` is
# excluded on purpose: ``(f : ℕ → ℝ)`` is a function, not a hypothesis.
_PROP_SYMBOLS = (
    "=",
    "≠",
    "≤",
    "<",
    "≥",
    ">",
    "∀",
    "∃",
    "¬",
    "↔",
    "∧",
    "∨",
    "∈",
    "∉",
    "⊆",
    "∣",
)
_PROP_HEADS = (
    "Monotone",
    "StrictMono",
    "Antitone",
    "StrictAnti",
    "Continuous",
    "Differentiable",
    "HasDerivAt",
    "Function.Injective",
    "Function.Surjective",
    "Nonempty",
    "IsIsolated",
    "True",
    "False",
)


@dataclass(frozen=True)
class Unparsed:
    module: str
    name: str
    line: int
    reason: str


@dataclass(frozen=True)
class RealDef:
    """``def f (x y : ℝ) : ℝ := body`` — inlined by the numeric evaluator."""

    name: str
    params: tuple[str, ...]
    body: str


def is_hypothesis(names: tuple[str, ...], type_text: str, bracket: str) -> bool:
    if bracket == "[":
        return False
    stripped = type_text.strip()
    if any(sym in stripped for sym in _PROP_SYMBOLS):
        return True
    if any(stripped.startswith(head) for head in _PROP_HEADS):
        return True
    return any(
        n == "h" or (n.startswith("h") and len(n) > 1 and not n[1].isalpha())
        for n in names
    ) or any(n.startswith("h_") or n.startswith("hyp") for n in names)


def _match_bracket(text: str, start: int) -> int:
    """Index just past the bracket that closes ``text[start]``; -1 on failure."""
    stack: list[str] = []
    i = start
    while i < len(text):
        ch = text[i]
        if ch in OPEN:
            stack.append(ch)
        elif ch in CLOSE:
            if not stack or stack[-1] != CLOSE[ch]:
                return -1
            stack.pop()
            if not stack:
                return i + 1
        i += 1
    return -1


def _split_top_level(text: str, token: str) -> tuple[str, str] | None:
    """Split at the first occurrence of ``token`` at bracket depth 0."""
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in OPEN:
            depth += 1
        elif ch in CLOSE:
            depth -= 1
        elif depth == 0 and text.startswith(token, i):
            nxt = text[i + len(token) : i + len(token) + 1]
            if token == ":" and nxt == "=":
                i += 2
                continue
            return text[:i], text[i + len(token) :]
        i += 1
    return None


def _parse_binder(group: str) -> Binder:
    bracket = group[0]
    inner = group[1:-1].strip()
    if bracket == "[":
        split = _split_top_level(inner, ":")
        if split is None:
            return Binder(names=(), type=inner, bracket=bracket, is_hypothesis=False)
        names, type_text = split
        return Binder(tuple(names.split()), type_text.strip(), bracket, False)
    split = _split_top_level(inner, ":")
    if split is None:
        names_t = tuple(inner.split())
        return Binder(names_t, "", bracket, False)
    names, type_text = split
    names_t = tuple(names.split())
    type_text = type_text.strip()
    return Binder(
        names_t, type_text, bracket, is_hypothesis(names_t, type_text, bracket)
    )


def _namespaces_at(text: str) -> list[tuple[int, tuple[str, ...]]]:
    """Piecewise-constant map offset -> open namespaces (lexical)."""
    stack: list[str] = []
    events: list[tuple[int, tuple[str, ...]]] = [(0, ())]
    for m in _NAMESPACE.finditer(text):
        kw, name = m.group(1), m.group(2)
        if kw == "namespace" and name:
            stack.append(name)
        elif kw == "end" and stack and name == stack[-1]:
            stack.pop()
        events.append((m.end(), tuple(stack)))
    return events


def _namespace_for(
    events: list[tuple[int, tuple[str, ...]]], offset: int
) -> tuple[str, ...]:
    current: tuple[str, ...] = ()
    for at, ns in events:
        if at <= offset:
            current = ns
        else:
            break
    return current


def extract_statements(
    path: Path, module: str | None = None
) -> tuple[list[Statement], list[Unparsed]]:
    raw = path.read_text(encoding="utf-8")
    text = mask_comments_and_strings(raw)
    module = module or path.stem
    events = _namespaces_at(text)
    statements: list[Statement] = []
    unparsed: list[Unparsed] = []
    for m in _DECL.finditer(text):
        kind, name = m.group(1), m.group(2)
        line = text.count("\n", 0, m.start()) + 1
        i = m.end()
        binders: list[Binder] = []
        failure = ""
        while True:
            while i < len(text) and text[i] in " \t\r\n":
                i += 1
            if i < len(text) and text[i] in OPEN:
                end = _match_bracket(text, i)
                if end == -1:
                    failure = "unbalanced_binder"
                    break
                binders.append(_parse_binder(text[i:end]))
                i = end
                continue
            break
        if failure:
            unparsed.append(Unparsed(module, name, line, failure))
            continue
        if i >= len(text) or text[i] != ":" or text.startswith(":=", i):
            unparsed.append(Unparsed(module, name, line, "no_type_colon"))
            continue
        rest = text[i + 1 :]
        split = _split_top_level(rest, ":=")
        if split is None:
            unparsed.append(Unparsed(module, name, line, "no_assignment"))
            continue
        conclusion = " ".join(split[0].split())
        if not conclusion:
            unparsed.append(Unparsed(module, name, line, "empty_conclusion"))
            continue
        if "\n" in split[0] and re.search(r"^\s*\|", split[0], re.M):
            unparsed.append(Unparsed(module, name, line, "pattern_matching"))
            continue
        statements.append(
            Statement(
                module=module,
                name=name,
                kind=kind,
                line=line,
                binders=tuple(binders),
                conclusion=conclusion,
                namespaces=_namespace_for(events, m.start()),
            )
        )
    return statements, unparsed


def extract_real_defs(path: Path) -> dict[str, RealDef]:
    """Defs of the shape ``def f (a b : ℝ) ... : ℝ := <expression>``.

    The body runs to the next blank line or declaration. Anything else is
    simply not inlined; the evaluator then reports the identifier unsupported.
    """
    text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
    defs: dict[str, RealDef] = {}
    for m in _DEF.finditer(text):
        name = m.group(1)
        i = m.end()
        params: list[str] = []
        ok = True
        while True:
            while i < len(text) and text[i] in " \t":
                i += 1
            if i < len(text) and text[i] in OPEN:
                end = _match_bracket(text, i)
                if end == -1:
                    ok = False
                    break
                b = _parse_binder(text[i:end])
                if b.type.strip() != "ℝ" or b.bracket != "(":
                    ok = False
                    break
                params.extend(b.names)
                i = end
                continue
            break
        if not ok or i >= len(text) or text[i] != ":":
            continue
        rest = text[i + 1 :]
        split = _split_top_level(rest, ":=")
        if split is None or split[0].strip() != "ℝ":
            continue
        body_text = split[1]
        body_lines: list[str] = []
        for line in body_text.split("\n"):
            if body_lines and (not line.strip() or re.match(r"^\S", line)):
                break
            body_lines.append(line)
        body = " ".join(" ".join(body_lines).split())
        if body:
            defs[name] = RealDef(name, tuple(params), body)
    return defs


def module_path(module: str, root: Path = _LEAN_PROOFS) -> Path:
    return root / f"{module}.lean"
