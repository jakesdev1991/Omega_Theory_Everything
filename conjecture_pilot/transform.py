# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""The three pilot transformations: weakening, converse, generalization.

They are text-level and intentionally naive. The pilot measures what fraction
of their output survives the gates; it does not try to be clever, because a
clever generator would be a learning component and the pilot has none.
"""

from __future__ import annotations

import re
from typing import Iterator

from .schema import Binder, Candidate, Statement

TRANSFORMATIONS = ("weaken", "converse", "generalize_one", "generalize_all")

# ASCII numerals not glued to identifiers (h0, Φ₁ are excluded by construction).
_LITERAL = re.compile(r"(?<![\w.'₀-₉])(\d+(?:\.\d+)?)(?![\w.'₀-₉])")
_EXPONENT = re.compile(r"\^\s*$")


def _uses(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w'₀-₉.]){re.escape(name)}(?![\w'₀-₉])", text) is not None


def _fresh(base: str, taken: set[str]) -> str:
    name = base
    while name in taken:
        name += "'"
    return name


def weaken(s: Statement) -> Iterator[tuple[dict[str, object], Statement]]:
    """Drop one hypothesis. Survivors say the hypothesis was decorative."""
    for k, b in enumerate(s.binders):
        if not b.is_hypothesis or b.bracket != "(":
            continue
        later = " ".join(x.type for x in s.binders[k + 1 :]) + " " + s.conclusion
        if any(_uses(n, later) for n in b.names):
            continue
        binders = s.binders[:k] + s.binders[k + 1 :]
        yield (
            {"dropped": list(b.names), "dropped_type": b.type},
            Statement(
                s.module,
                f"{s.name}__weak_{'_'.join(b.names)}",
                s.kind,
                s.line,
                binders,
                s.conclusion,
                s.namespaces,
            ),
        )


def converse(s: Statement) -> Iterator[tuple[dict[str, object], Statement]]:
    """Swap one hypothesis with the conclusion."""
    for k, b in enumerate(s.binders):
        if not b.is_hypothesis or b.bracket != "(" or len(b.names) != 1:
            continue
        later = " ".join(x.type for x in s.binders[k + 1 :]) + " " + s.conclusion
        if _uses(b.names[0], later):
            continue
        taken = {n for x in s.binders for n in x.names}
        hname = _fresh("h_conv", taken)
        new_h = Binder((hname,), s.conclusion, "(", True)
        binders = s.binders[:k] + s.binders[k + 1 :] + (new_h,)
        yield (
            {"swapped": b.names[0], "new_conclusion": b.type},
            Statement(
                s.module,
                f"{s.name}__conv_{b.names[0]}",
                s.kind,
                s.line,
                binders,
                b.type,
                s.namespaces,
            ),
        )


def _literals(text: str) -> list[tuple[int, int, str]]:
    out = []
    for m in _LITERAL.finditer(text):
        if _EXPONENT.search(text[: m.start()]):
            continue  # exponents of ^ are naturals; generalizing them changes the operation
        out.append((m.start(), m.end(), m.group(1)))
    return out


def generalize_one(s: Statement) -> Iterator[tuple[dict[str, object], Statement]]:
    """Replace a single literal occurrence in the conclusion by a fresh real."""
    taken = s.identifiers()
    for start, end, lit in _literals(s.conclusion):
        var = _fresh("c", taken)
        concl = s.conclusion[:start] + var + s.conclusion[end:]
        binders = (Binder((var,), "ℝ", "(", False),) + s.binders
        yield (
            {"literal": lit, "occurrence_offset": start, "variable": var},
            Statement(
                s.module,
                f"{s.name}__gen1_{lit.replace('.', '_')}_{start}",
                s.kind,
                s.line,
                binders,
                concl,
                s.namespaces,
            ),
        )


def generalize_all(s: Statement) -> Iterator[tuple[dict[str, object], Statement]]:
    """Replace every occurrence of one literal value (hypotheses and conclusion)."""
    taken = s.identifiers()
    values = []
    for _, _, lit in _literals(s.conclusion):
        if lit not in values:
            values.append(lit)
    for lit in values:
        var = _fresh("c", taken)

        def sub(text: str) -> str:
            return "".join(
                _replace_literal(text, lit, var),
            )

        binders = [Binder((var,), "ℝ", "(", False)]
        for b in s.binders:
            binders.append(
                Binder(
                    b.names,
                    sub(b.type) if b.is_hypothesis else b.type,
                    b.bracket,
                    b.is_hypothesis,
                )
            )
        concl = sub(s.conclusion)
        yield (
            {"literal": lit, "variable": var, "scope": "all"},
            Statement(
                s.module,
                f"{s.name}__genall_{lit.replace('.', '_')}",
                s.kind,
                s.line,
                tuple(binders),
                concl,
                s.namespaces,
            ),
        )


def _replace_literal(text: str, lit: str, var: str) -> Iterator[str]:
    last = 0
    for start, end, found in _literals(text):
        if found != lit:
            continue
        yield text[last:start]
        yield var
        last = end
    yield text[last:]


def generate(
    s: Statement, family: str, transformations: tuple[str, ...] = TRANSFORMATIONS
) -> list[Candidate]:
    funcs = {
        "weaken": weaken,
        "converse": converse,
        "generalize_one": generalize_one,
        "generalize_all": generalize_all,
    }
    out: list[Candidate] = []
    for t in transformations:
        for params, statement in funcs[t](s):
            out.append(Candidate.make(family, t, s, dict(params), statement))
    return out
