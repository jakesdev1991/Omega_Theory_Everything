# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Numerical falsification for the real-arithmetic subset of Lean statements.

Semantics follow Lean/Mathlib, not textbook arithmetic: ``x / 0 = 0`` and
``Real.sqrt x = 0`` for ``x < 0``. Getting these wrong would let a candidate
pass here and fail in Lean (or, worse, the reverse — the adversarial audit
found a theorem whose content was masked by ``x / 0 = 0``).

A PASS is *evidence* (epistemic category NUMERICAL), never a proof. A FAIL
with a counterexample is a refutation up to floating-point tolerance and is
recorded with the witness so the assumption it breaks can be identified.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .extract import RealDef
from .schema import Statement

REAL_TYPES = {"ℝ"}
NAT_TYPES = {"ℕ"}
INT_TYPES = {"ℤ"}
TOL = 1e-9
SEED = 20260926  # pre-registered
MIN_SATISFYING = 30  # below this a PASS is only "inconclusive"
EQUATION_SAMPLE_FRACTION = (
    0.5  # share of samples steered onto an equation's solution set
)
SPECIAL_POINTS = (
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    0.6180339887498949,
    0.9,
    0.99,
    1e-3,
    3.0,
)


class Unsupported(Exception):
    pass


# --------------------------------------------------------------------------- tokens
_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?)"
    r"|(?P<id>[^\W\d][\w'₀-₉.]*)"
    r"|(?P<op>≠|≤|≥|↔|→|∧|∨|¬|√|\^|\*|/|\+|-|=|<|>|\(|\)|\||:))",
    re.UNICODE,
)


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise Unsupported(f"token at {text[pos : pos + 12]!r}")
        pos = m.end()
        kind = m.lastgroup or ""
        tokens.append((kind, m.group(kind)))
    return tokens


# --------------------------------------------------------------------------- AST
Node = tuple  # ('num', v) | ('var', name) | ('app', f, [args]) | ('un', op, x) | ('bin', op, a, b)


class Parser:
    """Pratt parser for the arithmetic/propositional fragment (see grammar in docs)."""

    BIN_PREC = {
        "↔": 20,
        "→": 25,
        "∨": 30,
        "∧": 35,
        "=": 50,
        "≠": 50,
        "<": 50,
        ">": 50,
        "≤": 50,
        "≥": 50,
        "+": 65,
        "-": 65,
        "*": 70,
        "/": 70,
        "^": 75,
    }
    RIGHT = {"→", "∧", "∨", "↔", "^"}

    def __init__(self, tokens: list[tuple[str, str]], functions: set[str]):
        self.toks = tokens
        self.i = 0
        self.functions = functions

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.expr(0)
        if self.peek() is not None:
            raise Unsupported(f"trailing token {self.peek()!r}")
        return node

    def expr(self, min_prec: int) -> Node:
        left = self.unary()
        while True:
            tok = self.peek()
            if tok is None or tok[0] != "op" or tok[1] not in self.BIN_PREC:
                return left
            op = tok[1]
            prec = self.BIN_PREC[op]
            if prec < min_prec:
                return left
            self.take()
            right = self.expr(prec if op in self.RIGHT else prec + 1)
            left = ("bin", op, left, right)

    def unary(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise Unsupported("unexpected end")
        if tok == ("op", "-"):
            self.take()
            return ("un", "-", self.expr(75))
        if tok == ("op", "¬"):
            self.take()
            return ("un", "¬", self.expr(40))
        if tok == ("op", "√"):
            self.take()
            return ("app", "Real.sqrt", [self.atom()])
        return self.application()

    def application(self) -> Node:
        head = self.atom()
        if head[0] == "var" and head[1] in self.functions:
            args = []
            while True:
                tok = self.peek()
                if tok is None or tok[0] == "op" and tok[1] != "(" and tok[1] != "√":
                    break
                if tok == ("op", "√"):
                    args.append(self.unary())
                else:
                    args.append(self.atom())
            return ("app", head[1], args)
        return head

    def atom(self) -> Node:
        tok = self.take()
        kind, value = tok
        if kind == "num":
            return ("num", float(value))
        if kind == "id":
            return ("var", value)
        if tok == ("op", "|"):  # absolute value bars
            node = self.expr(0)
            if self.take() != ("op", "|"):
                raise Unsupported("unbalanced |")
            return ("app", "abs", [node])
        if tok == ("op", "("):
            node = self.expr(0)
            nxt = self.peek()
            if nxt == ("op", ":"):  # type ascription (1 : ℝ)
                self.take()
                typ = self.take()
                if typ[1] not in REAL_TYPES | NAT_TYPES | INT_TYPES:
                    raise Unsupported(f"ascription to {typ[1]}")
            if self.take() != ("op", ")"):
                raise Unsupported("expected )")
            return node
        raise Unsupported(f"unexpected {tok!r}")


# --------------------------------------------------------------------------- evaluation
def lean_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def lean_sqrt(x: float) -> float:
    return math.sqrt(x) if x > 0 else 0.0


def lean_pow(a: float, b: float) -> float:
    if float(b).is_integer():
        n = int(b)
        if n >= 0:
            return a**n
        return lean_div(1.0, a ** (-n))
    if a < 0:
        raise Unsupported("real power of negative base")
    return math.pow(a, b)


BUILTINS: dict[str, Callable[..., float]] = {
    "Real.sqrt": lean_sqrt,
    "Real.exp": math.exp,
    "Real.log": lambda x: math.log(x) if x > 0 else (math.log(-x) if x < 0 else 0.0),
    "abs": abs,
    "max": max,
    "min": min,
}


class Evaluator:
    def __init__(self, defs: dict[str, RealDef], tol: float = TOL):
        self.defs = defs
        self.tol = tol
        self.functions = set(defs) | set(BUILTINS)
        self._cache: dict[str, Node] = {}

    def parse(self, text: str) -> Node:
        if text not in self._cache:
            self._cache[text] = Parser(tokenize(text), self.functions).parse()
        return self._cache[text]

    def eval(self, node: Node, env: dict[str, float], depth: int = 0) -> Any:
        if depth > 64:
            raise Unsupported("recursion depth")
        tag = node[0]
        if tag == "num":
            return node[1]
        if tag == "var":
            if node[1] in env:
                return env[node[1]]
            if node[1] in {"True", "False"}:
                return node[1] == "True"
            raise Unsupported(f"identifier {node[1]}")
        if tag == "app":
            _, fname, args = node
            vals = [self.eval(a, env, depth + 1) for a in args]
            if fname in BUILTINS:
                return BUILTINS[fname](*vals)
            d = self.defs[fname]
            if len(vals) != len(d.params):
                raise Unsupported(f"arity of {fname}")
            inner = dict(env)
            inner.update(zip(d.params, vals))
            return self.eval(self.parse(d.body), inner, depth + 1)
        if tag == "un":
            _, op, x = node
            v = self.eval(x, env, depth + 1)
            return -v if op == "-" else (not v)
        _, op, a, b = node
        if op == "→":
            return (not self.eval(a, env, depth + 1)) or bool(
                self.eval(b, env, depth + 1)
            )
        if op == "∧":
            return bool(self.eval(a, env, depth + 1)) and bool(
                self.eval(b, env, depth + 1)
            )
        if op == "∨":
            return bool(self.eval(a, env, depth + 1)) or bool(
                self.eval(b, env, depth + 1)
            )
        if op == "↔":
            return bool(self.eval(a, env, depth + 1)) == bool(
                self.eval(b, env, depth + 1)
            )
        x = self.eval(a, env, depth + 1)
        y = self.eval(b, env, depth + 1)
        if op == "+":
            return x + y
        if op == "-":
            return x - y
        if op == "*":
            return x * y
        if op == "/":
            return lean_div(x, y)
        if op == "^":
            return lean_pow(x, y)
        return self.compare(op, x, y)

    def compare(self, op: str, x: float, y: float) -> bool:
        if isinstance(x, bool) or isinstance(y, bool):
            if op == "=":
                return x == y
            if op == "≠":
                return x != y
            raise Unsupported("order on propositions")
        if op == "=":
            return abs(x - y) <= self.tol * (1 + abs(x) + abs(y))
        if op == "≠":
            return abs(x - y) > self.tol * (1 + abs(x) + abs(y))
        if op == "<":
            return x < y
        if op == ">":
            return x > y
        if op == "≤":
            return x <= y
        return x >= y

    def violation_margin(self, op: str, x: float, y: float) -> float:
        """Normalized amount by which a relation fails; 0.0 when it holds.

        Exact equality under a strict relation is a genuine failure (Lean's
        ``x / 0 = 0`` produces exact zeros), so it gets a nominal margin.
        """
        norm = 1 + abs(x) + abs(y)
        scale = self.tol * norm
        if op == "=":
            return max(0.0, (abs(x - y) - scale) / norm)
        if op == "≠":
            return 1.0 if x == y else 0.0
        if op == "<":
            return 1e-300 if x == y else max(0.0, (x - y - scale) / norm)
        if op == ">":
            return 1e-300 if x == y else max(0.0, (y - x - scale) / norm)
        if op == "≤":
            return max(0.0, (x - y - scale) / norm)
        if op == "≥":
            return max(0.0, (y - x - scale) / norm)
        raise Unsupported(op)


# --------------------------------------------------------------------------- verdicts
TOL_STRICT = 1e-12  # hypotheses must still hold here for a violation to count
MARGIN_CONFIRM = 1e-6  # and the conclusion must fail by at least this much


@dataclass
class NumericVerdict:
    status: str  # pass | fail | inconclusive | not_applicable
    reason: str
    samples: int = 0
    satisfying: int = 0
    violations: int = 0
    tolerance_ambiguous: int = (
        0  # violated at TOL but not confirmed at TOL_STRICT/MARGIN
    )
    skipped_nonfinite: int = 0
    counterexample: dict[str, float] | None = None
    counterexample_margin: float = 0.0
    relevance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "samples": self.samples,
            "satisfying": self.satisfying,
            "violations": self.violations,
            "tolerance_ambiguous": self.tolerance_ambiguous,
            "skipped_nonfinite": self.skipped_nonfinite,
            "counterexample": self.counterexample,
            "counterexample_margin": self.counterexample_margin,
            "relevance": self.relevance,
        }


def _sample(kind: str, rng: random.Random) -> float:
    r = rng.random()
    if kind == "nat":
        return float(rng.randint(0, 6))
    if kind == "int":
        return float(rng.randint(-6, 6))
    if r < 0.25:
        return rng.choice(SPECIAL_POINTS)
    if r < 0.6:
        return rng.uniform(-1.0, 1.0)
    return rng.uniform(-3.0, 3.0)


def _free_vars(node: Node, out: set[str] | None = None) -> set[str]:
    acc: set[str] = set() if out is None else out
    tag = node[0]
    if tag == "var":
        acc.add(node[1])
    elif tag == "app":
        for a in node[2]:
            _free_vars(a, acc)
    elif tag == "un":
        _free_vars(node[2], acc)
    elif tag == "bin":
        _free_vars(node[2], acc)
        _free_vars(node[3], acc)
    return acc


def _equations(node: Node, out: list[Node] | None = None) -> list[Node]:
    """All ``a = b`` subterms; their solution sets have measure zero, so plain
    sampling never lands on them (``h : f Φ = 1`` or an ``↔`` with an equation)."""
    acc: list[Node] = [] if out is None else out
    tag = node[0]
    if tag == "bin":
        if node[1] == "=":
            acc.append(node)
        _equations(node[2], acc)
        _equations(node[3], acc)
    elif tag == "un":
        _equations(node[2], acc)
    elif tag == "app":
        for a in node[2]:
            _equations(a, acc)
    return acc


def _solve_for(
    ev: Evaluator, eq: Node, var: str, env: dict[str, float], rng: random.Random
) -> float | None:
    """Root of lhs - rhs in ``var`` by sign-change scan + bisection; None if none."""
    lhs, rhs = eq[2], eq[3]

    def f(x: float) -> float:
        inner = dict(env)
        inner[var] = x
        a = ev.eval(lhs, inner)
        b = ev.eval(rhs, inner)
        if isinstance(a, bool) or isinstance(b, bool):
            raise Unsupported("equation between propositions")
        return a - b

    lo_bound, hi_bound = (-3.0, 3.0) if rng.random() < 0.7 else (-10.0, 10.0)
    grid = 61
    xs = [lo_bound + (hi_bound - lo_bound) * k / (grid - 1) for k in range(grid)]
    try:
        vals = [f(x) for x in xs]
    except (ArithmeticError, OverflowError, ValueError):
        return None
    brackets = [
        k
        for k in range(grid - 1)
        if math.isfinite(vals[k])
        and math.isfinite(vals[k + 1])
        and vals[k] * vals[k + 1] <= 0
    ]
    if not brackets:
        return None
    k = rng.choice(brackets)
    lo, hi = xs[k], xs[k + 1]
    flo = vals[k]
    if flo == 0:
        return lo
    if vals[k + 1] == 0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        try:
            fm = f(mid)
        except (ArithmeticError, OverflowError, ValueError):
            return None
        if fm == 0:
            return mid
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _margin(ev: Evaluator, concl: Node, env: dict[str, float]) -> float:
    """0.0 if the conclusion holds, else how badly it fails (normalized)."""
    if concl[0] == "bin" and concl[1] in {"=", "≠", "<", ">", "≤", "≥"}:
        x = ev.eval(concl[2], env)
        y = ev.eval(concl[3], env)
        if isinstance(x, bool) or isinstance(y, bool):
            return 0.0 if ev.compare(concl[1], x, y) else 1.0
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ArithmeticError
        return ev.violation_margin(concl[1], x, y)
    return 0.0 if bool(ev.eval(concl, env)) else 1.0


def _confirmed(
    strict: Evaluator, hyps: list[tuple[str, Node]], concl: Node, env: dict[str, float]
) -> float:
    """Re-check at strict tolerance: hypotheses must hold, conclusion must fail
    by MARGIN_CONFIRM. Returns the strict margin, 0.0 if not confirmed."""
    try:
        if not all(bool(strict.eval(node, env)) for _, node in hyps):
            return 0.0
        m = _margin(strict, concl, env)
    except (ArithmeticError, OverflowError, ValueError):
        return 0.0
    return m if m >= MARGIN_CONFIRM or m == 1e-300 else 0.0


def check(
    statement: Statement,
    defs: dict[str, RealDef],
    samples: int = 2000,
    seed: int = SEED,
) -> NumericVerdict:
    """Sample the hypothesis region, look for a confirmed violation of the conclusion.

    Also estimates, per hypothesis, whether it is load-bearing: if the
    conclusion never fails on samples that violate *only* that hypothesis,
    the hypothesis is numerically decorative (rung-1 evidence, not proof).
    """
    ev = Evaluator(defs)
    strict = Evaluator(defs, tol=TOL_STRICT)
    variables: dict[str, str] = {}
    for b in statement.variables():
        t = b.type.strip()
        if t in REAL_TYPES:
            kind = "real"
        elif t in NAT_TYPES:
            kind = "nat"
        elif t in INT_TYPES:
            kind = "int"
        else:
            return NumericVerdict("not_applicable", f"binder type {t!r} unsupported")
        for n in b.names:
            variables[n] = kind
    try:
        hyps = [
            (b.names[0] if b.names else f"h{k}", ev.parse(b.type))
            for k, b in enumerate(statement.hypotheses())
        ]
        concl = ev.parse(statement.conclusion)
    except Unsupported as exc:
        return NumericVerdict("not_applicable", f"parse: {exc}")

    rng = random.Random(seed)
    verdict = NumericVerdict("pass", "")
    only_violating_seen: dict[str, int] = {h: 0 for h, _ in hyps}
    only_violating_broken: dict[str, int] = {h: 0 for h, _ in hyps}
    equations: list[tuple[Node, list[str]]] = []
    for node in [n for _, n in hyps] + [concl]:
        for eq in _equations(node):
            names = sorted(v for v in _free_vars(eq) if variables.get(v) == "real")
            if names:
                equations.append((eq, names))
    try:
        for _ in range(samples):
            env = {n: _sample(kind, rng) for n, kind in variables.items()}
            if equations and rng.random() < EQUATION_SAMPLE_FRACTION:
                eq, names = rng.choice(equations)
                var = rng.choice(names)
                root = _solve_for(ev, eq, var, env, rng)
                if root is not None:
                    env[var] = root
            verdict.samples += 1
            try:
                truth = [bool(ev.eval(node, env)) for _, node in hyps]
                if all(truth):
                    verdict.satisfying += 1
                    if _margin(ev, concl, env) > 0.0:
                        m = _confirmed(strict, hyps, concl, env)
                        if m > 0.0:
                            verdict.violations += 1
                            if m > verdict.counterexample_margin:
                                verdict.counterexample = dict(env)
                                verdict.counterexample_margin = m
                        else:
                            verdict.tolerance_ambiguous += 1
                elif truth.count(False) == 1:
                    name = hyps[truth.index(False)][0]
                    only_violating_seen[name] += 1
                    if (
                        _margin(ev, concl, env) > 0.0
                        and _confirmed(
                            strict, [h for h in hyps if h[0] != name], concl, env
                        )
                        > 0.0
                    ):
                        only_violating_broken[name] += 1
            except (ArithmeticError, OverflowError, ValueError):
                verdict.skipped_nonfinite += 1
    except Unsupported as exc:
        return NumericVerdict("not_applicable", f"eval: {exc}")

    for name, _ in hyps:
        seen = only_violating_seen[name]
        if seen < 20:
            verdict.relevance[name] = "undetermined"
        elif only_violating_broken[name] == 0:
            verdict.relevance[name] = "not_load_bearing_numerically"
        else:
            verdict.relevance[name] = "load_bearing"

    if verdict.satisfying == 0:
        verdict.status = "fail"
        verdict.reason = "hypotheses_unsatisfied_in_sampled_domain"
    elif verdict.violations:
        verdict.status = "fail"
        verdict.reason = "counterexample"
    elif verdict.satisfying < MIN_SATISFYING or verdict.tolerance_ambiguous:
        verdict.status = "inconclusive"
        verdict.reason = (
            f"only_{verdict.satisfying}_satisfying_samples"
            if verdict.satisfying < MIN_SATISFYING
            else f"{verdict.tolerance_ambiguous}_tolerance_ambiguous_violations"
        )
    else:
        verdict.reason = f"consistent_on_{verdict.satisfying}_satisfying_samples"
    return verdict
