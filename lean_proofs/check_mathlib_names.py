#!/usr/bin/env python3
"""Check qualified Lean names used in this repository against the pinned sources.

There is no local Lean toolchain in this environment, so a mistake like
`Subalgebra.mem_top` (the real lemma is `StarSubalgebra.mem_top`) can only
surface in a CI build, and then only one dependency layer at a time: the build
stops at the first module that fails. This script makes most such mistakes
findable offline. It indexes every declaration in a Mathlib checkout plus the
Lean core sources, and reports dotted names used in `lean_proofs/*.lean` that
appear in neither the local sources nor those two trees.

It is a *lexical* check with known blind spots: it does not know about `open`
scoping, notation, or type classes, and it cannot see a name that exists but
requires an import the file does not have. It is a net for typos and renamed
lemmas, not a replacement for `lake build`.

Usage:
    python3 check_mathlib_names.py \
        --mathlib /path/to/mathlib4-4.32.0 \
        --lean /path/to/lean4-4.32.0 \
        [--only-new --baseline main]

Exit status is 1 when suspicious names are found and 0 otherwise, so it can be
wired into CI once a baseline is clean.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

DECL_KEYWORDS = (
    "theorem",
    "lemma",
    "def",
    "abbrev",
    "structure",
    "class",
    "inductive",
    "instance",
    "axiom",
    "opaque",
)

DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?"
    r"(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+|partial\s+|scoped\s+|local\s+|open\s+)*"
    r"(?:" + "|".join(DECL_KEYWORDS) + r")\s+([A-Za-z_][\w'.]*)"
)
DOTTED_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|structure|class|instance|inductive)\s+"
    r"([A-Z][\w']*(?:\.[A-Za-z_][\w']*)+)"
)
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][\w'.]*)\s*$")
NAME_USE_RE = re.compile(r"\b([A-Z][\w'\u2080-\u2089]*(?:\.[A-Za-z_][\w'\u2080-\u2089]*)+)")
BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
# a bound variable, e.g. `(C : Compactification)` or `{A B : Operator}` or
# `variable (ρ : StateSpace)`; field projections of these (`C.Gbulk`) are not
# Mathlib lookups at all.
BINDER_RE = re.compile(r"[(\[{]\s*([A-Za-z_][\w'\u2080-\u2089]*)\s*(?::|:=|\))")
VARIABLE_RE = re.compile(r"^\s*variable\s*[({[]([^)}\]]*)")


def strip_comments(text: str) -> str:
    return BLOCK_COMMENT_RE.sub("", text)


def declarations(path: pathlib.Path) -> tuple[set[str], set[str]]:
    qualified: set[str] = set()
    bare: set[str] = set()
    for line in strip_comments(path.read_text(encoding="utf-8", errors="replace")).splitlines():
        line = line.split("--", 1)[0]
        if m := NAMESPACE_RE.match(line):
            qualified.add(m.group(1))
            continue
        if m := DECL_RE.match(line):
            name = m.group(1)
            bare.add(name.split(".")[-1])
            qualified.add(name)
        elif m := DOTTED_DECL_RE.match(line):
            qualified.add(m.group(1))
            bare.add(m.group(1).split(".")[-1])
    return qualified, bare


def _decl_names(line: str) -> list[str]:
    if m := DECL_RE.match(line):
        return [m.group(1)]
    if m := DOTTED_DECL_RE.match(line):
        return [m.group(1)]
    return []


def index_tree(root: pathlib.Path) -> tuple[set[str], set[str]]:
    qualified: set[str] = set()
    bare: set[str] = set()
    for path in root.rglob("*.lean"):
        # Namespace blocks are tracked per file so `theorem mem_top` inside
        # `namespace StarSubalgebra` is registered as `StarSubalgebra.mem_top`.
        stack: list[str] = []
        for raw in strip_comments(path.read_text(encoding="utf-8", errors="replace")).splitlines():
            line = raw.split("--", 1)[0]
            if m := NAMESPACE_RE.match(line):
                stack.append(m.group(1))
                qualified.add(".".join(stack))
                continue
            if re.match(r"^\s*end\s+[A-Za-z_]", line):
                if stack:
                    stack.pop()
                continue
            for name in _decl_names(line):
                bare.add(name.split(".")[-1])
                full = ".".join(stack + [name]) if stack else name
                qualified.add(full)
                parts = full.split(".")
                qualified.update(".".join(parts[i:]) for i in range(len(parts)))
    qualified.discard("")
    return qualified, bare


def bound_variables(text: str) -> set[str]:
    """Identifiers introduced as binders or by `variable`, per file."""
    names: set[str] = set()
    for line in strip_comments(text).splitlines():
        line = line.split("--", 1)[0]
        if m := VARIABLE_RE.match(line):
            for part in m.group(1).split(")"):
                for token in re.split(r"[\s,{}(\[\]}:]+", part):
                    if token and token[0].isalpha():
                        names.add(token)
        for m in BINDER_RE.finditer(line):
            names.add(m.group(1))
    return names


def added_lines(home: pathlib.Path, revision: str) -> set[str]:
    out = subprocess.run(
        ["git", "diff", revision, "--", "lean_proofs/*.lean"],
        cwd=home,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        line[1:].strip()
        for line in out.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathlib", required=True, help="path to a mathlib4 checkout")
    parser.add_argument("--lean", default="", help="path to a lean4 (core) checkout")
    parser.add_argument("--repo", default="lean_proofs")
    parser.add_argument("--baseline", default="", help="git revision to diff against")
    parser.add_argument("--only-new", action="store_true", help="only check lines added since --baseline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    home = repo.parent

    known: set[str] = set()
    bare: set[str] = set()
    for tree in [pathlib.Path(args.mathlib)] + ([pathlib.Path(args.lean)] if args.lean else []):
        qualified, names = index_tree(tree)
        known |= qualified
        bare |= names
        print(f"indexed {tree.name}: {len(qualified)} names")
    repo_qualified, repo_bare = index_tree(repo)
    known |= repo_qualified
    bare |= repo_bare
    print(f"indexed repository: {len(repo_qualified)} names")

    new_lines = added_lines(home, args.baseline) if (args.only_new and args.baseline) else set()

    suspicious: dict[str, set[str]] = {}
    for path in sorted(repo.glob("*.lean")):
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        locals_ = bound_variables(text)
        for raw in text.splitlines():
            code = raw.split("--", 1)[0]
            if not code.strip():
                continue
            if new_lines and code.strip() not in new_lines:
                continue
            for name in NAME_USE_RE.findall(code):
                if name in known:
                    continue
                head = name.split(".")[0]
                if head in locals_ or len(head) <= 2:
                    continue  # field projection of a bound variable
                if name in repo_qualified or name in repo_bare:
                    continue
                suspicious.setdefault(path.name, set()).add(name)

    total = 0
    for filename, names in sorted(suspicious.items()):
        print(f"\n{filename}:")
        for name in sorted(names):
            marker = "NEW " if any(name in line for line in new_lines) else "    "
            print(f"  {marker}? {name}")
            total += 1
    print(f"\n{total} suspicious name(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
