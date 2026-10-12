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

Names the indexer cannot see at all are the failure mode that matters: every
miss turns into a false "suspicious" report and, once the count is a CI ratchet,
into pressure to raise the number. So the indexer registers, beyond plain
declaration lines: declarations whose attribute block spans preceding lines,
constructors of `inductive`/`structure`/`class` blocks with the names Lean
generates for them (`X.ctor.injEq`, `X.noConfusion`, `X.rec`, ...), structure
fields where they are declared, and namespaces. Scanning the repository masks
string literals and comments, and skips `import`/`open`/`export`/`set_option`
lines so module paths are not read as identifiers.

A non-zero count is a floor, not a certificate: each line is a name the indexer
could not find, and some of those will be real. `--max N` fails when the count
exceeds N, so the number can ratchet down as the remaining false positives are
classified and the indexer improves.

Usage:
    python3 check_mathlib_names.py \\
        --mathlib /path/to/mathlib4 \\
        --lean /path/to/lean4/src \\
        [--repo lean_proofs] [--max N] [--only-new --baseline main]

Exit status: 1 when the count of suspicious names exceeds `--max` (0 when the
flag is absent), 0 otherwise. A source tree that yields no declarations at all
(missing or mislocated checkout, wrong `--repo`) exits 2 instead: indexing
nothing would otherwise report a clean run, which is the one failure mode that
silently disarms the check.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lean_source import mask_comments_and_strings  # noqa: E402

# Lean identifiers may carry primes and dots; `\w` covers neither, so the
# pattern is spelled out and reused everywhere a name is parsed.
IDENT = r"[A-Za-z_][\w']*(?:\.[\w']+)*"

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

# Every modifier keyword the pinned sources put in front of a declaration
# keyword. `nonrec def` (Complex/Exponential.lean) and `public theorem` are both
# in use; missing one drops every declaration it modifies from the index.
MODIFIER = r"(?:(?:private|protected|noncomputable|unsafe|partial|nonrec|scoped|local|public|expose)\s+)*"

DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    + MODIFIER
    + r"(?:"
    + "|".join(DECL_KEYWORDS)
    + r")\s+("
    + IDENT
    + r")"
)
DOTTED_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    + MODIFIER
    + r"(?:theorem|lemma|def|abbrev|structure|class|instance|inductive)\s+"
    r"([A-Z][\w']*(?:\.[A-Za-z_][\w']*)+)"
)
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][\w'.]*)\s*$")
SECTION_RE = re.compile(r"^\s*section(?:\s+([A-Za-z_][\w'.]*))?\s*$")
END_RE = re.compile(r"^\s*end(?:\s+([A-Za-z_][\w'.]*))?\s*$")
NAME_USE_RE = re.compile(r"\b([A-Z][\w'\u2080-\u2089]*(?:\.[\w'\u2080-\u2089]+)+)")
# Lines that name modules, options or scoping rather than declarations: their
# dotted tokens are not identifier uses. `namespace` is handled separately.
NON_CODE_LINE_RE = re.compile(
    r"^\s*(?:import|open|export|set_option|attribute|deriving|variable|universe|section|end)\b"
)
# `inductive`/`structure`/`class` open a block whose constructors and fields are
# part of the namespace even though they carry no keyword of their own.
BLOCK_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:@\[[^\]]*\]\s*)*"
    + MODIFIER
    + r"(?P<kind>structure|class|inductive)\s+(?P<name>"
    + IDENT
    + r")"
)
CTOR_RE = re.compile(r"^[ \t]*\|[ \t]*([A-Za-z_][\w']*)")
FIELD_RE = re.compile(r"^[ \t]*(?:private\s+|protected\s+)?([a-zA-Z_][\w']*)\s*:[^=]")
DERIVING_RE = re.compile(r"^[ \t]*deriving\b")
ATTR_OPEN_RE = re.compile(r"^[ \t]*@\[")
ATTR_CONT_RE = re.compile(r"^[ \t]*[-\]]")
# a bound variable, e.g. `(C : Compactification)` or `{A B : Operator}` or
# `variable (ρ : StateSpace)`; field projections of these (`C.Gbulk`) are not
# Mathlib lookups at all.
BINDER_RE = re.compile(r"[({\[]\s*([A-Za-z_][\w'\u2080-\u2089]*)\s*(?::|:=|\))")
VARIABLE_RE = re.compile(r"^\s*variable\s*[({[]([^)}\]]*)")
# Names Lean generates for every inductive or structure, which never appear in
# the sources as declaration lines.
GENERATED_BLOCK_NAMES = (
    "rec",
    "recOn",
    "casesOn",
    "below",
    "brecOn",
    "noConfusion",
    "noConfusionType",
    "sizeOf_spec",
)


def strip_comments(text: str) -> str:
    """Blank comments and strings, preserving line structure. See lean_source."""
    return mask_comments_and_strings(text)


def _register(names: set[str], bare: set[str], name: str) -> None:
    """Register a name, its bare form, and every dotted suffix of it.

    Suffixes matter because a declaration inside `namespace Lean.Data.AC` is
    reachable as `Lean.Data.AC.Context`, `Data.AC.Context` and `AC.Context`,
    and the repository uses the shortest form that its `open` lines allow.
    """
    names.add(name)
    bare.add(name.split(".")[-1])
    parts = name.split(".")
    for i in range(1, len(parts)):
        names.add(".".join(parts[i:]))


def _register_block(
    names: set[str], bare: set[str], container: str, constructors: list[str]
) -> None:
    for suffix in GENERATED_BLOCK_NAMES:
        _register(names, bare, f"{container}.{suffix}")
    for ctor in constructors:
        _register(names, bare, f"{container}.{ctor}")
        _register(names, bare, f"{container}.{ctor}.injEq")


def starts_attribute_block(line: str) -> bool:
    """True when `line` opens an attribute block that carries no declaration itself.

    `@[simp] theorem foo` is handled by DECL_RE, so it is not a pending block;
    `@[simp]` alone is, and the declaration follows on a later line. Getting this
    wrong silently drops every `@[...] theorem ...` line, which Mathlib uses.
    """
    if not ATTR_OPEN_RE.match(line):
        return False
    return not (
        DECL_RE.match(line) or DOTTED_DECL_RE.match(line) or BLOCK_RE.match(line)
    )


def declarations(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """Top-level declaration names in one file, ignoring its own namespace stack."""
    qualified: set[str] = set()
    bare: set[str] = set()
    pending_attribute = False
    for raw in strip_comments(
        path.read_text(encoding="utf-8", errors="replace")
    ).splitlines():
        line = raw
        if pending_attribute and ATTR_CONT_RE.match(line):
            continue
        pending_attribute = starts_attribute_block(line)
        if pending_attribute:
            continue
        if m := NAMESPACE_RE.match(line):
            _register(qualified, bare, m.group(1))
            continue
        if m := DECL_RE.match(line):
            name = m.group(1)
            _register(qualified, bare, name)
        elif m := DOTTED_DECL_RE.match(line):
            _register(qualified, bare, m.group(1))
    return qualified, bare


def _decl_names(line: str) -> list[str]:
    if m := DECL_RE.match(line):
        return [m.group(1)]
    if m := DOTTED_DECL_RE.match(line):
        return [m.group(1)]
    return []


def namespaces(stack: list[tuple[str, str]]) -> list[str]:
    """The namespace components of a scope stack, in order."""
    return [name for kind, name in stack if kind == "namespace" and name]


def index_tree(root: pathlib.Path) -> tuple[set[str], set[str]]:
    qualified: set[str] = set()
    bare: set[str] = set()
    for path in root.rglob("*.lean"):
        # Namespace and section scopes are tracked per file so `theorem mem_top`
        # inside `namespace StarSubalgebra` is registered as
        # `StarSubalgebra.mem_top`, and so that `end bij` closing a *section*
        # cannot pop the enclosing namespace: Mathlib files can have three times
        # as many `end` lines as `namespace` lines, and popping on any `end`
        # silently drops the namespace from everything after the first one.
        stack: list[tuple[str, str]] = []
        container: str | None = None
        container_indent = 0
        constructors: list[str] = []
        pending_attribute = False
        for raw in strip_comments(
            path.read_text(encoding="utf-8", errors="replace")
        ).splitlines():
            if not raw.strip():
                continue
            line = raw
            if pending_attribute and ATTR_CONT_RE.match(line):
                continue
            pending_attribute = starts_attribute_block(line)
            if pending_attribute:
                continue
            indent = len(raw) - len(raw.lstrip())

            if container is not None:
                if m := CTOR_RE.match(line):
                    constructors.append(m.group(1))
                    continue
                if indent > container_indent and (m := FIELD_RE.match(line)):
                    _register(qualified, bare, f"{container}.{m.group(1)}")
                    continue
                if DERIVING_RE.match(line):
                    continue
                if indent <= container_indent:
                    _register_block(qualified, bare, container, constructors)
                    container, constructors = None, []

            if m := NAMESPACE_RE.match(line):
                stack.append(("namespace", m.group(1)))
                _register(qualified, bare, ".".join(namespaces(stack)))
                continue
            if m := SECTION_RE.match(line):
                stack.append(("section", m.group(1) or ""))
                continue
            if m := END_RE.match(line):
                name = m.group(1)
                if not stack:
                    continue
                if name is None:
                    stack.pop()
                else:
                    for depth in range(len(stack) - 1, -1, -1):
                        if stack[depth][1] == name:
                            del stack[depth:]
                            break
                continue
            if m := BLOCK_RE.match(line):
                name = ".".join(namespaces(stack) + [m.group("name")])
                _register(qualified, bare, name)
                container, container_indent, constructors = name, indent, []
                # `inductive Strategy | Cooperate | Defect` puts constructors on
                # the same line as the keyword; the rest of the block is read by
                # the CTOR_RE branch below.
                tail = line[m.end() :]
                constructors.extend(
                    ctor
                    for raw_ctor in re.findall(r"\|[ \t]*([A-Za-z_][\w']*)", tail)
                    if (ctor := raw_ctor)
                )
                # The block stays open either way: `inductive X where\n | a | b`,
                # `inductive X\n | a`, and `inductive X | a` are all in use, and
                # the CTOR_RE/indent rules below close it in every case.
                continue
            for name in _decl_names(line):
                prefix = namespaces(stack)
                _register(
                    qualified, bare, ".".join(prefix + [name]) if prefix else name
                )
        if container is not None:
            _register_block(qualified, bare, container, constructors)
    qualified.discard("")
    return qualified, bare


def bound_variables(text: str) -> set[str]:
    """Identifiers introduced as binders or by `variable`, per file."""
    names: set[str] = set()
    for line in strip_comments(text).splitlines():
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


# `@[to_additive]` generates the additive statement and name from the
# multiplicative one -- `Finset.sum_range_succ` from `Finset.prod_range_succ` --
# so a large part of Mathlib's additive lemma namespace appears in no source
# file. These are Mathlib's default transliterations; generating the twins of an
# indexed name is how the checker avoids reporting all of them.
ADDITIVE_TOKENS = {
    "sum": "prod",
    "prod": "sum",
    "add": "mul",
    "mul": "add",
    "zero": "one",
    "one": "zero",
    "neg": "inv",
    "inv": "neg",
    "sub": "div",
    "div": "sub",
    "nsmul": "npow",
    "zsmul": "zpow",
}
# `@[to_dual]` generates the order-dual statement and name the same way, which
# is where `Set.mem_Ioi` comes from when the source declares `Set.mem_Iio`.
DUAL_TOKENS = {
    "Ioi": "Iio",
    "Iio": "Ioi",
    "Ici": "Iic",
    "Iic": "Ici",
    "lt": "gt",
    "gt": "lt",
    "le": "ge",
    "ge": "le",
    "inf": "sup",
    "sup": "inf",
    "min": "max",
    "max": "min",
    "top": "bot",
    "bot": "top",
}
MAX_GENERATED_VARIANTS = 64


def generated_variants(name: str) -> list[str]:
    """Additive and order-dual twins of a dotted name, capped in number.

    Both attributes rewrite the *name* token by token, so the twins are
    generated the same way, treating `.` and `_` as separators. The original
    name is not included.
    """
    variants = [""]
    for component in name.split("."):
        tokens = component.split("_")
        spawns: list[list[str]] = [tokens]
        for table in (ADDITIVE_TOKENS, DUAL_TOKENS):
            swapped = [table.get(token, token) for token in tokens]
            if swapped != tokens:
                spawns.append(swapped)
        variants = [
            f"{prefix}{'.' if prefix else ''}{'_'.join(spawn)}"
            for prefix in variants
            for spawn in spawns
        ]
        if len(variants) > MAX_GENERATED_VARIANTS:
            variants = variants[:MAX_GENERATED_VARIANTS]
    unique = [
        variant for variant in dict.fromkeys(variants) if variant and variant != name
    ]
    return unique[:MAX_GENERATED_VARIANTS]


def name_is_known(name: str, known: set[str], bare: set[str]) -> bool:
    """Whether a dotted name is indexed, or explainable without being declared.

    Three cases are accepted besides an exact hit:

    * a generated twin of the name (`Finset.sum_range_succ` for the declared
      `Finset.prod_range_succ`, `Set.mem_Ioi` for `Set.mem_Iio`);
    * dot notation: `Set.mem_Icc.mp` is `Iff.mp (Set.mem_Icc _)`, so a known
      prefix plus a declared trailing name is a use, not a declaration;
    * dot notation on a generated name, which combines the two
      (`Set.mem_Ioi.mp` over the generated `Set.mem_Ioi`).

    Anything else is reported. Each rule only ever *removes* a false positive;
    none of them can turn a missing name into a found one.
    """
    if name in known:
        return True
    if any(variant in known for variant in generated_variants(name)):
        return True
    prefix, _, last = name.rpartition(".")
    if prefix and last in bare:
        if prefix in known:
            return True
        if any(variant in known for variant in generated_variants(prefix)):
            return True
    return False


def find_suspicious(
    repo: pathlib.Path,
    known: set[str],
    bare: set[str],
    new_lines: set[str],
) -> dict[str, set[str]]:
    """Dotted names used in the repository that the index did not find.

    A trailing component that is itself a declared name and whose prefix is a
    known name is dot notation, not a lookup: `Set.mem_Icc.mp` is
    `Iff.mp (Set.mem_Icc _)`, and `Finset.univ.sum` is `Finset.sum Finset.univ`.
    Those are accepted; everything else is reported.
    """
    suspicious: dict[str, set[str]] = {}
    for path in sorted(repo.glob("*.lean")):
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        locals_ = bound_variables(text)
        for line in text.splitlines():
            if not line.strip() or NON_CODE_LINE_RE.match(line):
                continue
            if new_lines and line.strip() not in new_lines:
                continue
            for name in NAME_USE_RE.findall(line):
                if name in known:
                    continue
                head = name.split(".")[0]
                if head in locals_ or len(head) <= 2:
                    continue  # field projection of a bound variable
                if name_is_known(name, known, bare):
                    continue
                suspicious.setdefault(path.name, set()).add(name)
    return suspicious


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathlib", required=True, help="path to a mathlib4 checkout")
    parser.add_argument("--lean", default="", help="path to a lean4 (core) checkout")
    parser.add_argument("--repo", default="lean_proofs")
    parser.add_argument("--baseline", default="", help="git revision to diff against")
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="only check lines added since --baseline",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="fail when more than this many suspicious names are found (default 0)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    repo = pathlib.Path(args.repo).resolve()
    home = repo.parent

    trees = [pathlib.Path(args.mathlib)] + (
        [pathlib.Path(args.lean)] if args.lean else []
    )
    trees.append(repo)
    known: set[str] = set()
    bare: set[str] = set()
    for tree in trees:
        qualified, names = index_tree(tree)
        # Every tree, the repository included: indexing nothing is how this
        # check would pass vacuously, so it is an error rather than a result.
        if not qualified:
            print(
                f"error: no declarations indexed from {tree}; a missing or empty "
                "source tree would turn this check into a no-op",
                file=sys.stderr,
            )
            return 2
        known |= qualified
        bare |= names
        label = "repository" if tree == repo else tree.name
        print(f"indexed {label}: {len(qualified)} names")

    new_lines = (
        added_lines(home, args.baseline) if (args.only_new and args.baseline) else set()
    )
    suspicious = find_suspicious(repo, known, bare, new_lines)

    total = 0
    for filename, names in sorted(suspicious.items()):
        print(f"\n{filename}:")
        for name in sorted(names):
            marker = "NEW " if any(name in line for line in new_lines) else "    "
            print(f"  {marker}? {name}")
            total += 1
    print(f"\n{total} suspicious name(s), limit {args.max}.")
    print(
        "This is a lexical floor, not a proof of well-typedness: a name reported here may "
        "exist through `open`, notation or type-class resolution that this cannot see."
    )
    return 1 if total > args.max else 0


if __name__ == "__main__":
    sys.exit(main())
