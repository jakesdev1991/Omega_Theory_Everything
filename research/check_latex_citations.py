# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Check that every Lean target cited by a LaTeX volume exists.

Phase 4 of docs/research/CLAIM_TIGHTENING_ROADMAP.md rewrote the volumes so that
each one names its formal counterpart and labels what that file does and does
not establish. Step 2 of that phase -- "a script check that every cited
declaration exists" -- was left open. This is that script.

What is checked, for each ``latex_docs/*.tex``:

1. Every ``\\texttt{lean\\_proofs/X.lean}`` citation names a module that exists
   in ``lean_proofs/``.
2. The volume number in the file name matches the volume number of a module it
   cites, so a status box cannot point at the neighbouring volume.
3. Every ``\\texttt{identifier}`` citation is a declaration that exists in the
   LaTeX file's own cited module, or elsewhere in the corpus (reported as a
   note: the volume then leans on a declaration the volume's file does not
   define), or is listed in ``EXTERNAL`` with a reason -- Lean core and Mathlib
   names such as ``Empty`` and ``HasDerivAt`` that no corpus file declares.
   An ``EXTERNAL`` entry that no volume cites any more is a stale exemption and
   fails.
4. A token inside ``\\texttt{}`` that is neither a known module path, an
   identifier, nor an ``identifier = value`` pair is an error rather than a
   silent skip, so a new citation style cannot slip past unclassified.
5. ``latex_docs/README.md`` states how many of the 54 volumes have a LaTeX file
   and which ones do not; both are compared with the directory.

What is *not* checked: that the cited declaration says what the prose claims.
That is the statement-level audit (Phase 6) and it is not mechanical.

Scope limits, stated so a green run is not oversold: citations are found in
``\\texttt{}`` only, TeX comments are ignored, and the declaration scan is the
same masked-lexical regex the other source audits use -- not a Lean parser. A
declaration that exists only inside a ``where`` clause or in a `namespace` with
a primed shadow will resolve by bare name.

Run: python research/check_latex_citations.py [--list]
Exit status is 0 only if there are no errors.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lean_proofs"))

from lean_source import mask_comments_and_strings  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LATEX_DIR = "latex_docs"
LEAN_DIR = "lean_proofs"

# Lean core and Mathlib names, cited by a volume but not declared by the corpus.
# Each entry needs a reason, and the entry must still be cited (see `check`).
EXTERNAL: dict[str, str] = {
    "Empty": "Lean core empty type; Vol54 formalizes facts about it, it is not a corpus declaration",
    "HasDerivAt": "Mathlib differentiability predicate; Vol17 takes it as a hypothesis, it is not a corpus declaration",
}

LEAN_PATH = re.compile(r"lean_proofs/([A-Za-z0-9_]+)\.lean")
TEXTTT = re.compile(r"\\texttt\{([^{}]*)\}")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_']*)\s*=\s*.*", re.S)
VOLUME = re.compile(r"Vol(\d+)")
README_COUNT = re.compile(r"(\d+)\s+of the\s+(\d+)\s+volumes are currently documented")
README_MISSING = re.compile(r"no LaTeX file yet:\s*([^.]*)\.")

DECLARATION = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\]\s*)*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|axiom|opaque|structure|class|inductive)"
    r"\s+([A-Za-z_][A-Za-z0-9_'.]*)",
    re.M,
)
RANGE = re.compile(r"Vol(\d+)\s*-\s*Vol(\d+)")


@dataclass(frozen=True)
class Citation:
    """One ``\\texttt{}`` citation, resolved or not."""

    tex: str
    line: int
    kind: str  # "module" | "declaration"
    value: str
    declared_in: tuple[str, ...] = ()

    @property
    def module_name(self) -> str:
        """The Lean module this volume is named after, ``VolNN_X.tex -> VolNN_X.lean``."""
        return f"{pathlib.Path(self.tex).stem}.lean"

    @property
    def resolves_locally(self) -> bool:
        return self.module_name in self.declared_in


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning" | "note"
    message: str


def strip_tex_comments(text: str) -> str:
    """Blank TeX comments (a ``%`` that is not ``\\%``) to end of line, keeping offsets."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        ending = line[len(line.rstrip("\r\n")) :]
        body = line[: len(line) - len(ending)]
        cut = len(body)
        for index, char in enumerate(body):
            if char == "%" and (index == 0 or body[index - 1] != "\\"):
                cut = index
                break
        out.append(body[:cut] + " " * (len(body) - cut) + ending)
    return "".join(out)


def unescape(token: str) -> str:
    """Undo the LaTeX escaping a ``\\texttt{}`` payload uses for Lean names."""
    for source, target in (
        ("\\_", "_"),
        ("\\{", "{"),
        ("\\}", "}"),
        ("\\&", "&"),
        ("\\$", "$"),
        ("\\#", "#"),
        ("\\%", "%"),
        ("\\'", "'"),
    ):
        token = token.replace(source, target)
    return token.strip()


def tex_citations(path: pathlib.Path) -> tuple[list[Citation], list[Issue]]:
    """Extract module and declaration citations from one TeX file."""
    text = strip_tex_comments(path.read_text(encoding="utf-8"))
    citations: list[Citation] = []
    issues: list[Issue] = []
    for match in TEXTTT.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        raw = match.group(1)
        token = unescape(raw)
        module = LEAN_PATH.search(token.replace("\\_", "_"))
        if module:
            citations.append(Citation(path.name, line, "module", module.group(1)))
            continue
        if token in EXTERNAL:
            citations.append(Citation(path.name, line, "declaration", token))
            continue
        assigned = ASSIGNMENT.fullmatch(token)
        candidate = assigned.group(1) if assigned else token
        if IDENTIFIER.fullmatch(candidate):
            citations.append(Citation(path.name, line, "declaration", candidate))
        else:
            issues.append(
                Issue(
                    "error",
                    f"{path.name}:{line}: unclassified \\texttt{{{raw}}}: not a Lean "
                    "module path, an identifier, or an 'identifier = value' pair",
                )
            )
    return citations, issues


def corpus_declarations(root: pathlib.Path) -> dict[str, tuple[str, ...]]:
    """Declaration name -> sorted module file names, comments masked out."""
    found: dict[str, list[str]] = {}
    for source in sorted((root / LEAN_DIR).glob("*.lean")):
        masked = mask_comments_and_strings(source.read_text(encoding="utf-8"))
        for name in DECLARATION.findall(masked):
            found.setdefault(name, []).append(source.name)
    return {name: tuple(sorted(set(files))) for name, files in found.items()}


def readme_issues(root: pathlib.Path, tex_names: set[str]) -> list[Issue]:
    """Compare latex_docs/README.md's coverage claims with the directory."""
    readme = root / LATEX_DIR / "README.md"
    if not readme.is_file():
        return [Issue("error", f"{LATEX_DIR}/README.md: missing")]
    text = readme.read_text(encoding="utf-8")
    issues: list[Issue] = []
    count = README_COUNT.search(text)
    if not count:
        issues.append(
            Issue("error", f"{readme.name}: no 'N of the 54 volumes' coverage sentence")
        )
    else:
        stated, total = int(count.group(1)), int(count.group(2))
        if stated != len(tex_names):
            issues.append(
                Issue(
                    "error",
                    f"{readme.name}: says {stated} volumes have LaTeX files; the directory has {len(tex_names)}",
                )
            )
        if total != 54:
            issues.append(
                Issue(
                    "warning",
                    f"{readme.name}: says the corpus has {total} volumes, not 54",
                )
            )
    missing = README_MISSING.search(text)
    if not missing:
        issues.append(Issue("error", f"{readme.name}: no 'no LaTeX file yet' list"))
        return issues
    listed: set[int] = set()
    for low, high in RANGE.findall(missing.group(1)):
        listed.update(range(int(low), int(high) + 1))
    present = {
        int(match.group(1)) for name in tex_names if (match := VOLUME.match(name))
    }
    actual = set(range(1, 55)) - present
    if listed != actual:
        issues.append(
            Issue(
                "error",
                f"{readme.name}: missing-volume list {sorted(listed)} != actual {sorted(actual)}",
            )
        )
    return issues


def collect(root: pathlib.Path) -> tuple[list[Citation], list[Issue]]:
    citations: list[Citation] = []
    issues: list[Issue] = []
    tex_files = sorted((root / LATEX_DIR).glob("Vol*.tex"))
    if not tex_files:
        return citations, [
            Issue("error", f"no {LATEX_DIR}/Vol*.tex files found under {root}")
        ]
    for path in tex_files:
        found, problems = tex_citations(path)
        citations.extend(found)
        issues.extend(problems)
    return citations, issues


def volume_of(name: str) -> str | None:
    """The volume number in a file or module name, as a string, or None."""
    match = VOLUME.match(name)
    return match.group(1) if match else None


def resolved_citations(
    citations: list[Citation], declarations: dict[str, tuple[str, ...]]
) -> list[Citation]:
    """Attach the declaring modules to each declaration citation."""
    return [
        Citation(
            citation.tex,
            citation.line,
            citation.kind,
            citation.value,
            declarations.get(citation.value, ()),
        )
        if citation.kind == "declaration" and citation.value not in EXTERNAL
        else citation
        for citation in citations
    ]


def check(root: pathlib.Path = ROOT) -> tuple[list[Citation], list[Issue]]:
    """Resolve every citation; return (citations, issues) with errors in issues."""
    citations, issues = collect(root)
    declarations = corpus_declarations(root)
    lean_files = {path.name for path in (root / LEAN_DIR).glob("*.lean")}
    cited_external: set[str] = set()
    citations = resolved_citations(citations, declarations)

    for citation in citations:
        if citation.kind == "module":
            if f"{citation.value}.lean" not in lean_files:
                issues.append(
                    Issue(
                        "error",
                        f"{citation.tex}:{citation.line}: cites missing module "
                        f"{LEAN_DIR}/{citation.value}.lean",
                    )
                )
        elif citation.value in EXTERNAL:
            cited_external.add(citation.value)
            if citation.value in declarations:
                issues.append(
                    Issue(
                        "error",
                        f"{citation.tex}:{citation.line}: {citation.value} is exempted as "
                        f"external but is declared in {declarations[citation.value]}",
                    )
                )
        elif not citation.declared_in:
            issues.append(
                Issue(
                    "error",
                    f"{citation.tex}:{citation.line}: cites {citation.value}, which no "
                    f"{LEAN_DIR}/*.lean file declares",
                )
            )
        elif not citation.resolves_locally:
            issues.append(
                Issue(
                    "note",
                    f"{citation.tex}:{citation.line}: {citation.value} is declared in "
                    f"{', '.join(citation.declared_in)}, not in this volume's file",
                )
            )

    for name in sorted(EXTERNAL):
        if name not in cited_external:
            issues.append(
                Issue(
                    "error",
                    f"{LATEX_DIR}: stale EXTERNAL exemption for {name}; no volume cites it",
                )
            )

    module_citations = [c for c in citations if c.kind == "module"]
    for tex in sorted({c.tex for c in citations}):
        cited = {c.value for c in module_citations if c.tex == tex}
        if not cited:
            issues.append(Issue("error", f"{tex}: cites no Lean module at all"))
            continue
        volume = volume_of(tex)
        if volume and not any(volume_of(name) == volume for name in cited):
            issues.append(
                Issue(
                    "error",
                    f"{tex}: cites no Vol{volume} module (cites {', '.join(sorted(cited))})",
                )
            )

    issues.extend(
        readme_issues(root, {path.name for path in (root / LATEX_DIR).glob("Vol*.tex")})
    )
    return citations, issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=pathlib.Path, default=ROOT, help="repository root to check"
    )
    parser.add_argument(
        "--list", action="store_true", help="print every citation and its resolution"
    )
    args = parser.parse_args(argv)

    citations, issues = check(args.root)
    if args.list:
        for citation in citations:
            where = (
                f"{LEAN_DIR}/{citation.value}.lean"
                if citation.kind == "module"
                else ", ".join(citation.declared_in) or "EXTERNAL"
            )
            print(f"{citation.tex}:{citation.line}  {citation.value:28s} {where}")
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    notes = [i for i in issues if i.severity == "note"]
    for issue in errors:
        print(f"[ERROR] {issue.message}")
    for issue in warnings:
        print(f"[WARN]  {issue.message}")
    for issue in notes:
        print(f"[note]  {issue.message}")
    declared = [
        c for c in citations if c.kind == "declaration" and c.value not in EXTERNAL
    ]
    tex_count = len({c.tex for c in citations})
    print(
        f"\n{tex_count} LaTeX volumes cite {len({c.value for c in citations if c.kind == 'module'})} modules "
        f"and {len({c.value for c in declared})} declarations "
        f"({len({c.value for c in declared if not c.resolves_locally})} declared outside the citing volume)."
    )
    print(f"{len(errors)} error(s), {len(warnings)} warning(s), {len(notes)} note(s).")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
