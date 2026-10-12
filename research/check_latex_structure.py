# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Structural checks for the LaTeX volumes, runnable without a TeX toolchain.

No LaTeX installation is available in this environment, so the rewritten volumes
of roadmap Phase 4 were checked for brace balance, environment nesting and
unescaped special characters by hand. This turns that hand check into a gate, so
a future edit cannot break a volume's structure without a red check.

Checked per ``latex_docs/*.tex``, with TeX comments removed first:

1. braces balance (``\\{`` and ``\\}`` are literal characters, not grouping);
2. ``\\begin{X}`` / ``\\end{X}`` nest correctly and every environment is closed;
3. exactly one ``\\documentclass``, one ``\\begin{document}`` and one
   ``\\end{document}``, with ``document`` as the outermost environment;
4. ``\\[`` and ``\\]`` display-math delimiters balance;
5. the number of unescaped ``$`` is even, so inline math is not left open.

What this does not check: that the volume compiles. A TeX toolchain would catch
a missing package, an undefined macro, a bad table, or math that is unbalanced
across a line in a way this cannot see. These five rules are a floor, chosen so
that they are exact on files that compile -- no heuristics that could pass a
broken file to keep the gate green.

Run: python research/check_latex_structure.py [--list]
Exit status is 0 only if every file passes every rule.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research.check_latex_citations import strip_tex_comments  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LATEX_DIR = "latex_docs"
ENVIRONMENT = re.compile(r"\\(begin|end)\{([^}]*)\}")
ESCAPED_BRACE = re.compile(r"\\[{}]")
UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


@dataclass(frozen=True)
class Finding:
    """One structural problem in one volume."""

    tex: str
    rule: str
    detail: str


def check_volume(path: pathlib.Path) -> list[Finding]:
    """Apply the five structural rules to one TeX file."""
    code = strip_tex_comments(path.read_text(encoding="utf-8"))
    findings: list[Finding] = []

    body = ESCAPED_BRACE.sub("", code)
    brace_depth = body.count("{") - body.count("}")
    if brace_depth != 0:
        findings.append(
            Finding(path.name, "braces", f"unbalanced braces: depth {brace_depth:+d}")
        )

    stack: list[str] = []
    for match in ENVIRONMENT.finditer(code):
        kind, name = match.group(1), match.group(2)
        if kind == "begin":
            stack.append(name)
        elif not stack or stack[-1] != name:
            findings.append(
                Finding(
                    path.name,
                    "environments",
                    f"\\end{{{name}}} does not match the open environments {stack[-3:] or '[]'}",
                )
            )
            break
        else:
            stack.pop()
    else:
        if stack:
            findings.append(
                Finding(path.name, "environments", f"never closed: {stack}")
            )

    for token, count in (
        (r"\documentclass", code.count(r"\documentclass")),
        (r"\begin{document}", code.count(r"\begin{document}")),
        (r"\end{document}", code.count(r"\end{document}")),
    ):
        if count != 1:
            findings.append(
                Finding(path.name, "document", f"{token} appears {count} times")
            )

    open_bracket = code.count(r"\[")
    close_bracket = code.count(r"\]")
    if open_bracket != close_bracket:
        findings.append(
            Finding(
                path.name,
                "display math",
                f"display-math delimiters: open {open_bracket}, close {close_bracket}",
            )
        )

    dollars = len(UNESCAPED_DOLLAR.findall(code))
    if dollars % 2:
        findings.append(
            Finding(path.name, "inline math", f"odd number of unescaped $ ({dollars})")
        )

    return findings


def check(root: pathlib.Path = ROOT) -> list[Finding]:
    """Check every volume under ``root``; return all findings."""
    files = sorted((root / LATEX_DIR).glob("Vol*.tex"))
    findings: list[Finding] = []
    if not files:
        return [Finding("", "files", f"no {LATEX_DIR}/Vol*.tex files under {root}")]
    for path in files:
        findings.extend(check_volume(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=pathlib.Path, default=ROOT, help="repository root"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the rules and the file list"
    )
    args = parser.parse_args(argv)

    files = sorted((args.root / LATEX_DIR).glob("Vol*.tex"))
    if args.list:
        for path in files:
            print(path.name)
        print("rules: braces, environments, document, display math, inline math")
        return 0

    findings = check(args.root)
    for finding in findings:
        print(f"[ERROR] {finding.tex}: {finding.rule}: {finding.detail}")
    print(
        f"\n{len(files)} volumes checked: {len(findings)} finding(s) "
        "(braces, environment nesting, document markers, math delimiters)."
    )
    print("Not checked: that the volumes compile -- no TeX toolchain here.")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
