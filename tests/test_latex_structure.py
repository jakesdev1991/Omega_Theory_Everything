# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""CI gate for research/check_latex_structure.py (structural LaTeX checks)."""

from __future__ import annotations

import pathlib

from research.check_latex_structure import ROOT, check, check_volume, main

GOOD = r"""\documentclass{article}
\begin{document}
\section{One}
Inline $a+b$ math, a literal \{ brace \} and a price of 100\% or \$5.
\[
  e^{i\pi} + 1 = 0
\]
% \begin{commented} is not an environment here
\end{document}
"""


def write(tmp_path: pathlib.Path, text: str, name: str = "Vol09_X.tex") -> pathlib.Path:
    (tmp_path / "latex_docs").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "latex_docs" / name
    path.write_text(text, encoding="utf-8")
    return path


def rules(tex: str, tmp_path: pathlib.Path) -> list[str]:
    return [finding.rule for finding in check_volume(write(tmp_path, tex))]


def test_repository_volumes_are_structurally_clean() -> None:
    assert check(ROOT) == []


def test_repository_gate_covers_the_41_volumes() -> None:
    tex_count = len(list((ROOT / "latex_docs").glob("Vol*.tex")))
    assert tex_count == 41
    assert check(ROOT) == []


def test_good_file_has_no_findings(tmp_path: pathlib.Path) -> None:
    assert check_volume(write(tmp_path, GOOD)) == []


def test_unbalanced_brace_is_found(tmp_path: pathlib.Path) -> None:
    assert rules(GOOD.replace("e^{i\\pi}", "e^{i\\pi"), tmp_path) == ["braces"]


def test_escaped_braces_are_literal(tmp_path: pathlib.Path) -> None:
    # A lone \{ is a literal character, not an unclosed group: counting it would
    # report depth +1 here, and the real volumes use \{ and \} in text.
    text = "\\documentclass{article}\n\\begin{document}\nliteral \\{ only\n\\end{document}\n"
    assert check_volume(write(tmp_path, text)) == []


def test_mismatched_environment_is_found(tmp_path: pathlib.Path) -> None:
    broken = GOOD.replace(r"\begin{document}", r"\begin{documents}")
    assert "environments" in rules(broken, tmp_path)


def test_unclosed_environment_is_found(tmp_path: pathlib.Path) -> None:
    broken = GOOD.replace(r"\end{document}", "")
    found = rules(broken, tmp_path)
    assert "environments" in found
    assert "document" in found


def test_missing_documentclass_is_found(tmp_path: pathlib.Path) -> None:
    broken = GOOD.replace("\\documentclass{article}\n", "")
    assert rules(broken, tmp_path) == ["document"]


def test_unbalanced_display_math_is_found(tmp_path: pathlib.Path) -> None:
    broken = GOOD.replace("\\]", "")
    assert rules(broken, tmp_path) == ["display math"]


def test_odd_inline_dollar_is_found(tmp_path: pathlib.Path) -> None:
    broken = GOOD.replace("Inline $a+b$ math", "Inline $a+b math")
    assert rules(broken, tmp_path) == ["inline math"]


def test_escaped_dollar_does_not_count_as_math(tmp_path: pathlib.Path) -> None:
    # One \$ is money, not an unbalanced math delimiter.
    text = "\\documentclass{article}\n\\begin{document}\ncosts \\$5\n\\end{document}\n"
    assert check_volume(write(tmp_path, text)) == []


def test_commented_out_environment_is_ignored(tmp_path: pathlib.Path) -> None:
    # A commented-out, unbalanced \begin{} must not break the nesting check.
    text = GOOD.replace(
        "% \\begin{commented} is not an environment here",
        "% \\begin{commented} without an end",
    )
    assert "environments" not in rules(text, tmp_path)


def test_check_without_files_reports_an_error(tmp_path: pathlib.Path) -> None:
    findings = check(tmp_path)
    assert [f.rule for f in findings] == ["files"]


def test_main_exit_status(tmp_path: pathlib.Path, capsys: object) -> None:
    write(tmp_path, GOOD)
    assert main(["--root", str(tmp_path), "--list"]) == 0
    assert main(["--root", str(tmp_path)]) == 0
    write(tmp_path, GOOD.replace("\\]", ""))
    assert main(["--root", str(tmp_path)]) == 1
