# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""CI gate for research/check_latex_citations.py (roadmap Phase 4, step 2)."""

from __future__ import annotations

import pathlib

from research.check_latex_citations import (
    EXTERNAL,
    Citation,
    check,
    strip_tex_comments,
    tex_citations,
    unescape,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def errors(root: pathlib.Path) -> list[str]:
    _, issues = check(root)
    return [issue.message for issue in issues if issue.severity == "error"]


def test_repository_citations_all_resolve() -> None:
    assert errors(ROOT) == []


def test_every_volume_cites_a_module_and_the_module_exists() -> None:
    citations, _ = check(ROOT)
    modules = [c for c in citations if c.kind == "module"]
    assert len({c.tex for c in modules}) == 41
    assert all((ROOT / "lean_proofs" / f"{c.value}.lean").is_file() for c in modules)


def test_declaration_citations_resolve_in_the_citing_volume() -> None:
    citations, _ = check(ROOT)
    declarations = [
        c for c in citations if c.kind == "declaration" and c.value not in EXTERNAL
    ]
    assert declarations, "the volumes are expected to cite at least one declaration"
    unresolved = [c for c in declarations if not c.resolves_locally]
    assert not unresolved, [(c.tex, c.value, c.declared_in) for c in unresolved]


def test_external_exemptions_are_exactly_the_uncitable_names() -> None:
    citations, issues = check(ROOT)
    cited = {c.value for c in citations if c.kind == "declaration"}
    assert set(EXTERNAL) <= cited, "a stale exemption would have been an error"
    assert not [m for m in issues if "stale EXTERNAL" in m.message]


def test_unescape_and_token_classification() -> None:
    assert (
        unescape("lean\\_proofs/Vol11\\_CondensedMatter.lean")
        == "lean_proofs/Vol11_CondensedMatter.lean"
    )
    assert unescape("chern\\_number\\_quantized") == "chern_number_quantized"
    assert unescape("NumberOfGenerations = 3").startswith("NumberOfGenerations")


def test_comment_and_prose_citations_are_extracted(tmp_path: pathlib.Path) -> None:
    tex = tmp_path / "Vol11_CondensedMatter.tex"
    tex.write_text(
        "% \\texttt{CommentedOut} is not a citation\n"
        "\\texttt{lean\\_proofs/Vol11\\_CondensedMatter.lean} and \\texttt{ChernBand} and\n"
        "\\texttt{NumberOfGenerations = 3}\n",
        encoding="utf-8",
    )
    citations, issues = tex_citations(tex)
    assert issues == []
    assert [(c.kind, c.value) for c in citations] == [
        ("module", "Vol11_CondensedMatter"),
        ("declaration", "ChernBand"),
        ("declaration", "NumberOfGenerations"),
    ]


def test_strip_tex_comments_keeps_offsets() -> None:
    text = "a % b\n100\\% c\n"
    stripped = strip_tex_comments(text)
    assert stripped.splitlines() == ["a    ", "100\\% c"]
    assert len(stripped) == len(text)
    assert [len(line) for line in stripped.splitlines()] == [
        len(line) for line in text.splitlines()
    ]


def _write_corpus(root: pathlib.Path, module: str, body: str) -> None:
    (root / "lean_proofs").mkdir(parents=True, exist_ok=True)
    (root / "lean_proofs" / f"{module}.lean").write_text(body, encoding="utf-8")


def _write_tex(root: pathlib.Path, name: str, body: str) -> None:
    (root / "latex_docs").mkdir(parents=True, exist_ok=True)
    (root / "latex_docs" / name).write_text(body, encoding="utf-8")


def test_missing_module_and_unknown_declaration_fail(tmp_path: pathlib.Path) -> None:
    _write_corpus(
        tmp_path, "Vol11_CondensedMatter", "theorem ChernBand_ok : True := by trivial\n"
    )
    _write_tex(
        tmp_path,
        "Vol11_CondensedMatter.tex",
        "\\texttt{lean\\_proofs/Vol11\\_CondensedMatter.lean} \\texttt{Nonexistent}\n",
    )
    messages = errors(tmp_path)
    assert any("declares" in m and "Nonexistent" in m for m in messages)
    _write_tex(
        tmp_path,
        "Vol11_CondensedMatter.tex",
        "\\texttt{lean\\_proofs/Vol12\\_QuantumInformation.lean}\n",
    )
    messages = errors(tmp_path)
    assert any("missing module" in m for m in messages)
    assert any("cites no Vol11 module" in m for m in messages)


def test_wrong_volume_module_and_unclassified_token_fail(
    tmp_path: pathlib.Path,
) -> None:
    _write_corpus(tmp_path, "Vol11_CondensedMatter", "def ChernBand : Nat := 0\n")
    _write_corpus(tmp_path, "Vol12_QuantumInformation", "def Qubit : Nat := 0\n")
    _write_tex(
        tmp_path,
        "Vol11_CondensedMatter.tex",
        "\\texttt{lean\\_proofs/Vol12\\_QuantumInformation.lean} \\texttt{a sentence with spaces}\n",
    )
    messages = errors(tmp_path)
    assert any("unclassified" in m for m in messages)
    assert any("cites no Vol11 module" in m for m in messages)


def test_stale_exemption_fails(tmp_path: pathlib.Path) -> None:
    _write_corpus(tmp_path, "Vol11_CondensedMatter", "def ChernBand : Nat := 0\n")
    _write_tex(
        tmp_path,
        "Vol11_CondensedMatter.tex",
        "\\texttt{lean\\_proofs/Vol11\\_CondensedMatter.lean} \\texttt{ChernBand}\n",
    )
    (tmp_path / "latex_docs" / "README.md").write_text(
        "41 of the 54 volumes are currently documented. The following have no LaTeX file yet: Vol01-Vol53.\n",
        encoding="utf-8",
    )
    messages = errors(tmp_path)
    assert [m for m in messages if "stale EXTERNAL exemption for Empty" in m]
    assert [m for m in messages if "stale EXTERNAL exemption for HasDerivAt" in m]


def test_readme_coverage_drift_fails(tmp_path: pathlib.Path) -> None:
    _write_corpus(tmp_path, "Vol11_CondensedMatter", "def ChernBand : Nat := 0\n")
    _write_tex(
        tmp_path,
        "Vol11_CondensedMatter.tex",
        "\\texttt{lean\\_proofs/Vol11\\_CondensedMatter.lean}\n",
    )
    (tmp_path / "latex_docs" / "README.md").write_text(
        "40 of the 54 volumes are currently documented. The following have no LaTeX file yet: Vol01-Vol53.\n",
        encoding="utf-8",
    )
    messages = errors(tmp_path)
    assert any(
        "says 40 volumes have LaTeX files; the directory has 1" in m for m in messages
    )
    assert any("missing-volume list" in m for m in messages)


def test_exempted_name_that_becomes_a_declaration_fails(tmp_path: pathlib.Path) -> None:
    _write_corpus(tmp_path, "Vol54_TheoryOfNothing", "def Empty : Nat := 0\n")
    _write_tex(
        tmp_path,
        "Vol54_TheoryOfNothing.tex",
        "\\texttt{lean\\_proofs/Vol54\\_TheoryOfNothing.lean} \\texttt{Empty}\n",
    )
    messages = errors(tmp_path)
    assert any("exempted as external but is declared" in m for m in messages)


def test_resolves_locally_uses_the_volume_module() -> None:
    citation = Citation(
        "Vol11_CondensedMatter.tex",
        1,
        "declaration",
        "ChernBand",
        ("Vol11_CondensedMatter.lean",),
    )
    assert citation.resolves_locally
    assert citation.module_name == "Vol11_CondensedMatter.lean"
    other = Citation(
        "Vol11_CondensedMatter.tex",
        1,
        "declaration",
        "Qubit",
        ("Vol12_QuantumInformation.lean",),
    )
    assert not other.resolves_locally
