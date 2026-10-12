# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Tests for lean_proofs/check_mathlib_names.py.

Synthetic Mathlib/Lean/repository trees rather than the real checkouts: the
point of these tests is the indexer's rules, not the pinned sources.

Written with `unittest`, like the rest of this directory, because
`.github/workflows/lean-ci.yml` runs these audits with
`python3 -m unittest discover`; pytest collects them too.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import tempfile
import unittest

from check_mathlib_names import (
    generated_variants,
    index_tree,
    main,
    name_is_known,
    starts_attribute_block,
    strip_comments,
)

FAKE_MATHLIB = """\
namespace Finset

section CommMonoid

@[to_additive]
theorem prod_range_succ (f : ℕ → M) (n : ℕ) : True := trivial

@[to_additive]
lemma prod_const_one : True := trivial

end CommMonoid

namespace Prod

theorem mem_range : True := trivial

end Prod

end Finset
"""

FAKE_LEAN = """\
namespace Set

@[to_dual (attr := simp)]
theorem mem_Iio : True := trivial

end Set
"""

FAKE_REPO = """\
import Mathlib

namespace Local

inductive Strategy | Cooperate | Defect

structure Point where
  x : ℕ
  y : ℕ

nonrec def toNat : ℕ := 0

@[simp] theorem labelled : True := trivial

theorem plain : True := trivial

end Local
"""


class TreeTestCase(unittest.TestCase):
    """A repository, a fake mathlib and a fake core tree, in a temp directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        self.mathlib = self.tmp / "mathlib4"
        self.lean = self.tmp / "lean4" / "src"
        for root, text in ((self.mathlib, FAKE_MATHLIB), (self.lean, FAKE_LEAN)):
            root.mkdir(parents=True)
            (root / "Source.lean").write_text(text, encoding="utf-8")
        self.repo = self.tmp / "lean_proofs"
        self.repo.mkdir()
        (self.repo / "Local.lean").write_text(FAKE_REPO, encoding="utf-8")

    def write_repo(self, text: str) -> pathlib.Path:
        (self.repo / "Local.lean").write_text(text, encoding="utf-8")
        return self.repo

    def run_main(self, repo: pathlib.Path | None = None) -> tuple[int, str, str]:
        """Run `main` against the fake trees; return (status, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(
                [
                    "--mathlib",
                    str(self.mathlib),
                    "--lean",
                    str(self.lean),
                    "--repo",
                    str(self.repo if repo is None else repo),
                ]
            )
        return status, out.getvalue(), err.getvalue()


class IndexTreeTests(TreeTestCase):
    def test_namespaces_constructors_and_sections(self) -> None:
        qualified, bare = index_tree(self.mathlib)
        self.assertLessEqual(
            {
                "Finset.prod_range_succ",
                "Finset.Prod.mem_range",
                "Finset.prod_const_one",
            },
            qualified,
        )
        # `end CommMonoid` closes a section, not the enclosing namespace.
        self.assertIn("prod_range_succ", bare)
        repo_qualified, _ = index_tree(self.repo)
        self.assertLessEqual(
            {
                "Local.Strategy",
                "Local.Strategy.Cooperate",
                "Local.Strategy.Defect",
            },
            repo_qualified,
        )
        self.assertLessEqual({"Local.Point.x", "Local.Point.y"}, repo_qualified)
        self.assertLessEqual(
            {"Local.toNat", "Local.labelled", "Local.plain"}, repo_qualified
        )

    def test_same_line_and_indented_constructors(self) -> None:
        (self.repo / "One.lean").write_text(
            "inductive One | first | second\ninductive Two\n  | only\n",
            encoding="utf-8",
        )
        qualified, _ = index_tree(self.repo)
        self.assertLessEqual({"One.first", "One.second", "Two.only"}, qualified)

    def test_multi_line_attribute_does_not_hide_the_declaration(self) -> None:
        text = "@[to_additive /-- doc -/]\ntheorem foo (f : ℕ → M) : True := trivial\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "Snippet.lean").write_text(strip_comments(text), encoding="utf-8")
            qualified, _ = index_tree(root)
        self.assertIn("foo", qualified)

    def test_dependency_checkouts_are_not_part_of_the_corpus(self) -> None:
        """In CI the repository tree contains the whole mathlib copy."""
        nested = self.repo / ".lake" / "packages" / "mathlib4" / "Mathlib"
        nested.mkdir(parents=True)
        (nested / "Big.lean").write_text(
            "theorem dependency_theorem : True := trivial\n", encoding="utf-8"
        )
        qualified, _ = index_tree(self.repo)
        self.assertNotIn("dependency_theorem", qualified)
        self.assertIn("Local.plain", qualified)

    def test_strings_and_comments_are_masked(self) -> None:
        repo = self.tmp / "masked"
        repo.mkdir()
        (repo / "S.lean").write_text(
            "def real : Nat := 0\n"
            'def s := "Not.A.Declaration"\n'
            "-- Other.Fake\n"
            "/-- Doc.Comment -/\n"
            "def other : Nat := 0\n",
            encoding="utf-8",
        )
        qualified, _ = index_tree(repo)
        self.assertEqual(qualified, {"real", "s", "other"})


class NameLookupTests(TreeTestCase):
    def test_generated_variants_cover_to_additive_and_to_dual(self) -> None:
        self.assertIn(
            "Finset.prod_range_succ", generated_variants("Finset.sum_range_succ")
        )
        self.assertIn("Set.mem_Iio", generated_variants("Set.mem_Ioi"))
        self.assertIn(
            "Finset.prod_const_one", generated_variants("Finset.sum_const_zero")
        )
        self.assertEqual(generated_variants("Local.toNat"), [])

    def test_name_is_known_accepts_generated_and_dot_notation(self) -> None:
        qualified, bare = index_tree(self.mathlib)
        known = qualified | {"mp"}
        bare = bare | {"mp"}
        self.assertTrue(name_is_known("Finset.prod_range_succ", known, bare))
        # generated by to_additive from the declared multiplicative lemma
        self.assertTrue(name_is_known("Finset.sum_range_succ", known, bare))
        self.assertTrue(name_is_known("Finset.prod_range_succ.mp", known, bare))
        self.assertFalse(name_is_known("Finset.prod_range_succs", known, bare))
        self.assertFalse(name_is_known("Finset.sum_range_succs", known, bare))

    def test_dot_notation_over_a_generated_name(self) -> None:
        qualified, bare = index_tree(self.lean)
        self.assertTrue(name_is_known("Set.mem_Ioi", qualified, bare))
        self.assertTrue(name_is_known("Set.mem_Ioi.mp", qualified, {"mp"} | bare))
        self.assertFalse(name_is_known("Set.mem_Ioi.mpq", qualified, {"mp"} | bare))


class AttributeTests(unittest.TestCase):
    def test_attribute_lines_and_pending_attributes(self) -> None:
        self.assertTrue(starts_attribute_block("@[simp]"))
        self.assertTrue(starts_attribute_block("@[to_additive (attr := simp)]"))
        self.assertFalse(
            starts_attribute_block("@[simp] theorem foo : True := trivial")
        )
        self.assertFalse(starts_attribute_block("@[simp] nonrec def foo : Nat := 0"))
        self.assertFalse(starts_attribute_block("theorem foo : True := trivial"))


class MainTests(TreeTestCase):
    def test_reports_a_missing_name_and_ratchets(self) -> None:
        self.write_repo("theorem uses : True := Real.sqrtt 4\n")
        status, out, _ = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("Real.sqrtt", out)
        args = ["--mathlib", str(self.mathlib), "--repo", str(self.repo), "--max", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)

    def test_accepts_a_clean_repository(self) -> None:
        self.write_repo("theorem uses : True := trivial\n")
        status, out, _ = self.run_main()
        self.assertEqual(status, 0)
        self.assertIn("0 suspicious name(s), limit 0.", out)

    def test_import_lines_are_not_scanned(self) -> None:
        self.write_repo(
            "import Mathlib.Nonexistent.Module\n\ntheorem uses : True := trivial\n"
        )
        status, out, _ = self.run_main()
        self.assertEqual(status, 0)
        self.assertIn("indexed repository: 1 names", out)

    def test_refuses_an_empty_source_tree(self) -> None:
        """An index with nothing in it must not read as a clean repository."""
        self.mathlib = self.tmp / "empty-mathlib"
        (self.mathlib / "Mathlib").mkdir(parents=True)
        self.write_repo("theorem uses : True := trivial\n")
        status, out, err = self.run_main()
        self.assertEqual(status, 2)
        self.assertIn("no declarations indexed", err)
        self.assertNotIn("suspicious", out)

    def test_refuses_a_missing_repository(self) -> None:
        """A wrong --repo would otherwise pass with zero names to check."""
        status, _, err = self.run_main(repo=self.tmp / "not-a-repo")
        self.assertEqual(status, 2)
        self.assertIn("no declarations indexed", err)


if __name__ == "__main__":
    unittest.main()
