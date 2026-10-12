"""Tests for the Phase 5 parity gate: comments cannot change a declaration.

These are lexical regression tests, not Lean proof checks. The two repository
tests at the bottom are the CI gates; the rest pin the behaviour of the
projection itself, including the directions in which it must stay sensitive.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_declaration_parity as parity
import sync_txt_mirrors as mirrors

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / parity.BASELINE_NAME
TXT_DIR = mirrors.TXT_DIR


class StripCommentTests(unittest.TestCase):
    def test_line_comment_removed(self):
        self.assertEqual(
            parity.strip_comments("def x := 1 -- noise\n"), "def x := 1 \n"
        )

    def test_nested_block_comment_removed(self):
        text = "def x := 1 /- outer /- axiom fake : False -/ inner -/\ndef y := 2\n"
        self.assertEqual(parity.strip_comments(text), "def x := 1 \ndef y := 2\n")

    def test_docstring_removed_but_body_kept(self):
        text = "/-- Docstring with an axiom : False inside. -/\ndef x := 1\n"
        self.assertEqual(parity.strip_comments(text), "\ndef x := 1\n")

    def test_comment_markers_inside_strings_are_kept(self):
        text = 'def s := "-- not a comment /- nor this -/"\n'
        self.assertEqual(parity.strip_comments(text), text)

    def test_escaped_quote_does_not_end_the_string(self):
        text = r'def s := "a\"-- b"' + "\ndef x := 1\n"
        self.assertEqual(parity.strip_comments(text), text)

    def test_char_literals_are_kept(self):
        text = "def q : Char := '\"'\ndef n : Char := '\\n'\n"
        self.assertEqual(parity.strip_comments(text), text)


class ProjectionTests(unittest.TestCase):
    def test_blank_lines_and_trailing_space_do_not_move_the_digest(self):
        base = "def x := 1\n\n\ndef y := 2   \n"
        noisy = "def x := 1\n\n\n\n\ndef y := 2\n"
        self.assertEqual(
            parity.projection_digest(base), parity.projection_digest(noisy)
        )

    def test_comment_only_edit_does_not_move_the_digest(self):
        before = "/-- Old label. -/\ndef x := 1\n"
        after = "/-- Machine-checked: x is one. -/\ndef x := 1\n"
        self.assertEqual(
            parity.projection_digest(before), parity.projection_digest(after)
        )

    def test_statement_change_moves_the_digest(self):
        before = "def x : Nat := 1\n"
        after = "def x : Nat := 2\n"
        self.assertNotEqual(
            parity.projection_digest(before), parity.projection_digest(after)
        )

    def test_proof_body_change_moves_the_digest(self):
        before = "theorem t : True := by trivial\n"
        after = "theorem t : True := by exact True.intro\n"
        self.assertNotEqual(
            parity.projection_digest(before), parity.projection_digest(after)
        )

    def test_declaration_count_ignores_commented_out_headers(self):
        text = "/- theorem hidden : False := by\n   sorry\n-/\ntheorem shown : True := by trivial\n"
        self.assertEqual(parity.declaration_count(text), 1)


class CompareTests(unittest.TestCase):
    def tree(self, name: str, digest: str) -> dict:
        return {name: {"sha256": digest, "declarations": 1, "code_lines": 1}}

    def test_identical_trees_report_nothing(self):
        tree = self.tree("A.lean", "abc")
        self.assertEqual(parity.compare(tree, dict(tree)), [])

    def test_added_and_removed_files_are_reported(self):
        current = self.tree("A.lean", "abc")
        expected = self.tree("B.lean", "abc")
        problems = parity.compare(current, expected)
        self.assertEqual(len(problems), 2)
        self.assertTrue(
            any("B.lean" in line and "removed" in line for line in problems)
        )
        self.assertTrue(any("A.lean" in line and "added" in line for line in problems))

    def test_changed_code_is_reported_with_counts(self):
        current = {"A.lean": {"sha256": "new", "declarations": 3, "code_lines": 9}}
        expected = {"A.lean": {"sha256": "old", "declarations": 2, "code_lines": 8}}
        self.assertEqual(
            parity.compare(current, expected),
            ["A.lean: code changed (declarations 2 -> 3, code lines 8 -> 9)"],
        )


class SyncMirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lean = Path(self.temp.name) / "lean_proofs"
        self.txt = Path(self.temp.name) / "txt_proofs"
        self.lean.mkdir()
        self.txt.mkdir()

    def test_write_then_check_round_trip(self):
        (self.lean / "ToE.lean").write_text("import Vol01\n", encoding="utf-8")
        (self.lean / "lakefile.lean").write_text("import Lake\n", encoding="utf-8")
        changed = mirrors.sync(self.lean, self.txt)
        self.assertEqual(changed, ["ToE.txt"])
        self.assertEqual(mirrors.check(self.lean, self.txt), [])
        self.assertEqual(
            (self.txt / "ToE.txt").read_text(encoding="utf-8"), "import Vol01\n"
        )
        self.assertFalse((self.txt / "lakefile.txt").exists())

    def test_stale_mirror_is_detected_then_fixed(self):
        source = self.lean / "ToE.lean"
        source.write_text("def x := 1\n", encoding="utf-8")
        (self.txt / "ToE.txt").write_text("def x := 0\n", encoding="utf-8")
        self.assertEqual(
            mirrors.check(self.lean, self.txt), ["ToE.txt: mirror out of date"]
        )
        self.assertEqual(mirrors.sync(self.lean, self.txt), ["ToE.txt"])
        self.assertEqual(mirrors.check(self.lean, self.txt), [])

    def test_orphan_is_reported_but_not_deleted(self):
        (self.lean / "ToE.lean").write_text("def x := 1\n", encoding="utf-8")
        orphan = self.txt / "Vol99_Gone.txt"
        orphan.write_text("archived\n", encoding="utf-8")
        self.assertEqual(
            mirrors.check(self.lean, self.txt),
            [
                "ToE.txt: mirror missing",
                "Vol99_Gone.txt: orphan (no Lean source; not deleted)",
            ],
        )
        self.assertTrue(orphan.exists())


class RepositoryGateTests(unittest.TestCase):
    """CI gates: the committed baseline and the mirrors track the sources."""

    def test_corpus_matches_declaration_baseline(self):
        if not BASELINE.exists():
            self.fail(
                f"missing {BASELINE}; run "
                "python3 lean_proofs/check_declaration_parity.py --update"
            )
        expected = parity.load_baseline(BASELINE)
        self.assertEqual(parity.compare(parity.project_tree(ROOT), expected), [])

    def test_txt_mirrors_are_in_sync(self):
        self.assertEqual(mirrors.check(ROOT, TXT_DIR), [])


if __name__ == "__main__":
    unittest.main()
