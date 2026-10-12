# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Tests for lean_proofs/check_statement_audit.py.

Synthetic volume trees rather than the real corpus: the point of these tests is
the gate's rules -- one row per candidate, a witness/source that exists, a quoted
condition that occurs, a retired name that does not survive -- not the Phase 6
judgements themselves.  One test does run the gate against this directory's real
`statement_audit.json`, because the shipped audit must cover the shipped corpus.

Written with `unittest`, like the rest of this directory, because
`.github/workflows/lean-ci.yml` runs these audits with
`python3 -m unittest discover`; pytest collects them too.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from check_statement_audit import PHYSICS_TOKENS, main, physics_words

ROOT = pathlib.Path(__file__).resolve().parent

FAKE_VOLUME = """\
namespace OmegaProtocol.Vol01

/-- A concrete bracket. -/
noncomputable def poissonBracket (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x - g x

theorem poisson_antisymm (f g : ℝ → ℝ) (x : ℝ) :
    poissonBracket f g x = -poissonBracket g f x :=
  sub_eq_neg_sub.symm

theorem energy_conserved (m : ℝ) (hm : m ≠ 0) : m = m := rfl

noncomputable def photonMass : ℝ := 0

theorem photon_massless : photonMass = 0 := rfl

theorem bridge_vol01_metric_self_zero (x : ℝ) : x - x = 0 := sub_self x

end OmegaProtocol.Vol01
"""


def row(name: str, disposition: str, **extra: str) -> dict[str, str]:
    """A well-formed row for the fixture volume, with its required extra field."""
    payload = {
        "name": name,
        "file": "Vol01_Fixture.lean",
        "disposition": disposition,
        "note": f"Recorded judgement for {name} in the fixture corpus.",
    }
    payload.update(extra)
    return payload


def fixture_rows() -> list[dict[str, str]]:
    return [
        row("poisson_antisymm", "witness", witness="poissonBracket"),
        row("energy_conserved", "conditional", stated_condition="hm : m ≠ 0"),
        row("photon_massless", "carried", source="photonMass"),
        row(
            "bridge_vol01_metric_self_zero",
            "renamed",
            renamed_to="bridge_vol01_metric_self_zero",
            previous_name="metric_from_omega",
        ),
    ]


class GateTestCase(unittest.TestCase):
    """A temp volume tree whose audit can be mutated one field at a time."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        (self.root / "Vol01_Fixture.lean").write_text(FAKE_VOLUME, encoding="utf-8")
        self.write_audit(fixture_rows())

    def write_audit(self, rows: list[dict[str, str]], cap: object = 1) -> None:
        payload: dict[str, object] = {"rows": rows}
        if cap is not None:
            payload["carried_cap"] = cap
        (self.root / "statement_audit.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def audit(self) -> list[dict[str, str]]:
        return json.loads(
            (self.root / "statement_audit.json").read_text(encoding="utf-8")
        )["rows"]

    def run_main(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(["--repo", str(self.root), *args])
        return status, out.getvalue(), err.getvalue()

    def assert_fails(self, *args: str) -> str:
        status, out, _ = self.run_main(*args)
        self.assertEqual(status, 1, out)
        self.assertIn("FAILED", out)
        return out

    # --- the gate passes on a well-formed fixture ---------------------------

    def test_well_formed_fixture_passes(self) -> None:
        status, out, _ = self.run_main()
        self.assertEqual(status, 0, out)
        self.assertIn("4 physics-sounding declaration(s)", out)
        self.assertIn("witness 1, conditional 1, renamed 1, carried 1", out)
        self.assertIn("within the ratchet", out)
        self.assertIn("OK: every physics-sounding volume declaration", out)

    def test_list_names_every_row(self) -> None:
        status, out, _ = self.run_main("--list")
        self.assertEqual(status, 0, out)
        for name in ("poisson_antisymm", "energy_conserved", "photon_massless"):
            self.assertIn(name, out)

    def test_review_prints_candidate_statements(self) -> None:
        status, out, _ = self.run_main("--review")
        self.assertEqual(status, 0, out)
        self.assertIn("Vol01_Fixture.lean", out)
        self.assertIn("poisson_antisymm", out)
        self.assertIn("words=", out)

    # --- missing and malformed rows ----------------------------------------

    def test_missing_row_fails(self) -> None:
        rows = [r for r in self.audit() if r["name"] != "energy_conserved"]
        self.write_audit(rows)
        out = self.assert_fails()
        self.assertIn("1 candidate(s) lack a row", out)
        self.assertIn("energy_conserved", out)

    def test_missing_row_within_the_limit_passes(self) -> None:
        rows = [r for r in self.audit() if r["name"] != "energy_conserved"]
        self.write_audit(rows)
        status, out, _ = self.run_main("--max-missing", "1")
        self.assertEqual(status, 0, out)
        self.assertIn("1 candidate(s) lack a row, limit 1", out)

    def test_unknown_row_name_fails(self) -> None:
        rows = self.audit() + [
            row("no_such_declaration", "witness", witness="photonMass")
        ]
        self.write_audit(rows)
        self.assertIn("no such declaration in the corpus", self.assert_fails())

    def test_duplicate_row_fails(self) -> None:
        rows = self.audit() + [self.audit()[0]]
        self.write_audit(rows)
        self.assertIn("duplicate row", self.assert_fails())

    def test_wrong_file_fails(self) -> None:
        rows = self.audit()
        rows[0]["file"] = "Vol99_Elsewhere.lean"
        self.write_audit(rows)
        self.assertIn("declared in Vol01_Fixture.lean", self.assert_fails())

    def test_short_note_fails(self) -> None:
        rows = self.audit()
        rows[0]["note"] = "too short"
        self.write_audit(rows)
        self.assertIn("note is too short", self.assert_fails())

    def test_unknown_disposition_fails(self) -> None:
        rows = self.audit()
        rows[0]["disposition"] = "vibes"
        self.write_audit(rows)
        self.assertIn("is not one of", self.assert_fails())

    # --- per-disposition checks --------------------------------------------

    def test_witness_must_exist_and_differ(self) -> None:
        rows = self.audit()
        rows[0]["witness"] = "poisson_antisymm"
        self.write_audit(rows)
        self.assertIn("witness is the theorem itself", self.assert_fails())
        rows[0]["witness"] = "not_a_declaration"
        self.write_audit(rows)
        self.assertIn("is not a declaration", self.assert_fails())

    def test_conditional_condition_must_occur(self) -> None:
        rows = self.audit()
        rows[1]["stated_condition"] = "hm : m > 0"
        self.write_audit(rows)
        self.assertIn("does not occur in the statement", self.assert_fails())

    def test_conditional_condition_must_be_quoted(self) -> None:
        rows = self.audit()
        rows[1].pop("stated_condition")
        self.write_audit(rows)
        self.assertIn("must quote stated_condition", self.assert_fails())

    def test_carried_source_must_exist_and_differ(self) -> None:
        rows = self.audit()
        rows[2]["source"] = "photon_massless"
        self.write_audit(rows)
        self.assertIn("source is the declaration itself", self.assert_fails())
        rows[2]["source"] = "not_a_declaration"
        self.write_audit(rows)
        self.assertIn("is not a declaration", self.assert_fails())

    def test_carried_rows_are_capped(self) -> None:
        self.write_audit(self.audit(), cap=0)
        out = self.assert_fails()
        self.assertIn("OVER the ratchet", out)
        self.assertIn("carried row(s) (cap 0)", out)

    def test_renamed_target_must_exist(self) -> None:
        rows = self.audit()
        rows[3]["renamed_to"] = "not_a_declaration"
        self.write_audit(rows)
        self.assertIn("renamed_to", self.assert_fails())

    def test_retired_name_must_not_survive(self) -> None:
        rows = self.audit()
        rows[3]["previous_name"] = "photonMass"
        self.write_audit(rows)
        self.assertIn("still exists", self.assert_fails())

    def test_retired_name_check_masks_comments(self) -> None:
        path = self.root / "Vol01_Fixture.lean"
        path.write_text(
            FAKE_VOLUME
            + "-- the retired name metric_from_omega is only mentioned here\n",
            encoding="utf-8",
        )
        status, out, _ = self.run_main()
        self.assertEqual(status, 0, out)
        path.write_text(
            FAKE_VOLUME + "def old_alias : ℝ := metric_from_omega\n",
            encoding="utf-8",
        )
        self.assertIn("is still mentioned in", self.assert_fails())

    # --- unreadable audits --------------------------------------------------

    def test_missing_audit_exits_two(self) -> None:
        (self.root / "statement_audit.json").unlink()
        status, _, err = self.run_main()
        self.assertEqual(status, 2)
        self.assertIn("error:", err)

    def test_audit_without_cap_exits_two(self) -> None:
        self.write_audit(self.audit(), cap=None)
        status, _, err = self.run_main()
        self.assertEqual(status, 2)
        self.assertIn("carried_cap", err)


class LexiconTests(unittest.TestCase):
    def test_words_split_on_separators_and_case(self) -> None:
        self.assertEqual(physics_words("conservation_of_energy"), ["energy"])
        self.assertEqual(
            physics_words("twoLevel_kms_zeroth_law"),
            ["kms"],
        )
        self.assertEqual(physics_words("unrelated_lemma_name"), [])

    def test_lexicon_has_no_prose_words(self) -> None:
        # The literal used to be split straight from its own comment lines, so
        # words like `the` and `of` became tokens and matched unrelated names.
        for prose in ("the", "of", "names", "line", "tokens", "comment"):
            self.assertNotIn(prose, PHYSICS_TOKENS)


class ShippedAuditTests(unittest.TestCase):
    def test_the_repository_audit_covers_the_repository_corpus(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = main(["--repo", str(ROOT)])
        text = out.getvalue()
        self.assertEqual(status, 0, text)
        self.assertIn("0 candidate(s) lack a row, limit 0", text)
        self.assertIn("within the ratchet", text)


if __name__ == "__main__":
    unittest.main()
