"""Regression fixtures for lexical audit coverage (not Lean proof checks)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_axioms import declarations
from audit_vacuity import (
    alias_baseline_errors,
    audit,
    axiom_named_declarations,
    decorative_hypotheses,
    identity_proofs,
    statement_aliases,
    trivial_proofs,
    trivially_true_hypotheses,
)
from lean_source import find_import_cycle, mask_comments_and_strings, module_imports


class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, text, name="Vol01_Test.lean"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_nested_comments_and_strings_preserve_positions(self):
        source = (
            "/- outer\n /- axiom hidden : False -/\n -/\n"
            'def text := "axiom fake : False -- /-"\n'
            "axiom actual : False\n"
        )
        self.source(source)
        masked = mask_comments_and_strings(source)
        self.assertEqual(len(masked), len(source))
        self.assertEqual(masked.count("\n"), source.count("\n"))
        self.assertEqual(
            [(line, name) for _, line, name in declarations(self.root)], [(5, "actual")]
        )

    def test_escaped_quote_does_not_end_string(self):
        source = 'def text := "escaped \\" -- /-"\naxiom actual : False\n'
        self.source(source)
        self.assertEqual([name for _, _, name in declarations(self.root)], ["actual"])

    def test_commented_axioms_and_structure_fields_are_not_commands(self):
        self.source(
            "-- axiom fake : False\n/- axiom fake2 : False -/\n"
            "structure Model where\n  axiom_field : False\n"
        )
        self.assertEqual(declarations(self.root), [])

    def test_attributes_modifiers_and_unicode_axiom_names(self):
        self.source(
            "@[someAttribute] private axiom 假设 : False\n"
            "protected axiom next : False\n"
        )
        self.assertEqual(
            [name for _, _, name in declarations(self.root)], ["假设", "next"]
        )

    def test_nested_sources_but_not_downloaded_dependencies(self):
        self.source("axiom localClaim : False", "Nested/Local.lean")
        self.source("axiom dependency : False", ".lake/packages/Other.lean")
        self.assertEqual(
            [name for _, _, name in declarations(self.root)], ["localClaim"]
        )

    def test_parameterized_and_multiline_unit_types(self):
        self.source(
            "def Form (_degree : ℕ) : Type := Unit\n"
            "abbrev Indexed\n  {n : ℕ}\n  (x : Fin n) : Type := Unit\n"
        )
        self.assertEqual(len(audit(self.root)[2]), 2)

    def test_comment_and_string_unit_types_are_not_counted(self):
        self.source(
            "/- def Fake : Type := Unit -/\n"
            'def message := "def NotAType : Type := Unit"\n'
        )
        self.assertEqual(audit(self.root)[2], [])

    def test_direct_misleading_statement_and_bridge_exemption(self):
        self.source(
            "theorem physical_law (R : QRegion) : d R R = 0 := by exact h\n"
            "theorem bridge_self (R : QRegion) : d R R = 0 := by exact h\n"
        )
        self.assertEqual(audit(self.root)[0], ["Vol01_Test.lean: physical_law"])

    def test_legacy_alias_export_cannot_hide_behind_proposition_name(self):
        self.source(
            "def Law_Stmt : Prop := 0 = 0\ntheorem physicalLaw : Law_Stmt := by rfl\n"
        )
        self.assertEqual(
            statement_aliases(self.root), ["Vol01_Test.lean: physicalLaw -> Law_Stmt"]
        )

    def test_qualified_unicode_alias_and_long_theorem(self):
        self.source(
            "theorem 定理\n"
            + "\n".join(f"  (x{i} : Nat)" for i in range(20))
            + "\n  : Model.Law_Stmt := proof\n"
        )
        self.assertEqual(
            statement_aliases(self.root), ["Vol01_Test.lean: 定理 -> Model.Law_Stmt"]
        )

    def test_new_alias_cannot_replace_retired_alias_at_same_count(self):
        baseline = self.root / "baseline.json"
        baseline.write_text(json.dumps(["Vol01_Test.lean: old -> Old_Stmt"]))
        self.source("theorem new : New_Stmt := proof\n")
        new, stale = alias_baseline_errors(self.root, baseline)
        self.assertEqual(new, ["Vol01_Test.lean: new -> New_Stmt"])
        self.assertEqual(stale, ["Vol01_Test.lean: old -> Old_Stmt"])

    def test_exact_reviewed_alias_baseline_passes(self):
        self.source("theorem old : Old_Stmt := proof\n")
        baseline = self.root / "baseline.json"
        baseline.write_text(json.dumps(statement_aliases(self.root)))
        self.assertEqual(alias_baseline_errors(self.root, baseline), ([], []))

    def test_import_scanner_masks_comments_and_handles_nested_modules(self):
        self.source(
            "/- import Fake -/\nimport Base Mathlib\npublic import Other\n",
            "Nested/Local.lean",
        )
        self.source("import Lake\n", "lakefile.lean")
        self.source("import Ignored\n", ".lake/Cache.lean")
        self.assertEqual(
            module_imports(self.root), {"Nested.Local": {"Base", "Mathlib", "Other"}}
        )

    def test_import_cycle_detection_and_shared_dependencies(self):
        self.assertEqual(find_import_cycle({"A": {"A"}}), ["A", "A"])
        self.assertEqual(find_import_cycle({"A": {"B"}, "B": {"A"}}), ["A", "B", "A"])
        self.assertIsNone(
            find_import_cycle(
                {"A": {"B", "C"}, "B": {"D"}, "C": {"D"}, "D": {"Mathlib"}}
            )
        )

    def test_opaque_policy_masks_comments_and_checks_nested_sources(self):
        self.source(
            "/- opaque hidden : Nat -/\n@[tag] private opaque actual : Nat := 1",
            "Nested/Policy.lean",
        )
        self.assertEqual([name for _, _, name in declarations(self.root)], ["actual"])

    def test_trivial_only_proof_including_multiline_and_comment(self):
        self.source(
            "theorem a : True := by\n  trivial -- comment\n"
            "theorem b : True := by trivial\n"
        )
        self.assertEqual(len(trivial_proofs(self.root)), 2)

    def test_trivial_text_in_comments_and_strings_is_not_a_proof(self):
        self.source(
            "/- theorem a : True := by trivial -/\n"
            'def message := ":= by trivial"\n'
            "theorem b : True := by exact True.intro\n"
        )
        self.assertEqual(trivial_proofs(self.root), [])

    def test_axiom_named_declarations_and_fields_are_flagged(self):
        self.source(
            "theorem great_filter_axiom : Nonempty Filter := witness\n"
            "theorem axiom_kms_law : True := trivial\n"
            "structure Model where\n"
            "  axiom_metric : ℕ\n"
            "instance : Model where\n"
            "  axiom_metric := 0\n"
        )
        self.assertEqual(
            axiom_named_declarations(self.root),
            [
                "Vol01_Test.lean: axiom_kms_law",
                "Vol01_Test.lean: axiom_metric",
                "Vol01_Test.lean: great_filter_axiom",
            ],
        )

    def test_quantified_trivial_conclusion_is_flagged_and_bridge_is_exempt(self):
        self.source(
            "theorem hidden : ∀ (R : QRegion), d R R = 0 := by intro R; rfl\n"
            "theorem bridge_hidden : ∀ (R : QRegion), d R R = 0 := by intro R; rfl\n"
        )
        self.assertEqual(
            audit(self.root)[0],
            ["Vol01_Test.lean: hidden (quantified conclusion)"],
        )

    def test_hypothesis_chained_trivial_conclusion_is_flagged(self):
        self.source(
            "theorem hidden (R₁ R₂ : QRegion) : "
            "mutualInformation R₁ R₂ = 0 → d R₁ R₂ ≥ 0 := by intro h; rfl\n"
        )
        self.assertEqual(
            audit(self.root)[0],
            ["Vol01_Test.lean: hidden (quantified conclusion)"],
        )

    def test_decorative_hypothesis_flagged_but_statement_use_exempts(self):
        self.source(
            "theorem decorative (base : ℝ) (h : 0 ≤ base) : base ≤ base := by"
            "\n  exact le_rfl\n"
            "theorem statement_use (x : ℝ) (h : 0 ≤ x) : consume x h = x := by"
            "\n  rfl\n"
            "theorem automation_use (x : ℝ) (h : 0 ≤ x) : 0 ≤ x + 1 := by"
            "\n  linarith\n"
            "theorem bridge_decorative (base : ℝ) (h : 0 ≤ base) : base ≤ base :="
            " by\n  exact le_rfl\n"
        )
        self.assertEqual(
            decorative_hypotheses(self.root),
            [
                "Vol01_Test.lean: decorative [h : 0 ≤ base]",
                "Vol01_Test.lean: bridge_decorative [h : 0 ≤ base]",
            ],
        )

    def test_primed_hypothesis_used_in_proof_is_not_decorative(self):
        self.source(
            "theorem primed (a' : ℝ) (ha' : |a'| ≤ 1) : |a'| ≤ 1 := by\n  exact ha'\n"
        )
        self.assertEqual(decorative_hypotheses(self.root), [])

    def test_trivially_true_hypotheses_are_flagged(self):
        self.source(
            "theorem zero_bound (h : (0 : ℝ) ≤ 0) : d R R = 0 := by rfl\n"
            "theorem reflexive_bound (n : ℕ) (h : n = n) : d R R = 0 := by rfl\n"
            "theorem real_bound (x : ℝ) (h : 0 ≤ x) : d R R = 0 := by rfl\n"
        )
        self.assertEqual(
            trivially_true_hypotheses(self.root),
            [
                "Vol01_Test.lean: zero_bound [h : (0 : ℝ) ≤ 0]",
                "Vol01_Test.lean: reflexive_bound [h : n = n]",
            ],
        )

    def test_identity_proofs_are_flagged(self):
        self.source(
            "theorem pass_through (h : P) : Q := by exact h\n"
            "theorem introduced (R : QRegion) (h : isFrozen R) : "
            "forwardFlux R ≤ freezeBoundaryThreshold := by\n  intro h\n  exact h\n"
            "theorem term_pass (a b c d : ℝ) (hdet : a * d - b * c = 1) : "
            "a * d - b * c = 1 := hdet\n"
            "theorem wrapped (h : P) : Q := by exact ⟨h⟩\n"
            "theorem control_alias : Q := global_lemma\n"
            "theorem control_projection (S : QubitState) : S.trace = 1 := "
            "S.tr_one\n"
        )
        self.assertEqual(
            identity_proofs(self.root),
            [
                "Vol01_Test.lean: pass_through",
                "Vol01_Test.lean: introduced",
                "Vol01_Test.lean: term_pass (term-mode)",
            ],
        )

    def test_axiom_prose_and_law_fields_are_not_flagged(self):
        self.source(
            "/- the legacy name `foo_axiom` was retired: no axioms exist -/\n"
            'def note := "not an _axiom name"\n'
            "structure Model where\n"
            "  law_einstein_equations : ℕ\n"
            "theorem bell_violation : True := trivial\n"
        )
        self.assertEqual(axiom_named_declarations(self.root), [])


if __name__ == "__main__":
    unittest.main()
