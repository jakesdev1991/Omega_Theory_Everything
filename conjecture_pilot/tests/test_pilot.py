# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Unit tests for the pilot scaffold. Run: python -m pytest conjecture_pilot -q
(or python -m unittest discover -s conjecture_pilot/tests)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conjecture_pilot.extract import RealDef, extract_real_defs, extract_statements
from conjecture_pilot.freeze import (
    HELDOUT_MODULES,
    MASKED_MODULES,
    TRAIN_MODULES,
    freeze,
)
from conjecture_pilot.gates import evaluate, lean_available, outcome
from conjecture_pilot.numeric import (
    Evaluator,
    check,
    lean_div,
    lean_pow,
    lean_sqrt,
    tokenize,
)
from conjecture_pilot.schema import (
    Binder,
    Candidate,
    Epistemic,
    Ledger,
    Statement,
    Status,
)
from conjecture_pilot.transform import (
    converse,
    generalize_all,
    generalize_one,
    generate,
    weaken,
)

ROOT = Path(__file__).resolve().parents[2]
LEAN = ROOT / "lean_proofs"

SAMPLE = """
import Mathlib

namespace Demo

/-- A doc comment with a fake `theorem inside_comment (x : ℝ) : x = x := rfl`. -/
noncomputable def sq1 (Φ : ℝ) : ℝ :=
  (1 - Φ ^ 2) ^ 2

theorem bounded (Φ : ℝ) (h0 : 0 ≤ Φ) (h1 : Φ ≤ 1) :
    sq1 Φ ≤ 1 := by
  sorry_free_placeholder

lemma no_binders : (2 : ℝ) + 2 = 4 := by norm_num

theorem with_instance {α : Type*} [Nonempty α] (f : ℕ → ℝ) (hf : ∀ n, 0 ≤ f n) (n : ℕ) :
    0 ≤ f n := hf n

end Demo
"""


def _stmt(binders: list[Binder], conclusion: str) -> Statement:
    return Statement("T", "t", "theorem", 1, tuple(binders), conclusion)


def _real(*names: str) -> Binder:
    return Binder(tuple(names), "ℝ", "(", False)


def _hyp(name: str, prop: str) -> Binder:
    return Binder((name,), prop, "(", True)


class ExtractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "Demo.lean"
        self.path.write_text(SAMPLE, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_statements_and_binders(self) -> None:
        statements, unparsed = extract_statements(self.path, "Demo")
        self.assertEqual(unparsed, [])
        names = [s.name for s in statements]
        self.assertEqual(names, ["bounded", "no_binders", "with_instance"])
        bounded = statements[0]
        self.assertEqual([b.names for b in bounded.binders], [("Φ",), ("h0",), ("h1",)])
        self.assertEqual(
            [b.is_hypothesis for b in bounded.binders], [False, True, True]
        )
        self.assertEqual(bounded.conclusion, "sq1 Φ ≤ 1")
        self.assertEqual(bounded.namespaces, ("Demo",))
        self.assertEqual(statements[1].binders, ())
        inst = statements[2]
        self.assertEqual([b.bracket for b in inst.binders], ["{", "[", "(", "(", "("])
        self.assertFalse(
            inst.binders[2].is_hypothesis
        )  # (f : ℕ → ℝ) is a function, not a hypothesis
        self.assertTrue(inst.binders[3].is_hypothesis)

    def test_comment_is_not_a_declaration(self) -> None:
        statements, _ = extract_statements(self.path, "Demo")
        self.assertNotIn("inside_comment", [s.name for s in statements])

    def test_real_defs(self) -> None:
        defs = extract_real_defs(self.path)
        self.assertEqual(defs, {"sq1": RealDef("sq1", ("Φ",), "(1 - Φ ^ 2) ^ 2")})

    def test_repo_heldout_files_fully_parse(self) -> None:
        for module in HELDOUT_MODULES:
            statements, unparsed = extract_statements(LEAN / f"{module}.lean", module)
            self.assertTrue(statements, module)
            self.assertEqual(unparsed, [], module)


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s = _stmt(
            [_real("Φ"), _hyp("h0", "0 ≤ Φ"), _hyp("h1", "Φ ≤ 1")], "Φ ^ 2 ≤ 1"
        )

    def test_weaken_drops_one_hypothesis_each(self) -> None:
        out = list(weaken(self.s))
        self.assertEqual([p["dropped"] for p, _ in out], [["h0"], ["h1"]])
        self.assertEqual(len(out[0][1].binders), 2)

    def test_converse_swaps(self) -> None:
        out = list(converse(self.s))
        self.assertEqual(len(out), 2)
        _, st = out[1]
        self.assertEqual(st.conclusion, "Φ ≤ 1")
        self.assertEqual(st.binders[-1].type, "Φ ^ 2 ≤ 1")

    def test_generalize_skips_exponents_and_identifier_digits(self) -> None:
        s = _stmt([_real("Φ"), _hyp("h1", "Φ ≤ 1")], "Φ ^ 2 ≤ 1")
        one = list(generalize_one(s))
        self.assertEqual(
            [p["literal"] for p, _ in one], ["1"]
        )  # the exponent 2 is not generalized
        allv = list(generalize_all(s))
        self.assertEqual(allv[0][1].binders[0].names, ("c",))
        self.assertEqual(allv[0][1].binders[2].type, "Φ ≤ c")
        self.assertEqual(allv[0][1].conclusion, "Φ ^ 2 ≤ c")

    def test_fresh_variable_avoids_clash(self) -> None:
        s = _stmt([_real("c")], "c + 1 = 1 + c")
        out = list(generalize_one(s))
        self.assertEqual(out[0][1].binders[0].names, ("c'",))

    def test_generate_ids_are_deterministic(self) -> None:
        a = generate(self.s, "T")
        b = generate(self.s, "T")
        self.assertEqual([c.id for c in a], [c.id for c in b])
        self.assertTrue(all(c.epistemic is Epistemic.CONJECTURE for c in a))


class NumericTests(unittest.TestCase):
    def test_lean_semantics(self) -> None:
        self.assertEqual(lean_div(1.0, 0.0), 0.0)
        self.assertEqual(lean_sqrt(-4.0), 0.0)
        self.assertEqual(lean_pow(2.0, 3.0), 8.0)
        self.assertEqual(lean_pow(0.0, -1.0), 0.0)

    def test_parser_precedence(self) -> None:
        ev = Evaluator({})
        self.assertEqual(ev.eval(ev.parse("-2 ^ 2"), {}), -4.0)
        self.assertEqual(ev.eval(ev.parse("(1 : ℝ) + 2 * 3"), {}), 7.0)
        self.assertEqual(ev.eval(ev.parse("Real.sqrt (1 - x ^ 2)"), {"x": 0.6}), 0.8)
        self.assertEqual(ev.eval(ev.parse("|x - 3|"), {"x": 1.0}), 2.0)
        self.assertTrue(ev.eval(ev.parse("0 < 1 ∧ ¬ (1 < 0)"), {}))
        self.assertEqual(len(tokenize("√(1 - Φ₁ ^ 2)")), 8)

    def test_true_statement_passes_and_flags_decorative_hypothesis(self) -> None:
        s = _stmt(
            [
                _real("Φ", "y"),
                _hyp("h0", "0 ≤ Φ"),
                _hyp("hy", "y ≤ 2"),
                _hyp("h1", "Φ ^ 2 ≤ 1"),
            ],
            "Φ ≤ 1",
        )
        v = check(s, {}, samples=1500)
        self.assertEqual(v.status, "pass", v.reason)
        self.assertEqual(v.relevance["h1"], "load_bearing")
        self.assertEqual(v.relevance["hy"], "not_load_bearing_numerically")
        # A hypothesis whose violation set is never sampled stays honestly undetermined.
        s2 = _stmt([_real("Φ"), _hyp("hd", "Φ ≠ 7")], "Φ ^ 2 ≥ 0")
        self.assertEqual(check(s2, {}, samples=300).relevance["hd"], "undetermined")

    def test_false_statement_fails_with_counterexample(self) -> None:
        s = _stmt([_real("Φ"), _hyp("h0", "0 ≤ Φ")], "Φ ^ 2 ≤ 1")
        v = check(s, {}, samples=500)
        self.assertEqual(v.status, "fail")
        self.assertEqual(v.reason, "counterexample")
        assert v.counterexample is not None
        self.assertGreater(v.counterexample["Φ"], 1.0)

    def test_equation_hypothesis_is_solved_not_just_sampled(self) -> None:
        defs = {"g": RealDef("g", ("Φ",), "(1 - Φ ^ 2) ^ 2")}
        s = _stmt([_real("Φ"), _hyp("h1", "Φ ≤ 1"), _hyp("h", "g Φ = 1")], "Φ = 0")
        v = check(s, defs, samples=2000)
        self.assertEqual(v.status, "fail")
        assert v.counterexample is not None
        self.assertAlmostEqual(v.counterexample["Φ"], -(2**0.5), places=6)

    def test_division_by_zero_semantics_change_the_verdict(self) -> None:
        defs = {"k": RealDef("k", ("Φ",), "(1 + Φ) ^ 2 / Real.sqrt (1 - Φ ^ 2)")}
        with_h = _stmt([_real("Φ"), _hyp("hΦ", "Φ ^ 2 < 1")], "0 < k Φ")
        without = _stmt([_real("Φ")], "0 < k Φ")
        self.assertEqual(check(with_h, defs, samples=800).status, "pass")
        v = check(without, defs, samples=800)
        self.assertEqual(
            v.status, "fail"
        )  # outside the disc k Φ = 0 in Lean, so 0 < 0 fails

    def test_unsatisfiable_hypotheses_are_flagged(self) -> None:
        s = _stmt([_real("x"), _hyp("h", "x < 0"), _hyp("h2", "0 < x")], "x = 1")
        v = check(s, {}, samples=300)
        self.assertEqual(v.reason, "hypotheses_unsatisfied_in_sampled_domain")

    def test_unsupported_is_not_applicable(self) -> None:
        s = _stmt([Binder(("sys",), "EntropyProcess", "(", False)], "0 ≤ 1")
        self.assertEqual(check(s, {}).status, "not_applicable")
        s2 = _stmt([_real("x")], "Continuous (fun y => y * x)")
        self.assertEqual(check(s2, {}).status, "not_applicable")


class GateAndLedgerTests(unittest.TestCase):
    def test_outcome_labels_without_lean(self) -> None:
        s = _stmt([_real("Φ"), _hyp("h0", "0 ≤ Φ")], "Φ ^ 2 ≤ 1")
        cand = Candidate.make("T", "weaken", s, {}, s)
        results = evaluate(cand, {}, use_lean=False, samples=300)
        self.assertEqual(results[0].status, Status.FAIL)
        self.assertEqual(outcome(results), "refuted_numerically")
        good = _stmt(
            [_real("Φ"), _hyp("h0", "0 ≤ Φ"), _hyp("h1", "Φ ≤ 1")], "Φ ^ 2 ≤ 1"
        )
        cand2 = Candidate.make("T", "weaken", good, {}, good)
        self.assertEqual(
            outcome(evaluate(cand2, {}, use_lean=False, samples=300)),
            "numeric_pass_lean_unavailable",
        )

    def test_lean_gate_reports_unavailable_not_pass(self) -> None:
        if lean_available():
            self.skipTest("Lean present; this test covers the absent-toolchain path")
        s = _stmt([_real("Φ")], "0 ≤ Φ ^ 2")
        cand = Candidate.make("T", "weaken", s, {}, s)
        results = evaluate(cand, {}, use_lean=True, samples=100)
        self.assertEqual({r.gate for r in results}, {"numeric", "wellformed"})
        self.assertEqual(results[1].status, Status.UNAVAILABLE)

    def test_ledger_roundtrip_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "l.jsonl")
            ledger.append({"kind": "run", "n": 1})
            ledger.append({"kind": "gate", "status": "pass"})
            self.assertEqual([r["kind"] for r in ledger.records()], ["run", "gate"])
            self.assertEqual(len(ledger.sha256()), 64)

    def test_render_is_valid_looking_lean(self) -> None:
        s = _stmt([_real("Φ"), _hyp("h0", "0 ≤ Φ")], "Φ ^ 2 ≤ 1")
        text = s.render(proof="by nlinarith")
        self.assertTrue(
            text.startswith(
                "theorem t (Φ : ℝ) (h0 : 0 ≤ Φ) :\n    Φ ^ 2 ≤ 1 := by nlinarith"
            )
        )


class FreezeTests(unittest.TestCase):
    def test_split_is_disjoint_and_masked(self) -> None:
        self.assertFalse(set(HELDOUT_MODULES) & set(TRAIN_MODULES))
        self.assertFalse(
            set(MASKED_MODULES) & (set(TRAIN_MODULES) | set(HELDOUT_MODULES))
        )

    def test_freeze_finds_downstream_leakage(self) -> None:
        data = freeze(LEAN)
        masked = {entry["module"] for entry in data["downstream_leakage_masked"]}
        self.assertIn("OmegaProtocol", masked)
        self.assertIn("ProofRegression", masked)
        self.assertEqual(
            sum(len(v["statements"]) for v in data["inventory"].values()), 49
        )
        json.dumps(data)  # serializable

    def test_committed_freeze_matches_working_tree(self) -> None:
        path = ROOT / "conjecture_pilot" / "heldout_freeze.json"
        if not path.exists():
            self.skipTest("freeze artifact not generated")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        live = freeze(LEAN)
        for module, entry in frozen["inventory"].items():
            self.assertEqual(
                entry["sha256"], live["inventory"][module]["sha256"], module
            )


if __name__ == "__main__":
    unittest.main()
