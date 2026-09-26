<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Conjecture-generation pilot — pre-registration (DRAFT v0.1, awaiting sign-off)

**Status:** draft. The numeric thresholds in §7 are proposals. They become
frozen when this file is tagged `pilot-prereg-v1`; after that tag, nothing in
§2–§7 changes for the duration of the pilot. Anything learned during the pilot
goes into a v2 for the *next* run, never into this one.

**What the pilot is:** a two-week, measurement-first experiment that asks one
question — *does a naive conjecture generator, run over the Omega Lean corpus
and filtered by honest gates, produce a large enough stream of nontrivial,
auditable tasks at a sustainable cost to be worth learning from?*

**What the pilot is not:** a learning system. No weight updates, no adaptive
allocation of generation effort, no strategy changes mid-run, no human
selection of "interesting" candidates. Every source statement receives every
transformation; every candidate receives every applicable gate; in fixed order.

---

## 1. Why this comes first

The problem supply is the bottleneck for any system that is supposed to change
how it reasons rather than what it retrieves. Expert iteration on formal
mathematics worked because it had hundreds of thousands of statements to
attempt; the Omega corpus has 797 theorem declarations, many of them toy, and
perhaps a dozen that matter. Generating *formally valid* tasks is cheap;
generating *useful learning opportunities* is not, and the difference is only
visible once the yield at each gate is measured. This pilot measures it.

## 2. Held-out families (frozen)

Two families, chosen because each has clean formal statements *and* a physics
tag attached in the theory notes, so rung 4 (§4) is decidable:

- **radial**: `RadialDilation.lean` (20 statements), `RadialMetric.lean` (14)
- **entropy**: `Vol16_NonEquilibriumThermodynamics.lean` (10), `Vol21_ArrowOfTime.lean` (5)

Total: **49 statements**, all of which the extractor parses (0 unparsed).
Exact inventory, file SHA-256 hashes and the commit they were frozen at are in
[`heldout_freeze.json`](heldout_freeze.json) (`python -m conjecture_pilot.run_pilot freeze`).
`generate --split heldout` refuses to run if any held-out file hash has changed.

**Training/generation modules** (the ~10-module core; legacy volumes excluded):
`InformationPhysics`, `DynamicPlanckScale`, `DynamicCODScale`,
`LogCorrelationMetric`, `OmegaAxioms`, `CBwK_Budget_Pacer`, `APPA_Context_Branching`
(133 statements, 0 unparsed).

**Leakage policy.**

- *Upstream (allowed, recorded):* held-out modules import `DynamicCODScale`,
  `OmegaAxioms`, `OmegaUnifiedFoundation`. Held-out means the *statements* are
  unseen, not their prerequisites — a student who knows the lemmas but has not
  seen the exam.
- *Downstream (masked):* modules that restate or re-export held-out content are
  excluded from every generator input and every retrieval corpus during
  evaluation. The freeze scan finds three: `OmegaProtocol` (2 re-exports:
  `non_equilibrium_axiom`, `arrow_of_time_axiom`), `ProofRegression` (14
  references in regression examples), `ToE` (4 imports). Without this mask,
  retrieval would simply return the answers.

## 3. Epistemic categories (never merged)

Every ledger record carries exactly one of:

1. **formal** — kernel-checked mathematical consequence;
2. **conjecture** — proposed relationship, not yet verified (all generated candidates start here);
3. **numerical** — result under stated assumptions, with seed, sample counts, tolerances and commit;
4. **physical** — interpretation requiring empirical support.

A successful Lean proof moves a record from 2 to 1. Nothing moves a record to 4
except the rung-4 procedure in §4. This is what prevents a proof from being
read as evidence that the theory is physically correct.

## 4. Discovery ladder and gates

| Rung | Criterion | Decided by | Gate in this scaffold |
|---|---|---|---|
| 0 | kernel-checked | Lean | `wellformed` + a later proof attempt |
| 1 | hypotheses jointly satisfiable, each load-bearing | automated | `numeric` (satisfiability by sampling incl. equation solving; relevance by leave-one-out) — *evidence*, to be confirmed formally |
| 2 | not closable by cheap automation from the existing library (30 s: `rfl`, `norm_num`, `simp`, `positivity`, `linarith`, `nlinarith`, `aesop`, `exact?`) | automated | `not_closable_by_automation` |
| 3 | reused by ≥ 2 later proofs, or shortens an existing one | measured | not in pilot (no proofs are produced) |
| 4 | physically significant: changes a tag in the theory notes ([Identified]→[Derived], [Open]→resolved, canonical claim→refuted) or a pre-registered prediction | blinded human | not in pilot |

**Rung 3 is utility, not discovery.** Discovery claims require rungs 0–2 plus
rung 4. Rung 3 is reported as a separate outcome.

**Numerical falsification runs first** because it is cheap and it yields a
concrete counterexample, which identifies the assumption that a weakened
statement actually needs. It uses Lean's total-function semantics
(`x / 0 = 0`, `Real.sqrt x = 0` for `x < 0`, `Real.log 0 = 0`); getting these
wrong would let candidates pass here and fail in Lean, or the reverse. A
violation counts only if it is confirmed at strict tolerance (hypotheses hold
at 1e-12, conclusion fails by ≥ 1e-6); otherwise it is recorded as
`tolerance_ambiguous` and the verdict is *inconclusive*, not *fail*.

**Lean-unavailable policy.** When `lake` is not on PATH, Lean-backed gates
return `UNAVAILABLE`, never a guess (same policy as
`lean_proofs/leandojo_tactic_harness.py`). Scratch files are written outside
`lean_proofs/` so the CI grep for `sorry` never sees them.

## 5. Transformations (fixed for the pilot)

Text-level and deliberately naive:

- **weaken** — drop one hypothesis; survivors say the hypothesis was decorative (or that Lean's total semantics made it so — see §8, finding F3).
- **converse** — swap one hypothesis with the conclusion.
- **generalize_one** — replace a single literal occurrence in the conclusion by a fresh real `c`.
- **generalize_all** — replace every occurrence of one literal value in hypotheses and conclusion by `c`.

Exponents of `^` are never generalized (it would change the operation).
Boundary analysis, structural transfer and equivalence discovery are *not* in
the pilot; if S1 (§7) fails, they are the next step — not learning.

## 6. Metrics (all computed from the append-only ledger)

1. Candidates generated, per transformation × source module (after de-duplication by statement hash).
2. Pass rate at each gate and the resulting outcome label per candidate: `refuted_numerically`, `hypotheses_unsatisfiable`, `numeric_pass`/`numeric_inconclusive`/`numeric_na`, `ill_formed`, `closable_by_automation`, `open_task`.
3. Formal proof completion — *not in pilot*; recorded as 0 by design.
4. Reuse and proof-shortening — *not in pilot*.
5. Compute cost: wall seconds per gate and per surviving candidate (generator + gate time both charged).
6. Failure reasons (`unparsed` by reason, numeric `not_applicable` by reason, Lean error classes) and per-cell yields — the raw inputs of the learning-progress signal a later system would use. The pilot records them; it does not act on them.
7. Coverage honesty: fraction of candidates with a *definite* numeric verdict, per module.

## 7. Success criterion for the pilot (proposed thresholds — approve before tagging)

The pilot **succeeds** if, with Lean gates live, over the seven training modules:

- **S1 (volume):** ≥ 200 distinct candidates reach `open_task` (well-formed, not numerically refuted, not closable by the 30-second automation set).
- **S2 (nontriviality):** ≥ 50% of `open_task` candidates have every hypothesis either load-bearing or undetermined (none flagged decorative), and hypotheses jointly satisfiable.
- **S3 (auditability):** 100% of candidates carry provenance (source statement, transformation, parameters) and 100% of verdicts carry evidence (counterexample, sample counts, or Lean diagnostics). Enforced by tests, reported anyway.
- **S4 (cost):** median ≤ 60 CPU-seconds per `open_task`, all stages charged, on a laptop.

The pilot **fails** if S1 < 50. Then the finding is "naive transformations do
not supply enough", and the next step is richer transformations, not a learner.
Between 50 and 200 is reported as partial, with the per-cell table.

Nothing above rewards impressive-looking theorem counts: `closable_by_automation`
and `refuted_numerically` are the two largest buckets by construction and are
reported as such.

## 8. Sandbox dry run, 2026-09-26 (Lean unavailable — smoke test, not a result)

Run on commit `356fb8c` (held-out and training files byte-identical to `main`
at `045b5c7`). Full table in [`DRY_RUN_2026-09-26.md`](DRY_RUN_2026-09-26.md).

- Training split: 133 statements → 196 candidates (+45 duplicates). Numeric
  verdicts: 19 refuted, 4 consistent, 173 not applicable. 2.7 s total.
- Held-out split: 49 statements → 164 candidates (+24 duplicates). Numeric
  verdicts: 98 refuted, 16 consistent, 50 not applicable. 31 s total.

Findings worth carrying into the design:

- **F1 — coverage is the first engineering problem, not the learner.** The
  arithmetic-subset falsifier gives a definite verdict on 70% of radial-family
  candidates and on 12% of training-module candidates, because the core
  modules quantify over structures (`Environment`, `CODEnvironment`, `Ledger`,
  `QRegion`) and use `∀` over structure types. Sampling structures from their
  field constraints is the next increment.
- **F2 — the numeric screen is doing real work.** Weakenings of
  `gRR_lpOnly_eq_one_imp` are refuted at `Φ = ±√2`, found by solving the
  equality hypothesis rather than by luck; the converse
  `Φ² ≤ 1 → Φ ≤ 1` passes and its extra hypothesis `0 ≤ Φ` is flagged
  decorative, which is correct.
- **F3 — Lean's total semantics generate "true" weakenings that are not
  physics.** `kappa_outruns_inverse_lp` survives dropping `Φ₁² < 1` because
  outside the disc `√(1−Φ²) = 0` and `x/0 = 0` make the left side vanish. It is
  a kernel-provable statement with no physical content — exactly the class the
  epistemic split in §3 and the rung-4 rule exist to keep out of "discovery".
  A learner rewarded on kernel acceptance would farm these.
- **F4 — de-duplication matters even for naive transformations** (45 and 24
  duplicates); with richer transformations the α-equivalence check becomes
  essential.

## 9. Procedure (two weeks)

- **Day 1:** approve §7 thresholds; tag `pilot-prereg-v1`; record tag + `heldout_freeze.json` hash in a Lab Note.
- **Days 2–3:** run on a machine with the pinned Lean toolchain (`lean_proofs/lean-toolchain`, Mathlib v4.32.0); verify Lean gates on ten hand-picked candidates (five expected well-formed, five expected ill-formed).
- **Days 4–10:** `generate --split train` daily from a clean ledger; snapshot ledger hashes; no code changes to transformations or gates (bug fixes allowed only if they do not change any verdict already recorded — otherwise restart the run).
- **Days 11–12:** `generate --split heldout` once; the resulting ledger is the frozen evaluation pool for any later learning experiment (hash recorded; never regenerated).
- **Days 13–14:** `report`; write `RESULTS.md` in the `rcod/RESULTS.md` format — verdict first, then the table, then findings F1…Fn, then limitations.

## 10. Known limitations of this scaffold

- Transformations are text-level; the extractor is lexical (not a Lean parser). Anything it cannot parse is counted, not skipped silently.
- The numeric evaluator covers real arithmetic with `Real.sqrt`, `Real.exp`, `Real.log`, `|·|`, `^`, the six relations and propositional connectives, plus inlined single-expression `ℝ` defs. Structures, functions, `∀`/`∃`, derivatives and `Monotone`-style predicates are `not_applicable`.
- Numeric PASS is evidence at 2,000 seeded samples in `[-3, 3]` (plus special points); it is not a proof and is labeled `numerical`.
- The rung-2 automation set and its 30-second budget are conventions, not facts about difficulty.
- No learner exists here. That is the point of the pilot.
