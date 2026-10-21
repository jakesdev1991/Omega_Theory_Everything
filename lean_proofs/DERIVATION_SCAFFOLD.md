# EM Derivation Scaffold — Hodge star, codifferential bridge, Φ-conformal bridge

Branch: `claude/em-derivation-scaffold` · Started: 2026-10-21
Toolchain: `leanprover/lean4:v4.32.0` (pinned in `lean-toolchain` / `lakefile.lean`)
Baseline at scaffold creation: `lake build Vol02_Electromagnetism` green on the
branch point (`91dc170`, identical to `main` at creation — see §6).

This document is the design and status record for closing the one openly
carried gap in `Vol02_Electromagnetism.lean`: Mathlib (v4.32.0) provides the
exterior derivative `extDeriv` on differential forms but **no Hodge star on
differential forms**, so the inhomogeneous Maxwell law `d ⋆F = ⋆J` is currently
carried as model-law data (`SourcedField.law_dual_maxwell`) rather than stated
with a proved operator. The scaffold builds the missing operator and the
standard reformulations around it — without pretending the physical law itself
has been derived. The law stays carried; what changes is that its statement
can use a real, proved `⋆` and `δ`, and its equivalence to the codifferential
form `δF = J` can be proved rather than asserted.

Nothing in this document is a claim about physics beyond what the Lean
statements say. Dispositions follow the corpus convention used in
`PROOF_AUDIT.md` / `statement_audit.json`:

- **derived** — proved from definitions / Mathlib, with no extra physical
  hypothesis beyond regularity (e.g. `ContDiff ℝ ∞`) already in the statement;
- **conditional** — proved from explicitly named hypotheses or structure
  fields that appear verbatim in the statement;
- **carried** — still stated as data (a structure field, a postulate
  definition, or a dictionary entry). Carried items are named `law_*` /
  documented as postulates, never dressed as theorems.

---

## 1. Layered design

The scaffold is organised in four layers. A layer may only depend on layers
below it, and crossing a layer boundary upward (using a postulate as if it
were a theorem) is exactly the failure mode the honesty gates exist to catch.

### Layer 0 — Explicit physics postulates / dictionary (carried)

Everything assumed about the physical world, stated once, in the open, as
data. No theorem lives in this layer.

| Item | Where | Status |
|---|---|---|
| Masslessness of the model photon, `PhotonMass := 0` | `Vol02_Electromagnetism.lean` (definition + docstring "Model postulate") | existing, carried |
| Dual Maxwell law `law_dual_maxwell : d starF = starJ` on `SourcedField` | `Vol02_Electromagnetism.lean` | existing, carried — T2 re-expresses it, does not derive it |
| Signature convention for `⋆⋆`: Euclidean `ℝ⁴` (`+` involution on 2-forms) | to be stated in the T1 module docstring | T1 |
| Sign convention for the codifferential `δ = ⋆ d ⋆` on Euclidean `ℝ⁴` | to be stated in the T2 module docstring | T2 |
| `PhiEMDictionary`: at a point, the physical metric is the constant conformal rescaling of the flat metric by a positive factor `c = ℓ_P(Φ)` | new structure, T3 | T3, carried by design — named and docstringed as a postulate layer |

Rule for this layer: a postulate is a `def` with an explicit docstring or a
structure field named as data/`law_*`. It is never a Lean `axiom` (the corpus
has zero; `audit_axioms.py --max 0` enforces this) and never a theorem whose
name suggests a derivation.

### Layer 1 — Mathematical primitives (definitions)

Concrete operators built from Mathlib, with no physical interpretation
baked into the definition beyond the Layer 0 conventions.

| Primitive | Definition | Status |
|---|---|---|
| Exterior derivative `d` | `Vol02.d := extDeriv` on `Form n := Spacetime4 → ContinuousAlternatingMap ℝ Spacetime4 ℝ (Fin n)`, `Spacetime4 := EuclideanSpace ℝ (Fin 4)` | existing |
| Hodge star `⋆` on 2-forms | T1: explicit operator via an orthonormal basis of 2-forms / bivectors over `EuclideanSpace ℝ (Fin 4)` (or, if scoped down, over the finite-dimensional space of constant-coefficient 2-forms — the scope actually achieved must be recorded in §3) | T1 |
| Codifferential `δ` | T2: `δ := ⋆ ∘ d ∘ ⋆` with the Layer 0 sign convention, on smooth forms in the scope T1 supports | T2 |

### Layer 2 — Derived theorems (about the primitives)

Theorems proved from Layer 1 definitions and Mathlib alone. Regularity
hypotheses (`ContDiff`) are part of the statements, as in Vol02 today.

Existing (Vol02, all derived):

- `d_squared_zero` — `d (d η) = 0` for smooth `η` (Mathlib `extDeriv_extDeriv`).
- `contDiff_d`, `contDiff_d_of_contDiff_fderiv` — `d` preserves smoothness.
- `d_add`, `d_smul` — `d` is ℝ-linear on smooth forms.
- `homogeneous_maxwell` — `d F = 0` for `F = d A`, from `d² = 0`.
- `field_strength_gauge_invariant` — `d (A + dχ) = d A`.
- `gauge_transform_comp` — gauge transformations compose additively.

Planned:

- T1: linearity of `⋆` (`map_add` / `map_smul`, or a `LinearMap` structure);
  involution `⋆⋆ = +id` on 2-forms in the Euclidean convention; the induced
  nondegenerate pairing identity if reachable in scope.
- T2: `δ ∘ δ = 0` (from `d² = 0` and the T1 involution); linearity of `δ`.

### Layer 3 — Conditional bridge theorems

Theorems whose statements take a Layer 0 item as an explicit hypothesis or
structure argument. The conclusion is exactly as strong as the hypothesis —
no stronger reading is licensed by the name or docstring.

Existing:

- `inhomogeneous_maxwell` / `maxwell_equations_of_sourced_field`
  (`Vol02_Electromagnetism.lean`) — restate the carried law of a
  `SourcedField`; only the homogeneous conjunct is derived. (Renamed in the
  Phase 6 pass for exactly this reason.)

Planned:

- T2 bridge theorem: for field strength `F` and current 1-form `J` (in T1's
  scope), `d (⋆F) = ⋆J ↔ δF = J`, proved from the T1 involution. The
  *equivalence* is Layer 2 mathematics; its use to rewrite the carried law
  is Layer 3 — the law itself remains carried data.
- T3(c): given a `PhiEMDictionary` value as an explicit hypothesis, the
  vacuum Maxwell system (`dF = 0 ∧ δF = 0`) in the Φ-metric coincides with
  the flat one. T3(b) — `⋆` on 2-forms in dimension 4 is unchanged under
  constant positive conformal rescaling `g ↦ c²g` — is Layer 2 algebra at
  whatever generality T1 supports (full or basis-level; see §3).

---

## 2. Conventions inherited from Vol02 / the corpus

- Namespace `OmegaProtocol.*` (Vol02 uses `OmegaProtocol.Vol02`); new modules
  follow the same scheme and Vol02's header-comment style: what is proved,
  what is carried, and why — stated at the top of the file.
- Form variable is `η`, not `ω` (`open scoped ContDiff` turns `ω` into
  notation — see Vol02).
- Smoothness is always an explicit `ContDiff ℝ ∞` hypothesis, discharged via
  `contDiff_infty` / `contDiff_d` as in Vol02, never assumed silently.
- Any new module that should join the main build is registered in
  `lakefile.lean` roots and imported from `ToE.lean`; if a scaffold module
  is intentionally standalone, the reason is recorded in §3.

---

## 3. Target list and status

| Target | Title | Depends on | Status |
|---|---|---|---|
| T1 | Concrete Hodge star on 2-forms over `ℝ⁴` | — | **pending** |
| T2 | Codifferential bridge for the sourced Maxwell law | T1 | **pending** |
| T3 | Φ-metric conformal bridge (conditional theorems) | T1, T2 (`δ` for T3c) | **pending** |

### T1 — Concrete Hodge star on 2-forms over `ℝ⁴`

Create a new module (suggested name `EMHodgeStar.lean`) defining a concrete
Hodge star operator on alternating 2-forms over `EuclideanSpace ℝ (Fin 4)`
(the `Spacetime4` carrier Vol02 uses), via an explicit orthonormal basis of
2-forms / bivectors. Prove as real theorems (finite basis checks via
extensionality and `norm_num`/`simp` are fine; `native_decide` is banned):

- linearity (`map_add`, `map_smul`, or a `LinearMap` structure);
- the involution identity `⋆⋆ = +id` on 2-forms for the Euclidean signature
  (signature convention stated in the docstring);
- the induced nondegenerate pairing identity, if reachable.

Scope rule: if the full alternating-map generality resists, scope down to
the largest honest version (e.g. `⋆` on the finite-dimensional space of
constant-coefficient 2-forms) and record exactly what was scoped and why
here and in the module docstring. A scoped-down T1 is a success if labelled
as such; an overclaimed T1 is a failure.

### T2 — Codifferential bridge

Using T1's `⋆`, define the codifferential `δ = ⋆ d ⋆` (sign convention for
Euclidean `ℝ⁴` fixed and documented) on smooth forms, in the module where it
fits best (extend `EMHodgeStar.lean` or a new `EMMaxwellBridge.lean`). Prove:

- `δ ∘ δ = 0`, from `d² = 0` and the T1 involution;
- linearity of `δ`;
- the bridge theorem: `d (⋆F) = ⋆J ↔ δF = J` for field strength `F` and
  current 1-form `J`, given `⋆` is the T1 operator with its proved
  involution — i.e. Vol02's carried `SourcedField.law_dual_maxwell` is
  exactly the codifferential Maxwell equation, with the equivalence proved
  rather than asserted.

Explicit non-goal: T2 does **not** derive the law itself. It remains carried
data, re-expressed in standard codifferential form with the seam precisely
located. Defining `J := δF` (or any equivalent smuggling of the conclusion
into a definition) is forbidden. If T1 landed in reduced scope, T2 adapts
to that scope and documents it.

### T3 — Φ-metric conformal bridge (conditional)

Context: the radial-metric work uses the conformal spatial factor
`ℓ_P(Φ)²` (`DynamicCODScale.lean`: `ℓ_P(Φ) = ℓ_P0 · √(1 − Φ²)`;
`RadialMetric.lean` / `RadialDilation.lean`), i.e. `h = ℓ_P(Φ)² · γ`. The
Hodge star on 2-forms in 4 dimensions is conformally invariant — the one
place the Φ-geometry can touch electromagnetism without new physics.

- (a) Define an explicit dictionary structure (e.g. `PhiEMDictionary`)
  whose fields state, **as data**, that the physical metric at a point is
  the constant conformal rescaling of the flat metric by a positive factor
  `c = ℓ_P(Φ)` for the Φ value in question. Postulate layer: named and
  docstringed as such.
- (b) Prove the algebraic theorem: the T1 Hodge star on 2-forms is
  unchanged under constant positive conformal rescaling of the metric
  (`⋆_{c²g} = ⋆_g` on 2-forms in dimension 4), at the level of generality
  T1 supports. If the full statement resists, prove the basis-level
  version and document the gap precisely.
- (c) Prove the conditional consequence: **given** the dictionary (explicit
  hypothesis in every such theorem), the vacuum Maxwell system
  (`dF = 0 ∧ δF = 0`) in the Φ-metric coincides with the flat one.

---

## 4. Honesty rules (binding for every commit on this branch)

From the working brief and `.github/workflows/lean-ci.yml`. A commit that
fails any gate is not a commit — fix it or scope the target down honestly.

1. **Grep gate** — no `sorry`, `admit`, `native_decide`, `sorryAx`,
   `@[extern`, `implemented_by`, `opaque`, `unsafe`, `partial def`,
   `: True :=`, or `:= by trivial` in `.lean` sources.
2. **Zero axioms** — `python3 audit_axioms.py --max 0`.
3. **Vacuity ratchets at zero** —
   `python3 audit_vacuity.py --max-misleading 0 --max-axiom-names 0 --max-unit-stubs 0 --max-unit-types 0 --max-decorative-hypotheses 0 --max-trivial-hypotheses 0 --max-identity-proofs 0 --max-trivial-proofs 0 --alias-baseline legacy_statement_aliases.json`.
4. **Declaration parity + mirrors** — after any declaration change, run
   `python3 check_declaration_parity.py --update` and
   `python3 sync_txt_mirrors.py --write` in the **same** commit.
5. **Tooling tests** — `python3 -m unittest discover -s . -p 'test_*.py'`
   from `lean_proofs/` passes.
6. **Builds** — every commit leaves the repo building (`lake build` of
   every module touched; full `lake build ToE` at integration).
7. **Naming discipline** — a theorem's name and docstring claim only what
   its statement proves. Anything assumed appears as an explicit hypothesis
   or structure field, never hidden. Never rename or reword a claim to
   sound more derived than it is.
8. **Blockers are documented, not routed around** — if a target cannot be
   completed honestly, complete the largest honest sub-part and record the
   exact blocker in §5 and in the module docstring.

Working rules: work only on `claude/em-derivation-scaffold`; never touch
`main`, never merge, never force-push; one commit per completed target with
a descriptive message.

---

## 5. Blocker log

_(empty at scaffold creation — entries are appended, never erased, as
targets are attempted)_

---

## 6. Setup record

- Toolchain verified 2026-10-21: `lake --version` →
  `Lake version 5.0.0-src+8c9756b (Lean version 4.32.0)`;
  `lean --version` → `Lean (version 4.32.0, …)`. `.lake/packages/mathlib4`
  present — no cache download needed.
- Branch `claude/em-derivation-scaffold` already existed at `91dc170`
  ("Add OmegaScienceMinter: PoKW receipt-claim leg for the Sepolia
  pilot"), identical to `main` (`91dc170`, `8819772`, `97b0fc4`); checked
  out, working tree clean, no diff vs `main` before this document.
- Baseline: `lake build Vol02_Electromagnetism` — **green**
  ("Build completed successfully (8658 jobs)", ~26 s, exit 0) before any
  scaffold change. "Main was green" for the module this work extends is
  therefore established at the branch point; a full `lake build ToE`
  baseline is deferred to the integration step.
- Sources read before writing this design: `Vol02_Electromagnetism.lean`
  (full), `lean_proofs/README.md` (proof-honesty policy, passes 1–16),
  `lean_proofs/PROOF_AUDIT.md` (Vol02 inventory row: "Next: a Hodge-star
  development, if Mathlib ever provides exterior-calculus duals" — this
  scaffold builds it concretely instead of waiting), and
  `.github/workflows/lean-ci.yml` (gate sequence in §4).
