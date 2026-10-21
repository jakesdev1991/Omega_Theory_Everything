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
| Signature convention for `⋆⋆`: Euclidean `ℝ⁴` (`+` involution on 2-forms) | stated in the `EMHodgeStar.lean` module header (Euclidean `(+, +, +, +)`, orientation `e⁰ ∧ e¹ ∧ e² ∧ e³`) | T1, done |
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
| Hodge star `⋆` on 2-forms | T1: explicit operator via an orthonormal basis of 2-forms / bivectors over `EuclideanSpace ℝ (Fin 4)` (or, if scoped down, over the finite-dimensional space of constant-coefficient 2-forms — the scope actually achieved must be recorded in §3) | T1, done at the scoped-down level: `EMHodgeStar.hodgeStar`, a `LinearMap` on the coefficient space `Fin 6 → ℝ` of constant-coefficient 2-forms (see §3) |
| Codifferential `δ` | T2: `δ := ⋆ ∘ d ∘ ⋆` with the Layer 0 sign convention, on smooth forms in the scope T1 supports | T2, done conditionally: `EMBridge.codifferential` / `codifferentialOne`, defined relative to a `SmoothStarExtension` value (the smooth star itself is a hypothesis field, not a construction — see the T2 outcome note) |

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

- T1 (done, scoped as recorded in §3): linearity of `⋆` (the `LinearMap`
  `EMHodgeStar.hodgeStar`, restated as `hodgeStar_add` / `hodgeStar_smul`);
  involution `⋆⋆ = +id` on 2-forms in the Euclidean convention
  (`hodgeStar_hodgeStar`, `hodgeStar_involutive`, `hodgeStar_bijective`);
  the induced pairing identity (`wedgeTopCoeff_eq_coeffInner`: the
  volume-form coefficient of `f ∧ ⋆g` equals the coefficient inner
  product), inner-product preservation (`coeffInner_hodgeStar`), and
  pairing nondegeneracy (`eq_zero_of_wedgeTopCoeff_eq_zero_all`).
- T2 (done, conditional on a `SmoothStarExtension` — the `d² = 0` input is
  Vol02's derived theorem, the star inputs are the structure's named
  fields): `δ ∘ δ = 0` (`EMBridge.codifferentialOne_codifferential`);
  linearity of `δ` (`codifferential_add`, `codifferential_smul`).

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

- T2 bridge theorem (done, conditional): for field strength `F` and current
  1-form `J`, relative to a `SmoothStarExtension` `E`,
  `d (⋆F) = ⋆J ↔ δF = J` (`EMBridge.dual_maxwell_iff_codifferential`),
  proved from the extension's cross-degree inverse fields (the smooth
  analogue of the T1 involution); and
  `EMBridge.carried_law_iff_codifferential` states that *if* a
  `SourcedField`'s carried duals are `E`'s stars, its carried law is
  exactly `δF = J`. The *equivalence* is mathematics; its use to rewrite
  the carried law is Layer 3 — the law itself remains carried data, and
  no `SmoothStarExtension` value is constructed anywhere.
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
| T1 | Concrete Hodge star on 2-forms over `ℝ⁴` | — | **done (scoped)** — `EMHodgeStar.lean`: star as a `LinearMap` on the constant-coefficient coefficient space `Fin 6 → ℝ`; linearity, Euclidean involution `⋆⋆ = +id`, basis actions, pairing identity and nondegeneracy all proved. Scope limit recorded below. |
| T2 | Codifferential bridge for the sourced Maxwell law | T1 | **done (conditional, level B)** — `EMMaxwellBridge.lean`: `SmoothStarExtension` hypothesis structure extending T1's star to smooth forms; relative to it, `δ` defined, linearity and `δ² = 0` proved, and the bridge equivalence `d(⋆F) = ⋆J ↔ δF = J` proved. The law itself remains carried. See the T2 outcome note below §3's T2 spec. |
| T3 | Φ-metric conformal bridge (conditional theorems) | T1, T2 (`δ` for T3c) | **in progress** — T3a (the algebraic core, item (b)) done (derived, see outcome note below); the dictionary (a) and the vacuum consequence (c) are pending |

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

**T1 outcome (2026-10-21): done at the scoped-down level.** The operator
landed as `OmegaProtocol.EMHodge.hodgeStar : TwoFormCoeff →ₗ[ℝ]
TwoFormCoeff` in `EMHodgeStar.lean`, where `TwoFormCoeff := Fin 6 → ℝ` is
the space of constant-coefficient 2-forms on Euclidean `ℝ⁴` identified
through the ordered orthonormal basis `e⁰¹, e⁰², e⁰³, e¹², e¹³, e²³`.
Partner indices are `i ↔ i.rev` with sign `-1` exactly on indices `1, 4`.
Proved: the `LinearMap` linearity fields (restated as `hodgeStar_add` /
`hodgeStar_smul`), `hodgeStar_hodgeStar` (`⋆⋆ = +id`, Euclidean
signature `(+, +, +, +)` stated in the module header), involutivity and
bijectivity, all six basis actions (`hodgeStar_apply_zero` …
`hodgeStar_apply_five`, plus the basis-vector witness
`hodgeStar_single_five`), inner-product preservation
(`coeffInner_hodgeStar`), the wedge-pairing identity
(`wedgeTopCoeff_eq_coeffInner`), and pairing nondegeneracy
(`eq_zero_of_wedgeTopCoeff_eq_zero_all`).

**What was scoped, and why:** the operator is *not* yet defined on
arbitrary point-dependent alternating maps (`Vol02.Form 2` values, i.e.
`ContinuousAlternatingMap ℝ Spacetime4 ℝ (Fin 2)` at a point, let alone
smooth form fields). Mathlib v4.32.0 provides no basis / volume-form /
alternating-map Hodge infrastructure to lift a coefficient formula to
that generality without first building that infrastructure, which is a
larger development than T1. The constant-coefficient space is the
largest scope in which the star, its involution, and its pairing are
proved here without new unproved infrastructure; the module header
states this scope verbatim. Lifting `⋆` pointwise to `Form 2` is the
recorded next step (blocker log, §5), and T2 must be stated in this
coefficient scope or wait for that lift — it must not silently use
`hodgeStar` as if it acted on all smooth forms.

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

**T2 outcome (2026-10-21): done at the conditional level (level B).**
Level A — an unconditional codifferential from a coefficient-level
exterior derivative — is not honestly available: the only coefficient-level
derivative compatible with Vol02's smooth forms vanishes on
constant-coefficient forms (a constant field has zero derivative), so a
coefficient-level `δ` would be identically zero and the bridge would
degenerate. Level B landed as `EMMaxwellBridge.lean`
(namespace `OmegaProtocol.EMBridge`): the structure
`SmoothStarExtension` packages the smooth extension as explicit named
fields — ℝ-linear stars `star22`, `star14`, `star32`, `star04`, the
constant-coefficient embedding `const2`, the restriction field
`star22_const` (agreement with T1's proved `hodgeStar` on constants),
`star22_involutive`, the cross-degree inverse fields `star32_star14` /
`star14_star32`, and `smooth_star22`. Relative to any such `E`:
`codifferential E F = star32 (d (star22 F))` (Euclidean sign `+1`,
i.e. `δ = ⋆ d ⋆` on the nose), with `codifferential_add` /
`codifferential_smul` (linearity), `codifferentialOne_codifferential`
(`δ₁ ∘ δ₂ = 0`, from Vol02's derived `d² = 0` plus the named fields),
`dual_maxwell_iff_codifferential` (`d(⋆₂F) = ⋆₁J ↔ δF = J`), and
`carried_law_iff_codifferential` (a `SourcedField` whose carried duals
are `E`'s stars satisfies its carried law iff `δF = J`). **Dispositions:
every T2 theorem is conditional on the `SmoothStarExtension` argument;
`sourced law` remains carried.** No value of `SmoothStarExtension` is
constructed — constructing one is exactly the pointwise-lift blocker in
§5.

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

**T3a outcome (2026-10-21): item (b) done, DERIVED, at T1's coefficient
scope.** `EMConformalStar.lean` (namespace `OmegaProtocol.EMConformal`)
defines, for a constant `c`, the rescaled pairing
`coeffInnerScaled c f g = c⁴ * coeffInner f g` and the rescaled star
`hodgeStarScaled c` by the explicit-formula route: T1's signed partner
formula performed in the rescaled orthonormal basis — partner
coefficient taken with the factor `(c²)⁻¹` (each basis 2-form of the
rescaled metric `c²g` is `c²` times the flat one), converted back with
the factor `c²`. Proved: the coefficientwise cancellation
`hodgeStarScaled_apply_eq` (`c² · (c²)⁻¹ = 1`, for `0 < c`), the main
theorem `hodgeStarScaled_eq_hodgeStar` (`⋆_{c²g} = ⋆_g` for every
`c > 0`, as `LinearMap`s on `TwoFormCoeff`), involutivity of the
rescaled star, the rescaled pairing identity
`wedgeTopCoeffScaled_eq_coeffInnerScaled`, and the volume scaling
`wedgeTopCoeffScaled_eq_scaling` (the top pairing scales by exactly
`c⁴`, the star itself unchanged). Route note: the from-characterization
definition was not used; the explicit formula plus the proved rescaled
pairing identity is the route taken, as §T3(b)'s fallback permits.
Items (a) (the `PhiEMDictionary` postulate structure) and (c) (the
conditional vacuum-Maxwell consequence) remain pending.

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

- **T1 scope limit (2026-10-21, not a failure — the §3 scope rule applied):**
  `EMHodgeStar.hodgeStar` acts on constant-coefficient 2-forms via their
  six coefficients in an explicit orthonormal basis, not on arbitrary
  `ContinuousAlternatingMap` values or smooth `Vol02.Form 2` fields.
  Mathlib v4.32.0 has no Hodge star, volume form, or orthonormal-basis
  infrastructure for alternating maps, so a pointwise lift requires
  building that infrastructure first. Until that lift exists, T2's
  codifferential must be scoped to constant-coefficient forms (or to the
  coefficient level) and must say so; using T1's operator as if it acted
  on all smooth forms would be an overclaim.
- **T2 level taken (2026-10-21):** T2 was completed at the conditional
  level (level B in the working brief) rather than the coefficient level,
  for the reason recorded in the T2 outcome note: a coefficient-level
  exterior derivative compatible with Vol02 vanishes on constants, making
  a coefficient-level `δ` identically zero. The remaining blocker is
  unchanged and precise: **no `SmoothStarExtension` value exists yet** —
  constructing one requires the pointwise lift of `⋆` to smooth
  `Vol02.Form` values (alternating-map basis / volume-form infrastructure
  absent from Mathlib v4.32.0). Until that lift is built, every T2
  statement stays conditional and the sourced Maxwell law stays carried.

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
