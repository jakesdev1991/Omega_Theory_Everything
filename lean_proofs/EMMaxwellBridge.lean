import Mathlib
import EMHodgeStar
import Vol02_Electromagnetism

/-
  EMMaxwellBridge.lean
  The codifferential bridge for the sourced Maxwell law — CONDITIONAL level.

  What this module is, and what it is not
  ---------------------------------------
  T1 (`EMHodgeStar.lean`) delivered the Hodge star only at the
  constant-coefficient level: `EMHodge.hodgeStar` is a `LinearMap` on the
  six-dimensional coefficient space `TwoFormCoeff` of constant-coefficient
  2-forms on Euclidean `ℝ⁴`.  It is *not* an operator on smooth
  `Vol02.Form 2` fields, and Mathlib (v4.32.0) provides no pointwise star
  on smooth forms to bridge the gap.

  Why there is no unconditional codifferential here: a coefficient-level
  exterior derivative compatible with `Vol02`'s smooth forms would have
  to vanish on every constant-coefficient form (a constant field has zero
  derivative), so a coefficient-level `δ = ⋆ d ⋆` is identically zero and
  the bridge equivalence degenerates.  Building the genuine pointwise
  lift of `⋆` to smooth forms is the recorded blocker (see
  `DERIVATION_SCAFFOLD.md` §5).  What *can* be proved honestly today is
  the conditional statement, and that is what this module proves:

  * `SmoothStarExtension` (below) packages, as explicit named fields,
    exactly what a smooth extension of T1's star would have to provide:
    ℝ-linear stars on smooth forms of the degrees the Maxwell
    bookkeeping uses (`⋆₂` on 2-forms, `⋆₁ : 1-forms → 3-forms`,
    its inverse `⋆₃ : 3-forms → 1-forms`, and `⋆₄ : 4-forms → 0-forms`),
    the requirement that `⋆₂` restricts to T1's `hodgeStar` on constant
    coefficients, the cross-degree inverse identities, the pointwise
    involution of `⋆₂`, and preservation of smoothness by `⋆₂`.  Every
    one of these is a *hypothesis field*, not a theorem: no value of
    this structure is constructed in this module, and none is claimed
    to exist.

  * Relative to such an extension `E`, the codifferential on 2-forms is
    *defined* as `δ F = ⋆₃ (d (⋆₂ F))` — the Euclidean sign convention
    `δ = ⋆ d ⋆` with sign `+1`, matching T1's Euclidean involution
    `⋆⋆ = +id` (no Minkowski sign is used anywhere).  From the named
    hypotheses alone this module proves: linearity of `δ`, `δ² = 0`
    (the composite `δ₁ ∘ δ₂`, using Vol02's derived `d² = 0`), and the
    bridge equivalence
        `d (⋆₂ F) = ⋆₁ J  ↔  δ F = J`.

  * Disposition: every theorem here is **conditional** on the explicit
    `SmoothStarExtension` argument in its statement.  In particular,
    Vol02's carried law `SourcedField.law_dual_maxwell` is **not**
    derived by anything in this file, and it remains carried data.  The
    proved content is exactly: *if* a smooth extension as packaged
    exists *and* a sourced field's carried duals are that extension's
    stars (`carried_law_iff_codifferential` below), *then* the carried
    law holds if and only if the codifferential equation `δ F = J`
    holds.
-/

namespace OmegaProtocol.EMBridge

open OmegaProtocol.Vol02
open scoped ContDiff

/-- **Extension hypotheses for a smooth Hodge star (data, not a
    construction).**  A value of this structure would be a pointwise
    Hodge star on smooth `Vol02` forms extending T1's coefficient-level
    `EMHodge.hodgeStar`.  No value is constructed in this module; the
    fields are the explicitly named hypotheses of every conditional
    theorem below.

    Degree bookkeeping (spacetime dimension four): `star22` is the star
    on 2-forms; `star14` sends a current 1-form `J` to its dual 3-form
    `⋆J`; `star32` is its inverse, sending a 3-form back to a 1-form;
    `star04` sends a 4-form to a 0-form (used only for the second
    codifferential in `δ² = 0`).  All stars are ℝ-linear maps. -/
structure SmoothStarExtension where
  /-- The star on smooth 2-forms. -/
  star22 : Form 2 →ₗ[ℝ] Form 2
  /-- The star sending smooth 1-forms to smooth 3-forms. -/
  star14 : Form 1 →ₗ[ℝ] Form 3
  /-- The star sending smooth 3-forms back to smooth 1-forms. -/
  star32 : Form 3 →ₗ[ℝ] Form 1
  /-- The star sending smooth 4-forms to smooth 0-forms. -/
  star04 : Form 4 →ₗ[ℝ] Form 0
  /-- The constant-coefficient embedding of T1's coefficient space into
      smooth 2-forms: `const2 c` is the constant 2-form field whose
      coefficients in T1's ordered basis are `c`. -/
  const2 : EMHodge.TwoFormCoeff →ₗ[ℝ] Form 2
  /-- Restriction hypothesis: on constant-coefficient 2-forms, `star22`
      agrees with T1's proved coefficient-level star `hodgeStar`. -/
  star22_const : ∀ c : EMHodge.TwoFormCoeff,
    star22 (const2 c) = const2 (EMHodge.hodgeStar c)
  /-- Involution hypothesis: `star22` is pointwise involutive on smooth
      2-forms, extending T1's proved `⋆⋆ = +id` on coefficients. -/
  star22_involutive : ∀ F : Form 2, star22 (star22 F) = F
  /-- Cross-degree inverse hypothesis: `star32` undoes `star14`. -/
  star32_star14 : ∀ J : Form 1, star32 (star14 J) = J
  /-- Cross-degree inverse hypothesis: `star14` undoes `star32`. -/
  star14_star32 : ∀ η : Form 3, star14 (star32 η) = η
  /-- Smoothness hypothesis: `star22` sends smooth 2-forms to smooth
      2-forms, so Vol02's `d² = 0` applies after starring. -/
  smooth_star22 : ∀ F : Form 2, ContDiff ℝ ∞ F → ContDiff ℝ ∞ (star22 F)

/-- **The codifferential on smooth 2-forms, relative to an extension
    `E`**: `δ F = ⋆₃ (d (⋆₂ F))`, with `d` Vol02's exterior derivative.
    Euclidean sign convention: the sign is `+1`, i.e. `δ = ⋆ d ⋆` on the
    nose, matching the Euclidean involution `⋆⋆ = +id` proved in T1. -/
noncomputable def codifferential (E : SmoothStarExtension) (F : Form 2) :
    Form 1 :=
  E.star32 (Vol02.d (E.star22 F))

/-- **The codifferential on smooth 1-forms, relative to an extension
    `E`**: `δ G = ⋆₀ (d (⋆₁ G))`, same Euclidean sign convention as
    `codifferential`.  This is the second factor in the composite
    `δ² = δ₁ ∘ δ₂` below. -/
noncomputable def codifferentialOne (E : SmoothStarExtension)
    (G : Form 1) : Form 0 :=
  E.star04 (Vol02.d (E.star14 G))

/-- Conditional linearity of the codifferential, additivity: relative to
    an extension `E`, `δ (F + G) = δ F + δ G` for smooth `F`, `G`.  Uses
    the linearity fields of `E` and Vol02's `d_add`; the smoothness
    hypotheses are exactly what `d_add` requires after starring. -/
theorem codifferential_add (E : SmoothStarExtension) {F G : Form 2}
    (hF : ContDiff ℝ ∞ F) (hG : ContDiff ℝ ∞ G) :
    codifferential E (F + G) = codifferential E F + codifferential E G := by
  unfold codifferential
  rw [map_add, Vol02.d_add (E.smooth_star22 F hF) (E.smooth_star22 G hG), map_add]

/-- Conditional linearity of the codifferential, scalar compatibility:
    relative to an extension `E`, `δ (c • F) = c • δ F`.  Uses the
    linearity fields of `E` and Vol02's `d_smul`. -/
theorem codifferential_smul (E : SmoothStarExtension) (c : ℝ)
    (F : Form 2) :
    codifferential E (c • F) = c • codifferential E F := by
  unfold codifferential
  rw [map_smul, Vol02.d_smul, map_smul]

/-- **Conditional `δ² = 0`**: relative to an extension `E`, the composite
    codifferential `δ₁ (δ₂ F)` is zero for every smooth 2-form `F`.
    Proof: the middle stars cancel by the named hypothesis
    `star14_star32`, leaving `⋆₀ (d (d (⋆₂ F)))`; the inner double
    derivative vanishes by Vol02's derived `d_squared_zero`, applicable
    because `star22` preserves smoothness by the named hypothesis
    `smooth_star22`; a linear map sends `0` to `0`. -/
theorem codifferentialOne_codifferential (E : SmoothStarExtension)
    {F : Form 2} (hF : ContDiff ℝ ∞ F) :
    codifferentialOne E (codifferential E F) = 0 := by
  unfold codifferentialOne codifferential
  rw [E.star14_star32, Vol02.d_squared_zero _ (E.smooth_star22 F hF), map_zero]

/-- **Conditional bridge equivalence**: relative to an extension `E`,
    for a field strength `F` (2-form) and a current `J` (1-form),
    the dual form of the inhomogeneous Maxwell law, `d (⋆₂ F) = ⋆₁ J`,
    holds if and only if the codifferential equation `δ F = J` holds.
    Both directions are the cross-degree inverse hypotheses
    `star32_star14` / `star14_star32` applied to the two sides.  This
    equivalence is conditional mathematics about `E`; it does not
    assert that the law itself holds for any `F`, `J`. -/
theorem dual_maxwell_iff_codifferential (E : SmoothStarExtension)
    (F : Form 2) (J : Form 1) :
    Vol02.d (E.star22 F) = E.star14 J ↔ codifferential E F = J := by
  constructor
  · intro h
    have h2 := congrArg E.star32 h
    rwa [E.star32_star14] at h2
  · intro h
    have h2 := congrArg E.star14 h
    unfold codifferential at h2
    rwa [E.star14_star32] at h2

/-- **Conditional re-expression of Vol02's carried law**: let `S` be a
    `SourcedField` whose carried duals happen to be an extension `E`'s
    stars of some `F` and `J` (the two named hypotheses `hF`, `hJ`).
    Then `S`'s carried law `d S.starF = S.starJ` holds if and only if
    the codifferential equation `δ F = J` holds relative to `E`.

    This theorem does **not** derive the carried law: `S.law_dual_maxwell`
    remains carried data in Vol02, and the hypotheses `hF`, `hJ` — that
    the carried duals are given by a smooth star extension at all — are
    exactly the unproved seam, stated here in the open. -/
theorem carried_law_iff_codifferential (E : SmoothStarExtension)
    (S : SourcedField) (F : Form 2) (J : Form 1)
    (hF : S.starF = E.star22 F) (hJ : S.starJ = E.star14 J) :
    (Vol02.d S.starF = S.starJ) ↔ codifferential E F = J := by
  rw [hF, hJ]
  exact dual_maxwell_iff_codifferential E F J

end OmegaProtocol.EMBridge
