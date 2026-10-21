import Mathlib
import EMHodgeStar

/-
  EMConformalStar.lean
  Conformal invariance of the Hodge star on 2-forms — DERIVED (algebra).

  What this module is, and what it is not
  ---------------------------------------
  This is target T3a of `DERIVATION_SCAFFOLD.md`: the algebraic heart of
  the Φ-conformal bridge (item (b) of the scaffold's T3), at exactly the
  scope T1 supports — constant-coefficient 2-forms on Euclidean `ℝ⁴`,
  identified through the ordered orthonormal basis
  `e⁰¹, e⁰², e⁰³, e¹², e¹³, e²³` of `EMHodgeStar.lean`.  Nothing here is
  a statement about smooth forms, and nothing here touches the carried
  Maxwell law: the theorem proved is pure algebra about T1's operator.

  The rescaling.  Fix a constant `c > 0` and rescale the metric
  `g ↦ c²g`.  Three things happen to the coefficient bookkeeping, and
  the whole content of the theorem is that the first is exactly
  cancelled by the other two in the star:

  * Pairing: on 2-form coefficients the induced pairing scales by `c⁴`
    relative to T1's `coeffInner` — this is `coeffInnerScaled` below.
  * Volume: the top-degree (volume-form) coefficient scales by `c⁴` —
    this is the factor in `wedgeTopCoeffScaled` below.
  * Orthonormal basis: each basis 2-form `eⁱ ∧ eʲ` of the rescaled
    metric is `c²` times the flat one (`θⁱ = c·eⁱ` gives
    `θⁱ ∧ θʲ = c² · eⁱ ∧ eʲ`), so the coefficients of a fixed form in
    the rescaled orthonormal basis are `(c²)⁻¹` times its flat
    coefficients, and converting a form back to flat coefficients
    multiplies by `c²`.

  Route taken (the scaffold permits the explicit-formula route):
  `hodgeStarScaled c` is defined explicitly by performing T1's index
  formula in the rescaled orthonormal basis — take the partner
  coefficient in the rescaled basis (a factor `(c²)⁻¹`), apply the
  signed partner formula of `EMHodgeStar.hodgeStarCoeff`, and convert
  back (a factor `c²`).  The rescaled pairing identity
  (`wedgeTopCoeffScaled_eq_coeffInnerScaled`) is then proved for this
  operator, and conformal invariance
  (`hodgeStarScaled_eq_hodgeStar`: `⋆_{c²g} = ⋆_g` for every `c > 0`)
  follows from the cancellation `c² · (c²)⁻¹ = 1` in each coefficient.
  Disposition: **derived** — proved from T1's definitions and field
  algebra alone, with no hypothesis beyond the explicit `0 < c` in the
  statements.
-/

namespace OmegaProtocol.EMConformal

open OmegaProtocol.EMHodge

/-- The coefficient pairing on `TwoFormCoeff` induced by the conformally
    rescaled metric `g ↦ c²g`: on 2-form coefficients in dimension four
    the pairing scales by `c⁴` relative to T1's `coeffInner`.  This is a
    definition of the rescaled bookkeeping, not a theorem about a
    separately constructed metric. -/
def coeffInnerScaled (c : ℝ) (f g : TwoFormCoeff) : ℝ :=
  c ^ 4 * coeffInner f g

/-- The coefficient function of the conformally rescaled Hodge star:
    the coefficient at index `i` is obtained by passing to the rescaled
    orthonormal basis (factor `(c²)⁻¹` on the partner coefficient),
    applying T1's signed partner formula there, and converting back
    (factor `c²`).  See the module header for the route taken. -/
noncomputable def hodgeStarScaledCoeff (c : ℝ) (f : TwoFormCoeff) : TwoFormCoeff :=
  fun i => c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * f i.rev))

/-- **The Hodge star for the conformally rescaled metric `c²g`**, as an
    ℝ-linear map on the constant-coefficient space `TwoFormCoeff`, for a
    constant rescaling factor `c` (the invariance theorem below assumes
    `0 < c`; the operator is defined for every real `c`).  Defined by the
    explicit rescaled-basis formula `hodgeStarScaledCoeff`. -/
noncomputable def hodgeStarScaled (c : ℝ) : TwoFormCoeff →ₗ[ℝ] TwoFormCoeff where
  toFun := hodgeStarScaledCoeff c
  map_add' := by
    intro f g
    funext i
    show c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * (f i.rev + g i.rev)))
        = c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * f i.rev))
          + c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * g i.rev))
    ring
  map_smul' := by
    intro a f
    funext i
    show c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * (a * f i.rev)))
        = a * (c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * f i.rev)))
    ring

/-- Pointwise formula for the rescaled Hodge star, by definition of
    `hodgeStarScaled`. -/
theorem hodgeStarScaled_apply (c : ℝ) (f : TwoFormCoeff) (i : Fin 6) :
    hodgeStarScaled c f i
      = c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * f i.rev)) := rfl

/-- The rescaling factors in `hodgeStarScaled` cancel coefficientwise
    when `0 < c`: the rescaled star's coefficient at index `i` is the
    signed partner coefficient `starSign i * f i.rev`, exactly T1's
    formula.  This is the coefficient-level content of conformal
    invariance; the operator-level statement is
    `hodgeStarScaled_eq_hodgeStar`. -/
theorem hodgeStarScaled_apply_eq (hc : 0 < c) (f : TwoFormCoeff) (i : Fin 6) :
    hodgeStarScaled c f i = starSign i * f i.rev := by
  rw [hodgeStarScaled_apply]
  have hc2 : c ^ 2 ≠ 0 := pow_ne_zero 2 (ne_of_gt hc)
  calc c ^ 2 * (starSign i * ((c ^ 2)⁻¹ * f i.rev))
      = (c ^ 2 * (c ^ 2)⁻¹) * (starSign i * f i.rev) := by ring
    _ = starSign i * f i.rev := by rw [mul_inv_cancel₀ hc2, one_mul]

/-- **Conformal invariance of the Hodge star on 2-forms in dimension
    four (DERIVED)**: for every constant `c > 0`, the Hodge star of the
    rescaled metric `c²g` equals T1's Hodge star of the flat metric,
    `⋆_{c²g} = ⋆_g`, as linear maps on constant-coefficient 2-forms.
    Proved coefficientwise from `hodgeStarScaled_apply_eq` (the
    cancellation `c² · (c²)⁻¹ = 1`) and T1's defining formula
    `hodgeStar_apply`. -/
theorem hodgeStarScaled_eq_hodgeStar (hc : 0 < c) :
    hodgeStarScaled c = hodgeStar := by
  apply LinearMap.ext
  intro f
  funext i
  rw [hodgeStarScaled_apply_eq hc, hodgeStar_apply]

/-- The rescaled star is involutive for `0 < c`, because it equals T1's
    involutive star (`hodgeStarScaled_eq_hodgeStar` and T1's
    `hodgeStar_hodgeStar`). -/
theorem hodgeStarScaled_hodgeStarScaled (hc : 0 < c) (f : TwoFormCoeff) :
    hodgeStarScaled c (hodgeStarScaled c f) = f := by
  rw [hodgeStarScaled_eq_hodgeStar hc]
  exact hodgeStar_hodgeStar f

/-- The top-degree pairing through the rescaled star, with the rescaled
    volume coefficient: the volume-form coefficient of
    `f ∧ ⋆_{c²g} g` in the rescaled metric, i.e. the flat wedge pairing
    of `f` with `⋆_{c²g} g` weighted by the `c⁴` volume scaling. -/
noncomputable def wedgeTopCoeffScaled (c : ℝ) (f g : TwoFormCoeff) : ℝ :=
  ∑ i, c ^ 4 * (starSign i * f i * hodgeStarScaled c g i.rev)

/-- **Rescaled wedge-pairing identity (DERIVED)**: the top-degree
    pairing through the rescaled star equals the rescaled coefficient
    pairing, `wedgeTopCoeffScaled c f g = coeffInnerScaled c f g`, for
    `0 < c`.  Term by term this is T1's `wedgeTopCoeff_eq_coeffInner`
    argument (each `(starSign i)² = 1`) carried through the common `c⁴`
    factor, using `hodgeStarScaled_apply_eq` to evaluate the rescaled
    star at the complementary index. -/
theorem wedgeTopCoeffScaled_eq_coeffInnerScaled (hc : 0 < c)
    (f g : TwoFormCoeff) :
    wedgeTopCoeffScaled c f g = coeffInnerScaled c f g := by
  unfold wedgeTopCoeffScaled coeffInnerScaled coeffInner
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  have h1 : hodgeStarScaled c g i.rev = starSign i * g i := by
    rw [hodgeStarScaled_apply_eq hc, starSign_rev, Fin.rev_rev]
  rw [h1]
  calc c ^ 4 * (starSign i * f i * (starSign i * g i))
      = c ^ 4 * ((starSign i * starSign i) * (f i * g i)) := by ring
    _ = c ^ 4 * (f i * g i) := by rw [starSign_mul_self]; ring

/-- **Volume scaling of the top pairing (DERIVED)**: the rescaled
    top-degree pairing is `c⁴` times T1's flat pairing,
    `wedgeTopCoeffScaled c f g = c⁴ * wedgeTopCoeff f g`, for `0 < c` —
    the `c⁴` volume scaling of the module header, with the star itself
    unchanged.  From `wedgeTopCoeffScaled_eq_coeffInnerScaled` and T1's
    `wedgeTopCoeff_eq_coeffInner`. -/
theorem wedgeTopCoeffScaled_eq_scaling (hc : 0 < c) (f g : TwoFormCoeff) :
    wedgeTopCoeffScaled c f g = c ^ 4 * wedgeTopCoeff f g := by
  rw [wedgeTopCoeffScaled_eq_coeffInnerScaled hc]
  unfold coeffInnerScaled
  rw [wedgeTopCoeff_eq_coeffInner]

end OmegaProtocol.EMConformal
