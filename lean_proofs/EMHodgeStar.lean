import Mathlib
import Vol02_Electromagnetism

/-
  EMHodgeStar.lean
  A concrete Hodge star on constant-coefficient 2-forms over Euclidean `ℝ⁴`.

  What this module is, and what it is not
  ---------------------------------------
  `Vol02_Electromagnetism.lean` carries the inhomogeneous Maxwell law as
  model-law data because Mathlib (v4.32.0) has no Hodge star on differential
  forms.  This module builds the missing operator at the level that can be
  built honestly from Mathlib today, and labels that level exactly:

  * A constant-coefficient alternating 2-form on `ℝ⁴` is determined by its
    six coefficients in the ordered orthonormal basis of 2-forms
      `e⁰¹, e⁰², e⁰³, e¹², e¹³, e²³`
    indexed by `Fin 6` in that order (superscripts are coordinate indices
    `0, 1, 2, 3` of `Spacetime4 = EuclideanSpace ℝ (Fin 4)`).
    `TwoFormCoeff := Fin 6 → ℝ` is that coefficient space.  It is the space
    of *constant-coefficient* 2-forms, identified through this basis — not
    the space of all (point-dependent) `Vol02.Form 2` values.  Extending
    the operator pointwise to smooth `Form 2` values is future work and is
    recorded as the T1 scope limit in `DERIVATION_SCAFFOLD.md`.

  * Signature convention: the metric is the Euclidean metric of
    `EuclideanSpace ℝ (Fin 4)`, signature `(+, +, +, +)`, with orientation
    `e⁰ ∧ e¹ ∧ e² ∧ e³`.  In this convention the Hodge star on 2-forms in
    dimension four squares to `+1`, and that is the involution proved
    below.  No Minkowski-signature sign is claimed anywhere in this file.

  * In the basis above, with the stated orientation, the star acts by
      `⋆e⁰¹ = e²³`,  `⋆e⁰² = -e¹³`,  `⋆e⁰³ = e¹²`,
      `⋆e¹² = e⁰³`,  `⋆e¹³ = -e⁰²`,  `⋆e²³ = e⁰¹`.
    In coefficient-index terms the partner of index `i` is `i.rev`
    (pairing `0 ↔ 5`, `1 ↔ 4`, `2 ↔ 3`) and the sign is `-1` exactly on
    indices `1` and `4`.  The operator below is that formula, as a genuine
    `LinearMap`, and the file proves: linearity (the `LinearMap` fields,
    restated as `hodgeStar_add` / `hodgeStar_smul`), the involution
    `⋆⋆ = id` (`hodgeStar_hodgeStar`, `hodgeStar_involutive`), bijectivity,
    the six basis actions, preservation of the coefficient inner product,
    and the wedge-pairing identity (the volume-form coefficient of
    `f ∧ ⋆g` equals the coefficient inner product `Σᵢ fᵢ gᵢ`), from which
    the pairing nondegeneracy corollary is proved.

  Nothing here derives the inhomogeneous Maxwell law.  That law stays
  carried data in Vol02; this module only supplies the operator its
  statement was missing, at the scope stated above.
-/

namespace OmegaProtocol.EMHodge

/-- The coefficient space of constant-coefficient 2-forms on Euclidean
    `ℝ⁴`, in the ordered orthonormal basis
    `e⁰¹, e⁰², e⁰³, e¹², e¹³, e²³` (see the module header for the scope:
    constant-coefficient forms only, identified through this basis). -/
abbrev TwoFormCoeff := Fin 6 → ℝ

/-- The sign of the Hodge star on the basis 2-form at coefficient index
    `i`: `-1` exactly for the `e⁰²` and `e¹³` basis elements (indices `1`
    and `4`), `+1` otherwise — see the module header for the full table. -/
def starSign (i : Fin 6) : ℝ := if i = 1 ∨ i = 4 then -1 else 1

/-- The Hodge star as a plain coefficient function (the `LinearMap`
    `hodgeStar` below wraps exactly this function): the coefficient at
    index `i` is the signed coefficient at the partner index `i.rev`. -/
def hodgeStarCoeff (f : TwoFormCoeff) : TwoFormCoeff :=
  fun i => starSign i * f i.rev

/-- **The concrete Hodge star on constant-coefficient 2-forms over
    Euclidean `ℝ⁴`**, as an ℝ-linear map on the coefficient space
    `TwoFormCoeff`.  Euclidean signature `(+, +, +, +)` and the orientation
    of the module header are the conventions in force. -/
noncomputable def hodgeStar : TwoFormCoeff →ₗ[ℝ] TwoFormCoeff where
  toFun := hodgeStarCoeff
  map_add' := by
    intro f g
    funext i
    show starSign i * (f i.rev + g i.rev)
        = starSign i * f i.rev + starSign i * g i.rev
    ring
  map_smul' := by
    intro c f
    funext i
    show starSign i * (c * f i.rev) = c * (starSign i * f i.rev)
    ring

/-- Pointwise formula for the Hodge star, by definition of `hodgeStar`. -/
theorem hodgeStar_apply (f : TwoFormCoeff) (i : Fin 6) :
    hodgeStar f i = starSign i * f i.rev := rfl

/-- Linearity of the Hodge star, additivity: `⋆(f + g) = ⋆f + ⋆g`.  This
    is the `map_add` field of the `LinearMap` `hodgeStar`. -/
theorem hodgeStar_add (f g : TwoFormCoeff) :
    hodgeStar (f + g) = hodgeStar f + hodgeStar g :=
  map_add hodgeStar f g

/-- Linearity of the Hodge star, scalar compatibility:
    `⋆(c • f) = c • ⋆f`.  This is the `map_smul` field of the `LinearMap`
    `hodgeStar`. -/
theorem hodgeStar_smul (c : ℝ) (f : TwoFormCoeff) :
    hodgeStar (c • f) = c • hodgeStar f :=
  map_smul hodgeStar c f

/-- The star sign is unchanged by passing to the partner index:
    `starSign i.rev = starSign i`, checked on all six indices. -/
theorem starSign_rev (i : Fin 6) : starSign i.rev = starSign i := by
  fin_cases i <;> simp [starSign]

/-- Every star sign squares to one: `starSign i * starSign i = 1`,
    checked on all six indices. -/
theorem starSign_mul_self (i : Fin 6) : starSign i * starSign i = 1 := by
  fin_cases i <;> simp [starSign]

/-- **Involution of the Euclidean Hodge star on 2-forms**:
    `⋆(⋆f) = f` for every constant-coefficient 2-form `f`.  The sign is
    `+1` because the signature is Euclidean `(+, +, +, +)` and
    `(-1)^{2·(4-2)} = 1`; no Minkowski-signature statement is made. -/
theorem hodgeStar_hodgeStar (f : TwoFormCoeff) :
    hodgeStar (hodgeStar f) = f := by
  funext i
  show starSign i * (starSign i.rev * f i.rev.rev) = f i
  rw [starSign_rev, Fin.rev_rev]
  calc starSign i * (starSign i * f i)
      = (starSign i * starSign i) * f i := by ring
    _ = f i := by rw [starSign_mul_self]; ring

/-- The Hodge star is involutive as a function: `Function.Involutive`. -/
theorem hodgeStar_involutive : Function.Involutive hodgeStar :=
  fun f => hodgeStar_hodgeStar f

/-- The Hodge star is bijective, because it is its own inverse. -/
theorem hodgeStar_bijective : Function.Bijective hodgeStar :=
  hodgeStar_involutive.bijective

/-- Action on the basis coefficient at index `0` (`e⁰¹ ↦ e²³`). -/
theorem hodgeStar_apply_zero (f : TwoFormCoeff) :
    hodgeStar f 0 = f 5 := by
  rw [hodgeStar_apply]
  simp [starSign]

/-- Action on the basis coefficient at index `1` (`e⁰² ↦ -e¹³`). -/
theorem hodgeStar_apply_one (f : TwoFormCoeff) :
    hodgeStar f 1 = - f 4 := by
  rw [hodgeStar_apply]
  simp [starSign]

/-- Action on the basis coefficient at index `2` (`e⁰³ ↦ e¹²`). -/
theorem hodgeStar_apply_two (f : TwoFormCoeff) :
    hodgeStar f 2 = f 3 := by
  rw [hodgeStar_apply]
  simp [starSign]

/-- Action on the basis coefficient at index `3` (`e¹² ↦ e⁰³`). -/
theorem hodgeStar_apply_three (f : TwoFormCoeff) :
    hodgeStar f 3 = f 2 := by
  rw [hodgeStar_apply]
  simp [starSign]

/-- Action on the basis coefficient at index `4` (`e¹³ ↦ -e⁰²`). -/
theorem hodgeStar_apply_four (f : TwoFormCoeff) :
    hodgeStar f 4 = - f 1 := by
  rw [hodgeStar_apply]
  simp [starSign]

/-- Action on the basis coefficient at index `5` (`e²³ ↦ e⁰¹`). -/
theorem hodgeStar_apply_five (f : TwoFormCoeff) :
    hodgeStar f 5 = f 0 := by
  rw [hodgeStar_apply]
  simp [starSign]

/-- The Hodge star sends the `e²³` basis vector (coefficient `1` at index
    `5`, `0` elsewhere) to the `e⁰¹` basis vector.  A concrete witness
    that the operator is not the zero operator. -/
theorem hodgeStar_single_five :
    hodgeStar (Pi.single (5 : Fin 6) (1 : ℝ)) = Pi.single (0 : Fin 6) 1 := by
  funext i
  fin_cases i <;>
    simp [hodgeStar_apply, starSign]

/-- The coefficient inner product on `TwoFormCoeff`:
    `⟨f, g⟩ = Σᵢ fᵢ gᵢ` over the six basis coefficients. -/
def coeffInner (f g : TwoFormCoeff) : ℝ := ∑ i, f i * g i

/-- The star of `g`, evaluated at the complementary index of `i`, is the
    signed `i`-th coefficient of `g`: `(⋆g)ᵢᵣₑᵥ = starSign i * gᵢ`. -/
theorem hodgeStar_rev_apply (g : TwoFormCoeff) (i : Fin 6) :
    hodgeStar g i.rev = starSign i * g i := by
  rw [hodgeStar_apply, starSign_rev, Fin.rev_rev]

/-- The top-degree pairing of two constant-coefficient 2-forms through
    the star.  The wedge product `f ∧ ⋆g` pairs the `i`-th coefficient of
    `f` with the *complementary* component of `⋆g`, weighted by the sign
    of the permutation that orders the two index pairs — and that
    permutation sign is exactly `starSign i` (see the module header's
    table).  So the volume-form coefficient of `f ∧ ⋆g` is
    `Σᵢ starSign i * fᵢ * (⋆g)ᵢᵣₑᵥ`, which is this definition. -/
noncomputable def wedgeTopCoeff (f g : TwoFormCoeff) : ℝ :=
  ∑ i, starSign i * f i * hodgeStar g i.rev

/-- **The star is an isometry of the coefficient inner product**:
    `⟨⋆f, ⋆g⟩ = ⟨f, g⟩`, by expansion over the six basis coefficients
    (each sign squares to `1` and the partner pairing is a permutation). -/
theorem coeffInner_hodgeStar (f g : TwoFormCoeff) :
    coeffInner (hodgeStar f) (hodgeStar g) = coeffInner f g := by
  unfold coeffInner
  rw [Fin.sum_univ_six, Fin.sum_univ_six]
  simp only [hodgeStar_apply_zero, hodgeStar_apply_one, hodgeStar_apply_two,
    hodgeStar_apply_three, hodgeStar_apply_four, hodgeStar_apply_five]
  ring

/-- **Wedge-pairing identity**: the top-degree pairing through the star
    equals the coefficient inner product.  Term by term,
    `(⋆g)ᵢᵣₑᵥ = starSign i * gᵢ` (`hodgeStar_rev_apply`), so the `i`-th
    summand is `(starSign i)² * fᵢ * gᵢ = fᵢ * gᵢ`. -/
theorem wedgeTopCoeff_eq_coeffInner (f g : TwoFormCoeff) :
    wedgeTopCoeff f g = coeffInner f g := by
  unfold wedgeTopCoeff coeffInner
  apply Finset.sum_congr rfl
  intro i _
  rw [hodgeStar_rev_apply]
  calc starSign i * f i * (starSign i * g i)
      = (starSign i * starSign i) * (f i * g i) := by ring
    _ = f i * g i := by rw [starSign_mul_self]; ring

/-- **Nondegeneracy of the wedge pairing**: if `f` pairs to zero with
    every constant-coefficient 2-form `g` through the star, then `f = 0`.
    Proved by testing against the coefficient vector whose only possibly
    nonzero entry is `f i` at index `i`, which isolates `(f i)² = 0`. -/
theorem eq_zero_of_wedgeTopCoeff_eq_zero_all (f : TwoFormCoeff)
    (h : ∀ g : TwoFormCoeff, wedgeTopCoeff f g = 0) : f = 0 := by
  funext i
  have hi := h (fun j => if j = i then f i else 0)
  rw [wedgeTopCoeff_eq_coeffInner] at hi
  have hsum : coeffInner f (fun j => if j = i then f i else 0) = f i * f i := by
    unfold coeffInner
    rw [Finset.sum_eq_single i]
    · show f i * (if i = i then f i else 0) = f i * f i
      rw [if_pos rfl]
    · intro j _ hji
      show f j * (if j = i then f i else 0) = 0
      rw [if_neg hji, mul_zero]
    · intro hmem
      exact (hmem (Finset.mem_univ i)).elim
  rw [hsum] at hi
  have : f i * f i = 0 := hi
  simpa using mul_self_eq_zero.mp this

end OmegaProtocol.EMHodge
