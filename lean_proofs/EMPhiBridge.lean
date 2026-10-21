import Mathlib
import EMHodgeStar
import EMConformalStar
import DynamicCODScale

/-
  EMPhiBridge.lean
  The Φ–EM dictionary (postulate layer) and its conditional vacuum-Maxwell
  consequence — T3b of `DERIVATION_SCAFFOLD.md`, items (a) and (c) of T3.

  What this module is, and what it is not
  ---------------------------------------
  T3a (`EMConformalStar.lean`) proved, as derived algebra at T1's
  constant-coefficient scope, that the Hodge star on 2-forms is unchanged
  under a constant positive conformal rescaling of the metric:
  `hodgeStarScaled c = hodgeStar` for every `c > 0`.  This module supplies
  the two remaining T3 pieces, at exactly that scope:

  * (a) `PhiEMDictionary` — the POSTULATE layer (carried data, by design).
    A value of this structure is the dictionary entry that identifies, at
    a point, the physical metric with the constant conformal rescaling of
    the flat Euclidean metric by the positive factor `c = ℓ_P(Φ)`, where
    `Φ` is the Chain-Overlap Density value in question and `ℓ_P(Φ)` is
    the profile `ℓ_P(Φ) = ℓ_P0 · √(1 − Φ²)` of `DynamicCODScale.lean`
    (`DynamicCODScale.dynamicPlanckCOD`; the formula is a phenomenological
    choice adopted there, §2.2 of the v4.0 technical note as cited in that
    file — it is used here as the dictionary's source formula, not derived
    here).  The structure's fields are that identification stated as data:
    the `Φ` value, the COD environment (baseline `ℓ_P0 > 0`), the physical
    range hypothesis `Φ² < 1`, and the factor itself together with the
    equation identifying it with `ℓ_P(Φ)`.  Nothing in the structure is a
    theorem, no metric object is constructed, and no value of the
    structure is constructed in this module: the dictionary is carried,
    exactly as the scaffold's Layer 0 prescribes.

  * (c) The conditional vacuum consequence (conditional).  At the
    coefficient scope there is no exterior derivative: a coefficient-level
    `d` compatible with Vol02's smooth forms would vanish on constants
    (the T2 outcome note in `DERIVATION_SCAFFOLD.md`), so the vacuum
    system is stated here through an explicit closedness parameter
    `closed : TwoFormCoeff → Prop`, standing for the exterior-closedness
    test `d _ = 0` applied to a coefficient 2-form and to its star-dual.
    The vacuum system for a star `S` is then `closed F ∧ closed (S F)` —
    the two closedness statements `dF = 0` and `d(⋆F) = 0` of the vacuum
    Maxwell system, of which only the second involves the star (the
    codifferential form `δF = 0` is the T2 bridge's re-expression of the
    second, conditional there on a `SmoothStarExtension`).  Relative to
    an explicit `PhiEMDictionary` hypothesis, the theorems below prove
    that this system stated with the physical star
    (`hodgeStarScaled d.factor`, T3a's rescaled star at the dictionary's
    factor) coincides with the same system stated with the flat
    `hodgeStar`.  Every such theorem takes the dictionary as an explicit
    argument, and every proof goes through T3a's derived invariance
    theorem plus the positivity of the dictionary's factor, itself
    derived from the dictionary's identification with `ℓ_P(Φ)`.

  Dispositions: the dictionary is **carried**; the factor positivity and
  the star equality are **derived from the dictionary's fields**; the
  vacuum-Maxwell coincidence is **conditional** on the dictionary.  The
  sourced Maxwell law remains carried data in Vol02, untouched here.
-/

namespace OmegaProtocol.EMPhiBridge

open OmegaProtocol.EMHodge
open OmegaProtocol.EMConformal

/-- **The Φ–EM dictionary (POSTULATE layer — carried data, not a
    derivation).**  A value of this structure states, as data, that at a
    point the physical metric is the constant conformal rescaling of the
    flat Euclidean metric by the positive factor `factor`, and that this
    factor is identified with the dynamic Planck scale at the
    Chain-Overlap Density value `phi`:
    `factor = ℓ_P(phi) = ℓ_P0 · √(1 − phi²)`, the profile of
    `DynamicCODScale.dynamicPlanckCOD` in `DynamicCODScale.lean` (a
    phenomenological choice adopted there; cited here as the source
    formula, not derived here).  The field `phi_sq_lt_one` restricts to
    the range in which that profile is strictly positive.  No value of
    this structure is constructed in this module, and no theorem in this
    module holds without a value being supplied as an explicit
    hypothesis. -/
structure PhiEMDictionary where
  /-- The Chain-Overlap Density value `Φ` at the point in question. -/
  phi : ℝ
  /-- The COD environment: the baseline Planck scale `lP0` and its
      positivity `hlP0`. -/
  env : DynamicCODScale.CODEnvironment
  /-- Physical range data: `phi² < 1`, the range in which the profile
      `ℓ_P(phi)` is strictly positive (`dynamicPlanckCOD_pos`). -/
  phi_sq_lt_one : phi ^ 2 < 1
  /-- The conformal factor `c` of the dictionary: the physical metric at
      the point is, by this postulate, the constant rescaling of the
      flat metric by this factor (the rescaled-metric bookkeeping of
      `EMConformalStar` is the metric `factor² • g`). -/
  factor : ℝ
  /-- Identification data: the conformal factor is the dynamic Planck
      scale at `phi`, `factor = ℓ_P(phi)` with `ℓ_P` the
      `DynamicCODScale` profile `ℓ_P0 · √(1 − phi²)`. -/
  factor_eq_lP : factor = DynamicCODScale.dynamicPlanckCOD env phi

/-- The conformal factor of a dictionary value, written out in the
    source formula of `DynamicCODScale.lean`:
    `d.factor = d.env.lP0 * √(1 − d.phi²)`.  This is the dictionary's
    identification field `factor_eq_lP` unfolded through the definition
    of `dynamicPlanckCOD`; the formula itself remains the carried
    profile choice of `DynamicCODScale`. -/
theorem PhiEMDictionary.factor_eq_formula (d : PhiEMDictionary) :
    d.factor = d.env.lP0 * Real.sqrt (1 - d.phi ^ 2) := by
  rw [d.factor_eq_lP]
  rfl

/-- The conformal factor named by a dictionary value is strictly
    positive: from the dictionary's identification
    `factor = ℓ_P(phi)`, its baseline positivity field, and its range
    field `phi_sq_lt_one`, via `DynamicCODScale.dynamicPlanckCOD_pos`. -/
theorem PhiEMDictionary.factor_pos (d : PhiEMDictionary) : 0 < d.factor := by
  rw [d.factor_eq_lP]
  exact DynamicCODScale.dynamicPlanckCOD_pos d.env d.phi d.phi_sq_lt_one

/-- **The physical Hodge star named by a dictionary value**: by the
    dictionary (carried data), the star of the physical metric at the
    point is T3a's rescaled star at the dictionary's factor,
    `hodgeStarScaled d.factor` on constant-coefficient 2-forms.  This is
    a definition recording the dictionary's identification, not a
    construction of a star from a metric. -/
noncomputable def physicalStar (d : PhiEMDictionary) :
    TwoFormCoeff →ₗ[ℝ] TwoFormCoeff :=
  hodgeStarScaled d.factor

/-- **Conditional star equality (on a `PhiEMDictionary`)**: given a
    dictionary value `d`, the physical star it names equals T1's flat
    Hodge star, as linear maps on constant-coefficient 2-forms.  Proved
    from T3a's derived invariance theorem
    `hodgeStarScaled_eq_hodgeStar` at the dictionary's factor, whose
    positivity is `d.factor_pos`. -/
theorem physicalStar_eq_hodgeStar (d : PhiEMDictionary) :
    physicalStar d = hodgeStar := by
  unfold physicalStar
  exact hodgeStarScaled_eq_hodgeStar d.factor_pos

/-- **Conditional star equality, applied form**: given a dictionary
    value `d`, the physical star it names sends every constant-coefficient
    2-form `F` to the same coefficients as the flat Hodge star.  From
    `physicalStar_eq_hodgeStar`. -/
theorem physicalStar_apply_eq (d : PhiEMDictionary) (F : TwoFormCoeff) :
    physicalStar d F = hodgeStar F := by
  rw [physicalStar_eq_hodgeStar d]

/-- **The coefficient-level vacuum Maxwell predicate for a star `S`**,
    relative to an explicit closedness test `closed` standing for the
    exterior-closedness condition `d _ = 0` (supplied as a parameter
    because no coefficient-level exterior derivative is constructed in
    this scaffold — see the module header): a constant-coefficient
    2-form `F` satisfies it when `F` is closed and its star-dual `S F`
    is closed, i.e. the two closedness statements `dF = 0` and
    `d(⋆F) = 0` of the vacuum Maxwell system.  Only the second statement
    involves the star `S`.  This is a definition of the statement form,
    not a claim that either closedness holds for any `F`. -/
def CoeffVacuumMaxwell (closed : TwoFormCoeff → Prop)
    (S : TwoFormCoeff →ₗ[ℝ] TwoFormCoeff) (F : TwoFormCoeff) : Prop :=
  closed F ∧ closed (S F)

/-- **Conditional vacuum-Maxwell coincidence, scaled-star form**: given
    a dictionary value `d` as an explicit hypothesis, the coefficient-level
    vacuum Maxwell system stated with the physical star
    `hodgeStarScaled d.factor` (T3a's rescaled star at the dictionary's
    factor) holds for `F` if and only if the same system stated with the
    flat `hodgeStar` holds for `F`, for every closedness test `closed`.
    Proved by rewriting with T3a's derived invariance theorem
    `hodgeStarScaled_eq_hodgeStar` at `d.factor_pos`. -/
theorem vacuum_maxwell_scaled_iff_flat (d : PhiEMDictionary)
    (closed : TwoFormCoeff → Prop) (F : TwoFormCoeff) :
    CoeffVacuumMaxwell closed (hodgeStarScaled d.factor) F
      ↔ CoeffVacuumMaxwell closed hodgeStar F := by
  unfold CoeffVacuumMaxwell
  rw [hodgeStarScaled_eq_hodgeStar d.factor_pos]

/-- **Conditional vacuum-Maxwell coincidence (T3(c))**: given a
    dictionary value `d` as an explicit hypothesis, the coefficient-level
    vacuum Maxwell system stated with the physical star named by `d`
    holds for `F` if and only if the same system stated with the flat
    `hodgeStar` holds for `F`, for every closedness test `closed`.  This
    is `vacuum_maxwell_scaled_iff_flat` restated through the definition
    of `physicalStar`; the dictionary is the hypothesis, T3a's invariance
    is the mathematics, and no vacuum equation is asserted to hold for
    any particular `F`. -/
theorem vacuum_maxwell_physical_iff_flat (d : PhiEMDictionary)
    (closed : TwoFormCoeff → Prop) (F : TwoFormCoeff) :
    CoeffVacuumMaxwell closed (physicalStar d) F
      ↔ CoeffVacuumMaxwell closed hodgeStar F := by
  unfold physicalStar
  exact vacuum_maxwell_scaled_iff_flat d closed F

/-- **Conditional dual-closedness coincidence**: given a dictionary
    value `d`, the star-dependent half of the vacuum system — closedness
    of the dual field — stated with the physical star named by `d`
    coincides with the same statement for the flat star, for every
    closedness test `closed` and every `F`.  From
    `physicalStar_apply_eq`. -/
theorem dual_closed_physical_iff_flat (d : PhiEMDictionary)
    (closed : TwoFormCoeff → Prop) (F : TwoFormCoeff) :
    closed (physicalStar d F) ↔ closed (hodgeStar F) := by
  rw [physicalStar_apply_eq d F]

end OmegaProtocol.EMPhiBridge
