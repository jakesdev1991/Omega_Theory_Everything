import Mathlib.Analysis.Real.Sqrt

/-!
# Dynamic Planck Scale Driven by Reverse Chain-Overlap Density (RCOD)

Companion to `DynamicPlanckScale` (density-driven profile). Here the local Planck
length is driven by the Reverse Chain-Overlap Density `μ`, a measure of structural
novelty / dissonance, which is itself derived from the correlation density `Φ`:

    correlation density Φ  ⟹  RCOD μ(Φ) = √(1 − Φ²)  ⟹  dynamic scale ℓ_P(μ)

## Proven results
1. `rcod_nonneg`, `rcod_le_one`, `rcod_pos` — the RCOD map lands in `[0, 1]` and
   is strictly positive exactly when the correlation is imperfect (`Φ² < 1`).
2. `dynamicPlanckRCOD_pos` — the dynamic scale is always positive.
3. `dynamicPlanckRCOD_zero` — with no dissonance (`μ = 0`) the scale is the baseline.
4. `rcod_scale_contraction` — any positive dissonance contracts the scale below baseline.
5. `rcod_monotonic_contraction` — more dissonance yields strictly more contraction.
6. `correlation_deficit_contracts_planck_scale` — the full pipeline: any imperfect
   correlation (`Φ² < 1`) contracts the Planck scale.
7. `perfect_correlation_baseline` — perfect correlation (`Φ = 1`) leaves it at baseline.

## Model assumptions (stated, not derived)
* The profile `ℓ_P(μ) = ℓ_P0 / √(1 + k·μ)` for `μ ≥ 0` is a phenomenological choice, not
  a consequence of the Omega axioms. It is a different parametrisation from the
  `Φ`-profile `ℓ_P0 · exp((1 − Φ)/φ_c)` of the v4.0 technical note (§2.2). Note the
  direction: low `Φ` means high `μ`, and this profile *contracts* the scale there.
* `μ(Φ) = √(1 − Φ²)` is taken as the definition of RCOD in terms of `Φ`.
-/

namespace DynamicRCODScale

/-- Physical parameters coupling the dynamic Planck scale to RCOD `μ`. -/
structure RCODEnvironment where
  /-- Baseline Planck scale. -/
  lP0  : ℝ
  /-- Scale coupling constant. -/
  k    : ℝ
  hlP0 : lP0 > 0
  hk   : k > 0

/-- Reverse Chain-Overlap Density `μ` derived from the correlation density `Φ`. -/
noncomputable def rcod (Φ : ℝ) : ℝ :=
  Real.sqrt (1 - Φ ^ 2)

/-- Dynamic Planck length as a function of RCOD `μ`: as structural novelty / dissonance
`μ` increases, the local Planck length contracts. Below `μ = 0` (outside the physical
range) the baseline is returned. -/
noncomputable def dynamicPlanckRCOD (env : RCODEnvironment) (μ : ℝ) : ℝ :=
  if μ ≥ 0 then env.lP0 / Real.sqrt (1 + env.k * μ) else env.lP0

/-! ### Properties of the RCOD map -/

theorem rcod_nonneg (Φ : ℝ) : 0 ≤ rcod Φ :=
  Real.sqrt_nonneg _

theorem rcod_le_one (Φ : ℝ) : rcod Φ ≤ 1 := by
  have h : Real.sqrt (1 - Φ ^ 2) ≤ Real.sqrt 1 :=
    Real.sqrt_le_sqrt (by linarith [sq_nonneg Φ])
  rwa [Real.sqrt_one] at h

/-- Imperfect correlation (`Φ² < 1`) gives strictly positive dissonance. -/
theorem rcod_pos (Φ : ℝ) (hΦ : Φ ^ 2 < 1) : 0 < rcod Φ :=
  Real.sqrt_pos.mpr (by linarith)

/-- Perfect correlation gives zero dissonance. -/
theorem rcod_one : rcod 1 = 0 := by
  unfold rcod
  norm_num

/-! ### Properties of the dynamic scale -/

/-- The dynamic scale is always strictly positive. -/
theorem dynamicPlanckRCOD_pos (env : RCODEnvironment) (μ : ℝ) :
    0 < dynamicPlanckRCOD env μ := by
  unfold dynamicPlanckRCOD
  split_ifs with h
  · have hkμ : 0 ≤ env.k * μ := mul_nonneg env.hk.le h
    exact div_pos env.hlP0 (Real.sqrt_pos.mpr (by linarith))
  · exact env.hlP0

/-- Zero dissonance leaves the scale at its baseline. -/
theorem dynamicPlanckRCOD_zero (env : RCODEnvironment) :
    dynamicPlanckRCOD env 0 = env.lP0 := by
  unfold dynamicPlanckRCOD
  rw [if_pos (le_refl (0 : ℝ)), mul_zero, add_zero, Real.sqrt_one, div_one]

/-- **Theorem 1: Dissonance-Driven Scale Contraction.**
When RCOD is positive (`μ > 0`), the operational rendering resolution contracts
(`ℓ_P(μ) < ℓ_P0`). -/
theorem rcod_scale_contraction (env : RCODEnvironment) (μ : ℝ) (hμ : μ > 0) :
    dynamicPlanckRCOD env μ < env.lP0 := by
  have h1 : 1 < 1 + env.k * μ := by linarith [mul_pos env.hk hμ]
  have hsqrt : 1 < Real.sqrt (1 + env.k * μ) := by
    have hlt := Real.sqrt_lt_sqrt zero_le_one h1
    rwa [Real.sqrt_one] at hlt
  unfold dynamicPlanckRCOD
  rw [if_pos hμ.le]
  exact div_lt_self env.hlP0 hsqrt

/-- **Theorem 2: Monotonic Resolution Scaling.**
Higher RCOD (`μ₂ > μ₁ > 0`) yields strictly greater spatial scale contraction. -/
theorem rcod_monotonic_contraction (env : RCODEnvironment) (μ₁ μ₂ : ℝ)
    (hμ1 : μ₁ > 0) (hgt : μ₂ > μ₁) :
    dynamicPlanckRCOD env μ₂ < dynamicPlanckRCOD env μ₁ := by
  have hμ2 : μ₂ > 0 := lt_trans hμ1 hgt
  have hkμ1 : 0 < env.k * μ₁ := mul_pos env.hk hμ1
  have hsum : 1 + env.k * μ₁ < 1 + env.k * μ₂ := by
    linarith [mul_lt_mul_of_pos_left hgt env.hk]
  have hden1 : 0 < Real.sqrt (1 + env.k * μ₁) :=
    Real.sqrt_pos.mpr (by linarith)
  have hsqrt_lt : Real.sqrt (1 + env.k * μ₁) < Real.sqrt (1 + env.k * μ₂) :=
    Real.sqrt_lt_sqrt (by linarith) hsum
  unfold dynamicPlanckRCOD
  rw [if_pos hμ2.le, if_pos hμ1.le]
  exact div_lt_div_of_pos_left env.hlP0 hden1 hsqrt_lt

/-! ### The full pipeline Φ ⟹ μ ⟹ ℓ_P -/

/-- Any imperfect correlation (`Φ² < 1`) contracts the Planck scale below baseline. -/
theorem correlation_deficit_contracts_planck_scale (env : RCODEnvironment) (Φ : ℝ)
    (hΦ : Φ ^ 2 < 1) :
    dynamicPlanckRCOD env (rcod Φ) < env.lP0 :=
  rcod_scale_contraction env _ (rcod_pos Φ hΦ)

/-- Perfect correlation (`Φ = 1`) leaves the Planck scale at its baseline. -/
theorem perfect_correlation_baseline (env : RCODEnvironment) :
    dynamicPlanckRCOD env (rcod 1) = env.lP0 := by
  rw [rcod_one]
  exact dynamicPlanckRCOD_zero env

end DynamicRCODScale
