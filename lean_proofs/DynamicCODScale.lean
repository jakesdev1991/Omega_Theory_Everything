import Mathlib.Analysis.Real.Sqrt

/-!
# Dynamic Planck Scale Driven by Chain-Overlap Density (COD, `Φ`)

Companion to `DynamicPlanckScale` (density-driven profile). Here the local Planck
length is parametrised directly by the Chain-Overlap Density `Φ ∈ [0, 1]`, the
fundamental primitive of Omega Theory:

    ℓ_P(Φ) = ℓ_P0 · √(1 − Φ²)

* `Φ → 0` (pure vacuum, no shared state): the scale stays at its relaxed baseline.
* `Φ → 1` (bound matter, maximal overlap): the spatial distance between the
  overlapping states vanishes and the operational scale collapses to zero.

## Proven results
1. `dynamicPlanckCOD_zero` — `ℓ_P(0) = ℓ_P0` (vacuum baseline).
2. `dynamicPlanckCOD_one` — `ℓ_P(1) = 0` (maximal overlap collapses distance).
3. `dynamicPlanckCOD_nonneg`, `dynamicPlanckCOD_pos` — the scale is never negative
   and is strictly positive whenever the overlap is imperfect (`Φ² < 1`).
4. `dynamicPlanckCOD_le` — the scale never exceeds the baseline.
5. `cod_scale_contraction` — any positive overlap (`Φ > 0`) contracts the scale
   strictly below baseline.
6. `cod_monotonic_contraction` — higher overlap (`0 ≤ Φ₁ < Φ₂ ≤ 1`) yields strictly
   greater contraction.
7. `dynamicPlanckCOD_strictAntiOn` — the same fact packaged as `StrictAntiOn` on
   `Set.Icc 0 1`.

## Model assumptions (stated, not derived)
* The profile `ℓ_P(Φ) = ℓ_P0 · √(1 − Φ²)` is a phenomenological choice, not a
  consequence of the Omega axioms. It is the canonical form adopted in §2.2 of the
  v4.0 technical note and in `Sim3_Dynamic_Scale.py`. It supersedes the earlier
  `ℓ_P0 · exp((1 − Φ)/φ_c)` heuristic, which pointed the *opposite* way (stretching
  the scale as `Φ` falls); the relation to that macroscopic picture is an open item.
* The definition is total on `ℝ` (Mathlib's `Real.sqrt` is `0` on negative inputs), but
  the theorems below only claim anything on the physical range `0 ≤ Φ ≤ 1`.
-/

namespace DynamicCODScale

/-- Physical parameters for the COD-driven Planck scale. -/
structure CODEnvironment where
  /-- Baseline (relaxed, vacuum) Planck scale. -/
  lP0  : ℝ
  hlP0 : lP0 > 0

/-- Dynamic Planck length as a function of the Chain-Overlap Density `Φ`:
`ℓ_P(Φ) = ℓ_P0 · √(1 − Φ²)`. -/
noncomputable def dynamicPlanckCOD (env : CODEnvironment) (Φ : ℝ) : ℝ :=
  env.lP0 * Real.sqrt (1 - Φ ^ 2)

/-! ### Boundary values -/

/-- Vacuum (`Φ = 0`): the scale is the relaxed baseline. -/
theorem dynamicPlanckCOD_zero (env : CODEnvironment) : dynamicPlanckCOD env 0 = env.lP0 := by
  unfold dynamicPlanckCOD
  norm_num

/-- Maximal overlap (`Φ = 1`): spatial distance collapses to zero. -/
theorem dynamicPlanckCOD_one (env : CODEnvironment) : dynamicPlanckCOD env 1 = 0 := by
  unfold dynamicPlanckCOD
  norm_num

/-! ### Sign and bounds -/

theorem dynamicPlanckCOD_nonneg (env : CODEnvironment) (Φ : ℝ) :
    0 ≤ dynamicPlanckCOD env Φ :=
  mul_nonneg env.hlP0.le (Real.sqrt_nonneg _)

/-- Imperfect overlap (`Φ² < 1`) keeps the scale strictly positive. -/
theorem dynamicPlanckCOD_pos (env : CODEnvironment) (Φ : ℝ) (hΦ : Φ ^ 2 < 1) :
    0 < dynamicPlanckCOD env Φ :=
  mul_pos env.hlP0 (Real.sqrt_pos.mpr (by linarith))

/-- The scale never exceeds the baseline. -/
theorem dynamicPlanckCOD_le (env : CODEnvironment) (Φ : ℝ) :
    dynamicPlanckCOD env Φ ≤ env.lP0 := by
  have h : Real.sqrt (1 - Φ ^ 2) ≤ Real.sqrt 1 :=
    Real.sqrt_le_sqrt (by linarith [sq_nonneg Φ])
  rw [Real.sqrt_one] at h
  unfold dynamicPlanckCOD
  calc env.lP0 * Real.sqrt (1 - Φ ^ 2) ≤ env.lP0 * 1 :=
        mul_le_mul_of_nonneg_left h env.hlP0.le
    _ = env.lP0 := mul_one _

/-! ### Contraction theorems -/

/-- **Theorem 1: Overlap-Driven Scale Contraction.**
Any positive state overlap (`Φ > 0`) contracts the operational scale strictly below
its baseline (`ℓ_P(Φ) < ℓ_P0`). -/
theorem cod_scale_contraction (env : CODEnvironment) (Φ : ℝ) (hΦ : Φ > 0) :
    dynamicPlanckCOD env Φ < env.lP0 := by
  have hsq : 0 < Φ ^ 2 := pow_pos hΦ 2
  have hsqrt : Real.sqrt (1 - Φ ^ 2) < 1 :=
    (Real.sqrt_lt' one_pos).mpr (by linarith)
  unfold dynamicPlanckCOD
  exact mul_lt_of_lt_one_right env.hlP0 hsqrt

/-- **Theorem 2: Monotonic Resolution Scaling.**
Higher state overlap (`0 ≤ Φ₁ < Φ₂ ≤ 1`) yields strictly greater spatial scale
contraction toward zero. -/
theorem cod_monotonic_contraction (env : CODEnvironment) (Φ₁ Φ₂ : ℝ)
    (h0 : 0 ≤ Φ₁) (hgt : Φ₂ > Φ₁) (h1 : Φ₂ ≤ 1) :
    dynamicPlanckCOD env Φ₂ < dynamicPlanckCOD env Φ₁ := by
  have hΦ2 : 0 < Φ₂ := lt_of_le_of_lt h0 hgt
  have hsq : Φ₁ ^ 2 < Φ₂ ^ 2 := by
    nlinarith [mul_pos (sub_pos.mpr hgt) (add_pos_of_nonneg_of_pos h0 hΦ2)]
  have hsq2 : Φ₂ ^ 2 ≤ 1 := by nlinarith
  have hsqrt : Real.sqrt (1 - Φ₂ ^ 2) < Real.sqrt (1 - Φ₁ ^ 2) :=
    Real.sqrt_lt_sqrt (by linarith) (by linarith)
  unfold dynamicPlanckCOD
  exact mul_lt_mul_of_pos_left hsqrt env.hlP0

/-- Theorem 2 packaged as strict antitonicity on the physical range `[0, 1]`. -/
theorem dynamicPlanckCOD_strictAntiOn (env : CODEnvironment) :
    StrictAntiOn (dynamicPlanckCOD env) (Set.Icc 0 1) := by
  intro a ha b hb hab
  exact cod_monotonic_contraction env a b (Set.mem_Icc.mp ha).1 hab (Set.mem_Icc.mp hb).2

end DynamicCODScale
