import Mathlib.Analysis.Real.Sqrt
import OmegaAxioms
import DynamicCODScale

/-!
# Q-region → scalar-Gauss-Bonnet coupling derivation (attempt 1)

Bridge from the Q-region overlap density `Φ` and mutual information `I`
(OmegaAxioms) to the SGB coupling `η` of GAUSS_BONNET_SCALARIZATION.

Chain of identities (all definitions from OmegaAxioms / DynamicCODScale):

1. Pair overlap:      Φ R₁ R₂ = min d₁ d₂ (distinct), 1 (self)
2. Mutual information: I R₁ R₂ = Φ(1-Φ)·(S₁+S₂)/2
3. Planck factor:      ℓ_P(Φ) = ℓ_P0·√(1-Φ²)   [canonicalCODEnv: ℓ_P0 = 1]
4. Curvature scale:    𝒢 ~ 1/ℓ_P⁴ evaluated at the pair overlap
5. Coupling ansatz:    η = ℓ_P(Φ)²  (dimensionally length², matching η = f''(0)/Z₀)

Result: η(Φ) = ℓ_P0²·(1-Φ²), and at the horizon of a mass-M black hole the
dimensionless ratio is η/M² = (ℓ_P0/M)²·(1-Φ²).

The derivation is *defined*, not derived from dynamics: step 5 is the single
unproven identification. Everything above it is machine-checked.
-/

open OmegaProtocol

namespace QRegionCoupling

/-- Step 5, stated as an explicit definition: the SGB coupling equals the
    squared dynamic Planck length at the pair overlap. -/
noncomputable def etaFromPhi (env : DynamicCODScale.CODEnvironment) (φ : ℝ) : ℝ :=
  (DynamicCODScale.dynamicPlanckCOD env φ) ^ 2

/-- Closed form: η(Φ) = ℓ_P0²·(1-Φ²). -/
theorem etaFromPhi_eq (env : DynamicCODScale.CODEnvironment) (φ : ℝ) (hφ : φ ^ 2 ≤ 1) :
    etaFromPhi env φ = env.lP0 ^ 2 * (1 - φ ^ 2) := by
  unfold etaFromPhi DynamicCODScale.dynamicPlanckCOD
  have h : 0 ≤ 1 - φ ^ 2 := by linarith
  rw [mul_pow, Real.sq_sqrt h]

/-- η is nonnegative and bounded by ℓ_P0². -/
theorem etaFromPhi_nonneg (env : DynamicCODScale.CODEnvironment) (φ : ℝ) (hφ : φ ^ 2 ≤ 1) :
    0 ≤ etaFromPhi env φ := by
  rw [etaFromPhi_eq env φ hφ]
  exact mul_nonneg (by positivity) (by linarith)

theorem etaFromPhi_le (env : DynamicCODScale.CODEnvironment) (φ : ℝ) (hφ : φ ^ 2 ≤ 1) :
    etaFromPhi env φ ≤ env.lP0 ^ 2 := by
  rw [etaFromPhi_eq env φ hφ]
  nlinarith [sq_nonneg φ, env.hlP0]

/-- At maximal overlap (self-pair) the coupling vanishes: η(1) = 0. -/
theorem etaFromPhi_one (env : DynamicCODScale.CODEnvironment) :
    etaFromPhi env 1 = 0 := by
  rw [etaFromPhi_eq env 1 (by norm_num)]; norm_num

/-- At zero overlap (vacuum pair) the coupling is the baseline: η(0) = ℓ_P0². -/
theorem etaFromPhi_zero (env : DynamicCODScale.CODEnvironment) :
    etaFromPhi env 0 = env.lP0 ^ 2 := by
  rw [etaFromPhi_eq env 0 (by norm_num)]; norm_num

/-- Monotonicity: higher overlap ⇒ smaller coupling (on [0,1]). -/
theorem etaFromPhi_anti {env : DynamicCODScale.CODEnvironment} {φ₁ φ₂ : ℝ}
    (h0 : 0 ≤ φ₁) (h1 : φ₁ ≤ φ₂) (h2 : φ₂ ≤ 1) :
    etaFromPhi env φ₂ ≤ etaFromPhi env φ₁ := by
  have hq1 : φ₁ ^ 2 ≤ 1 := by nlinarith
  have hq2 : φ₂ ^ 2 ≤ 1 := by nlinarith
  rw [etaFromPhi_eq env φ₂ hq2, etaFromPhi_eq env φ₁ hq1]
  have hsq : φ₁ ^ 2 ≤ φ₂ ^ 2 := by nlinarith
  nlinarith

/-- The pair-level coupling: η evaluated at the Q-region overlap Φ R₁ R₂.
    Since Φ ∈ (0,1], η is always ≤ ℓ_P0² and equals ℓ_P0²·(1-Φ²). -/
noncomputable def etaPair (env : DynamicCODScale.CODEnvironment) (R₁ R₂ : QRegion) : ℝ :=
  etaFromPhi env (Φ R₁ R₂)

theorem etaPair_eq (env : DynamicCODScale.CODEnvironment) (R₁ R₂ : QRegion) :
    etaPair env R₁ R₂ = env.lP0 ^ 2 * (1 - (Φ R₁ R₂) ^ 2) := by
  have hq : (Φ R₁ R₂) ^ 2 ≤ 1 := by
    have h := Φ_le_one R₁ R₂
    have hpos := Φ_pos R₁ R₂
    nlinarith
  exact etaFromPhi_eq env (Φ R₁ R₂) hq

theorem etaPair_pos (env : DynamicCODScale.CODEnvironment) (R₁ R₂ : QRegion)
    (hΦlt : Φ R₁ R₂ < 1) : 0 < etaPair env R₁ R₂ := by
  rw [etaPair_eq]
  have hΦ := Φ_pos R₁ R₂
  have hsq : (Φ R₁ R₂) ^ 2 < 1 := by nlinarith
  refine mul_pos (pow_pos env.hlP0 2) (by linarith)

/-- Self-pair: Φ = 1, so the coupling vanishes — a self-region carries no
    SGB source, consistent with ℓ_P(1) = 0. -/
theorem etaPair_self (env : DynamicCODScale.CODEnvironment) (R : QRegion) :
    etaPair env R R = 0 := by
  rw [etaPair, Φ_self]; exact etaFromPhi_one env

/-- Dimensionless ratio at a black hole of mass M: η/M². -/
noncomputable def etaOverM2 (env : DynamicCODScale.CODEnvironment) (M : ℝ) (φ : ℝ) : ℝ :=
  etaFromPhi env φ / M ^ 2

/-- The threshold λ₁ ≈ 0.7256092 from GAUSS_BONNET_SCALARIZATION corresponds to
    the overlap Φ* satisfying ℓ_P0²(1-Φ*²)/M² = λ₁, i.e. Φ* = √(1 - λ₁ M²/ℓ_P0²).
    This is a *prediction of the required overlap*, not a derivation of M. -/
noncomputable def thresholdOverlap (env : DynamicCODScale.CODEnvironment) (M lam₁ : ℝ) : ℝ :=
  Real.sqrt (1 - lam₁ * M ^ 2 / env.lP0 ^ 2)

theorem thresholdOverlap_sq (env : DynamicCODScale.CODEnvironment) (M lam₁ : ℝ)
    (h : lam₁ * M ^ 2 ≤ env.lP0 ^ 2) :
    (thresholdOverlap env M lam₁) ^ 2 = 1 - lam₁ * M ^ 2 / env.lP0 ^ 2 := by
  rw [thresholdOverlap]
  have hpos : 0 < env.lP0 ^ 2 := pow_pos env.hlP0 2
  have h : 0 ≤ 1 - lam₁ * M ^ 2 / env.lP0 ^ 2 := by
    have hM := sq_nonneg M
    have hdiv : lam₁ * M ^ 2 / env.lP0 ^ 2 ≤ 1 := by
      exact div_le_one_of_le₀ h (by positivity)
    linarith
  rw [Real.sq_sqrt h]

/-- Consistency: at the threshold overlap, η/M² equals λ₁.
    Requires λ₁ ≥ 0 (true for the physical threshold ≈ 0.7256). -/
theorem etaOverM2_at_threshold (env : DynamicCODScale.CODEnvironment) (M lam₁ : ℝ)
    (hlam : 0 ≤ lam₁) (hM : M ≠ 0) (h : lam₁ * M ^ 2 ≤ env.lP0 ^ 2) :
    etaOverM2 env M (thresholdOverlap env M lam₁) = lam₁ := by
  rw [etaOverM2]
  have hq : (thresholdOverlap env M lam₁) ^ 2 ≤ 1 := by
    rw [thresholdOverlap_sq env M lam₁ h]
    have hdiv : lam₁ * M ^ 2 / env.lP0 ^ 2 ≤ 1 :=
      div_le_one_of_le₀ h (by positivity)
    have hnonneg : 0 ≤ lam₁ * M ^ 2 / env.lP0 ^ 2 :=
      div_nonneg (mul_nonneg hlam (sq_nonneg M)) (by positivity)
    linarith
  rw [etaFromPhi_eq env (thresholdOverlap env M lam₁) hq, thresholdOverlap_sq env M lam₁ h]
  have hM2 : M ^ 2 ≠ 0 := pow_ne_zero 2 hM
  have hlP0 : env.lP0 ≠ 0 := ne_of_gt env.hlP0
  have hlP : env.lP0 ^ 2 ≠ 0 := pow_ne_zero 2 hlP0
  field_simp [hM2, hlP]
  have hsq : env.lP0 * env.lP0 * (env.lP0⁻¹) ^ 2 = 1 := by
    field_simp [hlP0]
  nlinarith

end QRegionCoupling
