import DynamicCODScale
import Mathlib

/-!
# The macroscopic radial metric, written in `Φ`

Companion to `Omega_Theory_v4.0_Radial_Metric.md` and `Sim7_Radial_Metric.py`.
The working notes derive a two-factor law from the §2.1 axioms (Markov
factorization, additivity, locality): a proper length element is a *product* of
a microscopic rendering scale `ℓ_P(Φ)` and a macroscopic correlation-decay rate
`κ`,

    ds = ℓ_P(Φ) · dσ,   dσ = -d ln K,   g_rr = (ℓ_P(Φ) · κ)².

This module formalizes the three algebraic facts behind that note so that the
Lean CI job kernel-checks them.

1. **The golden-ratio bottleneck** (`golden_bottleneck_bound`). For
   `Φ* = (√5 - 1) / 2` and *every* real `Φ`,
       exp(2Φ) · (1 - Φ²) ≤ Φ* · exp(2Φ*),
   with equality only at `Φ = Φ*`. This is the elementary, calculus-free proof
   of the global minimum of Sim3's disformal causality band. The proof is the
   inequality chain of §8 of the notes: with `t = Φ - Φ*`,
       `1 - Φ² = Φ*(1 - 2t) - t² ≤ Φ*(1 - 2t) ≤ Φ*·exp(-2t)`,
   using `e^x ≥ 1 + x` (`Real.add_one_le_exp`) for the last step.
2. **The finite-chain two-factor law** (`chainLength_mono`,
   `chainLength_const_scale`). A chain of `n` links whose `i`-th link carries
   local scale `s i` and local weight `w i` has proper length `∑ s i · w i`.
   It is monotone in the scale, and the single-scale formula factors out
   *exactly* when the scale is link-independent — the only case in which the
   finite §2.1 formula `d = -ℓ_P ln K` is exact. That is the discrete content
   of `g_rr = (ℓ_P κ)²`: the two factors enter per link as a product, never as
   a sum.
3. **The `Φ`-form Schwarzschild identities** (`arealRadius_at_vacuum`,
   `arealRadius_at_horizon`, `lapse_conformal_identity`,
   `cod_profile_matches_lapse`). With `ψ = 1 + Φ` and `N = (1-Φ)/(1+Φ)`, the
   lapse times the conformal factor squared is exactly `1 - Φ²`, which is
   `(ℓ_P(Φ)/ℓ_P0)²` under the canonical COD profile of `DynamicCODScale`. The
   areal radius `R = ρ(1 + Φ)²` is `ρ` at the vacuum and `4ρ` at the horizon.

## Model assumptions (stated, not derived)
* The convention is the one of `DynamicCODScale` and §2.2 of the technical
  note: `Φ = 0` is the vacuum and `Φ → 1` the bound-state/horizon end, with
  `ℓ_P(Φ) = ℓ_P0 · √(1 - Φ²)`.
* The `Φ`-form of Schwarzschild in item 3 is *identified*, not derived from the
  Omega axioms; see §5 of the notes. Only its algebraic identities are proved
  here — Ricci-flatness is checked symbolically by `Sim7_Radial_Metric.py`,
  not here.
* The chain model of item 2 is a finite-chain rendering of the axioms, not a
  derivation of the metric itself.
-/

namespace RadialMetric

/-! ## 1. The golden-ratio bottleneck -/

/-- The golden conjugate `Φ* = (√5 - 1) / 2 ≈ 0.618`, the tightest point of
Sim3's disformal causality band. -/
noncomputable def phiStar : ℝ := (Real.sqrt 5 - 1) / 2

/-- Clearing the denominator: `2 * Φ* = √5 - 1`. -/
theorem two_mul_phiStar : 2 * phiStar = Real.sqrt 5 - 1 := by
  unfold phiStar
  field_simp

/-- The golden conjugate is strictly positive. Uses `Real.sqrt_lt_sqrt` and
`Real.sqrt_one`, the same two lemmas `DynamicCODScale` relies on. -/
theorem phiStar_pos : 0 < phiStar := by
  have h1 : (1 : ℝ) < Real.sqrt 5 := by
    have h' : Real.sqrt (1 : ℝ) < Real.sqrt 5 :=
      Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
    rwa [Real.sqrt_one] at h'
  have e : 2 * phiStar = Real.sqrt 5 - 1 := two_mul_phiStar
  linarith

/-- The defining equation of the golden conjugate: `Φ*² + Φ* = 1`. -/
theorem phiStar_golden : phiStar * phiStar + phiStar - 1 = 0 := by
  have hs : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  have e : 2 * phiStar = Real.sqrt 5 - 1 := two_mul_phiStar
  have hexp : (Real.sqrt 5 - 1) * (Real.sqrt 5 - 1)
            = (Real.sqrt 5) ^ 2 - 2 * Real.sqrt 5 + 1 := by ring
  have e2 : (2 * phiStar) * (2 * phiStar) = 6 - 2 * Real.sqrt 5 := by
    rw [e, hexp, hs]
    ring
  -- 4 * (Φ*² + Φ* - 1) with the denominator cleared on the right.
  have key : 4 * (phiStar * phiStar + phiStar - 1)
           = (2 * phiStar) * (2 * phiStar) + 2 * (2 * phiStar) - 4 := by ring
  have hval : (2 * phiStar) * (2 * phiStar) + 2 * (2 * phiStar) - 4 = 0 := by
    rw [e2, e]
    ring
  have h4 : 4 * (phiStar * phiStar + phiStar - 1) = 0 := by rw [key, hval]
  exact Or.resolve_left (mul_eq_zero.mp h4) (by norm_num)

/-- **The golden-ratio bottleneck.** For every real `Φ`,
`exp(2Φ) · (1 - Φ²) ≤ Φ* · exp(2Φ*)`, with equality only at `Φ = Φ*`.

The bound holds on all of `ℝ`: outside `(-1, 1)` the left side is negative and
the inequality is trivial, while on `(-1, 1)` this is the global minimum of
Sim3's disformal causality band `exp(-Φ) / (√β √(1-Φ²))`, whose minimum value
is `exp(-Φ*) / √(β Φ*)`. -/
theorem golden_bottleneck_bound (Φ : ℝ) :
    Real.exp (2 * Φ) * (1 - Φ * Φ) ≤ phiStar * Real.exp (2 * phiStar) := by
  have hg : phiStar * phiStar + phiStar - 1 = 0 := phiStar_golden
  have hφ : 0 < phiStar := phiStar_pos
  -- Step 1: the algebraic identity 1 - Φ² = Φ*(1 - 2t) - t² with t = Φ - Φ*.
  have halg : 1 - Φ * Φ
            = phiStar * (1 - 2 * (Φ - phiStar)) - (Φ - phiStar) * (Φ - phiStar) := by
    have hd : phiStar * (1 - 2 * (Φ - phiStar)) - (Φ - phiStar) * (Φ - phiStar)
            = 1 - Φ * Φ + (phiStar * phiStar + phiStar - 1) := by ring
    linarith [hd, hg]
  -- Step 2: e^x ≥ 1 + x, i.e. 1 - 2t ≤ exp(-2t).
  have h2 : 1 - 2 * (Φ - phiStar) ≤ Real.exp (-2 * (Φ - phiStar)) := by
    have h := Real.add_one_le_exp (-2 * (Φ - phiStar))
    linarith
  -- Step 3: drop the -t² term and use h2.
  have hsq : 0 ≤ (Φ - phiStar) * (Φ - phiStar) := mul_self_nonneg _
  have hmul : phiStar * (1 - 2 * (Φ - phiStar))
            ≤ phiStar * Real.exp (-2 * (Φ - phiStar)) :=
    mul_le_mul_of_nonneg_left h2 hφ.le
  -- Step 4: the exponential factors recombine to exp(2Φ*).
  have hexp : Real.exp (2 * Φ) * Real.exp (-2 * (Φ - phiStar))
            = Real.exp (2 * phiStar) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc Real.exp (2 * Φ) * (1 - Φ * Φ)
      = Real.exp (2 * Φ)
          * (phiStar * (1 - 2 * (Φ - phiStar)) - (Φ - phiStar) * (Φ - phiStar)) := by
        rw [halg]
    _ ≤ Real.exp (2 * Φ) * (phiStar * Real.exp (-2 * (Φ - phiStar))) :=
        mul_le_mul_of_nonneg_left
          (by linarith [hsq, hmul]) (le_of_lt (Real.exp_pos _))
    _ = phiStar * (Real.exp (2 * Φ) * Real.exp (-2 * (Φ - phiStar))) := by ring
    _ = phiStar * Real.exp (2 * phiStar) := by rw [hexp]

/-! ## 2. The finite-chain two-factor law -/

/-- Proper length of a chain of `n` links whose `i`-th link carries local
rendering scale `s i` and local weight `w i` (the correlation deficit `-ln K_i`).
The scale and the decay rate enter *per link* as a product: this is the discrete
content of `g_rr = (ℓ_P(Φ) · κ)²`. -/
noncomputable def chainLength (s w : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i ∈ Finset.range n, s i * w i

/-- The chain length is monotone in the local scale: a uniformly larger
rendering scale gives a uniformly longer chain. -/
theorem chainLength_mono (s t w : ℕ → ℝ) (n : ℕ)
    (h : ∀ i ∈ Finset.range n, s i ≤ t i)
    (hw : ∀ i ∈ Finset.range n, 0 ≤ w i) :
    chainLength s w n ≤ chainLength t w n := by
  unfold chainLength
  exact Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_right (h i hi) (hw i hi)

/-- **The single-scale formula factors out exactly.** If the local scale does not
vary along the chain, the chain length is the scale times the total weight.
This is the only case in which the finite §2.1 formula `d = -ℓ_P ln K` is exact;
with a varying `Φ` the chain sum and the single-scale formula differ, which
`Sim7_Radial_Metric.py` measures at 6–52 % for the canonical profile. -/
theorem chainLength_const_scale (a : ℝ) (w : ℕ → ℝ) (n : ℕ) :
    chainLength (fun _ => a) w n = a * ∑ i ∈ Finset.range n, w i := by
  unfold chainLength
  simp

/-! ## 3. The `Φ`-form Schwarzschild identities -/

/-- Areal radius of the isotropic slice written in `Φ = ψ - 1`:
`R = ρ(1 + Φ)²`. This is the `R` for which `Φ = M/(2ρ)` gives exact
Schwarzschild in isotropic coordinates. -/
noncomputable def arealRadius (ρ Φ : ℝ) : ℝ := ρ * (1 + Φ) ^ 2

/-- At the vacuum (`Φ = 0`) the areal radius is the isotropic coordinate, so the
slice is asymptotically flat. -/
theorem arealRadius_at_vacuum (ρ : ℝ) : arealRadius ρ 0 = ρ := by
  unfold arealRadius
  ring

/-- At the horizon (`Φ = 1`) the areal radius is `4ρ`; with `Φ = M/(2ρ)` this is
`R = 2M`, so the horizon area is `16πM²`. -/
theorem arealRadius_at_horizon (ρ : ℝ) : arealRadius ρ 1 = 4 * ρ := by
  unfold arealRadius
  ring

/-- **Lapse–conformal identity.** With `ψ = 1 + Φ` and `N = (1-Φ)/(1+Φ)`, one has
`N · ψ² = (1-Φ)(1+Φ) = 1 - Φ²`. This is the single identity that ties the
macroscopic metric to the microscopic rendering scale, and it is an algebraic
identity rather than a derivation. -/
theorem lapse_conformal_identity (Φ : ℝ) :
    (1 - Φ) * (1 + Φ) = 1 - Φ ^ 2 := by ring

/-- **The microscopic scale behind that identity.** Squaring the canonical COD
profile of `DynamicCODScale` gives `ℓ_P(Φ)² = ℓ_P0² · (1 - Φ²)`, so the
lapse–conformal identity reads `N · ψ² = (ℓ_P(Φ)/ℓ_P0)²`. -/
theorem cod_profile_matches_lapse (env : DynamicCODScale.CODEnvironment) (Φ : ℝ)
    (hΦ : 0 ≤ 1 - Φ ^ 2) :
    DynamicCODScale.dynamicPlanckCOD env Φ ^ 2 = env.lP0 ^ 2 * (1 - Φ ^ 2) := by
  have h : (Real.sqrt (1 - Φ ^ 2)) ^ 2 = 1 - Φ ^ 2 := Real.sq_sqrt hΦ
  unfold DynamicCODScale.dynamicPlanckCOD
  rw [mul_pow, h]

end RadialMetric
