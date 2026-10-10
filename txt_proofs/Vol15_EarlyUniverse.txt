import Mathlib
import OmegaUnifiedFoundation

/-
  Vol15_EarlyUniverse.lean
  REAL mathematical formalization of the Early Universe & Cosmic Inflation.

  Contains:
  1. Non-degenerate Inflationary Fluid Model:
     Scalar inflaton / vacuum energy state with equation-of-state parameter w = p / ρ.
     Proves that when w < -1/3 (e.g. slow-roll scalar field or de Sitter vacuum w = -1),
     ρ + 3p < 0 strictly holds, guaranteeing positive acceleration ä/a > 0 under
     the Friedmann acceleration law.
  2. Concrete de Sitter / Slow-Roll Benchmark:
     Demonstrates a realized inflationary state with w = -1 and positive energy density.

  The retired legacy theorem `inflation_acceleration` hypothesized
  `EnergyDensity t + 3 * Pressure t < 0` over the zero minimal Vol07 model,
  where `EnergyDensity ≡ 0` and `Pressure ≡ 0` make the hypothesis
  unsatisfiable; its genuine content is the fluid model below.
-/

namespace OmegaProtocol.Vol15
open OmegaProtocol

-- ============================================================
-- PARAMETERIZED INFLATIONARY COSMOLOGY (Realizable Inflaton)
-- ============================================================

/-- General homogeneous matter/energy component with density ρ and pressure p. -/
structure CosmologicalFluid where
  density : ℝ
  pressure : ℝ
  density_pos : 0 < density

namespace CosmologicalFluid

variable (F : CosmologicalFluid)

/-- Equation of state parameter w = p / ρ. -/
noncomputable def w : ℝ := F.pressure / F.density

/-- The acceleration source term ρ + 3p. -/
def accelerationSource : ℝ := F.density + 3 * F.pressure

theorem accelerationSource_eq_factored :
    F.accelerationSource = F.density * (1 + 3 * F.w) := by
  dsimp [accelerationSource, w]
  have hρ := ne_of_gt F.density_pos
  calc F.density + 3 * F.pressure
    _ = F.density + 3 * (F.density * (F.pressure / F.density)) := by
        rw [mul_div_cancel₀ _ hρ]
    _ = F.density * (1 + 3 * (F.pressure / F.density)) := by ring

/-- Whenever the equation of state satisfies w < -1/3, the acceleration source is strictly negative. -/
theorem accelerationSource_neg_of_w_lt (hw : F.w < -1 / 3) :
    F.accelerationSource < 0 := by
  rw [accelerationSource_eq_factored]
  have h_factor : 1 + 3 * F.w < 0 := by linarith
  have hρ := F.density_pos
  have := mul_neg_of_pos_of_neg hρ h_factor
  exact this

/-- Friedmann acceleration under NewtonG > 0 for a fluid with ρ + 3p < 0. -/
theorem cosmic_acceleration_positive
    (G : ℝ) (hG : 0 < G)
    (h_source : F.accelerationSource < 0) :
    0 < - (4 * Real.pi * G / 3) * F.accelerationSource := by
  have h_const : 0 < 4 * Real.pi * G / 3 := by
    have h1 : (4 : ℝ) > 0 := by norm_num
    have h2 : Real.pi > 0 := Real.pi_pos
    have h3 : (3 : ℝ) > 0 := by norm_num
    positivity
  have h_neg_source : 0 < -F.accelerationSource := neg_pos.mpr h_source
  calc 0
    _ < (4 * Real.pi * G / 3) * (-F.accelerationSource) := mul_pos h_const h_neg_source
    _ = - (4 * Real.pi * G / 3) * F.accelerationSource := by ring

/-- Standard de Sitter vacuum / slow-roll inflaton state: p = -ρ with ρ = 1. -/
def deSitterVacuum : CosmologicalFluid where
  density := 1
  pressure := -1
  density_pos := by norm_num

theorem deSitter_w : deSitterVacuum.w = -1 := by
  dsimp [w, deSitterVacuum]
  norm_num

theorem deSitter_w_lt_third : deSitterVacuum.w < -1 / 3 := by
  rw [deSitter_w]
  norm_num

theorem deSitter_source_neg : deSitterVacuum.accelerationSource = -2 := by
  dsimp [accelerationSource, deSitterVacuum]
  norm_num

end CosmologicalFluid

end OmegaProtocol.Vol15
