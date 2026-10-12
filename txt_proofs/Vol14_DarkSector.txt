import Mathlib
import OmegaUnifiedFoundation

/-
  Vol14_DarkSector.lean
  Machine-checked model of the Dark Sector (Dark Matter & Dark Energy).

  Contains:
  1. Flat Universe Cosmic Sum: The legacy (0, 0, 1) concrete baseline.
  2. Parameterized Cosmic Density Budget: Real cosmological fractions on the 2-simplex,
     proving that for any non-negative fractions summing to 1, dark energy is bounded
     and uniquely determined by the matter sectors.
  3. Strict Bounds: If baryonic and dark matter are strictly positive, dark energy
     fraction is strictly below 1.
-/

namespace OmegaProtocol.Vol14
open OmegaProtocol

-- Legacy minimal constants maintained for backwards compatibility
def OmegaBaryon : ℝ := 0
def OmegaDarkMatter : ℝ := 0
def OmegaDarkEnergy : ℝ := 1

/-- The legacy constants above (`0`, `0`, `1`) sum to 1 by `norm_num`: flatness
    here is a chosen constant, not a derived property, and the parameterized
    model is `CosmicDensityBudget`.  Renamed from `flat_universe_sum` in the
    Phase 6 rename pass. -/
theorem legacy_density_constants_sum_eq_one : OmegaBaryon + OmegaDarkMatter + OmegaDarkEnergy = 1 := by
  norm_num [OmegaBaryon, OmegaDarkMatter, OmegaDarkEnergy]

/-- Over the legacy constants, the dark-energy parameter is 1 − Ω_b − Ω_m by
    `linarith`; the parameterized deduction is
    `CosmicDensityBudget.dark_energy_eq`.  Renamed from `dark_energy_deduction`
    in the Phase 6 rename pass. -/
theorem legacy_density_constants_dark_energy_eq :
  OmegaDarkEnergy = 1 - OmegaBaryon - OmegaDarkMatter := by
  have h := legacy_density_constants_sum_eq_one
  calc OmegaDarkEnergy 
      = 1 - OmegaBaryon - OmegaDarkMatter := by linarith

-- ============================================================
-- PARAMETERIZED COSMIC DENSITY BUDGET (Non-degenerate Simplex)
-- ============================================================

/-- A flat cosmological density budget on the 2-simplex:
    Ω_b ≥ 0, Ω_dm ≥ 0, Ω_de ≥ 0, and Ω_b + Ω_dm + Ω_de = 1. -/
structure CosmicDensityBudget where
  omegaB : ℝ
  omegaDM : ℝ
  omegaDE : ℝ
  hb_nonneg : 0 ≤ omegaB
  hdm_nonneg : 0 ≤ omegaDM
  hde_nonneg : 0 ≤ omegaDE
  h_sum : omegaB + omegaDM + omegaDE = 1

namespace CosmicDensityBudget

variable (b : CosmicDensityBudget)

/-- In any flat budget, dark energy is uniquely determined by the matter components. -/
theorem dark_energy_eq : b.omegaDE = 1 - b.omegaB - b.omegaDM := by
  linarith [b.h_sum]

/-- Total matter density fraction Ω_m = Ω_b + Ω_dm. -/
def omegaMatter : ℝ := b.omegaB + b.omegaDM

theorem omegaMatter_nonneg : 0 ≤ b.omegaMatter := by
  dsimp [omegaMatter]
  linarith [b.hb_nonneg, b.hdm_nonneg]

theorem matter_dark_energy_complement : b.omegaMatter + b.omegaDE = 1 := by
  dsimp [omegaMatter]
  linarith [b.h_sum]

theorem dark_energy_le_one : b.omegaDE ≤ 1 := by
  have h := b.matter_dark_energy_complement
  have hm := b.omegaMatter_nonneg
  linarith

/-- If matter components are strictly positive, dark energy is strictly less than 1. -/
theorem dark_energy_lt_one_of_matter_pos
    (hb : 0 < b.omegaB) (hdm : 0 ≤ b.omegaDM) : b.omegaDE < 1 := by
  have h_sum := b.h_sum
  linarith

/-- Concrete realistic ΛCDM cosmological benchmark:
    Ω_b ≈ 0.05, Ω_dm ≈ 0.26, Ω_de ≈ 0.69. -/
noncomputable def standardLCDM : CosmicDensityBudget where
  omegaB := 5 / 100
  omegaDM := 26 / 100
  omegaDE := 69 / 100
  hb_nonneg := by norm_num
  hdm_nonneg := by norm_num
  hde_nonneg := by norm_num
  h_sum := by norm_num

theorem standardLCDM_matter :
    standardLCDM.omegaMatter = 31 / 100 := by
  dsimp [omegaMatter, standardLCDM]
  norm_num

theorem standardLCDM_dark_energy_dominant :
    standardLCDM.omegaMatter < standardLCDM.omegaDE := by
  dsimp [omegaMatter, standardLCDM]
  norm_num

end CosmicDensityBudget

end OmegaProtocol.Vol14
