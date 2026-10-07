import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol44_UniversalExpansion.lean
  Formalization of Universal Expansion (de Sitter horizon entropy).

  Model scope (stated honestly):
  * `DeSitterSpace` carries variable positive Λ and G. Horizon radius,
    area and entropy are proven positive, the exact scaling law
    `S = 3π/(GΛ)` is derived, and entropy is antitone in Λ at fixed G.
    A unit benchmark (`S = 3π`) is included. This is static horizon
    thermodynamics in a stated model, not cosmic dynamics.
  * The retired legacy cluster fixed `Λ := 1` in bare definitions
    (`Lambda_positive`, `DeSitterRadiusSq`, `DeSitterArea`,
    `DeSitterEntropy`); that duplicate Λ-notion was removed so this
    module has exactly one cosmological constant: the structure field
    `DeSitterSpace.Lambda`. ΩProtocol now delegates to the
    parameterized theorems.
-/

namespace OmegaProtocol.Vol44
open OmegaProtocol

/-- Consistency bridge (VOL44): the Omega-metric `d` is reflexive (`d R R = 0`).
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `expansion_distance_self` evoked. -/
theorem bridge_vol44_metric_self_zero (R : QRegion) : d R R = 0 := by
  exact bridge_qregion_self_distance_zero R

/-- Consistency bridge (VOL44): the coupling `Φ` is nonnegative.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `expansion_phi_nonneg` evoked. -/
theorem bridge_vol44_coupling_nonneg (R₁ R₂ : QRegion) : Φ R₁ R₂ ≥ 0 := by
  exact bridge_overlapDensity_nonneg R₁ R₂

-- ============================================================
-- DE SITTER SPACE AT VARIABLE COSMOLOGICAL CONSTANT
-- Horizon radius, area and entropy as functions of Λ > 0 and
-- G > 0, with the exact scaling law and antitonicity in Λ.
-- ============================================================

/-- De Sitter data: positive cosmological constant and positive
    Newton constant (kept general, not fixed to 1). -/
structure DeSitterSpace where
  Lambda : ℝ
  G : ℝ
  Lambda_pos : 0 < Lambda
  G_pos : 0 < G

namespace DeSitterSpace

/-- Horizon-radius-squared `r² = 3/Λ`. -/
noncomputable def radiusSq (S : DeSitterSpace) : ℝ := 3 / S.Lambda

/-- Horizon area `A = 4πr²`. -/
noncomputable def area (S : DeSitterSpace) : ℝ := 4 * Real.pi * S.radiusSq

/-- Gibbons–Hawking entropy `S = A/(4G)`. -/
noncomputable def entropy (S : DeSitterSpace) : ℝ := S.area / (4 * S.G)

theorem radiusSq_pos (S : DeSitterSpace) : 0 < S.radiusSq := by
  unfold radiusSq
  exact div_pos (by norm_num) S.Lambda_pos

theorem area_pos (S : DeSitterSpace) : 0 < S.area := by
  have h := S.radiusSq_pos
  have hpi := Real.pi_pos
  unfold area
  positivity

theorem entropy_pos (S : DeSitterSpace) : 0 < S.entropy := by
  have hA := S.area_pos
  have hG := S.G_pos
  unfold entropy
  exact div_pos hA (by positivity)

/-- Exact scaling law: `S = 3π/(GΛ)`. -/
theorem entropy_formula (S : DeSitterSpace) :
    S.entropy = 3 * Real.pi / (S.G * S.Lambda) := by
  unfold entropy area radiusSq
  have hG : S.G ≠ 0 := ne_of_gt S.G_pos
  have hL : S.Lambda ≠ 0 := ne_of_gt S.Lambda_pos
  field_simp

/-- Larger Λ means smaller horizon entropy (at fixed G). -/
theorem entropy_antitone {S₁ S₂ : DeSitterSpace}
    (hG : S₁.G = S₂.G) (hΛ : S₁.Lambda < S₂.Lambda) :
    S₂.entropy < S₁.entropy := by
  rw [entropy_formula, entropy_formula, hG]
  have hpi : (0 : ℝ) < 3 * Real.pi := by positivity
  have h1 : 0 < S₂.G * S₁.Lambda := mul_pos S₂.G_pos S₁.Lambda_pos
  have h2 : 0 < S₂.G * S₂.Lambda := mul_pos S₂.G_pos S₂.Lambda_pos
  rw [div_lt_div_iff₀ h2 h1]
  have hGΛ : S₂.G * S₁.Lambda < S₂.G * S₂.Lambda :=
    mul_lt_mul_of_pos_left hΛ S₂.G_pos
  exact mul_lt_mul_of_pos_left hGΛ hpi

/-- Unit benchmark: Λ = G = 1 gives `S = 3π`. -/
def unitDeSitter : DeSitterSpace :=
  { Lambda := 1, G := 1, Lambda_pos := by norm_num, G_pos := by norm_num }

theorem unitDeSitter_entropy : unitDeSitter.entropy = 3 * Real.pi := by
  rw [entropy_formula]
  norm_num [unitDeSitter]

end DeSitterSpace

end OmegaProtocol.Vol44
