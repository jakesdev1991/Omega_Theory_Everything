import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol48_ExtraDimensions.lean
  Formalization of Extra Dimensions (Kaluza-Klein).

  Model scope (stated honestly):
  * `Compactification` carries variable positive bulk coupling and
    compact volume. The effective coupling `G_eff = G/V` is proven
    positive, inverse to the volume, antitone in the volume at fixed
    bulk coupling, and arbitrarily small at large volume. A unit
    benchmark is included. This is dimensional-reduction algebra in a
    stated model, not a compactification derivation.
  * The retired legacy cluster fixed `G_higherdim := 1` and
    `CompactVolume := 1` in bare definitions (`G_effective`); that
    duplicate coupling-notion was removed so this module has exactly
    one: the `Compactification` structure fields. `OmegaProtocol.lean`
    now delegates to the parameterized theorems.
  * Omega-Protocol mapping: branes = Q-regions (0D), bulk fields = Φ
    (1D), compactification distance = Ω-metric (2D).
-/

namespace OmegaProtocol.Vol48
open OmegaProtocol

/-- Consistency bridge (VOL48): the Omega-metric `d` is reflexive (`d R R = 0`).
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `extradim_distance_self` evoked. -/
theorem bridge_vol48_metric_self_zero (R : QRegion) : d R R = 0 := by
  exact bridge_qregion_self_distance_zero R

/-- Consistency bridge (VOL48): the coupling `Φ` is nonnegative.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `extradim_phi_nonneg` evoked. -/
theorem bridge_vol48_coupling_nonneg (R₁ R₂ : QRegion) : Φ R₁ R₂ ≥ 0 := by
  exact bridge_overlapDensity_nonneg R₁ R₂

-- ============================================================
-- COMPACTIFICATION AT VARIABLE VOLUME
-- Effective 4D coupling `G_eff = G/V` with positivity, inverse
-- law, antitonicity in the compact volume, and the large-volume
-- weak-coupling limit. Algebra in a stated model.
-- ============================================================

/-- Bulk data: positive higher-dimensional Newton constant and
    positive compactification volume. -/
structure Compactification where
  Gbulk : ℝ
  volume : ℝ
  Gbulk_pos : 0 < Gbulk
  volume_pos : 0 < volume

namespace Compactification

/-- Effective 4D Newton constant. -/
noncomputable def Geff (C : Compactification) : ℝ := C.Gbulk / C.volume

theorem Geff_pos (C : Compactification) : 0 < C.Geff :=
  div_pos C.Gbulk_pos C.volume_pos

/-- Inverse law: `G_eff · V = G_bulk`. -/
theorem Geff_mul_volume (C : Compactification) :
    C.Geff * C.volume = C.Gbulk := by
  have hV : C.volume ≠ 0 := ne_of_gt C.volume_pos
  have hunfold : C.Geff = C.Gbulk / C.volume := rfl
  rw [hunfold, div_eq_mul_inv, mul_assoc, inv_mul_cancel₀ hV, mul_one]

/-- Larger compact volume means weaker effective 4D gravity. -/
theorem Geff_antitone {C₁ C₂ : Compactification}
    (hG : C₁.Gbulk = C₂.Gbulk) (hV : C₁.volume < C₂.volume) :
    C₂.Geff < C₁.Geff := by
  have h1 : C₁.Geff = C₁.Gbulk / C₁.volume := rfl
  have h2 : C₂.Geff = C₂.Gbulk / C₂.volume := rfl
  rw [h1, h2, hG, div_eq_mul_inv, div_eq_mul_inv]
  apply mul_lt_mul_of_pos_left _ C₂.Gbulk_pos
  exact (inv_lt_inv₀ C₂.volume_pos C₁.volume_pos).mpr hV

/-- Large-volume limit: at fixed bulk coupling, the effective coupling
    can be made smaller than any positive `ε` by taking the compact
    volume large. -/
theorem Geff_eventually_small (G ε : ℝ) (hG : 0 < G) (hε : 0 < ε) :
    ∃ V₀, ∀ V : ℝ, V₀ ≤ V →
      ∀ (C : Compactification), C.Gbulk = G → C.volume = V →
        C.Geff < ε := by
  refine ⟨G / ε + 1, fun V hV C hGbulk hVol => ?_⟩
  have hV0 : (0 : ℝ) < G / ε + 1 := by positivity
  have hVpos : 0 < V := lt_of_lt_of_le hV0 hV
  have hunfold : C.Geff = C.Gbulk / C.volume := rfl
  rw [hunfold, hGbulk, hVol, div_lt_iff₀ hVpos]
  have h1 : G / ε < V := by linarith
  have h3 := (div_lt_iff₀ hε).mp h1
  linarith

/-- Unit benchmark: unit bulk coupling and unit volume give unit
    effective coupling. -/
def unitCompact : Compactification :=
  { Gbulk := 1, volume := 1, Gbulk_pos := by norm_num,
    volume_pos := by norm_num }

theorem unitCompact_Geff : unitCompact.Geff = 1 := by
  unfold Geff
  norm_num [unitCompact]

end Compactification

end OmegaProtocol.Vol48
