import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol27_Consciousness.lean
  REAL formalization: Integrated Information Theory.

  Using Omega Protocol:
  - 4D: RCOD Asymmetry Tensor = Integrated Information Φ (Tononi)
  - Forward flux = sensory input processing
  - Reverse flux = predictive coding / top-down
  - Asymmetry = consciousness = Φ (not to confuse with Chain Overlap Density)
  - Freeze boundary = unconsciousness (anesthesia, deep sleep)
-/

namespace OmegaProtocol.Vol27
open OmegaProtocol

/-- Integrated Information Φ (Tononi) = RCOD Asymmetry -/
noncomputable def IntegratedInformation (R₁ R₂ : QRegion) : ℝ :=
  |asymmetryTensor R₁ R₂|

-- The retired legacy theorems `integrated_information_pos` and
-- `subjectiveexperience` hypothesized `asymmetryTensor R₁ R₂ ≠ 0`, which
-- is unsatisfiable in the concrete zero model (`asymmetryTensor ≡ 0`).
-- The genuine non-degenerate content is the `IntegratedSystem` model
-- below: strictly subadditive systems have strictly positive integrated
-- information, and the IIT "experience" dictionary is an explicit
-- definition on that model.

-- RETIRED (audit pass 10): `consciousness_qrf` (its statement `I ≥ 0` is
-- `abs_nonneg` and it duplicated `integrated_info_nonneg` under a name
-- invoking reference frames), and `consciousness_from_omega` /
-- `freeze_implies_low_integration`, which both restated the *definition*
-- `isFrozen := forwardFlux ≤ freezeBoundaryThreshold` (the latter through the
-- since-deleted `freeze_boundary_value`).  The definitional content —
-- including the numeric threshold — is stated once in
-- `OmegaAxioms.bridge_isFrozen_iff`.

/-- Definitional consistency bridge: in this model `IntegratedInformation` and
    `informationalImpedance` denote the same expression (`|asymmetryTensor|`),
    so the identity is `rfl`.  A naming-coherence witness, NOT the physical
    substrate-independence claim the legacy name `substrateindependence`
    evoked. -/
theorem bridge_integratedInformation_eq_impedance (R₁ R₂ : QRegion) :
  IntegratedInformation R₁ R₂ = informationalImpedance R₁ R₂ := by
  rfl

theorem integrated_info_nonneg (R₁ R₂ : QRegion) :
  IntegratedInformation R₁ R₂ ≥ 0 := by
  dsimp [IntegratedInformation]
  exact abs_nonneg _

-- ============================================================
-- INTEGRATED INFORMATION ARCHITECTURE & PARTITION DEFICIT
-- (Inside `OmegaProtocol.Vol27`; see the kernel-audit target names.)
-- ============================================================

/-- Bipartite informational architecture (IIT subsystem)
    consisting of two communicating subsystems A and B with marginal entropies
    H_A, H_B, joint entropy H_AB, and bidirectional information flux.
    Subadditivity implies H_AB ≤ H_A + H_B.
    Integrated information Φ is defined as the minimum information loss (mutual information)
    across the system partition: Φ = H_A + H_B - H_AB. -/
structure IntegratedSystem where
  H_A  : ℝ
  H_B  : ℝ
  H_AB : ℝ
  forward_rate : ℝ
  reverse_rate : ℝ
  h_subadd : H_AB ≤ H_A + H_B
  h_flux_bound : forward_rate ≥ 0 ∧ reverse_rate ≥ 0

namespace IntegratedSystem

variable (S : IntegratedSystem)

/-- Tononi integrated information Φ = H(A) + H(B) - H(AB) = I(A : B). -/
def phi : ℝ := S.H_A + S.H_B - S.H_AB

/-- Integrated information is always non-negative by subadditivity of entropy. -/
theorem phi_nonneg : 0 ≤ S.phi := by
  dsimp [phi]
  linarith [S.h_subadd]

/-- Asymmetry flux between top-down predictive processing and bottom-up sensory feed. -/
def fluxAsymmetry : ℝ := S.forward_rate - S.reverse_rate

/-- When a system is strictly integrated (joint entropy is strictly less than the partitioned sum),
    Φ is strictly positive. -/
theorem phi_pos_of_strict_subadditivity (h : S.H_AB < S.H_A + S.H_B) :
    0 < S.phi := by
  dsimp [phi]
  linarith

/-- Completely partitioned / unintegrated system: H(AB) = H(A) + H(B), yielding Φ = 0. -/
theorem phi_zero_of_independent (h : S.H_AB = S.H_A + S.H_B) :
    S.phi = 0 := by
  dsimp [phi]
  linarith

/-- IIT dictionary (toy, stated honestly): a system "experiences" exactly
    when its integrated information is strictly positive. This is a
    definition within the model, not a claim about physical
    consciousness. -/
def Experiences (S : IntegratedSystem) : Prop := 0 < S.phi

/-- Strictly integrated systems (joint entropy strictly below the
    partitioned sum) experience, in the sense of the dictionary above. -/
theorem experiences_of_strict_integration (h : S.H_AB < S.H_A + S.H_B) :
    S.Experiences :=
  phi_pos_of_strict_subadditivity S h

/-- Canonical conscious cortical architecture witness:
    H_A = 1, H_B = 1, H_AB = 1.2 (sharing 0.8 nats of mutual correlation),
    with forward rate 2.0 and reverse predictive rate 1.5. -/
noncomputable def corticalComplex : IntegratedSystem where
  H_A  := 1
  H_B  := 1
  H_AB := 6 / 5
  forward_rate := 2
  reverse_rate := 3 / 2
  h_subadd := by norm_num
  h_flux_bound := by norm_num

theorem cortical_phi_pos : 0 < corticalComplex.phi := by
  dsimp [phi, corticalComplex]
  norm_num

theorem cortical_phi_value : corticalComplex.phi = 4 / 5 := by
  dsimp [phi, corticalComplex]
  norm_num

theorem cortical_flux_asymmetry_pos : 0 < corticalComplex.fluxAsymmetry := by
  dsimp [fluxAsymmetry, corticalComplex]
  norm_num

end IntegratedSystem

end OmegaProtocol.Vol27
