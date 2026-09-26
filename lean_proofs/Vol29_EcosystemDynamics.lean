import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol29_EcosystemDynamics.lean
  REAL formalization: Ecosystem Dynamics (Lotka-Volterra).

  Using Omega Protocol:
  - 0D: Species = Q-Regions
  - 1D: Predator-prey interaction = Φ (mutual information)
  - 2D: Ω-Metric = ecological distance
  - 3D: Informational Viscosity = population dynamics timescale
  - 4D: RCOD Asymmetry = ecosystem stability
-/

namespace OmegaProtocol.Vol29
open OmegaProtocol

/-- Lotka-Volterra prey growth rate: dx/dt = αx - βxy -/
noncomputable def PreyRate (alpha beta x y : ℝ) : ℝ :=
  alpha * x - beta * x * y

/-- Lotka-Volterra predator growth rate: dy/dt = δxy - γy -/
noncomputable def PredatorRate (delta gamma x y : ℝ) : ℝ :=
  delta * x * y - gamma * y

/-- THEOREM: Equilibrium Point (GENUINE PROOF)
    At x = γ/δ, y = α/β, both rates are zero. -/
theorem lotka_volterra_equilibrium
  (alpha beta delta gamma : ℝ)
  (hd : delta ≠ 0) (hb : beta ≠ 0) :
  PreyRate alpha beta (gamma / delta) (alpha / beta) = 0 ∧
  PredatorRate delta gamma (gamma / delta) (alpha / beta) = 0 := by
  constructor
  · dsimp [PreyRate]
    field_simp
    ring
  · dsimp [PredatorRate]
    field_simp
    ring

/-- Consistency bridge (VOL29): the Omega-metric `d` is reflexive (`d R R = 0`).
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `ecosystem_from_omega` evoked. -/
theorem bridge_vol29_metric_self_zero (R : QRegion) : d R R = 0 := by
  exact qregion_self_distance_zero R

theorem predator_prey_symmetry (alpha beta delta gamma x y : ℝ) :
  PreyRate alpha beta x y + PredatorRate delta gamma x y = alpha * x - gamma * y + (delta - beta) * x * y := by
  dsimp [PreyRate, PredatorRate]; ring

/-- Consistency bridge (VOL29): the von Neumann entropy satisfies the bound `S ≤ π`.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `ecosystem_entropy_bound` evoked. -/
theorem bridge_vol29_entropy_bounded (R : QRegion) : vonNeumannEntropy R ≤ Real.pi := by
  exact entropy_bounded R

-- ============================================================
-- FULL EQUILIBRIUM CLASSIFICATION
-- With all rates nonzero, the only equilibria are extinction and
-- coexistence; with positive rates, coexistence is positive.
-- Stability analysis is not attempted here.
-- ============================================================

/-- Extinction is always an equilibrium. -/
theorem extinction_equilibrium (alpha beta delta gamma : ℝ) :
    PreyRate alpha beta 0 0 = 0 ∧ PredatorRate delta gamma 0 0 = 0 := by
  constructor <;> simp [PreyRate, PredatorRate]

/-- Classification: with nonzero rates, every equilibrium is either
    extinction or the coexistence point. -/
theorem equilibrium_classification (alpha beta delta gamma x y : ℝ)
    (ha : alpha ≠ 0) (hb : beta ≠ 0) (hd : delta ≠ 0) (hg : gamma ≠ 0)
    (hprey : PreyRate alpha beta x y = 0)
    (hpred : PredatorRate delta gamma x y = 0) :
    (x = 0 ∧ y = 0) ∨ (x = gamma / delta ∧ y = alpha / beta) := by
  have hfactor1 : x * (alpha - beta * y) = 0 := by
    have hform : x * (alpha - beta * y) = PreyRate alpha beta x y := by
      simp [PreyRate]
      ring
    rw [hform, hprey]
  have hfactor2 : y * (delta * x - gamma) = 0 := by
    have hform : y * (delta * x - gamma) = PredatorRate delta gamma x y := by
      simp [PredatorRate]
      ring
    rw [hform, hpred]
  rcases mul_eq_zero.mp hfactor1 with hx | hy
  · rcases mul_eq_zero.mp hfactor2 with hy2 | hx2
    · exact Or.inl ⟨hx, hy2⟩
    · exfalso
      rw [hx] at hx2
      simp only [mul_zero, zero_sub, neg_eq_zero] at hx2
      exact hg hx2
  · rcases mul_eq_zero.mp hfactor2 with hy2 | hx2
    · exfalso
      rw [hy2] at hy
      simp only [mul_zero, sub_zero] at hy
      exact ha hy
    · right
      constructor
      · rw [eq_div_iff hd]
        linarith [hx2]
      · rw [eq_div_iff hb]
        linarith [hy]

/-- With positive rates, the coexistence equilibrium is positive. -/
theorem coexistence_positive (alpha beta delta gamma : ℝ)
    (ha : 0 < alpha) (hb : 0 < beta) (hd : 0 < delta) (hg : 0 < gamma) :
    0 < gamma / delta ∧ 0 < alpha / beta :=
  ⟨div_pos hg hd, div_pos ha hb⟩

end OmegaProtocol.Vol29
