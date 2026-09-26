import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol37_Morphogenesis.lean
  Formalization of Morphogenesis (Turing Patterns).

  Model scope (stated honestly):
  * The legacy diffusion-constant comparison (`InhibitorDiffusion >
    ActivatorDiffusion`) is kept for `OmegaProtocol.lean`, but a bare
    diffusivity ratio is NOT a Turing instability proof.
  * `ReactionSystem` is a genuine two-species reaction-diffusion
    linearization: Jacobian plus positive diffusivities. The
    diffusion-driven instability criterion (drive + discriminant imply
    a negative-dispersion wavenumber), the exact inhibitor-faster
    ratio threshold, and a fully verified activator-inhibitor witness
    are proven below. This is textbook linear-stability algebra in a
    stated model, not a PDE derivation of pattern formation.
  * Omega-Protocol mapping: cells = Q-regions (0D), chemical gradients
    = Φ (1D), spatial pattern distance = Ω-metric (2D).
-/

namespace OmegaProtocol.Vol37
open OmegaProtocol

def ActivatorDiffusion : ℝ := 1
def InhibitorDiffusion : ℝ := 2
theorem turing_condition : InhibitorDiffusion > ActivatorDiffusion := by
  norm_num [InhibitorDiffusion, ActivatorDiffusion]

theorem turing_instability_ratio
  (h_pos : ActivatorDiffusion > 0) :
  InhibitorDiffusion / ActivatorDiffusion > 1 := by
  have h_turing := turing_condition
  rw [gt_iff_lt, one_lt_div h_pos]
  exact h_turing

/-- Consistency bridge (VOL37): the Omega-metric `d` is reflexive (`d R R = 0`).
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `morphogenesis_from_omega` evoked. -/
theorem bridge_vol37_metric_self_zero (R : QRegion) : d R R = 0 := by
  exact qregion_self_distance_zero R

theorem turing_ratio_pos (h_pos : ActivatorDiffusion > 0) :
  InhibitorDiffusion / ActivatorDiffusion > 0 := by
  have h := turing_instability_ratio h_pos
  linarith

/-- Consistency bridge (VOL37): the von Neumann entropy is nonnegative.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `morphogenesis_entropy_nonneg` evoked. -/
theorem bridge_vol37_entropy_nonneg (R : QRegion) : vonNeumannEntropy R ≥ 0 := by
  exact monotonicity_lemma R

-- ============================================================
-- REACTION-DIFFUSION STABILITY (TURING)
-- A 2-species Jacobian + diffusivities model with the classical
-- diffusion-driven instability criterion. Textbook algebra in a
-- stated model, not a PDE derivation.
-- ============================================================

/-- Two-species reaction-diffusion linearization: Jacobian entries
    `a b / c d` at a homogeneous steady state, plus diffusivities. -/
structure ReactionSystem where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  Du : ℝ
  Dv : ℝ
  Du_pos : 0 < Du
  Dv_pos : 0 < Dv

namespace ReactionSystem

/-- Jacobian determinant. -/
def detJ (S : ReactionSystem) : ℝ := S.a * S.d - S.b * S.c

/-- Jacobian trace. -/
def traceJ (S : ReactionSystem) : ℝ := S.a + S.d

/-- Dispersion relation: determinant of `J - k²D` as a function of
    `k2 = k² ≥ 0`. Diffusion-driven instability is `dispersion < 0`
    for some `k2 > 0`. -/
def dispersion (S : ReactionSystem) (k2 : ℝ) : ℝ :=
  S.detJ - k2 * (S.Du * S.d + S.Dv * S.a) + k2 ^ 2 * (S.Du * S.Dv)

/-- Stability without diffusion: the classical 2D trace/determinant
    conditions. -/
def StableNoDiffusion (S : ReactionSystem) : Prop :=
  S.traceJ < 0 ∧ 0 < S.detJ

/-- Activator-inhibitor sign pattern: `u` self-activates, `v`
    self-inhibits, `u` activates `v`, `v` inhibits `u`. -/
def IsActivatorInhibitor (S : ReactionSystem) : Prop :=
  0 < S.a ∧ S.b < 0 ∧ 0 < S.c ∧ S.d < 0

/-- Turing instability criterion: with positive diffusion drive and a
    positive discriminant, some wavenumber has negative dispersion.
    Proved by evaluating at the minimizing `k2* = P / (2Q)`. -/
theorem turing_instability (S : ReactionSystem)
    (hDrive : 0 < S.Du * S.d + S.Dv * S.a)
    (hDisc : 4 * (S.Du * S.Dv) * S.detJ
      < (S.Du * S.d + S.Dv * S.a) ^ 2) :
    ∃ k2 > 0, S.dispersion k2 < 0 := by
  have hDu := S.Du_pos
  have hDv := S.Dv_pos
  have hQ : 0 < S.Du * S.Dv := mul_pos hDu hDv
  have h2Q : (2 : ℝ) * (S.Du * S.Dv) ≠ 0 := by positivity
  have h4Q : (4 : ℝ) * (S.Du * S.Dv) ≠ 0 := by positivity
  refine ⟨(S.Du * S.d + S.Dv * S.a) / (2 * (S.Du * S.Dv)),
    div_pos hDrive (by positivity), ?_⟩
  have hval : S.dispersion ((S.Du * S.d + S.Dv * S.a) / (2 * (S.Du * S.Dv)))
      = S.detJ - (S.Du * S.d + S.Dv * S.a) ^ 2 / (4 * (S.Du * S.Dv)) := by
    unfold dispersion
    field_simp
    ring
  rw [hval]
  have h4Qpos : 0 < 4 * (S.Du * S.Dv) := by positivity
  rw [sub_lt_zero, lt_div_iff₀ h4Qpos]
  have hring : S.detJ * (4 * (S.Du * S.Dv))
      = 4 * (S.Du * S.Dv) * S.detJ := by ring
  rw [hring]
  exact hDisc

/-- The diffusion drive is positive iff the inhibitor/activator
    diffusivity ratio exceeds the reaction threshold `(-d)/a`.
    This is the precise form of "the inhibitor must diffuse faster". -/
theorem drive_iff_ratio (S : ReactionSystem) (ha : 0 < S.a)
    (hd : S.d < 0) :
    0 < S.Du * S.d + S.Dv * S.a ↔ S.Dv / S.Du > (-S.d) / S.a := by
  rw [gt_iff_lt, div_lt_div_iff₀ ha S.Du_pos]
  constructor <;> intro h <;> linarith

/-- Concrete activator-inhibitor witness: stable without diffusion,
    diffusion-driven unstable with the inhibitor diffusing 10× faster. -/
def turingWitness : ReactionSystem :=
  { a := 1, b := -2, c := 2, d := -3, Du := 1, Dv := 10,
    Du_pos := by norm_num, Dv_pos := by norm_num }

theorem turingWitness_stable : turingWitness.StableNoDiffusion := by
  constructor <;> norm_num [StableNoDiffusion, traceJ, detJ, turingWitness]

theorem turingWitness_ai : turingWitness.IsActivatorInhibitor := by
  norm_num [IsActivatorInhibitor, turingWitness]

theorem turingWitness_inhibitor_faster :
    turingWitness.Dv > turingWitness.Du := by
  norm_num [turingWitness]

theorem turingWitness_unstable :
    ∃ k2 > 0, turingWitness.dispersion k2 < 0 := by
  refine turing_instability turingWitness
    (by norm_num [turingWitness]) (by norm_num [turingWitness, detJ])

/-- The optimal wavenumber `k2* = 7/20` has dispersion `-9/40`. -/
theorem turingWitness_optimal_value :
    turingWitness.dispersion (7 / 20) = -9 / 40 := by
  norm_num [dispersion, detJ, turingWitness]

end ReactionSystem

end OmegaProtocol.Vol37
