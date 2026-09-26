import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-!
# Scalar feedback decay

The model is x(n) = initial * g^n, with x(n+1) = g*x(n).
The legacy `feedback_convergence` theorem only establishes a one-step inequality.
Actual convergence to zero is proved separately for 0 ≤ g < 1; at g = 1 a
nonzero initial state persists. This is scalar model analysis, not a derivation
of a controller or a stability guarantee for an arbitrary physical system.
-/

namespace OmegaProtocol.Vol32
open OmegaProtocol

/-- Legacy name: one-step nonincrease, not a limit statement. -/
theorem feedback_convergence (g : ℝ) (hg0 : 0 ≤ g) (hg1 : g ≤ 1) (n : ℕ) :
  g ^ (n + 1) ≤ g ^ n := by
  exact pow_le_pow_of_le_one hg0 hg1 (Nat.le_succ n)

/-- Legacy export name for the scalar power inequality. -/
theorem cybernetics_from_omega (g : ℝ) (hg0 : 0 ≤ g) (hg1 : g ≤ 1) (n : ℕ) :
  g ^ (n + 1) ≤ g ^ n := by
  exact feedback_convergence g hg0 hg1 n

/-- Consistency bridge (VOL32): the Omega-metric `d` is nonnegative.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `feedback_stability` evoked. -/
theorem bridge_vol32_metric_nonneg (R₁ R₂ : QRegion) : d R₁ R₂ ≥ 0 := by
  exact distance_nonneg R₁ R₂

/-- Consistency bridge (VOL32): the von Neumann entropy is nonnegative.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `control_loop_entropy` evoked. -/
theorem bridge_vol32_entropy_nonneg (R : QRegion) : vonNeumannEntropy R ≥ 0 := by
  exact monotonicity_lemma R

def feedbackState (g initial : ℝ) (n : ℕ) : ℝ := initial * g ^ n

theorem feedbackState_step (g initial : ℝ) (n : ℕ) :
    feedbackState g initial (n + 1) = g * feedbackState g initial n := by
  unfold feedbackState
  rw [pow_succ]
  ring

/-- Genuine asymptotic convergence; strictness of the gain bound is essential. -/
theorem feedback_tends_to_zero (g initial : ℝ) (hg0 : 0 ≤ g) (hg1 : g < 1) :
    Filter.Tendsto (feedbackState g initial) Filter.atTop (nhds 0) := by
  have hg : Filter.Tendsto (fun n : ℕ => g ^ n) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hg0 hg1
  have h : Filter.Tendsto (fun n : ℕ => initial * g ^ n)
      Filter.atTop (nhds (initial * 0)) := tendsto_const_nhds.mul hg
  change Filter.Tendsto (fun n : ℕ => initial * g ^ n) Filter.atTop (nhds 0)
  simpa only [mul_zero] using h

theorem unit_gain_preserves_state (initial : ℝ) (n : ℕ) :
    feedbackState 1 initial n = initial := by
  simp [feedbackState]

/-- A counterexample to interpreting the old non-strict bound as decay. -/
theorem unit_gain_not_decay (initial : ℝ) (hi : initial ≠ 0) :
    ¬ Filter.Tendsto (feedbackState 1 initial) Filter.atTop (nhds 0) := by
  intro hzero
  have hconstant : Filter.Tendsto (feedbackState 1 initial)
      Filter.atTop (nhds initial) := by
    change Filter.Tendsto (fun n : ℕ => initial * (1 : ℝ) ^ n) Filter.atTop (nhds initial)
    simpa only [one_pow, mul_one] using
      (tendsto_const_nhds : Filter.Tendsto (fun _ : ℕ => initial) Filter.atTop (nhds initial))
  exact hi (tendsto_nhds_unique hconstant hzero)

end OmegaProtocol.Vol32
