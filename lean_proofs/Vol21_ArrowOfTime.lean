import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation
import Vol16_NonEquilibriumThermodynamics

/-!
# A conditional entropy arrow along a differentiable history

This module now reuses Vol16's explicit entropy process rather than assigning
zero to both a singleton region's entropy and a supposed time derivative.
The arrow theorem is conditional on isolation and nonnegative production; it
is not a derivation of a cosmological time orientation from Omega's Q-regions.

There is no identification of physical time with computational latency. The
legacy latency arithmetic is retained as a separate, unrelated inequality.
-/
namespace OmegaProtocol.Vol21
open OmegaProtocol

abbrev CosmicTime := ℝ
abbrev CosmicHistory := Vol16.EntropyProcess

def CosmicEntropy (history : CosmicHistory) : CosmicTime → ℝ := Vol16.Entropy history
noncomputable def EntropyRate (history : CosmicHistory) (t : CosmicTime) : ℝ :=
  deriv (CosmicEntropy history) t

theorem entropy_rate_nonneg (history : CosmicHistory) (t : CosmicTime)
    (h : Vol16.EntropyFlux history t = 0) : 0 ≤ EntropyRate history t :=
  Vol16.isolated_entropy_nondecreasing history t h

/-- A derivative-to-monotonicity implication, using actual differentiability. -/
theorem entropy_derivative (history : CosmicHistory) (t₁ t₂ : CosmicTime) (h : t₁ ≤ t₂)
    (h_rate : ∀ t, 0 ≤ EntropyRate history t) :
    CosmicEntropy history t₁ ≤ CosmicEntropy history t₂ :=
  monotone_of_deriv_nonneg
    (fun t => (history.entropy_hasDerivAt t).differentiableAt) h_rate h

theorem arrow_of_time (history : CosmicHistory) (h_isolated : Vol16.IsIsolated history)
    (t₁ t₂ : CosmicTime) (h : t₁ ≤ t₂) :
    CosmicEntropy history t₁ ≤ CosmicEntropy history t₂ :=
  Vol16.isolated_entropy_monotone history h_isolated h

/-- Nonnegative initial entropy is propagated forward, not silently postulated
    for all negative and positive times. -/
theorem arrow_of_time_entropy_nonneg (history : CosmicHistory)
    (h_isolated : Vol16.IsIsolated history) (t₀ t : CosmicTime)
    (h₀ : 0 ≤ CosmicEntropy history t₀) (ht : t₀ ≤ t) :
    0 ≤ CosmicEntropy history t :=
  Vol16.entropy_nonneg_after_initial history h_isolated t₀ t h₀ ht

theorem computational_latency_nonneg (n : ℕ) : 0 ≤ computationalLatency n := by
  unfold computationalLatency
  split_ifs <;> positivity

end OmegaProtocol.Vol21
