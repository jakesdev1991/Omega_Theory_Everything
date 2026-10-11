import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol41_PostBiologicalEvolution.lean
  Formalization of Post-Biological Evolution.

  Model scope (stated honestly):
  * Substrates are three: carbon, silicon, hybrid. The transition-cost
    table is an explicit finite metric.
  * The substantive theorems are that the table is a genuine metric
    (self-distance zero, symmetry, triangle inequality) and that distinct
    substrates always cost at least one transition unit.
  * Omega-Protocol mapping: substrates = Q-regions (0D), information
    transfer = Φ (1D), substrate distance = Ω-metric (2D).
-/

namespace OmegaProtocol.Vol41
open OmegaProtocol

/-- Substrates available to post-biological evolution. -/
inductive Substrate
  | carbon
  | silicon
  | hybrid

/-- Legacy carrier name kept for `OmegaProtocol.lean`. -/
def SyntheticSubstrate : Type := Substrate

/-- Synthetic substrates exist, witnessed by the hybrid substrate. -/
theorem synthetic_substrate_witness : Nonempty SyntheticSubstrate :=
  ⟨Substrate.hybrid⟩

/-- Explicit transition-cost table between substrates. -/
def transitionCost : Substrate → Substrate → ℕ
  | .carbon, .carbon => 0
  | .silicon, .silicon => 0
  | .hybrid, .hybrid => 0
  | .carbon, .silicon => 2
  | .silicon, .carbon => 2
  | .carbon, .hybrid => 1
  | .hybrid, .carbon => 1
  | .silicon, .hybrid => 1
  | .hybrid, .silicon => 1

theorem cost_self_zero : ∀ s : Substrate, transitionCost s s = 0 := by
  intro s
  cases s <;> rfl

theorem cost_symmetric : ∀ s t : Substrate, transitionCost s t = transitionCost t s := by
  intro s t
  cases s <;> cases t <;> rfl

/-- The transition-cost table satisfies the triangle inequality. -/
theorem cost_triangle :
    ∀ s t u : Substrate,
      transitionCost s u ≤ transitionCost s t + transitionCost t u := by
  intro s t u
  cases s <;> cases t <;> cases u <;> decide

/-- Distinct substrates are never free to move between. -/
theorem distinct_substrate_cost_positive (s t : Substrate) (h : s ≠ t) :
    transitionCost s t ≥ 1 := by
  cases s <;> cases t <;> first | exact absurd rfl h | decide

/-- The hybrid substrate is the cheapest bridge between carbon and silicon. -/
theorem hybrid_cheapest_bridge :
    transitionCost .carbon .hybrid + transitionCost .hybrid .silicon ≤
    transitionCost .carbon .silicon := by
  decide

/-- Structural bridge: substrate coupling in the Q-region model is
    nonnegative. -/
theorem bridge_vol41_coupling_nonneg (R₁ R₂ : QRegion) : Φ R₁ R₂ ≥ 0 :=
  bridge_overlapDensity_nonneg R₁ R₂

/-- Structural bridge: the Ω-metric is reflexive on the Q-region model. -/
theorem bridge_vol41_metric_self_zero (R : QRegion) : d R R = 0 :=
  bridge_qregion_self_distance_zero R

-- ============================================================
-- MULTI-STEP SUBSTRATE MIGRATION PATHS
-- ============================================================

/-- Total transition cost of migrating from `s` along a sequence of
    intermediate substrates. -/
def pathCost : Substrate → List Substrate → ℕ
  | _, [] => 0
  | s, t :: rest => transitionCost s t + pathCost t rest

/-- Final destination reached by a migration path starting at `s`. -/
def pathEnd : Substrate → List Substrate → Substrate
  | s, [] => s
  | _, t :: rest => pathEnd t rest

/-- **Path triangle inequality**: any multi-step migration path from `s` to
    `pathEnd s p` costs at least the direct transition cost
    `transitionCost s (pathEnd s p)`. -/
theorem pathCost_ge_direct (s : Substrate) (p : List Substrate) :
    transitionCost s (pathEnd s p) ≤ pathCost s p := by
  induction p generalizing s with
  | nil =>
    dsimp [pathEnd, pathCost]
    rw [cost_self_zero]
  | cons t rest ih =>
    dsimp [pathEnd, pathCost]
    have htri := cost_triangle s t (pathEnd t rest)
    have hrest := ih t
    omega

/-- The two-step migration `carbon → hybrid → silicon` attains the direct
    metric bound `2`. -/
theorem hybrid_path_optimal :
    pathCost .carbon [.hybrid, .silicon] =
      transitionCost .carbon .silicon := by
  decide

end OmegaProtocol.Vol41
