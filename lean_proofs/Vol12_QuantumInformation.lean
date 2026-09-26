import Mathlib
import OmegaUnifiedFoundation
/-!
# Two-amplitude overlap obstruction to universal copying

The state carrier is genuinely two-dimensional. The algebraic copying condition
is stated as a hypothesis; no tensor-product unitary implementation is modeled.
A normalized, nonorthogonal pair witnesses that this necessary condition cannot
hold for all normalized states. The separate entropy-bound stub remains legacy.
-/

namespace OmegaProtocol.Vol12
open OmegaProtocol

-- ============================================================
-- QUANTUM STATE COPYING (NO-CLONING THEOREM)
-- ============================================================

/-- Two complex amplitudes; normalization is an explicit predicate. -/
def InfoState : Type := ℂ × ℂ

noncomputable def state_inner (ψ φ : InfoState) : ℂ :=
  starRingEnd ℂ ψ.1 * φ.1 + starRingEnd ℂ ψ.2 * φ.2

def IsNormalized (ψ : InfoState) : Prop := state_inner ψ ψ = 1

/-- A blank "target" state for the copying operation -/
def BlankState : InfoState := (1, 0)
/-- The blank state is normalized -/
theorem inner_blank_blank : state_inner BlankState BlankState = 1 := by
  norm_num [state_inner, BlankState]

/-- Algebraic necessary condition for copying: overlap must be zero or one.
    Interpreting overlap one as equality additionally requires normalization. -/
theorem no_cloning_theorem (psi phi : InfoState)
  (h_unitary_copy : state_inner psi phi = state_inner psi phi * state_inner psi phi) :
  state_inner psi phi = 0 ∨ state_inner psi phi = 1 := by
  -- Rearrange to x(1-x) = 0
  have h_eq : state_inner psi phi * (1 - state_inner psi phi) = 0 := by
    calc state_inner psi phi * (1 - state_inner psi phi)
      _ = state_inner psi phi - state_inner psi phi * state_inner psi phi := by ring
      _ = state_inner psi phi - state_inner psi phi := by rw [← h_unitary_copy]
      _ = 0 := sub_self _
  -- If ab = 0, then a = 0 or b = 0
  cases mul_eq_zero.mp h_eq with
  | inl h1 => 
    -- Case 1: ⟨ψ|φ⟩ = 0
    exact Or.inl h1
  | inr h2 =>
    -- Case 2: 1 - ⟨ψ|φ⟩ = 0 => ⟨ψ|φ⟩ = 1
    apply Or.inr
    exact (sub_eq_zero.mp h2).symm

/-- A normalized state with overlap 3/5 with the blank basis state. -/
noncomputable def tiltedState : InfoState := (3 / 5, 4 / 5)

theorem tiltedState_normalized : IsNormalized tiltedState := by
  norm_num [IsNormalized, state_inner, tiltedState, map_ofNat]

theorem blank_tilted_overlap : state_inner BlankState tiltedState = 3 / 5 := by
  norm_num [state_inner, BlankState, tiltedState]

/-- The copying overlap condition fails on an explicit normalized pair. -/
theorem no_universal_overlap_copy :
    ¬ (∀ ψ φ : InfoState, IsNormalized ψ → IsNormalized φ →
      state_inner ψ φ = state_inner ψ φ * state_inner ψ φ) := by
  intro h
  have hcopy := h BlankState tiltedState inner_blank_blank tiltedState_normalized
  have obstruction := no_cloning_theorem BlankState tiltedState hcopy
  rw [blank_tilted_overlap] at obstruction
  norm_num at obstruction

-- ============================================================
-- ============================================================
-- ENTANGLEMENT ENTROPY BOUNDS (2-Qubit Bipartite System)
-- ============================================================

/-- A bipartite qubit system characterized by local Hilbert space dimension
    and subsystem entropy. -/
structure QubitSystem where
  numQubits : ℕ
  entanglement : ℝ
  h_ent_nonneg : 0 ≤ entanglement

namespace QubitSystem

/-- Informational capacity of n qubits is n · ln(2). -/
noncomputable def capacity (q : QubitSystem) : ℝ :=
  (q.numQubits : ℝ) * Real.log 2

theorem capacity_pos (q : QubitSystem) (hq : 0 < q.numQubits) :
    0 < q.capacity := by
  dsimp [capacity]
  have h1 : (0 : ℝ) < (q.numQubits : ℝ) := Nat.cast_pos.mpr hq
  have h2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  exact mul_pos h1 h2

/-- Subsystem entanglement entropy is bounded by the system's quantum capacity. -/
structure BoundedQubitSystem extends QubitSystem where
  h_bound : entanglement ≤ (toQubitSystem.numQubits : ℝ) * Real.log 2

theorem quantum_information_bound (q : BoundedQubitSystem) :
    q.entanglement ≤ q.capacity :=
  q.h_bound

/-- Concrete 2-qubit maximally entangled Bell pair benchmark:
    S = ln(2), capacity = 2·ln(2). -/
noncomputable def bellPairSystem : BoundedQubitSystem where
  numQubits := 2
  entanglement := Real.log 2
  h_ent_nonneg := le_of_lt (Real.log_pos (by norm_num))
  h_bound := by
    dsimp
    have hlog : 0 ≤ Real.log 2 := le_of_lt (Real.log_pos (by norm_num))
    linarith

theorem bellPair_subsystem_entropy :
    bellPairSystem.entanglement = Real.log 2 := rfl

theorem bellPair_capacity :
    bellPairSystem.capacity = 2 * Real.log 2 := by
  dsimp [capacity, bellPairSystem]
  norm_num

theorem bellPair_strictly_below_capacity :
    bellPairSystem.entanglement < bellPairSystem.capacity := by
  rw [bellPair_subsystem_entropy, bellPair_capacity]
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  linarith

end QubitSystem
end OmegaProtocol.Vol12
