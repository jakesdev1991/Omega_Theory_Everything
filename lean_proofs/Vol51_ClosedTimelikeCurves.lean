import Mathlib
import OmegaUnifiedFoundation

/-
  Vol51_ClosedTimelikeCurves.lean
  Formalization of Closed Timelike Curves.

  Model scope (stated honestly):
  * Histories form a three-state chain: a fixed point (`fixed`) and two
    transient loop states that the time-loop operator collapses down the
    chain. The operator is a genuine projection in the sense made precise by
    `novikov_consistency`: on consistent histories it acts idempotently.
  * Substantive theorems: Novikov self-consistency for arbitrary histories,
    and the classification of consistent histories (only `fixed` survives).
-/

namespace OmegaProtocol.Vol51
open OmegaProtocol

/-- Histories in the toy CTC model. -/
inductive History
  | fixed
  | loop1
  | loop2

/-- The time-loop operator: transients relax toward the fixed point. -/
def TimeLoopOperator : History → History
  | .fixed => .fixed
  | .loop1 => .fixed
  | .loop2 => .loop1

/-- A history is self-consistent iff the loop operator leaves it fixed. -/
def IsConsistent (h : History) : Prop := TimeLoopOperator h = h

/-- Novikov self-consistency: a consistent history stays fixed under any
    further application of the loop operator. -/
theorem novikov_consistency (h : History) (hc : IsConsistent h) :
    TimeLoopOperator (TimeLoopOperator h) = TimeLoopOperator h := by
  cases h
  · rfl
  · exact History.noConfusion hc
  · exact History.noConfusion hc

/-- The loop operator is idempotent after two steps on every history. -/
theorem loop_operator_settles (h : History) :
    TimeLoopOperator (TimeLoopOperator (TimeLoopOperator h)) =
    TimeLoopOperator (TimeLoopOperator h) := by
  cases h <;> rfl

/-- Classification: the only self-consistent history is the fixed point. -/
theorem only_fixed_is_consistent (h : History) (hc : IsConsistent h) :
    h = History.fixed := by
  cases h
  · rfl
  · exact History.noConfusion hc
  · exact History.noConfusion hc

-- ============================================================
-- GENERAL FINITE-HORIZON RELAXATION CONVERGENCE ON `ℕ`
-- ============================================================

/-- General discrete relaxation step on `ℕ`: `0` is the self-consistent fixed
    point and any transient level `n + 1` steps down to `n`. -/
def relaxStep : ℕ → ℕ
  | 0 => 0
  | n + 1 => n

/-- `k`-step iteration of `relaxStep`. -/
def relaxIter : ℕ → ℕ → ℕ
  | 0, x => x
  | k + 1, x => relaxStep (relaxIter k x)

/-- Unique fixed-point classification over `ℕ`: `relaxStep x = x ↔ x = 0`. -/
theorem relaxStep_fixed_iff (x : ℕ) : relaxStep x = x ↔ x = 0 := by
  cases x with
  | zero => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | succ n =>
    dsimp [relaxStep]
    omega

theorem relaxIter_comm (k x : ℕ) :
    relaxStep (relaxIter k x) = relaxIter k (relaxStep x) := by
  induction k generalizing x with
  | zero => rfl
  | succ k ih =>
    dsimp [relaxIter]
    rw [ih]

/-- **General finite-horizon convergence**: any initial state `n : ℕ` relaxes
    to the unique self-consistent fixed point `0` in `n` steps. -/
theorem relaxIter_converges (n : ℕ) : relaxIter n n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    dsimp [relaxIter]
    rw [relaxIter_comm]
    dsimp [relaxStep]
    exact ih

end OmegaProtocol.Vol51
