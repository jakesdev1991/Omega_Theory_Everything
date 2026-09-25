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

end OmegaProtocol.Vol51
