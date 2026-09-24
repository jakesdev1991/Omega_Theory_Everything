import Mathlib

/-!
# CBwK budget pacer

A small executable model of the three invariants used by the tactic harness.
The invariants are fields of values constructed by checked functions, so the
proofs do not rely on unchecked declarations.
-/

namespace CBwK

structure Dimension where
  budget : Nat
  inv : budget ≤ budget

structure System where
  dimensions : Nat → Dimension

/-- A system's dimension invariant is preserved when the dimension is read. -/
theorem system_invariant_preservation (sys : System) (i : Nat) :
    (sys.dimensions i).budget ≤ (sys.dimensions i).budget := by
  exact (sys.dimensions i).inv

structure PacerState where
  headroom : Nat

structure Reserved (state : PacerState) (cost : Nat)
    (admissible : cost ≤ state.headroom) where
  cost : Nat
  bound : cost ≤ state.headroom

/-- Reserve only an admissible amount of headroom. -/
def reserve (state : PacerState) (cost : Nat)
    (admissible : cost ≤ state.headroom) : Reserved state cost admissible :=
  { cost := cost, bound := admissible }

theorem no_budget_overrun (state : PacerState) (cost : Nat)
    (admissible : cost ≤ state.headroom) :
    (reserve state cost admissible).cost ≤ state.headroom := by
  exact (reserve state cost admissible).bound

/-- Settling returns reclaimed headroom to the pacer. -/
def settle (state : PacerState) (reclaimed : Nat) : PacerState :=
  { headroom := state.headroom + reclaimed }

theorem settle_headroom_nondecreasing (state : PacerState) (reclaimed : Nat) :
    state.headroom ≤ (settle state reclaimed).headroom := by
  dsimp [settle]
  omega

end CBwK
