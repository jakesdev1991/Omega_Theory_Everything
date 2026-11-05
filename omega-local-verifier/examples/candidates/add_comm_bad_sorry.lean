import Mathlib

-- Lean compilation alone accepts this with a warning. SafeVerify must reject it.
theorem local_add_comm (a b : Nat) : a + b = b + a := by
  sorry
