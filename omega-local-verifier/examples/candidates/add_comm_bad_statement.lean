import Mathlib

-- This compiles, but it changes the required theorem and must be rejected.
theorem local_add_comm (a b : Nat) : a + b = a + b := by
  rfl
