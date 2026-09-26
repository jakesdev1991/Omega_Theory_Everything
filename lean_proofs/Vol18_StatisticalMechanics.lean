import Mathlib
import OmegaUnifiedFoundation

namespace OmegaProtocol.Vol18
open OmegaProtocol

noncomputable def BoltzmannWeight (beta E : ℝ) : ℝ :=
  Real.exp (-beta * E)

noncomputable def PartitionFunction2 (beta E1 E2 : ℝ) : ℝ :=
  BoltzmannWeight beta E1 + BoltzmannWeight beta E2

noncomputable def ProbState1 (beta E1 E2 : ℝ) : ℝ :=
  BoltzmannWeight beta E1 / PartitionFunction2 beta E1 E2

noncomputable def ProbState2 (beta E1 E2 : ℝ) : ℝ :=
  BoltzmannWeight beta E2 / PartitionFunction2 beta E1 E2

/-- The denominator is positive for every finite real input; no extra
    nonzero-partition postulate is necessary. -/
theorem partitionFunction2_pos (beta E1 E2 : ℝ) :
    0 < PartitionFunction2 beta E1 E2 := by
  exact add_pos (Real.exp_pos _) (Real.exp_pos _)

theorem total_probability_is_one 
  (beta E1 E2 : ℝ) 
  (h_Z_pos : PartitionFunction2 beta E1 E2 ≠ 0) :
  ProbState1 beta E1 E2 + ProbState2 beta E1 E2 = 1 := by
  dsimp [ProbState1, ProbState2, PartitionFunction2]
  have h_add : BoltzmannWeight beta E1 / (BoltzmannWeight beta E1 + BoltzmannWeight beta E2) + 
               BoltzmannWeight beta E2 / (BoltzmannWeight beta E1 + BoltzmannWeight beta E2) = 
               (BoltzmannWeight beta E1 + BoltzmannWeight beta E2) / (BoltzmannWeight beta E1 + BoltzmannWeight beta E2) := by
    exact (add_div (BoltzmannWeight beta E1) (BoltzmannWeight beta E2) (BoltzmannWeight beta E1 + BoltzmannWeight beta E2)).symm
  rw [h_add]
  exact div_self h_Z_pos

/-- Unconditional normalization; the legacy theorem remains available. -/
theorem probabilities_normalized (beta E1 E2 : ℝ) :
    ProbState1 beta E1 E2 + ProbState2 beta E1 E2 = 1 :=
  total_probability_is_one beta E1 E2 (ne_of_gt (partitionFunction2_pos beta E1 E2))

/-- Both states have strictly positive probability, strictly below one. -/
theorem probabilities_in_unit_interval (beta E1 E2 : ℝ) :
    0 < ProbState1 beta E1 E2 ∧ ProbState1 beta E1 E2 < 1 ∧
    0 < ProbState2 beta E1 E2 ∧ ProbState2 beta E1 E2 < 1 := by
  have h1 : 0 < ProbState1 beta E1 E2 :=
    div_pos (Real.exp_pos _) (partitionFunction2_pos beta E1 E2)
  have h2 : 0 < ProbState2 beta E1 E2 :=
    div_pos (Real.exp_pos _) (partitionFunction2_pos beta E1 E2)
  have hsum := probabilities_normalized beta E1 E2
  exact ⟨h1, by linarith, h2, by linarith⟩

end OmegaProtocol.Vol18
