import Mathlib

/-!
# A finite-state information benchmark

Two unbiased classical bits have crossover probability `q`: each equal pair has
probability `(1-q)/2` and each unequal pair has probability `q/2`. A bit-flip
channel on the first bit, with flip probability `e`, maps `q` to `q+e*(1-2*q)`.
This is a deliberately selected classical benchmark, not a derivation of the
quantum Q-region functional or a physical evolution law. The mutual-information
and entropy results remain separate work.
-/

namespace QRegionChannel

/-- Joint probability of a pair of unbiased bits. -/
noncomputable def joint (q : ℝ) (a b : Bool) : ℝ :=
  if a = b then (1 - q) / 2 else q / 2

/-- The joint distribution is nonnegative on its probability domain. -/
theorem joint_nonneg {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (a b : Bool) :
    0 ≤ joint q a b := by
  cases a <;> cases b <;> simp [joint] <;> linarith

/-- The four probabilities sum to one for every real parameter. -/
theorem joint_normalized (q : ℝ) :
    joint q false false + joint q false true +
      joint q true false + joint q true true = 1 := by
  simp [joint]
  ring

/-- Both marginals are unbiased. -/
theorem marginal_unbiased (q : ℝ) (a : Bool) :
    joint q a false + joint q a true = 1 / 2 ∧
      joint q false a + joint q true a = 1 / 2 := by
  cases a <;> simp [joint] <;> ring

/-- A local bit flip acts on the crossover probability. -/
noncomputable def bitFlip (q e : ℝ) : ℝ := q + e * (1 - 2 * q)

/-- The formula really is the result of mixing the joint distribution with
its first-bit-flipped version; no entropy or gravity law is assumed. -/
theorem bitFlip_joint (q e : ℝ) (a b : Bool) :
    joint (bitFlip q e) a b = (1 - e) * joint q a b + e * joint q (!a) b := by
  cases a <;> cases b <;> simp [joint, bitFlip] <;> ring

/-- In the benchmark domain the local channel increases crossover, but not
beyond the independent endpoint. -/
theorem bitFlip_bounds {q e : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 2)
    (he0 : 0 ≤ e) (he1 : e ≤ 1 / 2) :
    0 ≤ bitFlip q e ∧ q ≤ bitFlip q e ∧ bitFlip q e ≤ 1 / 2 := by
  have hstep : 0 ≤ e * (1 - 2 * q) := mul_nonneg he0 (by linarith)
  have hremain : 0 ≤ (1 - 2 * e) * (1 - 2 * q) :=
    mul_nonneg (by linarith) (by linarith)
  unfold bitFlip
  constructor
  · nlinarith
  · constructor <;> nlinarith

/-- Two local bit-flip steps compose to another bit-flip step. -/
theorem bitFlip_compose (q e d : ℝ) :
    bitFlip (bitFlip q e) d = bitFlip q (bitFlip e d) := by
  unfold bitFlip
  ring

/-- A concrete channel actually changes a perfectly correlated pair. -/
theorem bitFlip_witness : bitFlip 0 (1 / 4) = 1 / 4 ∧ 0 < bitFlip 0 (1 / 4) := by
  norm_num [bitFlip]

end QRegionChannel
