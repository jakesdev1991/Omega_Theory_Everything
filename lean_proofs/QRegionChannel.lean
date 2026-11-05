import Mathlib
import OmegaAxioms

/-!
# A finite-state information benchmark

Two unbiased classical bits have crossover probability `q`: each equal pair has
probability `(1-q)/2` and each unequal pair has probability `q/2`. A bit-flip
channel on the first bit, with flip probability `e`, maps `q` to `q+e*(1-2*q)`.
This is a deliberately selected classical benchmark, not a derivation of the
quantum Q-region functional or a physical evolution law. Mutual information is
computed from the four joint probabilities and their marginals in nats.
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

/-- Shannon entropy of the four joint probabilities, in nats. Mathlib's
`negMulLog` uses the continuous zero endpoint. -/
noncomputable def jointEntropy (q : ℝ) : ℝ :=
  Real.negMulLog (joint q false false) + Real.negMulLog (joint q false true) +
    Real.negMulLog (joint q true false) + Real.negMulLog (joint q true true)

/-- Mutual information computed as the two marginal entropies minus joint
entropy, rather than a chosen function of overlap. -/
noncomputable def mutualInformation (q : ℝ) : ℝ :=
  Real.negMulLog (joint q false false + joint q false true) +
    Real.negMulLog (joint q true false + joint q true true) +
    Real.negMulLog (joint q false false + joint q true false) +
    Real.negMulLog (joint q false true + joint q true true) - jointEntropy q

/-- The entropy of a fair bit. -/
theorem negMulLog_half : Real.negMulLog (1 / 2 : ℝ) = Real.log 2 / 2 := by
  rw [show (1 / 2 : ℝ) = (2 : ℝ)⁻¹ by norm_num, Real.negMulLog, Real.log_inv]
  ring

/-- Computing the joint entropy yields the fair-bit entropy plus the binary
entropy of the crossover parameter. -/
theorem jointEntropy_eq (q : ℝ) : jointEntropy q = Real.log 2 + Real.binEntropy q := by
  have hhalf : Real.negMulLog ((2 : ℝ)⁻¹) = Real.log 2 / 2 := by
    simpa using negMulLog_half
  rw [Real.binEntropy_eq_negMulLog_add_negMulLog_one_sub]
  simp only [jointEntropy, joint, Bool.false_eq_true, Bool.true_eq_false, if_false, if_true,
    div_eq_mul_inv, Real.negMulLog_mul, hhalf]
  ring

/-- Exact mutual information of the benchmark in nats, including `q = 0` and
`q = 1/2`. -/
theorem mutualInformation_eq (q : ℝ) :
    mutualInformation q = Real.log 2 - Real.binEntropy q := by
  have h1 := (marginal_unbiased q false).1
  have h2 := (marginal_unbiased q true).1
  have h3 := (marginal_unbiased q false).2
  have h4 := (marginal_unbiased q true).2
  unfold mutualInformation
  simp only [h1, h2, h3, h4, negMulLog_half, jointEntropy_eq]
  ring

/-- On the probability domain, mutual information lies between zero and one
fair bit, expressed in nats. -/
theorem mutualInformation_bounds {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    0 ≤ mutualInformation q ∧ mutualInformation q ≤ Real.log 2 := by
  rw [mutualInformation_eq]
  constructor
  · linarith [Real.binEntropy_le_log_two (p := q)]
  · linarith [Real.binEntropy_nonneg hq0 hq1]

/-- The perfectly correlated and independent endpoints are one bit and zero. -/
theorem mutualInformation_endpoints :
    mutualInformation 0 = Real.log 2 ∧ mutualInformation (1 / 2) = 0 := by
  simp [mutualInformation_eq]

/-- Local bit-flip noise cannot increase mutual information on the stated
classical benchmark. -/
theorem bitFlip_mutualInformation_le {q e : ℝ}
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 2) (he0 : 0 ≤ e) (he1 : e ≤ 1 / 2) :
    mutualInformation (bitFlip q e) ≤ mutualInformation q := by
  have hb := bitFlip_bounds hq0 hq1 he0 he1
  have hq : q ∈ Set.Icc (0 : ℝ) (2 : ℝ)⁻¹ := ⟨hq0, by norm_num at *; linarith⟩
  have hflip : bitFlip q e ∈ Set.Icc (0 : ℝ) (2 : ℝ)⁻¹ :=
    ⟨hb.1, by norm_num at *; linarith [hb.2.2]⟩
  have hentropy := Real.binEntropy_strictMonoOn.monotoneOn hq hflip hb.2.1
  rw [mutualInformation_eq, mutualInformation_eq]
  linarith

/-- Model-specific overlap: mutual information normalized by one fair bit.
It is not asserted to be the universal Omega COD functional. -/
noncomputable def normalizedOverlap (q : ℝ) : ℝ := mutualInformation q / Real.log 2

/-- The normalized benchmark overlap lies in the unit interval. -/
theorem normalizedOverlap_bounds {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    normalizedOverlap q ∈ Set.Icc (0 : ℝ) 1 := by
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hmi := mutualInformation_bounds hq0 hq1
  constructor
  · exact div_nonneg hmi.1 hlog.le
  · exact (div_le_one hlog).2 hmi.2

/-- Perfect correlation has unit overlap, while independence has zero overlap. -/
theorem normalizedOverlap_endpoints :
    normalizedOverlap 0 = 1 ∧ normalizedOverlap (1 / 2) = 0 := by
  have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  unfold normalizedOverlap
  rw [mutualInformation_endpoints.1, mutualInformation_endpoints.2]
  simp [hlog]

/-- The selected bit-flip channel cannot increase normalized overlap. -/
theorem bitFlip_normalizedOverlap_le {q e : ℝ}
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1 / 2) (he0 : 0 ≤ e) (he1 : e ≤ 1 / 2) :
    normalizedOverlap (bitFlip q e) ≤ normalizedOverlap q := by
  unfold normalizedOverlap
  exact div_le_div_of_nonneg_right (bitFlip_mutualInformation_le hq0 hq1 he0 he1)
    (Real.log_pos (by norm_num : (1 : ℝ) < 2)).le

/-- The independent endpoint of this benchmark cannot equal the current
strictly positive pairwise overlap on any existing `QRegion` pair. -/
theorem independent_not_existing_overlap (R S : OmegaProtocol.QRegion) :
    OmegaProtocol.Φ R S ≠ normalizedOverlap (1 / 2) := by
  rw [normalizedOverlap_endpoints.2]
  exact ne_of_gt (OmegaProtocol.Φ_pos R S)

end QRegionChannel
