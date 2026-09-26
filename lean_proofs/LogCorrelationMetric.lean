import Mathlib

/-!
# A non-degenerate logarithmic correlation metric

This independent model has real-valued points, not the singleton `QRegion`.
The correlation kernel is an explicit modelling choice, not a mutual-information
or data-processing theorem. Its negative logarithm recovers ordinary distance.
In particular, the multiplicative inequality needed for a triangle inequality
is a LOWER bound on direct correlation, not the upper bound in the legacy DPI.
-/
namespace LogCorrelationMetric

/-- General sufficient condition, with all positivity assumptions exposed.
    This applies to normalized positive correlations at a fixed common scale. -/
theorem neg_log_triangle_of_mul_le (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (h : a * b ≤ c) :
    -Real.log c ≤ -Real.log a + -Real.log b := by
  have hlog := Real.log_le_log (mul_pos ha hb) h
  rw [Real.log_mul (ne_of_gt ha) (ne_of_gt hb)] at hlog
  linarith

noncomputable def correlation (x y : ℝ) : ℝ := Real.exp (-|x - y|)
noncomputable def logDistance (x y : ℝ) : ℝ := -Real.log (correlation x y)

theorem correlation_pos (x y : ℝ) : 0 < correlation x y := Real.exp_pos _

theorem correlation_le_one (x y : ℝ) : correlation x y ≤ 1 := by
  unfold correlation
  simpa using Real.exp_le_exp.mpr (neg_nonpos.mpr (abs_nonneg (x - y)))

theorem logDistance_eq_abs (x y : ℝ) : logDistance x y = |x - y| := by
  simp [logDistance, correlation, Real.log_exp]

theorem logDistance_nonneg (x y : ℝ) : 0 ≤ logDistance x y := by
  rw [logDistance_eq_abs]
  exact abs_nonneg _

theorem logDistance_eq_zero_iff (x y : ℝ) : logDistance x y = 0 ↔ x = y := by
  rw [logDistance_eq_abs, abs_eq_zero, sub_eq_zero]

theorem logDistance_symm (x y : ℝ) : logDistance x y = logDistance y x := by
  simp only [logDistance_eq_abs, abs_sub_comm]

theorem logDistance_triangle (x y z : ℝ) :
    logDistance x z ≤ logDistance x y + logDistance y z := by
  simp only [logDistance_eq_abs]
  exact abs_sub_le x y z

/-- The sign reverses when passing from a negative logarithm to correlation. -/
theorem correlation_supermultiplicative (x y z : ℝ) :
    correlation x y * correlation y z ≤ correlation x z := by
  unfold correlation
  rw [← Real.exp_add]
  apply Real.exp_le_exp.mpr
  have h := abs_sub_le x y z
  linarith

/-- A concrete witness excludes the all-zero-distance model. -/
theorem distinct_points_distance : logDistance 0 1 = 1 := by
  norm_num [logDistance_eq_abs]

/-- The opposite multiplicative inequality does not hold for arbitrary triples. -/
theorem upper_multiplicative_bound_fails :
    ¬ correlation 0 0 ≤ correlation 0 1 * correlation 1 0 := by
  have h : Real.exp (-2 : ℝ) < Real.exp 0 :=
    Real.exp_lt_exp.mpr (by norm_num)
  norm_num [correlation, ← Real.exp_add] at h ⊢ <;> linarith

end LogCorrelationMetric
