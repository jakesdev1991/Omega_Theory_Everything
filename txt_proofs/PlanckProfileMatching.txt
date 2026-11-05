import DynamicPlanckScale
import DynamicCODScale

/-!
# Conditional matching of Planck-scale profiles

These results compare the existing profiles; they do not assume that physical
COD and density are related, nor do they construct the proposed dictionary.
Uniqueness is conditional on two matching physical-range values existing. The
endpoint obstruction applies even if the two baselines differ. The clamped
branch result explicitly assumes that the baselines coincide.

The dictionary below is a mathematical matching condition, not a derivation of
COD from Q-region data. See `docs/PLANCK_PROFILE_MATCHING.md`.
-/

namespace PlanckProfileMatching

/-- Any two physical-range overlaps matching the same density are equal. -/
theorem matching_overlap_unique (senv : DynamicGeometry.ScaleEnvironment)
    (cenv : DynamicCODScale.CODEnvironment) (rho phi psi : ℝ)
    (hphi : phi ∈ Set.Icc (0 : ℝ) 1) (hpsi : psi ∈ Set.Icc (0 : ℝ) 1)
    (hmatchphi : DynamicCODScale.dynamicPlanckCOD cenv phi =
      DynamicGeometry.dynamicPlanckLength senv rho)
    (hmatchpsi : DynamicCODScale.dynamicPlanckCOD cenv psi =
      DynamicGeometry.dynamicPlanckLength senv rho) : phi = psi := by
  apply DynamicCODScale.dynamicPlanckCOD_injOn cenv hphi hpsi
  exact hmatchphi.trans hmatchpsi.symm

/-- The zero COD scale at maximal overlap cannot match any finite real density. -/
theorem maximal_overlap_no_match (senv : DynamicGeometry.ScaleEnvironment)
    (cenv : DynamicCODScale.CODEnvironment) (rho : ℝ) :
    DynamicCODScale.dynamicPlanckCOD cenv 1 ≠
      DynamicGeometry.dynamicPlanckLength senv rho := by
  intro h
  rw [DynamicCODScale.dynamicPlanckCOD_one] at h
  have hpos := DynamicGeometry.dynamicPlanckLength_pos senv rho
  rw [← h] at hpos
  exact (lt_irrefl 0) hpos

/-- With a shared baseline, the clamped nonpositive-density branch matches
only zero overlap within the physical range. -/
theorem nonpositive_match_iff (senv : DynamicGeometry.ScaleEnvironment)
    (cenv : DynamicCODScale.CODEnvironment) (rho phi : ℝ)
    (hbase : cenv.lP0 = senv.lP0) (hrho : rho ≤ 0)
    (hphi : phi ∈ Set.Icc (0 : ℝ) 1) :
    DynamicCODScale.dynamicPlanckCOD cenv phi =
      DynamicGeometry.dynamicPlanckLength senv rho ↔ phi = 0 := by
  rw [DynamicGeometry.dynamicPlanckLength_of_nonpos senv hrho, ← hbase]
  exact DynamicCODScale.dynamicPlanckCOD_baseline_iff cenv phi hphi

/-- Any physical-range matching function must be strictly increasing on
positive densities, even though the two baselines need not be identified here.
This does not assert that such a function exists. -/
theorem matching_strictMonoOn_positive (senv : DynamicGeometry.ScaleEnvironment)
    (cenv : DynamicCODScale.CODEnvironment) (f : ℝ → ℝ)
    (hrange : ∀ rho ∈ Set.Ioi (0 : ℝ), f rho ∈ Set.Icc (0 : ℝ) 1)
    (hmatch : ∀ rho ∈ Set.Ioi (0 : ℝ),
      DynamicCODScale.dynamicPlanckCOD cenv (f rho) =
        DynamicGeometry.dynamicPlanckLength senv rho) :
    StrictMonoOn f (Set.Ioi 0) := by
  intro a ha b hb hab
  have hden := DynamicGeometry.dynamicPlanckLength_strictAntiOn senv ha hb hab
  have hma := hmatch a ha
  have hmb := hmatch b hb
  by_contra hn
  by_cases hrev : f b < f a
  · have hcod := DynamicCODScale.dynamicPlanckCOD_strictAntiOn cenv
      (hrange b hb) (hrange a ha) hrev
    rw [hma, hmb] at hcod
    exact (lt_asymm hcod) hden
  · have heq : f a = f b :=
      le_antisymm (le_of_not_gt hrev) (le_of_not_gt hn)
    rw [heq] at hma
    have hlen : DynamicGeometry.dynamicPlanckLength senv a =
        DynamicGeometry.dynamicPlanckLength senv b := hma.symm.trans hmb
    rw [hlen] at hden
    exact (lt_irrefl _) hden

/-- Candidate overlap matching the two phenomenological profiles. On the
physical domain this equals `√((rho / rho0) / (1 + rho / rho0))`. -/
noncomputable def overlapOfDensity (senv : DynamicGeometry.ScaleEnvironment) (rho : ℝ) : ℝ :=
  Real.sqrt (rho / (rho + senv.rho0))

/-- For finite nonnegative density the candidate is in the half-open physical range. -/
theorem overlapOfDensity_range (senv : DynamicGeometry.ScaleEnvironment) {rho : ℝ}
    (hrho : 0 ≤ rho) : overlapOfDensity senv rho ∈ Set.Ico (0 : ℝ) 1 := by
  have hden : 0 < rho + senv.rho0 := by linarith [senv.hrho0]
  have hquot0 : 0 ≤ rho / (rho + senv.rho0) := div_nonneg hrho hden.le
  have hquot1 : rho / (rho + senv.rho0) < 1 :=
    (div_lt_one hden).2 (by linarith [senv.hrho0])
  constructor
  · exact Real.sqrt_nonneg _
  · have h := Real.sqrt_lt_sqrt hquot0 hquot1
    simpa [overlapOfDensity] using h

/-- The candidate agrees with the existing dynamic profiles, assuming the two
models use the same positive vacuum baseline. -/
theorem overlapOfDensity_matches (senv : DynamicGeometry.ScaleEnvironment)
    (cenv : DynamicCODScale.CODEnvironment) {rho : ℝ} (hrho : 0 ≤ rho)
    (hbase : cenv.lP0 = senv.lP0) :
    DynamicCODScale.dynamicPlanckCOD cenv (overlapOfDensity senv rho) =
      DynamicGeometry.dynamicPlanckLength senv rho := by
  by_cases hzero : rho = 0
  · subst rho
    simp [overlapOfDensity, DynamicCODScale.dynamicPlanckCOD,
      DynamicGeometry.dynamicPlanckLength, hbase]
  · have hpos : 0 < rho := lt_of_le_of_ne hrho (Ne.symm hzero)
    have hden : 0 < rho + senv.rho0 := by linarith [senv.hrho0]
    have hquot0 : 0 ≤ rho / (rho + senv.rho0) := div_nonneg hrho hden.le
    have hsum : 1 + rho / senv.rho0 = (rho + senv.rho0) / senv.rho0 := by
      field_simp [ne_of_gt senv.hrho0]
      <;> ring
    have harg : 1 - (overlapOfDensity senv rho) ^ 2 =
        1 / (1 + rho / senv.rho0) := by
      rw [overlapOfDensity, Real.sq_sqrt hquot0, hsum, one_div_div]
      field_simp [ne_of_gt hden]
      <;> ring
    rw [DynamicCODScale.dynamicPlanckCOD, harg,
      Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 1), Real.sqrt_one, hbase]
    rw [DynamicGeometry.dynamicPlanckLength, if_pos hpos]
    simp [div_eq_mul_inv]

/-- The constructed matching value is the only one in the closed physical
range. In particular the excluded endpoint cannot match a finite density. -/
theorem overlapOfDensity_unique (senv : DynamicGeometry.ScaleEnvironment)
    (cenv : DynamicCODScale.CODEnvironment) {rho phi : ℝ}
    (hrho : 0 ≤ rho) (hbase : cenv.lP0 = senv.lP0)
    (hphi : phi ∈ Set.Icc (0 : ℝ) 1)
    (hmatch : DynamicCODScale.dynamicPlanckCOD cenv phi =
      DynamicGeometry.dynamicPlanckLength senv rho) :
    phi = overlapOfDensity senv rho := by
  have hrange := overlapOfDensity_range senv hrho
  apply matching_overlap_unique senv cenv rho phi (overlapOfDensity senv rho) hphi
    ⟨hrange.1, hrange.2.le⟩ hmatch
  exact overlapOfDensity_matches senv cenv hrho hbase

/-- The candidate is strictly increasing over the entire nonnegative domain,
including vacuum. -/
theorem overlapOfDensity_strictMonoOn (senv : DynamicGeometry.ScaleEnvironment) :
    StrictMonoOn (overlapOfDensity senv) (Set.Ici 0) := by
  intro a ha b hb hab
  have ha0 : 0 ≤ a := Set.mem_Ici.mp ha
  have hb0 : 0 ≤ b := Set.mem_Ici.mp hb
  have hda : 0 < a + senv.rho0 := by linarith [senv.hrho0]
  have hdb : 0 < b + senv.rho0 := by linarith [senv.hrho0]
  have hquot : a / (a + senv.rho0) < b / (b + senv.rho0) := by
    rw [div_lt_div_iff₀ hda hdb]
    nlinarith [mul_pos (sub_pos.mpr hab) senv.hrho0]
  exact Real.sqrt_lt_sqrt (div_nonneg ha0 hda.le) hquot

/-- Applying the inverse formula to a finite nonnegative density recovers it. -/
theorem overlapOfDensity_inverse (senv : DynamicGeometry.ScaleEnvironment)
    {rho : ℝ} (hrho : 0 ≤ rho) :
    senv.rho0 * (overlapOfDensity senv rho) ^ 2 /
      (1 - (overlapOfDensity senv rho) ^ 2) = rho := by
  have hden : 0 < rho + senv.rho0 := by linarith [senv.hrho0]
  have hquot0 : 0 ≤ rho / (rho + senv.rho0) := div_nonneg hrho hden.le
  have hquot1 : rho / (rho + senv.rho0) < 1 :=
    (div_lt_one hden).2 (by linarith [senv.hrho0])
  have hdiff : 1 - rho / (rho + senv.rho0) ≠ 0 := by linarith
  rw [overlapOfDensity, Real.sq_sqrt hquot0]
  field_simp [ne_of_gt hden, hdiff, ne_of_gt senv.hrho0]
  <;> ring

/-- Density corresponding to an overlap below saturation. -/
noncomputable def densityOfOverlap (senv : DynamicGeometry.ScaleEnvironment) (phi : ℝ) : ℝ :=
  senv.rho0 * phi ^ 2 / (1 - phi ^ 2)

/-- Every overlap in `[0, 1)` has the stated finite nonnegative inverse. -/
theorem densityOfOverlap_rightInverse (senv : DynamicGeometry.ScaleEnvironment)
    {phi : ℝ} (hphi : phi ∈ Set.Ico (0 : ℝ) 1) :
    0 ≤ densityOfOverlap senv phi ∧
      overlapOfDensity senv (densityOfOverlap senv phi) = phi := by
  have hphi0 : 0 ≤ phi := hphi.1
  have hphi1 : phi < 1 := hphi.2
  have hd : 0 < 1 - phi ^ 2 := by nlinarith
  have hrho : 0 ≤ densityOfOverlap senv phi :=
    div_nonneg (mul_nonneg senv.hrho0.le (sq_nonneg phi)) hd.le
  have hsum : 0 < densityOfOverlap senv phi + senv.rho0 := by
    linarith [senv.hrho0]
  have hid : densityOfOverlap senv phi * (1 - phi ^ 2) = senv.rho0 * phi ^ 2 := by
    unfold densityOfOverlap
    field_simp [ne_of_gt hd]
  have hquot : densityOfOverlap senv phi /
      (densityOfOverlap senv phi + senv.rho0) = phi ^ 2 := by
    apply (div_eq_iff (ne_of_gt hsum)).2
    nlinarith [hid]
  constructor
  · exact hrho
  · rw [overlapOfDensity, hquot, Real.sqrt_sq hphi0]

/-- For any supplied overlap trajectory in the physical range, a strictly
increasing overlap between two times yields a shorter dynamic length. This does
not specify or derive the trajectory's physical evolution. -/
theorem trajectory_overlap_contraction (cenv : DynamicCODScale.CODEnvironment)
    (phi : ℝ → ℝ) {t1 t2 : ℝ} (hzero : 0 ≤ phi t1)
    (hstep : phi t1 < phi t2) (hone : phi t2 ≤ 1) :
    DynamicCODScale.dynamicPlanckCOD cenv (phi t2) <
      DynamicCODScale.dynamicPlanckCOD cenv (phi t1) :=
  DynamicCODScale.cod_monotonic_contraction cenv (phi t1) (phi t2) hzero hstep hone

end PlanckProfileMatching
