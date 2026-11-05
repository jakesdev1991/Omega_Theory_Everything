import DynamicPlanckScale
import DynamicCODScale

/-!
# Conditional matching of Planck-scale profiles

These results compare the existing profiles; they do not assume that physical
COD and density are related, nor do they construct the proposed dictionary.
Uniqueness is conditional on two matching physical-range values existing. The
endpoint obstruction applies even if the two baselines differ. The clamped
branch result explicitly assumes that the baselines coincide.

The candidate dictionary's construction, zero-boundary monotonicity and inverse
remain to be proved separately. See `docs/PLANCK_PROFILE_MATCHING.md`.
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

end PlanckProfileMatching
