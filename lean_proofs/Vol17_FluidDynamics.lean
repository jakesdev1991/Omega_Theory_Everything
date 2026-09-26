import Mathlib
import OmegaUnifiedFoundation

/-!
# Continuity along a material trajectory

A positive density history satisfies the scalar material continuity equation
ρ' = -ρ div(v). The derivative is Mathlib's derivative, linked to the trajectory
by an explicit `HasDerivAt` assumption. Divergence is supplied as a scalar history;
no spatial velocity field or Navier–Stokes solution is constructed here.

Global incompressibility implies equal densities at any two times, not merely
a pointwise zero symbol. Positive density is essential for the converse: in a
vacuum the continuity equation cannot recover divergence from the density rate.
-/
namespace OmegaProtocol.Vol17
open OmegaProtocol

abbrev FluidTime := ℝ

structure FluidTrajectory where
  density : ℝ → ℝ
  divergence : ℝ → ℝ
  density_pos : ∀ t, 0 < density t
  continuity_hasDerivAt : ∀ t, HasDerivAt density (-density t * divergence t) t

def FluidDensity (flow : FluidTrajectory) : FluidTime → ℝ := flow.density
def FlowDivergence (flow : FluidTrajectory) : FluidTime → ℝ := flow.divergence
noncomputable def FluidTimeDeriv (f : FluidTime → ℝ) (t : FluidTime) : ℝ := deriv f t

theorem continuity_equation (flow : FluidTrajectory) (t : FluidTime) :
    FluidTimeDeriv (FluidDensity flow) t + FluidDensity flow t * FlowDivergence flow t = 0 := by
  have h := (flow.continuity_hasDerivAt t).deriv
  change deriv flow.density t + flow.density t * flow.divergence t = 0
  rw [h]
  ring

theorem incompressible_conservation (flow : FluidTrajectory) (t : FluidTime)
    (h : FlowDivergence flow t = 0) : FluidTimeDeriv (FluidDensity flow) t = 0 := by
  have hc := continuity_equation flow t
  simpa [h] using hc

theorem density_rate_zero_iff (flow : FluidTrajectory) (t : FluidTime) :
    FluidTimeDeriv (FluidDensity flow) t = 0 ↔ FlowDivergence flow t = 0 := by
  constructor
  · intro h
    have hc := continuity_equation flow t
    rw [h, zero_add] at hc
    exact (mul_eq_zero.mp hc).resolve_left (ne_of_gt (flow.density_pos t))
  · exact incompressible_conservation flow t

/-- A genuine constant-history conclusion using differentiability. -/
theorem incompressible_density_constant (flow : FluidTrajectory)
    (h : ∀ t, FlowDivergence flow t = 0) (s t : FluidTime) :
    FluidDensity flow s = FluidDensity flow t :=
  is_const_of_deriv_eq_zero
    (fun x => (flow.continuity_hasDerivAt x).differentiableAt)
    (fun x => incompressible_conservation flow x (h x)) s t

def constantDensity (ρ : ℝ) (hρ : 0 < ρ) : FluidTrajectory :=
  { density := fun _ => ρ
    divergence := fun _ => 0
    density_pos := fun _ => hρ
    continuity_hasDerivAt := fun t => by simpa using hasDerivAt_const t ρ }

/-- A nonconstant positive solution of the scalar continuity law with divergence one. -/
noncomputable def expandingFlow : FluidTrajectory :=
  { density := fun t => Real.exp (-t)
    divergence := fun _ => 1
    density_pos := fun t => Real.exp_pos (-t)
    continuity_hasDerivAt := fun t => by simpa using ((hasDerivAt_id t).neg).exp }

theorem expandingFlow_density_decreases (s t : FluidTime) (h : s < t) :
    FluidDensity expandingFlow t < FluidDensity expandingFlow s :=
  Real.exp_lt_exp.mpr (neg_lt_neg h)

end OmegaProtocol.Vol17
