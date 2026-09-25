import InformationPhysics
import Mathlib.Analysis.Real.Sqrt

/-!
# Dynamic Planck Scale

Second stage of the information-physics pipeline. The Landauer–Einstein bridge
(`InformationPhysics`) is imported as a prerequisite, and its information mass
sets the local density that scales the operational Planck length:

  `m(I)` (information-mass equivalence) ⟹ `ρ = m(I) / V` (local density)
    ⟹ `ℓ_P(ρ)` (dynamic scale)

The two modules are kept separate so that the kernel checks two small,
independent developments: thermodynamic primitives live in
`InformationPhysics`, scale geometry lives here, and the only coupling is the
density `ρ`. This module uses exactly one fact about the upstream model,
`InformationPhysics.informationMass_pos`, and never unfolds the Landauer formula.

Genuinely proven theorems:
1. **Positivity** — `ℓ_P(ρ)` is a strictly positive length at every density.
2. **Vacuum baseline** — for `ρ ≤ 0` the scale is exactly the baseline `ℓ_P0`.
3. **Planck-scale contraction** — every positive density gives `ℓ_P(ρ) < ℓ_P0`.
4. **Strict antitonicity** — on `ρ > 0`, denser regions have strictly shorter
   Planck lengths: the scale contracts in dense regions and relaxes back
   toward `ℓ_P0` in sparse ones.
5. **Information-driven contraction** — `I > 0` bits held in a volume `V > 0`
   contract the Planck scale through their Landauer information-mass density.

Model assumptions (definitions, not derivations):
- `ℓ_P(ρ) = ℓ_P0 / √(1 + ρ/ρ₀)` for `ρ > 0`, and `ℓ_P(ρ) = ℓ_P0` otherwise
  (the vacuum and unphysical negative densities sit at the baseline).
- `ρ = m(I) / V`, with `m(I) = I · k_B · T · ln 2 / c²` from `InformationPhysics`.

This density profile is a phenomenological choice. It is not derived from the
Omega axioms, and it is a different parametrisation from the `Φ`-dependent
profile `ℓ_P(Φ) = ℓ_P0 · exp((1 − Φ)/φ_c)` of the v4.0 technical note.
-/

namespace DynamicGeometry

/-- Physical parameters for dynamic scale rendering -/
structure ScaleEnvironment where
  lP0 : ℝ -- Baseline Planck length (lP0 > 0)
  rho0 : ℝ -- Critical saturation density (rho0 > 0)
  hlP0 : lP0 > 0
  hrho0 : rho0 > 0

/-- Dynamic Planck Length: contracts in high-density regions and relaxes back to
    the baseline `lP0` in sparse regions (`planck_scale_contraction`,
    `dynamicPlanckLength_strictAntiOn`). Non-positive densities give `lP0`. -/
noncomputable def dynamicPlanckLength (env : ScaleEnvironment) (rho : ℝ) : ℝ :=
  if rho > 0 then env.lP0 / Real.sqrt (1 + rho / env.rho0) else env.lP0

/-- The dynamic Planck length is a genuine, strictly positive length at every
    density. -/
theorem dynamicPlanckLength_pos (env : ScaleEnvironment) (rho : ℝ) :
    0 < dynamicPlanckLength env rho := by
  unfold dynamicPlanckLength
  split_ifs with h
  · have harg : 0 < 1 + rho / env.rho0 := by
      have hdiv : 0 < rho / env.rho0 := div_pos h env.hrho0
      linarith
    exact div_pos env.hlP0 (Real.sqrt_pos.mpr harg)
  · exact env.hlP0

/-- Vacuum baseline: with zero (or unphysical negative) density, the operational
    Planck length is exactly the baseline `lP0`. -/
theorem dynamicPlanckLength_of_nonpos (env : ScaleEnvironment) {rho : ℝ}
    (hrho : rho ≤ 0) : dynamicPlanckLength env rho = env.lP0 := by
  unfold dynamicPlanckLength
  split_ifs with h
  · exact absurd h (not_lt.mpr hrho)
  · rfl

/-- **Theorem: High Information Density Contracts the Operational Planck Scale**
    As information density `rho` increases above zero, the effective Planck
    length `lP(rho)` is strictly smaller than the baseline `lP0`. -/
theorem planck_scale_contraction (env : ScaleEnvironment) (rho : ℝ) (hrho : rho > 0) :
    dynamicPlanckLength env rho < env.lP0 := by
  dsimp [dynamicPlanckLength]
  split_ifs with h
  · have h1 : 1 + rho / env.rho0 > 1 := by
      have hdiv : rho / env.rho0 > 0 := div_pos hrho env.hrho0
      linarith
    -- Compare the square roots first, then rewrite only `√1` into `1`.
    -- (`rw [← Real.sqrt_one]` on the goal would also rewrite the `1` inside
    -- `1 + rho / env.rho0`, which no longer matches `h1`.)
    have hsqrt : Real.sqrt (1 + rho / env.rho0) > 1 := by
      have hlt := Real.sqrt_lt_sqrt zero_le_one h1
      rwa [Real.sqrt_one] at hlt
    -- `a / b < a` whenever `0 < a` and `1 < b`; no nonlinear arithmetic needed.
    exact div_lt_self env.hlP0 hsqrt
  · exact absurd hrho h

/-- **Contraction is monotone in density.** On positive densities the operational
    Planck length is strictly decreasing: a denser region has a strictly shorter
    scale than a sparser one, and every such scale stays below `lP0`. -/
theorem dynamicPlanckLength_strictAntiOn (env : ScaleEnvironment) :
    StrictAntiOn (dynamicPlanckLength env) (Set.Ioi 0) := by
  intro a ha b hb hab
  have ha' : 0 < a := Set.mem_Ioi.mp ha
  have hb' : 0 < b := Set.mem_Ioi.mp hb
  have harg : 0 < 1 + a / env.rho0 := by
    have hdiv : 0 < a / env.rho0 := div_pos ha' env.hrho0
    linarith
  have hstep : 1 + a / env.rho0 < 1 + b / env.rho0 := by
    have hdiv : a / env.rho0 < b / env.rho0 := div_lt_div_of_pos_right hab env.hrho0
    linarith
  unfold dynamicPlanckLength
  rw [if_pos hb', if_pos ha']
  exact div_lt_div_of_pos_left env.hlP0 (Real.sqrt_pos.mpr harg)
    (Real.sqrt_lt_sqrt harg.le hstep)

/-! ### Coupling to the Landauer–Einstein module -/

/-- **Local information-mass density** of `I` bits held in a region of volume
    `V`: `ρ = m(I) / V`, where `m(I) = I · k_B · T · ln 2 / c²` is
    `InformationPhysics.informationMass`. -/
noncomputable def informationDensity (env : InformationPhysics.Environment) (I V : ℝ) : ℝ :=
  InformationPhysics.informationMass env I / V

/-- Positive information held in a positive volume has strictly positive mass
    density. -/
theorem informationDensity_pos (env : InformationPhysics.Environment) {I V : ℝ}
    (hI : 0 < I) (hV : 0 < V) : 0 < informationDensity env I V := by
  unfold informationDensity
  exact div_pos (InformationPhysics.informationMass_pos env hI) hV

/-- **Information-driven Planck contraction** (the full pipeline
    `m(I) ⟹ ρ ⟹ ℓ_P(ρ)`): any positive amount of information `I` held in a
    positive volume `V` contracts the operational Planck length strictly below
    its baseline, through its Landauer information-mass density. -/
theorem information_contracts_planck_scale (penv : InformationPhysics.Environment)
    (senv : ScaleEnvironment) {I V : ℝ} (hI : 0 < I) (hV : 0 < V) :
    dynamicPlanckLength senv (informationDensity penv I V) < senv.lP0 :=
  planck_scale_contraction senv _ (informationDensity_pos penv hI hV)

end DynamicGeometry
