import Mathlib
import OmegaUnifiedFoundation

/-
  Vol02_Electromagnetism.lean
  Electromagnetism via Mathlib's genuine exterior calculus.

  The former version of this file declared a *zero* differential-form model:
  `ExteriorDerivative _ := 0`, `HodgeStar _ := 0` and `J := 0`, so `d² = 0`
  held by `rfl` and the Maxwell equations were `0 = 0` statements.  This file
  now uses Mathlib's `extDeriv` on concrete alternating forms:

  1. `d² = 0` for smooth forms (`d_squared_zero`, i.e. Mathlib's
     `extDeriv_extDeriv`, which rests on symmetry of second derivatives —
     a genuine theorem about a non-trivial operator);
  2. `d` is ℝ-linear on smooth forms (`d_add`, `d_smul`) and the derivative of
     a smooth form is smooth (`contDiff_d`);
  3. **gauge invariance**: `d (A + dχ) = d A` for smooth `A`, `χ`, so the
     field strength `F = dA` is unchanged by `A ↦ A + dχ`
     (`field_strength_gauge_invariant`), and gauge transformations compose
     additively (`gauge_transform_comp`).

  Scope: the *inhomogeneous* Maxwell law `d⋆F = ⋆J` is **not** derived — Mathlib
  has no Hodge star on differential forms — so it is carried explicitly as
  model-law data in `SourcedField.law_dual_maxwell`.  No zero placeholders
  remain; the only postulate left is the masslessness of the model photon,
  stated as a definition with an explicit docstring.
-/

namespace OmegaProtocol.Vol02
open OmegaProtocol
open scoped ContDiff

-- ============================================================
-- SPACETIME AND DIFFERENTIAL FORMS (Mathlib exterior calculus)
-- ============================================================

/-- Minkowski spacetime coordinates (t, x, y, z) — legacy coordinate
    interface, retained for downstream compatibility. -/
structure Spacetime where
  t : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- Spacetime as a normed space: `ℝ⁴` with coordinates indexed by `Fin 4`. -/
abbrev Spacetime4 := EuclideanSpace ℝ (Fin 4)

/-- A differential `n`-form on spacetime: at each point, an alternating
    `n`-linear functional (Mathlib's `ContinuousAlternatingMap`). -/
abbrev Form (n : ℕ) := Spacetime4 → ContinuousAlternatingMap ℝ Spacetime4 ℝ (Fin n)

/-- The exterior derivative: Mathlib's `extDeriv`, i.e.
    `dω(x; v₀,…,vₙ) = Σᵢ (-1)ⁱ Dω(x; …, v̂ᵢ, …)·vᵢ`. -/
noncomputable def d {n : ℕ} (η : Form n) : Form (n + 1) := extDeriv η

/-- **`d² = 0` for smooth forms** — the direction of the Poincaré lemma behind
    the homogeneous Maxwell equations.  This is Mathlib's
    `extDeriv_extDeriv`, which rests on symmetry of second derivatives
    (Clairaut/Schwarz); the smoothness hypothesis is genuine, and the operator
    is Mathlib's non-trivial exterior derivative rather than the former `0`.

    The regularity input is taken in the finite form `ContDiff ℝ 2 η`, supplied
    by `contDiff_infty` from the `C^∞` hypothesis, and the `minSmoothness`
    bound is discharged by the `@[simp]` fact `minSmoothness ℝ n = n`. -/
theorem d_squared_zero {n : ℕ} (η : Form n) (hω : ContDiff ℝ ∞ η) :
    d (d η) = 0 :=
  extDeriv_extDeriv ((contDiff_infty.mp hω) 2) (by simp)

/-- **Derivative regularity from derivative-map regularity**: `d η` is `fderiv η`
    followed by the continuous linear map
    `ContinuousAlternatingMap.alternatizeUncurryFinCLM` (whose defining
    property is `alternatizeUncurryFinCLM_apply`), so `d η` is `C^k` whenever
    `fderiv η` is. -/
theorem contDiff_d_of_contDiff_fderiv {n : ℕ} {k : ℕ∞ω} {η : Form n}
    (hf : ContDiff ℝ k (fderiv ℝ η)) : ContDiff ℝ k (d η) := by
  have hcomp : ContDiff ℝ k
      (fun x => ContinuousAlternatingMap.alternatizeUncurryFinCLM ℝ Spacetime4 ℝ
        (n := n) (fderiv ℝ η x)) :=
    (ContinuousLinearMap.contDiff
      (ContinuousAlternatingMap.alternatizeUncurryFinCLM ℝ Spacetime4 ℝ (n := n))).fun_comp hf
  have h_eq : d η = fun x =>
      ContinuousAlternatingMap.alternatizeUncurryFinCLM ℝ Spacetime4 ℝ (n := n)
        (fderiv ℝ η x) := by
    funext x
    simp only [d, extDeriv, ContinuousAlternatingMap.alternatizeUncurryFinCLM_apply]
  rw [h_eq]
  exact hcomp

/-- **Smoothness of the exterior derivative**: for a smooth form, `d η` is
    smooth.  `contDiff_infty_iff_fderiv` supplies the smoothness of `fderiv η`
    from `ContDiff ℝ ∞ η`, and `contDiff_d_of_contDiff_fderiv` composes with
    the linear map `alternatizeUncurryFinCLM`. -/
theorem contDiff_d {n : ℕ} {η : Form n} (hω : ContDiff ℝ ∞ η) :
    ContDiff ℝ ∞ (d η) :=
  contDiff_d_of_contDiff_fderiv (contDiff_infty_iff_fderiv.mp hω).2

/-- `d` is additive on smooth forms.  (The differentiability hypotheses of
    `extDeriv_add` are supplied in the finite form `C¹`, extracted from the
    `C^∞` hypotheses via `contDiff_infty`.) -/
theorem d_add {n : ℕ} {ω₁ ω₂ : Form n}
    (h₁ : ContDiff ℝ ∞ ω₁) (h₂ : ContDiff ℝ ∞ ω₂) :
    d (ω₁ + ω₂) = d ω₁ + d ω₂ := by
  funext x
  exact extDeriv_add (((contDiff_infty.mp h₁) 1).contDiffAt.differentiableAt_one)
    (((contDiff_infty.mp h₂) 1).contDiffAt.differentiableAt_one)

/-- `d` commutes with scalar multiplication. -/
theorem d_smul {n : ℕ} (c : ℝ) (η : Form n) : d (c • η) = c • d η := by
  funext x
  exact extDeriv_smul c η

-- ============================================================
-- THE ELECTROMAGNETIC FIELD
-- ============================================================

/-- The electromagnetic field: a smooth potential 1-form `A` and the field
    strength `F = d A`. -/
structure ElectromagneticField where
  A : Form 1
  F : Form 2
  smooth_A : ContDiff ℝ ∞ A
  F_eq_dA : F = d A

/-- **Homogeneous Maxwell equations**: `dF = 0` for `F = dA` with smooth
    potential.  Derived from `d² = 0`, not assumed. -/
theorem homogeneous_maxwell (em : ElectromagneticField) :
    d em.F = 0 := by
  simp only [em.F_eq_dA, d_squared_zero em.A em.smooth_A]

-- ============================================================
-- GAUGE INVARIANCE
-- ============================================================

/-- Gauge transformation of a 1-form: `A ↦ A + dχ` for a 0-form `χ`. -/
noncomputable def gaugeTransform (A : Form 1) (χ : Form 0) : Form 1 :=
  A + d χ

/-- **Gauge invariance of the field strength**: for smooth `A` and `χ`,
    `d (gaugeTransform A χ) = d A`, since `d (A + dχ) = dA + d(dχ)` and
    `d(dχ) = 0` by `d_squared_zero`. -/
theorem field_strength_gauge_invariant {A : Form 1} {χ : Form 0}
    (hA : ContDiff ℝ ∞ A) (hχ : ContDiff ℝ ∞ χ) :
    d (gaugeTransform A χ) = d A := by
  have hdχ : ContDiff ℝ ∞ (d χ) := contDiff_d hχ
  have hd2χ : d (d χ) = 0 := d_squared_zero χ hχ
  change d (A + d χ) = d A
  simp only [d_add hA hdχ, hd2χ, add_zero]

/-- Gauge transformations compose additively: `(A + dχ) + dψ = A + d(χ + ψ)`. -/
theorem gauge_transform_comp {A : Form 1} {χ ψ : Form 0}
    (hχ : ContDiff ℝ ∞ χ) (hψ : ContDiff ℝ ∞ ψ) :
    gaugeTransform (gaugeTransform A χ) ψ = gaugeTransform A (χ + ψ) := by
  have h : d (χ + ψ) = d χ + d ψ := d_add hχ hψ
  simp only [gaugeTransform, h, add_assoc]

-- ============================================================
-- INHOMOGENEOUS MAXWELL EQUATIONS (EXPLICIT MODEL LAW)
-- ============================================================

/-- A sourced Maxwell configuration.  Mathlib has no Hodge star on
    differential forms, so the dual field `⋆F` (a 2-form) and the dual current
    `⋆J` (a 3-form, as in the standard 4D bookkeeping) are carried as
    parameters related by the explicit model law
    `law_dual_maxwell : d ⋆F = ⋆J`.  This is a *law of the model* (data), not a
    derivation — stated as such rather than disguised by `HodgeStar := 0`,
    `J := 0`. -/
structure SourcedField where
  A : Form 1
  smooth_A : ContDiff ℝ ∞ A
  starF : Form 2
  starJ : Form 3
  law_dual_maxwell : d starF = starJ

/-- The inhomogeneous Maxwell equation of a sourced configuration: the model
    law carried by the data. -/
theorem inhomogeneous_maxwell (S : SourcedField) :
    d S.starF = S.starJ :=
  S.law_dual_maxwell

/-- Full Maxwell sector: the homogeneous equation is *derived* from `d² = 0`;
    the inhomogeneous one is the model law of the sourced configuration. -/
theorem maxwells_equations (S : SourcedField) :
    d (d S.A) = 0 ∧ d S.starF = S.starJ :=
  ⟨d_squared_zero S.A S.smooth_A, inhomogeneous_maxwell S⟩

-- ============================================================
-- PHYSICAL CONSTANTS
-- ============================================================

/-- Model postulate, stated as a definition: the gauge field of the minimal
    model is taken massless (`PhotonMass := 0`).  This is not a derived
    result; the content is the definitional restatement below. -/
noncomputable def PhotonMass : ℝ := 0

/-- Definitional restatement of the masslessness postulate above. -/
theorem photon_massless : PhotonMass = 0 := rfl

noncomputable def FineStructureConstant : ℝ := 1 / 137.035999
theorem fine_structure_positive : FineStructureConstant > 0 := by
  dsimp [FineStructureConstant]; norm_num

end OmegaProtocol.Vol02
