import Mathlib.Analysis.VonNeumannAlgebra.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.LinearAlgebra.Trace
import Mathlib.Topology.Instances.Complex
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Real.Pi.Bounds
import DynamicCODScale

namespace OmegaProtocol

/-!
# Concrete Omega model

The original version of this file declared every symbol and every desired
property as an `unchecked declaration`.  That made downstream statements look proved even
when the kernel had been handed the result.  This module now supplies a
small, explicit zero-information model instead:

* `StateSpace` is `ℂ`, the one-dimensional Hilbert space (the type-`I₁`
  tracial model of the modular pillar);
* `QRegion` is a genuine finite information region — a normalized von Neumann
  entropy and a chain-overlap density, both in `(0,1]` — so the region-level
  quantities (`vonNeumannEntropy`, `mutualInformation`, the overlap density
  `Φ`, the metric `d`) are non-constant, with `unitQRegion`/`halfQRegion` as
  distinct witnesses and concrete computed values (`witness_mutualInformation_eq`
  is `3/16`, `witness_reverseFlux_eq` is `13/16`);
* the state functional `ω_ρ` is the algebraic trace (a genuine, non-constant
  tracial state — see `ω_ρ_trace` in `OmegaUnifiedFoundation.lean`), and the
  functional calculus is the honest one for a one-dimensional state space,
  `A ^ z = (A 1) ^ z • 1` (see `op_pow`, with the algebraic laws `op_pow_zero`,
  `op_pow_one`, `op_pow_add`, `op_pow_of_one`); `CyclicSeparating` is the
  genuine cyclic-and-separating predicate (`not_cyclicSeparating_zero` shows it
  is not trivially true);
* the metric's Planck-scale factor is the canonical COD profile
  `ℓ_P(Φ) = ℓ_P0·√(1-Φ²)` from `DynamicCODScale` (evaluated at `ℓ_P0 = 1`),
  shared with every other file that needs a Planck scale
  (`planckFactor_eq_dynamicPlanckCOD`); and
* the properties that follow from this model are proved below.

The *remaining* minimal-choice layer, stated rather than hidden, is the
spacetime/operator stub layer: `state_at`, `tangent_at` and `vacuum_at` return
`0`, `QFIM`/`Hessian` are the zero functional pair, `CovariantDivergence` is
zero, `commutator A _ := A` and `lim_h_to_0 f := f 0` are placeholder
operations, and `StateType`/`Observable`/`ConnesInvariant` are `Unit` carriers
with `PureState`/`TraceOp` the constantly-false predicates used by the
`TypeIII1Algebra` bundle (whose concrete instance is therefore not a faithful
type-III algebra).

This is intentionally a model, not a claim that the physical Omega theory has
been derived.  Theorems depending on a richer physical model must add those
assumptions as hypotheses rather than hiding them in axioms.  Consequences of
the degeneracy are proved under the `bridge_*` naming convention (zero-model
consistency witnesses); no declaration in this file asserts a physical law,
and the substantive counterparts live in `InformationNetwork` and the
`dynamicPlanckCOD` development.  Retired names and their replacements are
recorded in `PROOF_AUDIT.md` (tenth pass).
-/

/-- The concrete Hilbert space used by the minimal model. -/
def StateSpace : Type := ℂ

noncomputable instance : NormedAddCommGroup StateSpace := inferInstanceAs (NormedAddCommGroup ℂ)
noncomputable instance : InnerProductSpace ℂ StateSpace := inferInstanceAs (InnerProductSpace ℂ ℂ)
instance : CompleteSpace StateSpace := inferInstanceAs (CompleteSpace ℂ)

/-- The top operator *-algebra serves as the concrete algebra for the model.
    (Mathlib does not (yet) supply a `Top` instance for the bundled
    `VonNeumannAlgebra` structure, so the minimal model uses the top
    `StarSubalgebra` of bounded operators on `StateSpace` instead.) -/
noncomputable def OmegaAlgebra : StarSubalgebra ℂ (StateSpace →L[ℂ] StateSpace) := ⊤

abbrev Operator := StateSpace →L[ℂ] StateSpace

/-- The model's scalar functional calculus `A ↦ A ^ z`.  The state space is
    one-dimensional, so an operator is determined by its single value `A 1`
    (`operator_eq_smul_one`), and the honest calculus is the complex power of
    that value transported back to the operator:
    `A ^ z = (A 1) ^ z • 1`.  This is *not* the constant placeholder `A ^ z = A`
    used before the eleventh pass: the algebraic laws `op_pow_zero`,
    `op_pow_one`, `op_pow_add` and `op_pow_of_one` below are genuine (the
    additive law needs the base to be nonzero, exactly as `Complex.cpow_add`
    does), and `law_modular_operator` in `OmegaUnifiedFoundation.lean` is now a
    computation in this calculus rather than a definitional restatement. -/
noncomputable def op_pow (A : Operator) (z : ℂ) : Operator := (A 1) ^ z • (1 : Operator)
@[default_instance] noncomputable instance instHPowOperator : HPow Operator ℂ Operator where
  hPow := op_pow

/-- **Every operator on the one-dimensional state space is scalar
    multiplication by its value at `1`.**  This is the structural fact that
    makes the model's functional calculus honest. -/
theorem operator_eq_smul_one (A : Operator) : A = A 1 • (1 : Operator) := by
  ext v
  calc A v = A (v • (1 : StateSpace)) := by rw [smul_eq_mul, mul_one]
    _ = v • A 1 := map_smul A v 1
    _ = (A 1 • (1 : Operator)) v := by
        rw [smul_apply, one_apply_eq_self, smul_eq_mul, mul_comm]

/-- Model calculus: the zeroth complex power is the identity operator
    (`Complex.cpow_zero` together with `1 • 1 = 1`). -/
theorem op_pow_zero (A : Operator) : A ^ (0 : ℂ) = 1 := by
  rw [op_pow, Complex.cpow_zero, one_smul]

/-- Model calculus: the first complex power is the operator itself, because the
    operator is determined by its value at `1` (`operator_eq_smul_one`). -/
theorem op_pow_one (A : Operator) : A ^ (1 : ℂ) = A := by
  rw [op_pow, Complex.cpow_one]
  exact (operator_eq_smul_one A).symm

/-- Model calculus: powers of the identity operator are the identity — the case
    exercised by the tracial modular operator `Δ = 1`. -/
theorem op_pow_of_one (z : ℂ) : (1 : Operator) ^ z = 1 := by
  rw [op_pow, one_apply_eq_self, Complex.one_cpow, one_smul]

/-- Model calculus is multiplicative on the exponent of a nonzero base: the
    complex-power law `x ^ (z + w) = x ^ z * x ^ w` for `x = A 1 ≠ 0`.  The
    hypothesis is genuine — `Complex.cpow_add` needs it, and `0 ^ z` is not
    multiplicative in `z`. -/
theorem op_pow_add (A : Operator) (h : A 1 ≠ 0) (z w : ℂ) :
    A ^ (z + w) = A ^ z * A ^ w := by
  rw [op_pow, op_pow, op_pow, Complex.cpow_add z w h]
  ext v
  simp only [smul_apply, mul_apply_eq_comp, one_apply_eq_self, smul_eq_mul]
  ring

@[default_instance] noncomputable instance instCoeOmegaAlgebraOperator :
    Coe ↥OmegaAlgebra Operator where
  coe A := A.1

@[default_instance] noncomputable instance instMulOperator : Mul Operator where
  mul A B := A.comp B

/-- **The model vector is cyclic and separating** for the top algebra, in the
    standard von Neumann-algebra sense: the orbit map `A ↦ A·ξ` has dense range
    in the state space (cyclicity) and `A·ξ = 0` forces `A = 0` (separating).

    Both conjuncts have content: `CyclicSeparating 0` is *false*
    (`not_cyclicSeparating_zero` below: the zero vector is not separating),
    while both are *proved* for the model's `Ω`-state `1` in
    `OmegaUnifiedFoundation.concreteModularTheory`
    (`law_omega_cyclic_separating`), where cyclicity uses surjectivity of
    `A ↦ A 1` and separatingness uses `operator_eq_smul_one`. -/
def CyclicSeparating (ξ : StateSpace) : Prop :=
  DenseRange (fun A : ↥OmegaAlgebra => (A : Operator) ξ) ∧
    (∀ A : ↥OmegaAlgebra, (A : Operator) ξ = 0 → A = 0)

/-- **Non-vacuity witness**: the zero vector is not cyclic-and-separating —
    the orbit through `0` vanishes for *every* algebra element, so separation
    fails at `A = 1 ≠ 0`. -/
theorem not_cyclicSeparating_zero : ¬ CyclicSeparating (0 : StateSpace) := by
  rintro ⟨_, hsep⟩
  have h0 : ((1 : ↥OmegaAlgebra) : Operator) (0 : StateSpace) = 0 := map_zero _
  have h1 : (1 : ↥OmegaAlgebra) = 0 := hsep 1 h0
  have h2 : (1 : Operator) = 0 := by
    simpa using congrArg (fun A : ↥OmegaAlgebra => (A : Operator)) h1
  have h3 : (1 : StateSpace) = 0 := by
    simpa using congrArg (fun f : Operator => f (1 : StateSpace)) h2
  exact one_ne_zero h3

/-- The model state functional: the algebraic trace of the operator in the
    one-dimensional representation, `A ↦ tr A`.  Non-constant (`tr 0 = 0` vs
    `tr 1 = 1`; `op_pow_one`/`ω_ρ_trace` in `OmegaUnifiedFoundation.lean`) and a
    trace by construction: Mathlib's basis-independent `LinearMap.trace_mul_comm`
    gives `ω_ρ (A ∘ B) = ω_ρ (B ∘ A)`, which is exactly the KMS condition in the
    tracial model where the modular flow is trivial.  A non-tracial functional
    falsifies that equation, so the KMS predicate is not vacuous. -/
noncomputable def ω_ρ (A : ↥OmegaAlgebra) : ℂ :=
  LinearMap.trace ℂ StateSpace ((A : Operator) : StateSpace →ₗ[ℂ] StateSpace)

abbrev CPTPMap := StateSpace → StateSpace

def QFIM (_ : StateSpace) (_ : Operator) (_ : Operator) : ℝ := 0
def Hessian (_ : StateSpace → ℝ) (_ : StateSpace → ℝ) (_ : StateSpace → ℝ) : ℝ := 0
def NewtonG : ℝ := 1

noncomputable def state_at {spacetime : Type*} (_ : spacetime) : StateSpace := 0
noncomputable def tangent_at {spacetime : Type*} (_ : spacetime) : Operator := 0
def CovariantDivergence {spacetime : Type*}
    (_ : spacetime → spacetime → ℝ) (_ : spacetime) : ℝ := 0

def StateType : Type := Unit
def PureState (_ : StateType) : Prop := False
def TraceOp (_ : (↥OmegaAlgebra → ℝ)) : Prop := False

def ConnesInvariant : Type := Unit
def III₁ : ConnesInvariant := ()

def Observable : Type := Unit
def commutator {T : Type*} (A _ : T) : T := A
def lim_h_to_0 {T : Type*} (f : ℝ → T) : T := f 0

noncomputable def vacuum_at {manifold : Type*} (_ : manifold) : StateSpace := 0

-- `QRegion` is a structure without a `DecidableEq` instance, so the region
-- model's case split on `R₁ = R₂` (inside `Φ`) needs classical decidability,
-- exactly as `OmegaUnifiedFoundation` records for state equality:
attribute [local instance] Classical.propDecidable

/-- **A Q-Region** — a finite information region of the model, carrying a
    normalized von Neumann entropy `S ∈ (0,1]` and the chain-overlap density
    `Φ_R ∈ (0,1]`, the fundamental primitive of Omega Theory.  The former
    `QRegion := Unit` made every region-level quantity a constant, so all the
    `d`/`Φ`/entropy statements were `0 = 0`-shaped; `unitQRegion` and
    `halfQRegion` below are two distinct witnesses of the present model, and
    the region functions (`vonNeumannEntropy`, `mutualInformation`, `Φ`, the
    metric `d`) are non-constant in these parameters. -/
structure QRegion where
  entropy : ℝ
  entropy_pos : 0 < entropy
  entropy_le_one : entropy ≤ 1
  density : ℝ
  density_pos : 0 < density
  density_le_one : density ≤ 1

/-- The saturated region: unit entropy and unit overlap density. -/
noncomputable def unitQRegion : QRegion :=
  ⟨1, by norm_num, le_rfl, 1, by norm_num, le_rfl⟩

/-- A half-saturated region: entropy `1/2` and overlap density `1/2`. -/
noncomputable def halfQRegion : QRegion :=
  ⟨1 / 2, by norm_num, by norm_num, 1 / 2, by norm_num, by norm_num⟩

theorem halfQRegion_ne_unitQRegion : halfQRegion ≠ unitQRegion := by
  intro h
  have hd : (1 / 2 : ℝ) = 1 := congrArg QRegion.density h
  norm_num at hd

/-- The model's von Neumann entropy of a region. -/
noncomputable def vonNeumannEntropy (R : QRegion) : ℝ := R.entropy

theorem vonNeumannEntropy_pos (R : QRegion) : 0 < vonNeumannEntropy R :=
  R.entropy_pos

theorem vonNeumannEntropy_le_one (R : QRegion) : vonNeumannEntropy R ≤ 1 :=
  R.entropy_le_one

/-- Joint entropy of two regions: the sum of the two region entropies (the
    model's subadditivity-saturating choice, a stated model law). -/
noncomputable def jointEntropy (R₁ R₂ : QRegion) : ℝ := R₁.entropy + R₂.entropy

theorem jointEntropy_pos (R₁ R₂ : QRegion) : 0 < jointEntropy R₁ R₂ := by
  rw [jointEntropy]
  exact add_pos R₁.entropy_pos R₂.entropy_pos

theorem jointEntropy_symm (R₁ R₂ : QRegion) : jointEntropy R₁ R₂ = jointEntropy R₂ R₁ := by
  rw [jointEntropy, jointEntropy, add_comm]

/-- **Chain-overlap density of a pair** (the model law behind the metric): a
    region overlaps itself completely, `Φ R R = 1`, and two *distinct* regions
    share the smaller of their densities, `Φ R₁ R₂ = min R₁.density R₂.density`.
    The `min` coupling is the hierarchical-similarity form, and
    `Φ_ultrametric` below is the resulting ultrametric law
    `Φ(X,Z) ≥ min (Φ X Y) (Φ Y Z)` — the property that makes the metric satisfy
    the triangle inequality.  A model law, stated as such and not derived. -/
noncomputable def Φ (R₁ R₂ : QRegion) : ℝ :=
  if R₁ = R₂ then 1 else min R₁.density R₂.density

theorem Φ_pos (R₁ R₂ : QRegion) : 0 < Φ R₁ R₂ := by
  rw [Φ]
  split_ifs
  · norm_num
  · exact lt_min R₁.density_pos R₂.density_pos

theorem Φ_le_one (R₁ R₂ : QRegion) : Φ R₁ R₂ ≤ 1 := by
  rw [Φ]
  split_ifs
  · exact le_rfl
  · exact (min_le_left _ _).trans R₁.density_le_one

theorem Φ_self (R : QRegion) : Φ R R = 1 := by
  rw [Φ]
  exact if_pos rfl

theorem Φ_eq_one_iff (R₁ R₂ : QRegion) :
    Φ R₁ R₂ = 1 ↔ R₁ = R₂ ∨ R₁.density = 1 ∧ R₂.density = 1 := by
  rw [Φ]
  split_ifs with h
  · exact ⟨fun _ => Or.inl h, fun _ => rfl⟩
  · constructor
    · intro hmin
      refine Or.inr ⟨le_antisymm R₁.density_le_one ?_, le_antisymm R₂.density_le_one ?_⟩
      · rw [← hmin]; exact min_le_left _ _
      · rw [← hmin]; exact min_le_right _ _
    · rintro (h' | ⟨h1, h2⟩)
      · exact absurd h' h
      · rw [h1, h2]; exact min_self 1

/-- **The ultrametric law of the overlap density**: for any three regions the
    outer pair's overlap is at least the smaller of the two intermediate
    overlaps.  This is the hierarchical-similarity property of the `min`
    coupling. -/
theorem Φ_ultrametric (R₁ R₂ R₃ : QRegion) :
    min (Φ R₁ R₂) (Φ R₂ R₃) ≤ Φ R₁ R₃ := by
  by_cases h13 : R₁ = R₃
  · rw [h13, Φ_self]
    exact (min_le_left _ _).trans (Φ_le_one _ _)
  · by_cases h12 : R₁ = R₂
    · rw [h12, Φ_self]
      exact min_le_right _ _
    · by_cases h23 : R₂ = R₃
      · rw [← h23, Φ_self]
        exact min_le_left _ _
      · rw [Φ, Φ, Φ, if_neg h12, if_neg h23, if_neg h13]
        exact le_min (le_trans (min_le_left _ _) (min_le_left _ _))
          (le_trans (min_le_right _ _) (min_le_right _ _))

theorem bridge_overlapDensity_nonneg (R₁ R₂ : QRegion) : Φ R₁ R₂ ≥ 0 :=
  le_of_lt (Φ_pos R₁ R₂)

theorem bridge_overlapDensity_symm (R₁ R₂ : QRegion) : Φ R₁ R₂ = Φ R₂ R₁ := by
  rw [Φ, Φ]
  split_ifs with h h'
  · rfl
  · exact absurd h.symm h'
  · exact absurd h' h.symm
  · exact min_comm _ _

/-- **Mutual information of a pair**: `Φ·(1-Φ)·(S₁+S₂)/2`, a genuine,
    non-constant function of the region parameters.  It vanishes exactly at
    complete overlap (`Φ = 1`, i.e. identical or fully saturated regions —
    `mutualInformation_eq_zero_iff`) and is strictly positive otherwise
    (`mutualInformation_pos_iff`, witnessed by `witness_mutualInformation_eq`).
    The former model had `mutualInformation ≡ 0`, which made the ER=EPR and
    entropy bridges `0 = 0`-shaped. -/
noncomputable def mutualInformation (R₁ R₂ : QRegion) : ℝ :=
  Φ R₁ R₂ * (1 - Φ R₁ R₂) * ((R₁.entropy + R₂.entropy) / 2)

theorem mutualInformation_nonneg (R₁ R₂ : QRegion) : 0 ≤ mutualInformation R₁ R₂ := by
  rw [mutualInformation]
  refine mul_nonneg (mul_nonneg (le_of_lt (Φ_pos R₁ R₂)) ?_) ?_
  · have h := Φ_le_one R₁ R₂; linarith
  · have h := add_pos R₁.entropy_pos R₂.entropy_pos; linarith

theorem mutualInformation_symm (R₁ R₂ : QRegion) :
    mutualInformation R₁ R₂ = mutualInformation R₂ R₁ := by
  rw [mutualInformation, mutualInformation, bridge_overlapDensity_symm,
    add_comm R₁.entropy R₂.entropy]

theorem mutualInformation_self (R : QRegion) : mutualInformation R R = 0 := by
  rw [mutualInformation, Φ_self]
  ring

theorem mutualInformation_le_quarter (R₁ R₂ : QRegion) :
    mutualInformation R₁ R₂ ≤ 1 / 4 := by
  rw [mutualInformation]
  have hS0 : 0 ≤ (R₁.entropy + R₂.entropy) / 2 := by
    have h := add_pos R₁.entropy_pos R₂.entropy_pos; linarith
  have hS1 : (R₁.entropy + R₂.entropy) / 2 ≤ 1 := by
    have h₁ := R₁.entropy_le_one
    have h₂ := R₂.entropy_le_one
    linarith
  have hprod : Φ R₁ R₂ * (1 - Φ R₁ R₂) ≤ 1 / 4 := by
    nlinarith [Φ_pos R₁ R₂, Φ_le_one R₁ R₂, sq_nonneg (Φ R₁ R₂ - 1 / 2)]
  calc Φ R₁ R₂ * (1 - Φ R₁ R₂) * ((R₁.entropy + R₂.entropy) / 2)
      ≤ (1 / 4) * 1 := mul_le_mul hprod hS1 hS0 (by norm_num)
    _ = 1 / 4 := by norm_num

/-- Mutual information vanishes exactly at complete overlap. -/
theorem mutualInformation_eq_zero_iff (R₁ R₂ : QRegion) :
    mutualInformation R₁ R₂ = 0 ↔ R₁ = R₂ ∨ R₁.density = 1 ∧ R₂.density = 1 := by
  have hS : ((R₁.entropy + R₂.entropy) / 2) ≠ 0 := by
    have h := add_pos R₁.entropy_pos R₂.entropy_pos
    linarith
  rw [mutualInformation]
  constructor
  · intro h
    have hfac : Φ R₁ R₂ * (1 - Φ R₁ R₂) = 0 := by
      rcases mul_eq_zero.mp h with h1 | h2
      · exact h1
      · exact absurd h2 hS
    rcases mul_eq_zero.mp hfac with h1 | h2
    · exact absurd h1 (ne_of_gt (Φ_pos R₁ R₂))
    · have hΦ : Φ R₁ R₂ = 1 := by linarith
      exact (Φ_eq_one_iff R₁ R₂).mp hΦ
  · rintro (h | ⟨h1, h2⟩)
    · rw [h, Φ_self]; ring
    · rw [(Φ_eq_one_iff R₁ R₂).mpr (Or.inr ⟨h1, h2⟩)]; ring

/-- Mutual information is strictly positive exactly below complete overlap. -/
theorem mutualInformation_pos_iff (R₁ R₂ : QRegion) :
    0 < mutualInformation R₁ R₂ ↔ Φ R₁ R₂ < 1 := by
  have hS : 0 < (R₁.entropy + R₂.entropy) / 2 := by
    have h := add_pos R₁.entropy_pos R₂.entropy_pos
    linarith
  rw [mutualInformation]
  constructor
  · intro h
    by_contra hc
    have hc' : 1 ≤ Φ R₁ R₂ := le_of_not_gt hc
    have h1 : Φ R₁ R₂ = 1 := le_antisymm (Φ_le_one _ _) hc'
    have hcontra : (0 : ℝ) < 0 := by simpa [h1] using h
    exact absurd hcontra (lt_irrefl 0)
  · intro hΦ
    have hΦ0 : 0 < Φ R₁ R₂ := Φ_pos _ _
    have h2 : 0 < 1 - Φ R₁ R₂ := by linarith
    exact mul_pos (mul_pos hΦ0 h2) hS

/-- The model's normalized information capacity: entropies are normalized to
    `≤ 1`, and `mutualInformation_le_quarter` gives `I ≤ 1/4`, so `1` is a
    genuine (if loose) upper bound. -/
noncomputable def maxMutualInformation : ℝ := 1

theorem bridge_mutualInformation_le_max (R₁ R₂ : QRegion) :
    mutualInformation R₁ R₂ ≤ maxMutualInformation := by
  rw [maxMutualInformation]
  have h := mutualInformation_le_quarter R₁ R₂
  linarith

/-- The pair chain-overlap density as the ratio of mutual information to joint
    entropy, `I/(S₁+S₂) = Φ(1-Φ)/2` — the entropy-normalized companion of the
    primitive pair coupling `Φ` that the metric uses directly. -/
noncomputable def chainOverlapDensity (R₁ R₂ : QRegion) : ℝ :=
  mutualInformation R₁ R₂ / jointEntropy R₁ R₂

theorem chainOverlapDensity_nonneg (R₁ R₂ : QRegion) : 0 ≤ chainOverlapDensity R₁ R₂ :=
  div_nonneg (mutualInformation_nonneg R₁ R₂) (le_of_lt (jointEntropy_pos R₁ R₂))


/-- The model's Planck-scale factor at overlap density `φ`: the canonical COD
    profile `ℓ_P(φ) = ℓ_P0·√(1-φ²)` at unit baseline
    (`planckFactor_eq_dynamicPlanckCOD`). -/
noncomputable def planckFactor (φ : ℝ) : ℝ := Real.sqrt (1 - φ * φ)

theorem planckFactor_eq_dynamicPlanckCOD (φ : ℝ) :
    planckFactor φ = DynamicCODScale.dynamicPlanckCOD canonicalCODEnv φ := by
  rw [planckFactor, DynamicCODScale.dynamicPlanckCOD, canonicalCODEnv, pow_two, mul_one]

theorem planckFactor_nonneg (φ : ℝ) : 0 ≤ planckFactor φ := Real.sqrt_nonneg _

theorem planckFactor_antitone {φ φ' : ℝ} (hφ' : 0 ≤ φ') (h : φ' ≤ φ) :
    planckFactor φ ≤ planckFactor φ' := by
  rw [planckFactor, planckFactor]
  exact Real.sqrt_le_sqrt (by linarith [mul_le_mul h h hφ' (hφ'.trans h)])

/-- The distance contributed by a pair at overlap density `φ`:
    `-ℓ_P(φ)·log φ`, the correlation-log distance weighted by the COD scale
    factor.  It is nonnegative on `(0,1]` (`pairDistance_nonneg`), antitone
    there (`pairDistance_antitone`), and strictly positive exactly below
    complete overlap (`pairDistance_pos_iff`). -/
noncomputable def pairDistance (φ : ℝ) : ℝ := -(planckFactor φ) * Real.log φ

theorem pairDistance_eq (φ : ℝ) : pairDistance φ = planckFactor φ * (-Real.log φ) := by
  rw [pairDistance]; ring

theorem pairDistance_nonneg {φ : ℝ} (hφ : 0 < φ) (h1 : φ ≤ 1) : 0 ≤ pairDistance φ := by
  rw [pairDistance_eq]
  exact mul_nonneg (planckFactor_nonneg φ)
    (neg_nonneg.mpr (Real.log_nonpos (le_of_lt hφ) h1))

theorem pairDistance_antitone {φ φ' : ℝ} (hφ' : 0 < φ') (h : φ' ≤ φ) (hφ1 : φ ≤ 1) :
    pairDistance φ ≤ pairDistance φ' := by
  have hφ0 : 0 < φ := lt_of_lt_of_le hφ' h
  have hℓ : planckFactor φ ≤ planckFactor φ' := planckFactor_antitone (le_of_lt hφ') h
  have hl : -Real.log φ ≤ -Real.log φ' := by
    have hlog := Real.log_le_log hφ' h
    linarith
  have hc : 0 ≤ -Real.log φ := neg_nonneg.mpr (Real.log_nonpos (le_of_lt hφ0) hφ1)
  rw [pairDistance_eq, pairDistance_eq]
  exact mul_le_mul hℓ hl hc (planckFactor_nonneg φ')

theorem pairDistance_pos_iff {φ : ℝ} (hφ : 0 < φ) (h1 : φ ≤ 1) :
    0 < pairDistance φ ↔ φ < 1 := by
  rw [pairDistance_eq]
  have hℓ : 0 < planckFactor φ ↔ φ < 1 := by
    have hsq : φ * φ < 1 ↔ φ < 1 := by
      constructor
      · intro h
        by_contra hc
        have hc1 : 1 ≤ φ := le_of_not_gt hc
        have h2 : 1 ≤ φ * φ := one_le_mul_of_one_le_of_one_le hc1 hc1
        linarith
      · intro h
        have h2 : φ * φ < φ * 1 := mul_lt_mul_of_pos_left h hφ
        linarith
    rw [planckFactor, Real.sqrt_pos, sub_pos]
    exact hsq
  have hl : 0 < -Real.log φ ↔ φ < 1 :=
    neg_pos.trans (Real.log_neg_iff hφ)
  constructor
  · intro h
    rcases mul_pos_iff.mp h with ⟨h1', _⟩ | ⟨h1', _⟩
    · exact hℓ.mp h1'
    · exact absurd h1' (not_lt.mpr (planckFactor_nonneg φ))
  · intro h
    exact mul_pos (hℓ.mpr h) (hl.mpr h)

/-- Core of the triangle inequality for the model metric: monotonicity of the
    pair distance together with the ultrametric law for `Φ` bounds the outer
    pair by the smaller intermediate one. -/
theorem pairDistance_ultrametric {φ ψ χ : ℝ} (hφ : 0 < φ) (hψ : 0 < ψ) (hχ : 0 < χ)
    (hφ1 : φ ≤ 1) (hψ1 : ψ ≤ 1) (hχ1 : χ ≤ 1) (hultra : min φ ψ ≤ χ) :
    pairDistance χ ≤ pairDistance φ + pairDistance ψ := by
  rcases le_total φ ψ with hle | hle
  · have hφχ : φ ≤ χ := le_trans (min_le_left φ ψ) hultra
    have hmono := pairDistance_antitone hφ hφχ hχ1
    have hnn := pairDistance_nonneg hψ hψ1
    linarith
  · have hψχ : ψ ≤ χ := le_trans (min_le_right φ ψ) hultra
    have hmono := pairDistance_antitone hψ hψχ hχ1
    have hnn := pairDistance_nonneg hφ hφ1
    linarith

/-- Unit-base-scale COD environment: the model evaluates the canonical
    `DynamicCODScale` profile at `ℓ_P0 = 1`, so the metric's Planck factor is
    `√(1 - Φ²)` — the same profile used everywhere else in the project.
    (A previous revision used a bespoke `√(πΦ+1)` here; see C3 in
    ADVERSARIAL_AUDIT.md.) -/
noncomputable def canonicalCODEnv : DynamicCODScale.CODEnvironment :=
  ⟨1, by norm_num⟩

/-- **The Ω-metric of a pair**: `d R₁ R₂ = -ℓ_P(Φ)·log Φ`, the COD-scaled
    correlation-log distance of the pair's chain-overlap density.  Unlike the
    former `QRegion := Unit` model (where the metric was identically zero), it
    is non-constant: `witness_distance_pos` separates the half-saturated and
    saturated regions strictly, and it vanishes exactly at complete overlap. -/
noncomputable def omegaMetric (R₁ R₂ : QRegion) : ℝ := pairDistance (Φ R₁ R₂)

noncomputable def d (R₁ R₂ : QRegion) : ℝ := omegaMetric R₁ R₂

theorem bridge_distance_nonneg (R₁ R₂ : QRegion) : d R₁ R₂ ≥ 0 := by
  rw [d, omegaMetric]
  exact pairDistance_nonneg (Φ_pos R₁ R₂) (Φ_le_one R₁ R₂)

theorem bridge_qregion_self_distance_zero (R : QRegion) : d R R = 0 := by
  rw [d, omegaMetric, Φ_self, pairDistance]
  simp [planckFactor]

theorem distance_symm (R₁ R₂ : QRegion) : d R₁ R₂ = d R₂ R₁ := by
  rw [d, d, omegaMetric, omegaMetric, bridge_overlapDensity_symm]

/-- **The Ω-metric satisfies the triangle inequality** — the genuine version:
    the former model's distance was identically zero, so the statement was
    `0 ≤ 0 + 0`.  Here it follows from the ultrametric law of `Φ` and the
    monotonicity of the COD-scaled log distance (`pairDistance_ultrametric`). -/
theorem bridge_distance_triangle_inequality (R₁ R₂ R₃ : QRegion) :
    d R₁ R₃ ≤ d R₁ R₂ + d R₂ R₃ := by
  by_cases h13 : R₁ = R₃
  · rw [h13, bridge_qregion_self_distance_zero]
    exact add_nonneg (bridge_distance_nonneg _ _) (bridge_distance_nonneg _ _)
  by_cases h12 : R₁ = R₂
  · rw [← h12, bridge_qregion_self_distance_zero, zero_add]
  by_cases h23 : R₂ = R₃
  · rw [← h23, bridge_qregion_self_distance_zero, add_zero]
  rw [d, d, d, omegaMetric, omegaMetric, omegaMetric]
  exact pairDistance_ultrametric (Φ_pos _ _) (Φ_pos _ _) (Φ_pos _ _)
    (Φ_le_one _ _) (Φ_le_one _ _) (Φ_le_one _ _) (Φ_ultrametric _ _ _)

/-- The metric vanishes exactly at complete overlap. -/
theorem distance_eq_zero_iff (R₁ R₂ : QRegion) :
    d R₁ R₂ = 0 ↔ Φ R₁ R₂ = 1 := by
  rw [d, omegaMetric]
  constructor
  · intro h
    by_contra hne
    have hlt : Φ R₁ R₂ < 1 := lt_of_le_of_ne (Φ_le_one _ _) hne
    have hpos := (pairDistance_pos_iff (Φ_pos R₁ R₂) (Φ_le_one R₁ R₂)).mpr hlt
    rw [h] at hpos
    exact absurd hpos (lt_irrefl 0)
  · intro hΦ
    rw [hΦ, pairDistance]
    simp [planckFactor]

/-- Complete overlap forces zero distance, and vice versa (the ER=EPR
    dictionary of the model): the metric's zero set is exactly the saturated
    pairs. -/
theorem bridge_distance_zero_when_I_zero (R₁ R₂ : QRegion)
    (h : mutualInformation R₁ R₂ = 0) : d R₁ R₂ = 0 :=
  (distance_eq_zero_iff R₁ R₂).mpr
    ((Φ_eq_one_iff R₁ R₂).mpr ((mutualInformation_eq_zero_iff R₁ R₂).mp h))

/-- Positive mutual information and positive distance coincide — the exact
    ER=EPR correspondence of the non-degenerate model. -/
theorem distance_pos_iff_mutualInformation_pos (R₁ R₂ : QRegion) :
    0 < d R₁ R₂ ↔ 0 < mutualInformation R₁ R₂ := by
  rw [mutualInformation_pos_iff]
  constructor
  · intro h
    have hne : Φ R₁ R₂ ≠ 1 := by
      intro h1
      have hzero : d R₁ R₂ = 0 := (distance_eq_zero_iff R₁ R₂).mpr h1
      rw [hzero] at h
      exact lt_irrefl 0 h
    exact lt_of_le_of_ne (Φ_le_one R₁ R₂) hne
  · intro hΦ
    have hne : d R₁ R₂ ≠ 0 := by
      intro hzero
      exact absurd ((distance_eq_zero_iff R₁ R₂).mp hzero) (ne_of_lt hΦ)
    exact lt_of_le_of_ne (bridge_distance_nonneg R₁ R₂) (Ne.symm hne)

/-- A concrete, fully computed witness: the half-saturated and saturated
    regions carry mutual information `3/16`. -/
theorem Φ_halfQRegion_unitQRegion : Φ halfQRegion unitQRegion = 1 / 2 := by
  rw [Φ, if_neg halfQRegion_ne_unitQRegion]
  have h : QRegion.density halfQRegion = 1 / 2 := rfl
  have h' : QRegion.density unitQRegion = 1 := rfl
  rw [h, h']
  exact min_eq_left (by norm_num)

theorem witness_mutualInformation_eq :
    mutualInformation halfQRegion unitQRegion = 3 / 16 := by
  rw [mutualInformation, Φ_halfQRegion_unitQRegion]
  have h : halfQRegion.entropy = 1 / 2 := rfl
  have h' : unitQRegion.entropy = 1 := rfl
  rw [h, h']
  norm_num

theorem witness_mutualInformation_pos : 0 < mutualInformation halfQRegion unitQRegion := by
  rw [witness_mutualInformation_eq]; norm_num

theorem witness_distance_pos : 0 < d halfQRegion unitQRegion := by
  rw [d, omegaMetric]
  rw [pairDistance_pos_iff (Φ_pos halfQRegion unitQRegion) (Φ_le_one halfQRegion unitQRegion),
    Φ_halfQRegion_unitQRegion]
  norm_num

theorem witness_codProfile_pos :
    0 < DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ halfQRegion unitQRegion) := by
  rw [← planckFactor_eq_dynamicPlanckCOD, Φ_halfQRegion_unitQRegion, planckFactor,
    Real.sqrt_pos, sub_pos]
  norm_num

/-- Structural property of the COD-scaled metric: the profile value of the outer
    pair never exceeds the sum of the two intermediate profile values (the
    profile is antitone and the overlap density is ultrametric).  This is the
    `bridge_codProfile_triangle` fact, which held trivially in the zero model
    (`1 ≤ 1 + 1`) and is a genuine inequality here. -/
theorem bridge_codProfile_triangle (R₁ R₂ R₃ : QRegion) :
    DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₁ R₃) ≤
      DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₁ R₂) +
      DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₂ R₃) := by
  rw [← planckFactor_eq_dynamicPlanckCOD, ← planckFactor_eq_dynamicPlanckCOD,
    ← planckFactor_eq_dynamicPlanckCOD]
  rcases le_total (Φ R₁ R₂) (Φ R₂ R₃) with h | h
  · have hultra := Φ_ultrametric R₁ R₂ R₃
    rw [min_eq_left h] at hultra
    have hmono := planckFactor_antitone (le_of_lt (Φ_pos R₁ R₂)) hultra
    linarith [planckFactor_nonneg (Φ R₂ R₃)]
  · have hultra := Φ_ultrametric R₁ R₂ R₃
    rw [min_eq_right h] at hultra
    have hmono := planckFactor_antitone (le_of_lt (Φ_pos R₂ R₃)) hultra
    linarith [planckFactor_nonneg (Φ R₁ R₂)]

theorem bridge_codProfile_pos (R₁ R₂ : QRegion) (h : Φ R₁ R₂ < 1) :
    0 < DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₁ R₂) := by
  rw [← planckFactor_eq_dynamicPlanckCOD, planckFactor, Real.sqrt_pos, sub_pos]
  have hφ := Φ_pos R₁ R₂
  calc Φ R₁ R₂ * Φ R₁ R₂ < Φ R₁ R₂ * 1 := mul_lt_mul_of_pos_left h hφ
    _ = Φ R₁ R₂ := mul_one _
    _ < 1 := h

/-- Zero-information transitivity: if two pairs are saturated, so is the third
    (a genuine law of the model, not the former `rfl` on a constant). -/
theorem bridge_zero_product_implies_zero (R₁ R₂ R₃ : QRegion)
    (h₁ : mutualInformation R₁ R₂ = 0) (h₂ : mutualInformation R₂ R₃ = 0) :
    mutualInformation R₁ R₃ = 0 := by
  rw [mutualInformation_eq_zero_iff] at h₁ h₂ ⊢
  rcases h₁ with h | ⟨h1, h2⟩
  · rw [h] at ⊢
    exact h₂
  · rcases h₂ with h | ⟨h1', h2'⟩
    · rw [← h]; exact Or.inr ⟨h1, h2⟩
    · exact Or.inr ⟨h1, h2'⟩

theorem bridge_metric_triangle_from_DPI (R₁ R₂ R₃ : QRegion) :
    d R₁ R₃ ≤ d R₁ R₂ + d R₂ R₃ :=
  bridge_distance_triangle_inequality R₁ R₂ R₃

/-- Arithmetic core of the "time dilation as processing lag" reading: a rate
    reduced by a novelty factor in `[0, 1]` never exceeds the base rate. This
    is the non-degenerate statement (all three hypotheses are used). -/
theorem speed_reduction_of_novelty (baseRate novelty : ℝ)
    (h_base : 0 ≤ baseRate) (h_novelty0 : 0 ≤ novelty) (h_novelty1 : novelty ≤ 1) :
    baseRate * (1 - novelty) ≤ baseRate := by
  calc baseRate * (1 - novelty) = baseRate - baseRate * novelty := by ring
    _ ≤ baseRate := by
      have hprod : 0 ≤ baseRate * novelty := mul_nonneg h_base h_novelty0
      linarith

noncomputable def informationalViscosity (updateRate : ℝ) : ℝ :=
  if updateRate = 0 then Real.pi else 1 / updateRate

structure DiscreteTime where
  step : ℕ
  state : QRegion

noncomputable def computationalLatency (Δupdates : ℕ) : ℝ :=
  if Δupdates = 0 then 0 else 1 / (Δupdates : ℝ)

/-- The model's novelty factor of a region: `sin S` of its entropy.  With the
    genuine entropy `S ∈ (0,1]` this is non-constant (`informationNovelty_pos`),
    where the former `Unit` model had `S ≡ 0` and hence zero novelty. -/
noncomputable def informationNovelty (R : QRegion) : ℝ :=
  Real.sin (vonNeumannEntropy R)

theorem informationNovelty_nonneg (R : QRegion) : 0 ≤ informationNovelty R := by
  rw [informationNovelty]
  exact Real.sin_nonneg_of_nonneg_of_le_pi (le_of_lt (vonNeumannEntropy_pos R))
    (by have h := vonNeumannEntropy_le_one R; linarith [Real.pi_gt_three])

theorem informationNovelty_le_one (R : QRegion) : informationNovelty R ≤ 1 := by
  rw [informationNovelty]
  exact Real.sin_le_one _

theorem informationNovelty_pos_witness : 0 < informationNovelty unitQRegion := by
  rw [informationNovelty]
  have h : vonNeumannEntropy unitQRegion = 1 := rfl
  rw [h]
  exact Real.sin_pos_of_pos_of_lt_pi (by norm_num) (by linarith [Real.pi_gt_three])

noncomputable def processingSpeed (R : QRegion) (baseRate : ℝ) : ℝ :=
  baseRate * (1 - informationNovelty R)

/-- The model's processing speed never exceeds the base rate: the novelty factor
    lies in `[0,1]`.  The former `bridge_processingSpeed_zero_model` recorded
    the degenerate `= baseRate` of the `Unit` model; here the reduction is
    genuine and strict for saturated regions (`witness_processingSpeed_lt`). -/
theorem processingSpeed_le_baseRate (R : QRegion) (baseRate : ℝ) (h : 0 ≤ baseRate) :
    processingSpeed R baseRate ≤ baseRate :=
  speed_reduction_of_novelty baseRate (informationNovelty R) h
    (informationNovelty_nonneg R) (informationNovelty_le_one R)

theorem witness_processingSpeed_lt (baseRate : ℝ) (h : 0 < baseRate) :
    processingSpeed unitQRegion baseRate < baseRate := by
  rw [processingSpeed]
  have hnov : 0 < informationNovelty unitQRegion := informationNovelty_pos_witness
  have hprod : 0 < baseRate * informationNovelty unitQRegion := mul_pos h hnov
  linarith

/-- A one-step recurrence trajectory is determined by its initial state:
    trajectories obeying the same law `ψ(t+1) = H (ψ t)` that agree at time
    zero agree everywhere. This replaces the retired
    `schrodinger_from_discrete_limit`, whose conclusion `∃ c, c = c` was
    satisfiable by every real number and whose hypothesis was unused — a
    tautology wearing a physics name. The present statement has content for
    an arbitrary state type; the Schrödinger limit itself is not formalized. -/
theorem discrete_recurrence_unique {α : Type*} (H : α → α) (ψ ψ' : ℕ → α)
    (hψ : ∀ t, ψ t.succ = H (ψ t)) (hψ' : ∀ t, ψ' t.succ = H (ψ' t))
    (h0 : ψ 0 = ψ' 0) : ψ = ψ' := by
  funext n
  induction n with
  | zero => exact h0
  | succ k ih => rw [hψ k, hψ' k, ih]

/-- The iterate trajectory solves the discrete recurrence: the successor point
    of `fun k => H^[k] x` is obtained from the current one by `H`. -/
theorem discrete_recurrence_iterate {α : Type*} (H : α → α) (x : α) :
    ∀ n : ℕ,
      (fun k : ℕ => (H^[k]) x) n.succ = H ((fun k : ℕ => (H^[k]) x) n) := by
  intro n
  show H^[n.succ] x = H (H^[n] x)
  rw [Function.iterate_succ_apply']

/-- Forward flux across a pair: the model's chain-overlap density. -/
noncomputable def forwardFlux (R₁ R₂ : QRegion) : ℝ := Φ R₁ R₂

/-- Reverse flux across an ordered pair: the entropy-normalized deficit
    `(S₂ - I)/S₂`.  With the genuine entropy `S₂ > 0` and mutual information
    `I` this is a real ratio (no zero-denominator branch), and on a region
    paired with itself it is `1` (`reverseFlux_self`) and otherwise strictly
    below `1` (`witness_reverseFlux_eq`). -/
noncomputable def reverseFlux (R₁ R₂ : QRegion) : ℝ :=
  (vonNeumannEntropy R₂ - mutualInformation R₁ R₂) / vonNeumannEntropy R₂

theorem reverseFlux_self (R : QRegion) : reverseFlux R R = 1 := by
  rw [reverseFlux, mutualInformation_self, sub_zero,
    div_self (ne_of_gt (vonNeumannEntropy_pos R))]

theorem witness_reverseFlux_eq : reverseFlux halfQRegion unitQRegion = 13 / 16 := by
  rw [reverseFlux, witness_mutualInformation_eq]
  have h : vonNeumannEntropy unitQRegion = 1 := rfl
  rw [h]
  norm_num

/-- Asymmetry of a pair: the forward flux minus the reverse flux of the
    reversed pair. -/
noncomputable def asymmetryTensor (R₁ R₂ : QRegion) : ℝ :=
  forwardFlux R₁ R₂ - reverseFlux R₂ R₁

/-- Impedance: the absolute asymmetry of a pair. -/
noncomputable def informationalImpedance (R₁ R₂ : QRegion) : ℝ :=
  |asymmetryTensor R₁ R₂|

def freezeBoundaryThreshold : ℝ := 0.012

def isFrozen (R₁ R₂ : QRegion) : Prop :=
  forwardFlux R₁ R₂ ≤ freezeBoundaryThreshold

/-- Consistency bridge (model core): the von Neumann entropy of a region is
    nonnegative — genuinely, from the region's entropy field (the former
    `QRegion := Unit` model had it identically zero). -/
theorem bridge_vonNeumannEntropy_nonneg :
  ∀ (R : QRegion), vonNeumannEntropy R ≥ 0 := by
  intro R
  exact le_of_lt (vonNeumannEntropy_pos R)

theorem bridge_vonNeumannEntropy_le_pi :
  ∀ (R : QRegion), vonNeumannEntropy R ≤ Real.pi := by
  intro R
  have h := vonNeumannEntropy_le_one R
  linarith [Real.pi_gt_three]

theorem bridge_mutualInformation_nonneg :
  ∀ (R₁ R₂ : QRegion), mutualInformation R₁ R₂ ≥ 0 := by
  intro R₁ R₂
  exact mutualInformation_nonneg R₁ R₂

-- RETIRED: `data_processing_inequality` and
-- `data_processing_inequality_multiplicative`. On the former zero model both
-- sides of each inequality were identically zero, so the names asserted a
-- data-processing theorem while proving `0 ≤ 0`. A data-processing inequality
-- needs a channel model; the honest, hypothesis-carrying ingredient used by
-- the metric layer is `InformationNetwork.log_inequality_of_multiplicative_DPI`
-- below. The remaining `bridge_*` declarations in this section are
-- model-level (as opposed to physical-law) statements about the concrete
-- region model; after the eleventh pass they are genuine theorems about a
-- non-degenerate model rather than consistency witnesses of a zero model.

-- The legacy vacuous-hypothesis bridges `log_ratio_nonpos`,
-- `log_inequality_from_DPI` and `perfect_overlap_identifies_regions`
-- (hypotheses `mutualInformation > 0` / `Φ = 1`, unsatisfiable in this
-- zero model) were RETIRED. Their genuine, non-degenerate restatements
-- live in the `InformationNetwork` namespace below:
-- `InformationNetwork.log_ratio_nonpos`,
-- `InformationNetwork.log_inequality_of_multiplicative_DPI`,
-- `InformationNetwork.overlap_one_identifies`.

/-- Definitional unfolding of the freeze predicate at the model's chosen
    threshold: `isFrozen` *is* `forwardFlux ≤ freezeBoundaryThreshold`, and
    `freezeBoundaryThreshold := 0.012`, so this `↔` is `Iff.rfl`.  Stated once
    here instead of proliferating as grand-named restatements (the retired
    `freeze_boundary_theorem`, `consciousness_from_omega` and
    `freeze_implies_low_integration` were all this same definitional
    content). -/
theorem bridge_isFrozen_iff (R₁ R₂ : QRegion) :
  isFrozen R₁ R₂ ↔ forwardFlux R₁ R₂ ≤ (0.012 : ℝ) :=
  Iff.rfl

/-- The impedance is positive exactly when the asymmetry tensor is nonzero —
    an identity about the absolute-value definition. The retired
    `asymmetry_generates_stress_energy` stated one direction under a
    hypothesis that is unsatisfiable in the zero model (`asymmetryTensor ≡ 0`),
    so it could never fire. -/
theorem informationalImpedance_pos_iff (R₁ R₂ : QRegion) :
    0 < informationalImpedance R₁ R₂ ↔ asymmetryTensor R₁ R₂ ≠ 0 := by
  dsimp [informationalImpedance]
  exact abs_pos

/-- Vanishing reverse flux is what makes the asymmetry tensor equal the
    forward flux. The retired `phase_transition_at_freeze` used a frozen-phase
    hypothesis that its proof never consumed; here the hypothesis is the one
    actually needed. -/
theorem asymmetryTensor_eq_forward_of_reverseFlux_zero (R₁ R₂ : QRegion)
    (h : reverseFlux R₂ R₁ = 0) : asymmetryTensor R₁ R₂ = forwardFlux R₁ R₂ := by
  dsimp [asymmetryTensor]
  rw [h, sub_zero]

/-!
### Non-degenerate Information Network & Witness

The minimal zero-information model above ensures internal consistency,
but cannot realize positive mutual information, non-zero asymmetry,
or genuine metric separation. Below we formalize the abstract class
of non-degenerate information networks and construct an explicit
two-region witness.
-/

/-- Abstract specification of an information network over regions.
    Encapsulates mutual information, joint entropy, and bounds. -/
structure InformationNetwork (α : Type*) where
  mutualInfo : α → α → ℝ
  jointEnt   : α → α → ℝ
  maxMI      : ℝ
  mi_symm    : ∀ A B, mutualInfo A B = mutualInfo B A
  mi_nonneg  : ∀ A B, 0 ≤ mutualInfo A B
  mi_le_max  : ∀ A B, mutualInfo A B ≤ maxMI
  je_symm    : ∀ A B, jointEnt A B = jointEnt B A
  je_pos     : ∀ A B, 0 < jointEnt A B
  mi_le_je   : ∀ A B, mutualInfo A B ≤ jointEnt A B

namespace InformationNetwork

variable {α : Type*} (net : InformationNetwork α)

/-- Chain overlap density Φ(A, B) = I(A : B) / S(A, B) in the general model. -/
noncomputable def overlap (A B : α) : ℝ :=
  net.mutualInfo A B / net.jointEnt A B

theorem overlap_nonneg (A B : α) : 0 ≤ net.overlap A B :=
  div_nonneg (net.mi_nonneg A B) (le_of_lt (net.je_pos A B))

theorem overlap_le_one (A B : α) : net.overlap A B ≤ 1 :=
  div_le_one_of_le₀ (net.mi_le_je A B) (le_of_lt (net.je_pos A B))

theorem overlap_symm (A B : α) : net.overlap A B = net.overlap B A := by
  dsimp [overlap]
  rw [net.mi_symm, net.je_symm]

/-- Omega metric induced by an InformationNetwork with a given Planck scale factor. -/
noncomputable def metric (planckScale : ℝ → ℝ) (A B : α) : ℝ :=
  let I := net.mutualInfo A B
  let phi := net.overlap A B
  if I = 0 then 0
  else -planckScale phi * Real.log (I / net.maxMI)

theorem metric_nonneg_of_le_max (planckScale : ℝ → ℝ)
    (h_scale_nonneg : ∀ x, 0 ≤ planckScale x)
    (h_max_pos : 0 < net.maxMI) (A B : α) :
    0 ≤ net.metric planckScale A B := by
  dsimp [metric]
  split_ifs with hI
  · exact le_rfl
  · have hI_pos : 0 < net.mutualInfo A B :=
      lt_of_le_of_ne (net.mi_nonneg A B) (Ne.symm hI)
    have h_ratio_pos : 0 < net.mutualInfo A B / net.maxMI :=
      div_pos hI_pos h_max_pos
    have h_ratio_le : net.mutualInfo A B / net.maxMI ≤ 1 :=
      div_le_one_of_le₀ (net.mi_le_max A B) (le_of_lt h_max_pos)
    have h_log_nonpos : Real.log (net.mutualInfo A B / net.maxMI) ≤ 0 := by
      have h_le := Real.log_le_log h_ratio_pos h_ratio_le
      rw [Real.log_one] at h_le
      exact h_le
    have h_neg_log_nonneg : 0 ≤ -Real.log (net.mutualInfo A B / net.maxMI) :=
      neg_nonneg.mpr h_log_nonpos
    have h_scale := h_scale_nonneg (net.overlap A B)
    calc 0
      _ ≤ planckScale (net.overlap A B) * (-Real.log (net.mutualInfo A B / net.maxMI)) :=
          mul_nonneg h_scale h_neg_log_nonneg
      _ = -planckScale (net.overlap A B) * Real.log (net.mutualInfo A B / net.maxMI) := by ring

/-- Non-degenerate restatement of the legacy `log_ratio_nonpos` bridge:
    for a network with positive capacity, any positive mutual information
    sits at or below capacity, so its normalized log is nonpositive. Unlike
    the retired zero-model version, the hypothesis `0 < mutualInfo A B` is
    realizable (e.g. by `BinaryRegion.binaryNetwork`). -/
theorem log_ratio_nonpos (h_max : 0 < net.maxMI) {A B : α}
    (h : 0 < net.mutualInfo A B) :
    Real.log (net.mutualInfo A B / net.maxMI) ≤ 0 := by
  have h_ratio_pos : 0 < net.mutualInfo A B / net.maxMI :=
    div_pos h h_max
  have h_ratio_le : net.mutualInfo A B / net.maxMI ≤ 1 :=
    div_le_one_of_le₀ (net.mi_le_max A B) (le_of_lt h_max)
  have h_le := Real.log_le_log h_ratio_pos h_ratio_le
  rwa [Real.log_one] at h_le

/-- Non-degenerate restatement of the legacy `log_inequality_from_DPI`
    bridge: the log-triangle ingredient behind the metric triangle
    inequality. If a multiplicative data-processing bound holds in
    normalized form,
    `I(A:C)/maxMI ≤ (I(A:B)/maxMI) · (I(B:C)/maxMI)`,
    with all three informations positive and positive capacity, then the
    normalized log of `I(A:C)` is at most the sum of the normalized logs.
    Every hypothesis is realizable in a non-degenerate network; in the
    retired zero-model version the positivity hypotheses were
    unsatisfiable. -/
theorem log_inequality_of_multiplicative_DPI (h_max : 0 < net.maxMI)
    (A B C : α) (hA : 0 < net.mutualInfo A C) (hB : 0 < net.mutualInfo A B)
    (hC : 0 < net.mutualInfo B C)
    (h : net.mutualInfo A C / net.maxMI ≤
        (net.mutualInfo A B / net.maxMI) * (net.mutualInfo B C / net.maxMI)) :
    Real.log (net.mutualInfo A C / net.maxMI) ≤
      Real.log (net.mutualInfo A B / net.maxMI) +
      Real.log (net.mutualInfo B C / net.maxMI) := by
  have h1 := Real.log_le_log (div_pos hA h_max) h
  rwa [Real.log_mul (ne_of_gt (div_pos hB h_max))
    (ne_of_gt (div_pos hC h_max))] at h1

/-- Non-degenerate restatement of the legacy
    `perfect_overlap_identifies_regions` bridge: if a network separates
    distinct regions strictly (`mutualInfo < jointEnt` whenever `X ≠ Y`),
    then unit chain-overlap density forces `A = B`. The separation
    hypothesis is an explicit non-degeneracy assumption, satisfiable e.g.
    by any network whose distinct regions share less information than
    their joint entropy; in the retired zero-model version the hypothesis
    `Φ = 1` was unsatisfiable. -/
theorem overlap_one_identifies
    (h_sep : ∀ X Y : α, X ≠ Y → net.mutualInfo X Y < net.jointEnt X Y)
    {A B : α} (h : net.overlap A B = 1) : A = B := by
  by_contra hne
  have hlt : net.mutualInfo A B < net.jointEnt A B := h_sep A B hne
  have hdiv : net.mutualInfo A B / net.jointEnt A B < 1 := by
    rw [div_lt_iff₀ (net.je_pos A B)]
    linarith
  dsimp [InformationNetwork.overlap] at h
  linarith

end InformationNetwork

/-- Two-region discrete label type to witness non-degeneracy. -/
inductive BinaryRegion : Type
  | alpha : BinaryRegion
  | beta  : BinaryRegion
  deriving DecidableEq

namespace BinaryRegion

/-- An explicit non-degenerate information witness on two regions where
    mutual information and joint entropy are strictly positive,
    and distinct regions have positive separation. -/
noncomputable def binaryMI : BinaryRegion → BinaryRegion → ℝ
  | BinaryRegion.alpha, BinaryRegion.alpha => 1
  | BinaryRegion.beta,  BinaryRegion.beta  => 1
  | _,                  _                  => 1 / 2

noncomputable def binaryNetwork : InformationNetwork BinaryRegion where
  mutualInfo := binaryMI
  jointEnt   := fun _ _ => 2
  maxMI      := 1
  mi_symm    := by
    intro A B
    cases A <;> cases B <;> rfl
  mi_nonneg  := by
    intro A B
    cases A <;> cases B <;> norm_num [binaryMI]
  mi_le_max  := by
    intro A B
    cases A <;> cases B <;> norm_num [binaryMI]
  je_symm    := by intros; rfl
  je_pos     := by intros; norm_num
  mi_le_je   := by
    intro A B
    cases A <;> cases B <;> norm_num [binaryMI]

theorem binary_overlap_self (A : BinaryRegion) :
    binaryNetwork.overlap A A = 1 / 2 := by
  dsimp [InformationNetwork.overlap, binaryNetwork, binaryMI]
  cases A <;> norm_num

theorem binary_overlap_cross :
    binaryNetwork.overlap BinaryRegion.alpha BinaryRegion.beta = 1 / 4 := by
  dsimp [InformationNetwork.overlap, binaryNetwork, binaryMI]
  norm_num

theorem binary_mi_pos (A B : BinaryRegion) :
    0 < binaryNetwork.mutualInfo A B := by
  cases A <;> cases B <;> norm_num [binaryNetwork, binaryMI]

theorem binary_distinct_regions :
    BinaryRegion.alpha ≠ BinaryRegion.beta := by
  intro h
  cases h

end BinaryRegion

end OmegaProtocol
