import Mathlib.Analysis.VonNeumannAlgebra.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Topology.Instances.Complex
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import DynamicCODScale

namespace OmegaProtocol

/-!
# Concrete Omega model

The original version of this file declared every symbol and every desired
property as an `unchecked declaration`.  That made downstream statements look proved even
when the kernel had been handed the result.  This module now supplies a
small, explicit zero-information model instead:

* `StateSpace` is `ℂ` and `QRegion` is `Unit`;
* all informational observables are zero;
* the operator and entropy constructions are real definitions;
* the metric's Planck-scale factor is the canonical COD profile
  `ℓ_P(Φ) = ℓ_P0·√(1-Φ²)` from `DynamicCODScale` (evaluated at `ℓ_P0 = 1`),
  shared with every other file that needs a Planck scale; and
* the properties that follow from this model are proved below.

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

/-- The minimal model has a zero functional calculus. -/
noncomputable def op_pow (_ : Operator) (_ : ℂ) : Operator := 0
@[default_instance] noncomputable instance instHPowOperator : HPow Operator ℂ Operator where
  hPow := op_pow

@[default_instance] noncomputable instance instCoeOmegaAlgebraOperator :
    Coe ↥OmegaAlgebra Operator where
  coe A := A.1

@[default_instance] noncomputable instance instMulOperator : Mul Operator where
  mul A B := A.comp B

def CyclicSeparating (_ : StateSpace) : Prop := True
def ω_ρ (_ : ↥OmegaAlgebra) : ℂ := 0

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

/-- A Q-Region is a single point in the minimal model. -/
def QRegion : Type := Unit
def vonNeumannEntropy (_ : QRegion) : ℝ := 0
def jointEntropy (_ _ : QRegion) : ℝ := 0
def mutualInformation (_ _ : QRegion) : ℝ := 0

noncomputable def chainOverlapDensity (R₁ R₂ : QRegion) : ℝ :=
  if jointEntropy R₁ R₂ = 0 then 0
  else mutualInformation R₁ R₂ / jointEntropy R₁ R₂

noncomputable def Φ (R₁ R₂ : QRegion) : ℝ :=
  chainOverlapDensity R₁ R₂

noncomputable def maxMutualInformation : ℝ := 1

/-- Unit-base-scale COD environment: the zero-model evaluates the canonical
    `DynamicCODScale` profile at `ℓ_P0 = 1`, so the metric's Planck factor is
    `√(1 - Φ²)` — the same profile used everywhere else in the project.
    (A previous revision used a bespoke `√(πΦ+1)` here; see C3 in
    ADVERSARIAL_AUDIT.md.) -/
noncomputable def canonicalCODEnv : DynamicCODScale.CODEnvironment :=
  ⟨1, by norm_num⟩

noncomputable def omegaMetric (R₁ R₂ : QRegion) : ℝ :=
  let Φ_val := Φ R₁ R₂
  let I := mutualInformation R₁ R₂
  if I = 0 then 0
  else -DynamicCODScale.dynamicPlanckCOD canonicalCODEnv Φ_val * Real.log (I / (1 : ℝ))

noncomputable def d (R₁ R₂ : QRegion) : ℝ := omegaMetric R₁ R₂

noncomputable def informationalViscosity (updateRate : ℝ) : ℝ :=
  if updateRate = 0 then Real.pi else 1 / updateRate

structure DiscreteTime where
  step : ℕ
  state : QRegion

noncomputable def computationalLatency (Δupdates : ℕ) : ℝ :=
  if Δupdates = 0 then 0 else 1 / (Δupdates : ℝ)

noncomputable def informationNovelty (R : QRegion) : ℝ :=
  Real.sin (vonNeumannEntropy R)

noncomputable def processingSpeed (R : QRegion) (baseRate : ℝ) : ℝ :=
  baseRate * (1 - informationNovelty R)

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

noncomputable def forwardFlux (R₁ R₂ : QRegion) : ℝ := Φ R₁ R₂

noncomputable def reverseFlux (R₁ R₂ : QRegion) : ℝ :=
  let I := mutualInformation R₁ R₂
  let S₂ := vonNeumannEntropy R₂
  if S₂ = 0 then 0 else (S₂ - I) / S₂

noncomputable def asymmetryTensor (R₁ R₂ : QRegion) : ℝ :=
  forwardFlux R₁ R₂ - reverseFlux R₂ R₁

noncomputable def informationalImpedance (R₁ R₂ : QRegion) : ℝ :=
  |asymmetryTensor R₁ R₂|

def freezeBoundaryThreshold : ℝ := 0.012

def isFrozen (R₁ R₂ : QRegion) : Prop :=
  forwardFlux R₁ R₂ ≤ freezeBoundaryThreshold

-- RETIRED: `data_processing_inequality` and
-- `data_processing_inequality_multiplicative`. On the zero model both sides
-- of each inequality are identically zero, so the names asserted a
-- data-processing theorem while proving `0 ≤ 0`. A data-processing
-- inequality needs a channel model; the honest, hypothesis-carrying
-- ingredient used by the metric layer is
-- `InformationNetwork.log_inequality_of_multiplicative_DPI` below.
-- The remaining `bridge_*` declarations in this section are consistency
-- witnesses of the degenerate model (every information quantity is zero),
-- named so that no physical law is attributed to them. Their non-degenerate
-- counterparts live in `InformationNetwork`.

theorem bridge_vonNeumannEntropy_nonneg :
  ∀ (R : QRegion), vonNeumannEntropy R ≥ 0 := by
  intro
  exact le_rfl

theorem bridge_vonNeumannEntropy_le_pi :
  ∀ (R : QRegion), vonNeumannEntropy R ≤ Real.pi := by
  intro
  have : (0 : ℝ) ≤ Real.pi := le_of_lt Real.pi_pos
  simpa [vonNeumannEntropy] using this

theorem bridge_mutualInformation_nonneg :
  ∀ (R₁ R₂ : QRegion), mutualInformation R₁ R₂ ≥ 0 := by
  intros
  exact le_rfl

theorem bridge_mutualInformation_le_max :
  ∀ (R₁ R₂ : QRegion), mutualInformation R₁ R₂ ≤ maxMutualInformation := by
  intros
  simp [mutualInformation, maxMutualInformation]

-- The legacy vacuous-hypothesis bridges `log_ratio_nonpos`,
-- `log_inequality_from_DPI` and `perfect_overlap_identifies_regions`
-- (hypotheses `mutualInformation > 0` / `Φ = 1`, unsatisfiable in this
-- zero model) were RETIRED. Their genuine, non-degenerate restatements
-- live in the `InformationNetwork` namespace below:
-- `InformationNetwork.log_ratio_nonpos`,
-- `InformationNetwork.log_inequality_of_multiplicative_DPI`,
-- `InformationNetwork.overlap_one_identifies`.

theorem bridge_overlapDensity_nonneg :
  ∀ (R₁ R₂ : QRegion), Φ R₁ R₂ ≥ 0 := by
  intros
  simp [Φ, chainOverlapDensity, jointEntropy]

theorem bridge_overlapDensity_symm :
  ∀ (R₁ R₂ : QRegion), Φ R₁ R₂ = Φ R₂ R₁ := by
  intros
  simp [Φ, chainOverlapDensity, jointEntropy]

theorem bridge_qregion_self_distance_zero :
  ∀ (R : QRegion), d R R = 0 := by
  intro
  simp [d, omegaMetric, mutualInformation]

theorem bridge_codProfile_triangle :
  ∀ (R₁ R₂ R₃ : QRegion),
    DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₁ R₃) ≤
      DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₁ R₂) +
      DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₂ R₃) := by
  intro R₁ R₂ R₃
  have hΦ : ∀ (A B : QRegion), Φ A B = 0 := fun A B => by
    norm_num [Φ, chainOverlapDensity, jointEntropy]
  rw [hΦ, hΦ, hΦ, DynamicCODScale.dynamicPlanckCOD_zero]
  norm_num [canonicalCODEnv]

theorem bridge_distance_zero_when_I_zero :
  ∀ (R₁ R₂ : QRegion),
    mutualInformation R₁ R₂ = 0 → d R₁ R₂ = 0 := by
  intros
  simp [d, omegaMetric, mutualInformation]

theorem bridge_distance_nonneg :
  ∀ (R₁ R₂ : QRegion), d R₁ R₂ ≥ 0 := by
  intros
  simp [d, omegaMetric, mutualInformation]

theorem bridge_codProfile_pos :
  ∀ (R₁ R₂ : QRegion),
    0 < DynamicCODScale.dynamicPlanckCOD canonicalCODEnv (Φ R₁ R₂) := by
  intro R₁ R₂
  have hΦ : Φ R₁ R₂ = 0 := by
    norm_num [Φ, chainOverlapDensity, jointEntropy]
  rw [hΦ, DynamicCODScale.dynamicPlanckCOD_zero]
  norm_num [canonicalCODEnv]

theorem bridge_zero_product_implies_zero :
  ∀ (R₁ R₂ R₃ : QRegion),
    mutualInformation R₁ R₂ = 0 → mutualInformation R₂ R₃ = 0 →
    mutualInformation R₁ R₃ = 0 := by
  intros
  rfl

theorem bridge_distance_triangle_inequality :
  ∀ (R₁ R₂ R₃ : QRegion),
    d R₁ R₃ ≤ d R₁ R₂ + d R₂ R₃ := by
  intros
  simp [d, omegaMetric, mutualInformation]

/-- Zero-model law: the reverse flux vanishes identically (both entropies
    are zero), so the retired hypothesis `isFrozen R₁ R₂` in
    `freeze_boundary_reverse_flux_vanishes` was decorative. -/
theorem bridge_reverseFlux_zero_zero_model :
  ∀ (R₁ R₂ : QRegion), reverseFlux R₁ R₂ = 0 := by
  intros
  simp [reverseFlux, vonNeumannEntropy]

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

/-- Consistency bridge (protocol core): the Omega-metric `d` satisfies the triangle inequality.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `triangle_inequality_from_DPI` evoked. -/
theorem bridge_metric_triangle_from_DPI (R₁ R₂ R₃ : QRegion) :
  d R₁ R₃ ≤ d R₁ R₂ + d R₂ R₃ :=
  bridge_distance_triangle_inequality R₁ R₂ R₃

/-- Arithmetic core of the "time dilation as processing lag" reading: a rate
    reduced by a novelty factor in `[0, 1]` never exceeds the base rate. This
    is the non-degenerate statement (all three hypotheses are used); the zero
    model cannot witness an actual lag, since its novelty is identically zero. -/
theorem speed_reduction_of_novelty (baseRate novelty : ℝ)
    (h_base : 0 ≤ baseRate) (h_novelty0 : 0 ≤ novelty) (h_novelty1 : novelty ≤ 1) :
    baseRate * (1 - novelty) ≤ baseRate := by
  calc baseRate * (1 - novelty) = baseRate - baseRate * novelty := by ring
    _ ≤ baseRate := by
      have hprod : 0 ≤ baseRate * novelty := mul_nonneg h_base h_novelty0
      linarith

/-- Zero-model identity (no physics content): the model's novelty factor is
    identically zero, so its processing speed equals the base rate. -/
theorem bridge_processingSpeed_zero_model (R : QRegion) (baseRate : ℝ) :
    processingSpeed R baseRate = baseRate := by
  simp [processingSpeed, informationNovelty, vonNeumannEntropy]

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
