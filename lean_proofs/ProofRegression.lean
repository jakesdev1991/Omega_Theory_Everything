import LogCorrelationMetric
import RadialMetric
import CBwK_Budget_Pacer
import APPA_Context_Branching
import Vol12_QuantumInformation
import Vol25_BlackHoleInformation
import Vol49_QuantumReferenceFrames
import DynamicPlanckScale
import Vol16_NonEquilibriumThermodynamics
import Vol17_FluidDynamics
import Vol21_ArrowOfTime
import Vol18_StatisticalMechanics
import Vol28_EvolutionaryAlgorithms
import Vol10_StandardModel
import Vol11_CondensedMatter
import Vol26_NetworkTheory
import Vol39_SocietalNetworks
import Vol30_PlanetarySystems
import Vol31_GameTheory
import Vol32_Cybernetics
import Vol38_Economics
import Vol53_UniversalCompiler

/-! Boundary and non-degeneracy regression checks. Included in `lake build ToE`. -/

example : LogCorrelationMetric.logDistance 0 1 ≠ 0 := by
  rw [LogCorrelationMetric.distinct_points_distance]
  norm_num

example : CBwK.runCosts 10 [6, 6] = none := by decide
example : CBwK.runCosts 10 [6, 4] = some 0 := by decide
example : CBwK.runCosts 7 [] = some 7 := rfl
example : CBwK.runCosts 0 [1] = none := by decide

example : APPA.try_declassify ⟨"payload", true, false⟩ = none := rfl
example : APPA.try_declassify ⟨"payload", false, true⟩ = none := rfl
example : APPA.try_declassify ⟨"payload", false, false⟩ = none := rfl
example : APPA.try_declassify ⟨"payload", true, true⟩ =
    some ⟨"payload", true, true⟩ := rfl

example : OmegaProtocol.Vol18.ProbState1 0 3 7 = 1 / 2 := by
  norm_num [OmegaProtocol.Vol18.ProbState1, OmegaProtocol.Vol18.PartitionFunction2,
    OmegaProtocol.Vol18.BoltzmannWeight]

example : OmegaProtocol.Vol28.fitness
    (OmegaProtocol.Vol28.selectBest [true, true] [false]) = 2 := by decide

/-- A varying scale can still factor at a chosen reference scale. -/
example : RadialMetric.chainLength (fun i => if i = 0 then 1 else 3) (fun _ => 1) 2 =
    2 * ∑ i ∈ Finset.range 2, (1 : ℝ) := by
  norm_num [RadialMetric.chainLength, Finset.sum_range_succ]

example : ¬ ∃ p : OmegaProtocol.Vol53.Prog,
    ∀ x, OmegaProtocol.Vol53.eval p x = x :=
  OmegaProtocol.Vol53.identity_not_representable

/-! Non-degenerate quantum/entropy witnesses added in the second audit pass. -/

example : OmegaProtocol.Vol12.IsNormalized OmegaProtocol.Vol12.tiltedState :=
  OmegaProtocol.Vol12.tiltedState_normalized

example : ¬ (∀ ψ φ : OmegaProtocol.Vol12.InfoState,
    OmegaProtocol.Vol12.IsNormalized ψ → OmegaProtocol.Vol12.IsNormalized φ →
    OmegaProtocol.Vol12.state_inner ψ φ =
      OmegaProtocol.Vol12.state_inner ψ φ * OmegaProtocol.Vol12.state_inner ψ φ) :=
  OmegaProtocol.Vol12.no_universal_overlap_copy

example : OmegaProtocol.Vol25.VonNeumannEntropy
    OmegaProtocol.Vol25.positiveMarginalWitness +
    OmegaProtocol.Vol25.RadiationEntropy OmegaProtocol.Vol25.positiveMarginalWitness = 2 := by
  norm_num [OmegaProtocol.Vol25.VonNeumannEntropy, OmegaProtocol.Vol25.RadiationEntropy,
    OmegaProtocol.Vol25.positiveMarginalWitness]

example : OmegaProtocol.Vol49.FrameTransform
    OmegaProtocol.Vol49.positiveFrame OmegaProtocol.Vol49.negativeFrame
      OmegaProtocol.Vol49.unitState ≠ OmegaProtocol.Vol49.unitState := by
  rw [OmegaProtocol.Vol49.frame_change_nontrivial]
  change (-1 : ℂ) ≠ 1
  norm_num

example (A B : OmegaProtocol.Vol49.ReferenceFrame) (ψ : OmegaProtocol.StateSpace) :
    OmegaProtocol.Vol49.FrameTransform B A
      (OmegaProtocol.Vol49.FrameTransform A B ψ) = ψ :=
  OmegaProtocol.Vol49.frame_round_trip A B ψ

example (env : DynamicCODScale.CODEnvironment) :
    ¬ Function.Injective (DynamicCODScale.dynamicPlanckCOD env) :=
  DynamicCODScale.dynamicPlanckCOD_not_globally_injective env

example (env : DynamicGeometry.ScaleEnvironment) :
    ¬ DynamicGeometry.dynamicPlanckLength env (-1) < env.lP0 := by
  rw [DynamicGeometry.planck_scale_contraction_iff]
  norm_num

example (env : InformationPhysics.Environment) :
    InformationPhysics.informationMass env 2 = 2 * InformationPhysics.informationMass env 1 :=
  InformationPhysics.informationMass_linear env 2

/-! Positive-parameter orbital/market witnesses and sharp feedback boundaries. -/

namespace StrengtheningRegression

def unitKeplerSystem : OmegaProtocol.Vol30.KeplerSystem :=
  ⟨1, 1, by norm_num, by norm_num⟩

example : 0 < OmegaProtocol.Vol30.OrbitalPeriod unitKeplerSystem 1 :=
  OmegaProtocol.Vol30.orbital_period_pos _ _ (by norm_num)

example (sys : OmegaProtocol.Vol30.KeplerSystem) : sys.centralMass ≠ 0 :=
  ne_of_gt sys.centralMass_pos

example (sys : OmegaProtocol.Vol30.KeplerSystem) :
    OmegaProtocol.Vol30.OrbitalPeriod sys 0 = 0 :=
  OmegaProtocol.Vol30.orbital_period_zero sys

/-- Including zero radius in the ratio law would be wrong even with positive mass. -/
example (sys : OmegaProtocol.Vol30.KeplerSystem) :
    OmegaProtocol.Vol30.OrbitalPeriod sys 0 ^ 2 / (0 : ℝ) ^ 3 ≠
      OmegaProtocol.Vol30.OrbitalPeriod sys 1 ^ 2 / (1 : ℝ) ^ 3 := by
  have hzero : OmegaProtocol.Vol30.OrbitalPeriod sys 0 ^ 2 / (0 : ℝ) ^ 3 = 0 := by
    norm_num
  have hpos : 0 < OmegaProtocol.Vol30.OrbitalPeriod sys 1 ^ 2 / (1 : ℝ) ^ 3 := by
    rw [OmegaProtocol.Vol30.kepler_third_law sys 1 (by norm_num)]
    simpa using OmegaProtocol.Vol30.keplerCoefficient_pos sys
  rw [hzero]
  exact ne_of_lt hpos

def sampleMarket : OmegaProtocol.Vol38.LinearMarket :=
  ⟨12, 2, 1, by norm_num, by norm_num, by norm_num⟩

example : OmegaProtocol.Vol38.equilibrium_price sampleMarket = 4 := by
  norm_num [OmegaProtocol.Vol38.equilibrium_price, sampleMarket]

example : OmegaProtocol.Vol38.Supply sampleMarket 4 = 8 ∧
    OmegaProtocol.Vol38.Demand sampleMarket 4 = 8 := by
  norm_num [OmegaProtocol.Vol38.Supply, OmegaProtocol.Vol38.Demand, sampleMarket]

example : OmegaProtocol.Vol38.ExcessDemand sampleMarket 3 = 3 ∧
    OmegaProtocol.Vol38.ExcessDemand sampleMarket 5 = -3 := by
  norm_num [OmegaProtocol.Vol38.ExcessDemand, OmegaProtocol.Vol38.Supply,
    OmegaProtocol.Vol38.Demand, sampleMarket]

example : ¬ OmegaProtocol.Vol31.IsNash .Cooperate .Cooperate := by
  rw [OmegaProtocol.Vol31.nash_iff_mutual_defection]
  simp

example : OmegaProtocol.Vol31.IsNash .Defect .Defect :=
  (OmegaProtocol.Vol31.nash_iff_mutual_defection _ _).mpr ⟨rfl, rfl⟩

example (initial : ℝ) : OmegaProtocol.Vol32.feedbackState 0 initial 0 = initial := by
  simp [OmegaProtocol.Vol32.feedbackState]

example (initial : ℝ) (n : ℕ) : OmegaProtocol.Vol32.feedbackState 0 initial (n + 1) = 0 := by
  simp [OmegaProtocol.Vol32.feedbackState]

example : Filter.Tendsto (OmegaProtocol.Vol32.feedbackState (1 / 2) 2)
    Filter.atTop (nhds 0) :=
  OmegaProtocol.Vol32.feedback_tends_to_zero _ _ (by norm_num) (by norm_num)

example : ¬ Filter.Tendsto (OmegaProtocol.Vol32.feedbackState 1 1)
    Filter.atTop (nhds 0) :=
  OmegaProtocol.Vol32.unit_gain_not_decay _ (by norm_num)

end StrengtheningRegression

/-! Independently counted graph quantities and sharp potential-minimum cases. -/

example : OmegaProtocol.Vol26.DegreeSum OmegaProtocol.Vol26.twoVertexEdge = 2 ∧
    OmegaProtocol.Vol26.NumEdges OmegaProtocol.Vol26.twoVertexEdge = 1 :=
  OmegaProtocol.Vol26.twoVertexEdge_counts

example (n : ℕ) : OmegaProtocol.Vol26.NumEdges (OmegaProtocol.Vol26.emptyGraph n) = 0 :=
  OmegaProtocol.Vol26.emptyGraph_numEdges n

example : OmegaProtocol.Vol39.NoIsolatedVertices (OmegaProtocol.Vol26.emptyGraph 0) := by
  intro v
  exact Fin.elim0 v

example : ¬ OmegaProtocol.Vol39.NoIsolatedVertices (OmegaProtocol.Vol26.emptyGraph 1) :=
  OmegaProtocol.Vol39.singleton_has_isolated_vertex

example : OmegaProtocol.Vol39.NoIsolatedVertices OmegaProtocol.Vol26.twoVertexEdge :=
  OmegaProtocol.Vol39.twoVertexEdge_no_isolated

example : ¬ OmegaProtocol.Vol39.NumNodes (OmegaProtocol.Vol26.emptyGraph 1) ≤
    2 * OmegaProtocol.Vol39.NumConnections (OmegaProtocol.Vol26.emptyGraph 1) :=
  OmegaProtocol.Vol39.unconditioned_bound_fails

example : OmegaProtocol.Vol10.HiggsPotential (-1) = 0 ∧
    (-1 : ℝ) ≠ OmegaProtocol.Vol10.VacuumExpectationValue := by
  norm_num [OmegaProtocol.Vol10.HiggsPotential, OmegaProtocol.Vol10.VacuumExpectationValue]

example : ∀ ψ : ℝ, OmegaProtocol.Vol10.HiggsPotential (-1) ≤
    OmegaProtocol.Vol10.HiggsPotential ψ := by
  apply (OmegaProtocol.Vol10.higgs_global_minimizers (-1)).mpr
  right
  rfl

example : OmegaProtocol.Vol11.LandauFreeEnergy 1 1 0 2 1 = -1 ∧
    OmegaProtocol.Vol11.LandauFreeEnergy 1 1 0 2 (-1) = -1 := by
  norm_num [OmegaProtocol.Vol11.LandauFreeEnergy]

example : ∀ x : ℝ, OmegaProtocol.Vol11.LandauFreeEnergy 1 1 0 2 1 ≤
    OmegaProtocol.Vol11.LandauFreeEnergy 1 1 0 2 x := by
  apply (OmegaProtocol.Vol11.landau_minimizer_iff 1 1 0 2 1 (by norm_num) (by norm_num)).mpr
  norm_num

example : ∀ x : ℝ, OmegaProtocol.Vol11.LandauFreeEnergy 1 1 0 2 (-1) ≤
    OmegaProtocol.Vol11.LandauFreeEnergy 1 1 0 2 x := by
  apply (OmegaProtocol.Vol11.landau_minimizer_iff 1 1 0 2 (-1) (by norm_num) (by norm_num)).mpr
  norm_num

example : OmegaProtocol.Vol11.LandauFreeEnergy 1 0 2 2 3 = 0 ∧ (3 : ℝ) ≠ 0 :=
  ⟨OmegaProtocol.Vol11.landau_flat_at_zero_quartic 1 2 3, by norm_num⟩

/-! Actual derivative models: growth, constancy and flux-driven decrease. -/

example : StrictMono (OmegaProtocol.Vol16.Entropy OmegaProtocol.Vol16.growingProcess) :=
  OmegaProtocol.Vol16.growingProcess_strictMono

example : ¬ Monotone (OmegaProtocol.Vol16.Entropy OmegaProtocol.Vol16.exportingProcess) :=
  OmegaProtocol.Vol16.nonnegative_production_not_enough

example : OmegaProtocol.Vol16.IsIsolated (OmegaProtocol.Vol16.constantProcess 5) :=
  fun _ => rfl

example (s t : ℝ) :
    OmegaProtocol.Vol16.Entropy (OmegaProtocol.Vol16.constantProcess 5) s =
      OmegaProtocol.Vol16.Entropy (OmegaProtocol.Vol16.constantProcess 5) t := rfl

example : OmegaProtocol.Vol21.CosmicEntropy OmegaProtocol.Vol16.growingProcess 0 ≤
    OmegaProtocol.Vol21.CosmicEntropy OmegaProtocol.Vol16.growingProcess 1 :=
  OmegaProtocol.Vol21.arrow_of_time _ OmegaProtocol.Vol16.growingProcess_isolated 0 1 (by norm_num)

example (s t : ℝ) :
    OmegaProtocol.Vol17.FluidDensity (OmegaProtocol.Vol17.constantDensity 3 (by norm_num)) s =
      OmegaProtocol.Vol17.FluidDensity (OmegaProtocol.Vol17.constantDensity 3 (by norm_num)) t :=
  OmegaProtocol.Vol17.incompressible_density_constant _ (fun _ => rfl) s t

example : OmegaProtocol.Vol17.FluidDensity OmegaProtocol.Vol17.expandingFlow 1 <
    OmegaProtocol.Vol17.FluidDensity OmegaProtocol.Vol17.expandingFlow 0 :=
  OmegaProtocol.Vol17.expandingFlow_density_decreases 0 1 (by norm_num)

/-- Outside the positive-density model, a vacuum rate cannot determine divergence. -/
example : deriv (fun _ : ℝ => (0 : ℝ)) 0 + 0 * 1 = 0 ∧ (1 : ℝ) ≠ 0 := by
  simp

/-! Executable aggregate-accounting traces; these do not model ticket ownership. -/

example : ((CBwK.runLedger (CBwK.startLedger 10)
    [.reserve 6, .settle 2 4]).map fun s => (s.available, s.reserved, s.spent)) =
      some (8, 0, 2) := by decide

example : ((CBwK.runLedger (CBwK.startLedger 10)
    [.reserve 6, .settle 2 3]).map fun s => (s.available, s.reserved, s.spent)) =
      some (7, 1, 2) := by decide

/-- The combined charge plus release, not each individually, must fit the reserve. -/
example : (CBwK.runLedger (CBwK.startLedger 10) [.reserve 6, .settle 2 5]).isNone = true := by
  decide

example : (CBwK.runLedger (CBwK.startLedger 10) [.reserve 6, .reserve 6]).isNone = true := by
  decide

example : (CBwK.runLedger (CBwK.startLedger 10)
    [.reserve 6, .settle 0 6, .settle 0 6]).isNone = true := by decide

example : (CBwK.runLedger (CBwK.startLedger 10)
    [.reserve 6, .settle 2 4, .reserve 8, .settle 8 0, .settle 0 1]).isNone = true := by decide

example : ((CBwK.runLedger (CBwK.startLedger 0) [.reserve 0, .settle 0 0]).map
    fun s => (s.available, s.reserved, s.spent)) = some (0, 0, 0) := by decide

example : (CBwK.dimensionOfLedger (CBwK.startLedger 10)).allocated = 0 := rfl

example (state : CBwK.Ledger) :
    (CBwK.dimensionOfLedger state).allocated ≤ (CBwK.dimensionOfLedger state).budget :=
  (CBwK.dimensionOfLedger state).inv

/-! Non-ideal erasure: real excess costs and cumulative feasibility. -/
namespace ErasureRegression
open InformationPhysics

private def env : Environment :=
  ⟨1, 1, 1, by norm_num, by norm_num, by norm_num⟩

/-- Permitted overhead without erased information; not a microscopic protocol. -/
private def overheadOnly : Erasure := ⟨0, 1, by norm_num, by norm_num⟩

example (e : Environment) : erasureCost e (idealErasure 0 le_rfl) = 0 := by
  simp [erasureCost, idealErasure, landauerEnergy]

example (e : Environment) : erasureCost e overheadOnly = 1 := by
  simp [erasureCost, overheadOnly, landauerEnergy]

example (e : Environment) : landauerEnergy e lossyBit.bits < erasureCost e lossyBit :=
  (erasureCost_strict_iff e lossyBit).mpr (by norm_num [lossyBit])

example (e : Environment) :
    ¬ (∀ job : Erasure, erasureCost e job = landauerEnergy e job.bits) :=
  not_every_erasure_is_ideal e

example (e : Environment) : batchCost e [] = 0 := rfl

example (e : Environment) :
    batchCost e [idealErasure 1 (by norm_num), idealErasure 2 (by norm_num)] =
      landauerEnergy e 3 := by
  rw [batchCost_decomposition]
  norm_num [batchBits, batchExcess, idealErasure]

example (e : Environment) :
    batchCost e [idealErasure 1 (by norm_num), overheadOnly] = landauerEnergy e 1 + 1 := by
  rw [batchCost_decomposition]
  norm_num [batchBits, batchExcess, idealErasure, overheadOnly]

example : remainingMass env 1 [overheadOnly] = 0 := by
  norm_num [remainingMass, batchCost, erasureCost, landauerEnergy, overheadOnly, env]

/-- Each identical job fits the initial budget separately, but two do not. -/
example : 0 ≤ remainingMass env 1 [overheadOnly] ∧
    remainingMass env 1 [overheadOnly, overheadOnly] < 0 := by
  norm_num [remainingMass, batchCost, erasureCost, landauerEnergy, overheadOnly, env]

example : remainingMass env 2 [overheadOnly, overheadOnly] = 0 := by
  norm_num [remainingMass, batchCost, erasureCost, landauerEnergy, overheadOnly, env]

example (e : Environment) (mass : ℝ) : remainingMass e mass [] = mass := by
  simp [remainingMass, batchCost]

example (e : Environment) (mass : ℝ) (xs ys : List Erasure) :
    remainingMass e mass (xs ++ ys) = remainingMass e (remainingMass e mass xs) ys :=
  remainingMass_append e mass xs ys

end ErasureRegression
