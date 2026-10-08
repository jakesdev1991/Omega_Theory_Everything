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
import Vol29_EcosystemDynamics
import Vol35_TheoryOfComputation
import Vol36_ToposTheory
import Vol37_Morphogenesis
import Vol44_UniversalExpansion
import Vol48_ExtraDimensions
import Vol54_TheoryOfNothing
import Vol02_Electromagnetism
import Vol03_Thermodynamics
import Vol05_GeneralRelativity
import Vol06_QuantumFieldTheory
import Vol07_Cosmology
import Vol08_BlackHoleThermodynamics
import Vol09_HolographicPrinciple
import Vol13_QuantumGravity
import Vol14_DarkSector
import Vol15_EarlyUniverse
import Vol19_ComplexSystems
import Vol20_ChaosTheory
import Vol22_EREqualsEPR
import Vol23_MeasurementProblem
import Vol24_NonLocality
import Vol27_Consciousness
import Vol33_AGI

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

/-! Non-degenerate Information Network regression examples. -/
namespace InformationNetworkRegression
open OmegaProtocol

example : 0 ≤ BinaryRegion.binaryNetwork.overlap BinaryRegion.alpha BinaryRegion.beta :=
  BinaryRegion.binaryNetwork.overlap_nonneg _ _

example : BinaryRegion.binaryNetwork.overlap BinaryRegion.alpha BinaryRegion.beta ≤ 1 :=
  BinaryRegion.binaryNetwork.overlap_le_one _ _

example : BinaryRegion.binaryNetwork.overlap BinaryRegion.alpha BinaryRegion.beta =
    BinaryRegion.binaryNetwork.overlap BinaryRegion.beta BinaryRegion.alpha :=
  BinaryRegion.binaryNetwork.overlap_symm _ _

example : BinaryRegion.binaryNetwork.overlap BinaryRegion.alpha BinaryRegion.beta = 1 / 4 :=
  BinaryRegion.binary_overlap_cross

example : 0 < BinaryRegion.binaryNetwork.mutualInfo BinaryRegion.alpha BinaryRegion.beta :=
  BinaryRegion.binary_mi_pos _ _

end InformationNetworkRegression

/-! Non-degenerate Cosmic Density & Dynamics regression examples. -/
namespace CosmicAndDynamicsRegression
open OmegaProtocol

example : Vol14.CosmicDensityBudget.standardLCDM.omegaMatter = 31 / 100 :=
  Vol14.CosmicDensityBudget.standardLCDM_matter

example : Vol14.CosmicDensityBudget.standardLCDM.omegaMatter <
    Vol14.CosmicDensityBudget.standardLCDM.omegaDE :=
  Vol14.CosmicDensityBudget.standardLCDM_dark_energy_dominant

example : Vol19.macroEntropy Vol19.disorderedState <
    Vol19.macroEntropy Vol19.organizedState :=
  Vol19.emergence_witness

example : Vol19.informationGain Vol19.disorderedState Vol19.organizedState = 2 :=
  Vol19.emergence_witness_gain

example (d0 : ℝ) : Vol20.ExpandingMap.doublingMap.separation d0 3 = 8 * d0 :=
  Vol20.ExpandingMap.doubling_separation_after_3_steps d0

example : Vol15.CosmologicalFluid.deSitterVacuum.w = -1 :=
  Vol15.CosmologicalFluid.deSitter_w

example : Vol15.CosmologicalFluid.deSitterVacuum.accelerationSource = -2 :=
  Vol15.CosmologicalFluid.deSitter_source_neg

example : 0 < - (4 * Real.pi * 1 / 3) * Vol15.CosmologicalFluid.deSitterVacuum.accelerationSource :=
  Vol15.CosmologicalFluid.cosmic_acceleration_positive Vol15.CosmologicalFluid.deSitterVacuum 1
    (by norm_num) (by rw [Vol15.CosmologicalFluid.deSitter_source_neg]; norm_num)

example : Vol12.QubitSystem.bellPairSystem.entanglement <
    Vol12.QubitSystem.bellPairSystem.capacity :=
  Vol12.QubitSystem.bellPair_strictly_below_capacity

example (planckScale : ℝ → ℝ) :
    Vol22.NetworkER_EPR.networkThroatArea BinaryRegion.binaryNetwork planckScale
      BinaryRegion.alpha BinaryRegion.alpha = 0 :=
  Vol22.NetworkER_EPR.binary_identical_zero_throat planckScale BinaryRegion.alpha

end CosmicAndDynamicsRegression

-- File-scope resolution for the bare `VolXX.*` references used by all
-- regression examples below (the earlier `open OmegaProtocol` commands
-- are scoped inside the namespaces above and do not reach this far).
open OmegaProtocol

example : Vol08.standardSchwarzschild.mass = 1 := rfl

example : 0 < Vol08.HawkingTemperature Vol08.standardSchwarzschild :=
  Vol08.hawkingtemperature_pos Vol08.standardSchwarzschild

example : Vol08.BHEntropy Vol08.standardSchwarzschild = 4 * Real.pi :=
  Vol08.standardSchwarzschild_entropy

example : Vol09.EntanglementEntropy Vol08.standardSchwarzschild =
    Vol08.BHEntropy Vol08.standardSchwarzschild :=
  Vol09.bridge_bh_entropy_is_holographic Vol08.standardSchwarzschild

example : Vol11.ChernBand.haldaneBand.chernNumber = 1 :=
  Vol11.ChernBand.haldane_chern_number

example (e h : ℝ) : Vol11.ChernBand.haldaneBand.hallConductance e h = e^2 / h :=
  Vol11.ChernBand.haldane_hall_conductance e h

example : 0 < Vol13.SpinNetwork.areaGap 1 1 (by norm_num) (by norm_num) :=
  Vol13.SpinNetwork.areaGap_pos 1 1 (by norm_num) (by norm_num)

example : Vol13.Graviton.plusGraviton.helicity ≠ Vol13.Graviton.minusGraviton.helicity :=
  Vol13.Graviton.distinct_helicities

example (H_m H_g : Operator) (Ψ : StateSpace) (h : (H_m + H_g) Ψ = 0) :
    H_m Ψ = - H_g Ψ :=
  Vol13.wheeler_dewitt_balance H_m H_g Ψ h

example (M : Vol13.MiniSuperspace) :
    ∃ a p : ℝ, 0 < a ∧
      Vol13.MiniSuperspace.matterDensity a p + M.gravityTerm a = 0 :=
  Vol13.MiniSuperspace.constraint_satisfiable (M := M)

example : Vol06.spacelike_separated Vol06.regionA Vol06.regionB :=
  Vol06.regionA_B_spacelike

example (h12 : Vol06.subset_region Vol06.regionA Vol06.regionA)
    (h34 : Vol06.subset_region Vol06.regionB Vol06.regionB)
    (A B : ↥OmegaAlgebra) (hA : A ∈ Vol06.LocalNet Vol06.regionA)
    (hB : B ∈ Vol06.LocalNet Vol06.regionB) :
    A * B = B * A :=
  Vol06.nested_microcausality Vol06.regionA Vol06.regionA Vol06.regionB Vol06.regionB
    h12 h34 Vol06.regionA_B_spacelike A B hA hB

example : Vol02.FineStructureConstant > 0 :=
  Vol02.fine_structure_positive

example (em : Vol02.ElectromagneticField) :
    Vol02.d em.F = 0 :=
  Vol02.homogeneous_maxwell em

example (em : Vol02.ElectromagneticField) :
    Vol02.d (Vol02.d em.A) = 0 :=
  Vol02.d_squared_zero em.A em.smooth_A

example (S : Vol02.SourcedField) :
    Vol02.d S.starF = S.starJ :=
  Vol02.inhomogeneous_maxwell S

example (ρ : StateSpace) (β : ℝ) :
    Vol03.MT.KMSState ρ β :=
  Vol03.kms_holds_at_every_temperature ρ β

example (T : StateSpace → ℝ) (ρ₁ ρ₂ ρ₃ : StateSpace)
    (h₁ : Vol03.InEquilibrium T ρ₁ ρ₂) (h₂ : Vol03.InEquilibrium T ρ₂ ρ₃) :
    Vol03.InEquilibrium T ρ₁ ρ₃ :=
  Vol03.zeroth_law T ρ₁ ρ₂ ρ₃ h₁ h₂

example (P : Vol03.ThermoProcess) :
    P.Q / P.T ≤ P.S₂ - P.S₁ :=
  Vol03.clausius_inequality P

example (bh : Vol08.BHGeometry) :
    (2 * Real.pi / bh.surfaceGravity) * Vol08.HawkingTemperature bh = 1 :=
  Vol08.kms_beta_hawking bh

example (s1 s2 : Vol03.ThermoState) :
    s2.entropy - s1.entropy = Vol03.heatExchange s1 s2 / s1.temp :=
  Vol03.clausius_reversible_equality s1 s2

example (TC TH Qin : ℝ) (hC : 0 < TC) (hH : 0 < TH) (h_ord : TC < TH) (hQ : 0 < Qin) :
    0 < (Vol03.HeatEngine.idealCarnotEngine TC TH hC hH h_ord Qin hQ).carnotEfficiency :=
  (Vol03.HeatEngine.idealCarnotEngine TC TH hC hH h_ord Qin hQ).carnot_efficiency_pos

example (TC TH Qin : ℝ) (hC : 0 < TC) (hH : 0 < TH) (h_ord : TC < TH) (hQ : 0 < Qin) :
    (Vol03.HeatEngine.idealCarnotEngine TC TH hC hH h_ord Qin hQ).efficiency =
    (Vol03.HeatEngine.idealCarnotEngine TC TH hC hH h_ord Qin hQ).carnotEfficiency :=
  Vol03.HeatEngine.idealCarnotEngine_attains_carnot TC TH hC hH h_ord Qin hQ

example (U : Vol07.FLRWUniverse) (G : ℝ) (hG : 0 < G) :
    0 < U.criticalDensity G :=
  U.criticalDensity_pos G hG

example (U : Vol07.FLRWUniverse) (G : ℝ) (hG : 0 < G) :
    U.hubbleRate ^ 2 = (8 * Real.pi * G / 3) * U.density ↔
    U.density = U.criticalDensity G :=
  U.flat_friedmann_critical_density_eq G hG

example (a_emit a_obs : ℝ) (ha : 0 < a_emit) (h : a_emit < a_obs) :
    0 < Vol07.FLRWUniverse.redshift a_emit a_obs :=
  Vol07.FLRWUniverse.redshift_positive_of_expansion a_emit a_obs ha h

example (S : Vol05.SphericallySymmetricSpacetime) (G r : ℝ) (hG : 0 < G) (hr : S.horizonRadius G < r) :
    0 < S.lapseFunction G r :=
  S.lapse_strictly_positive G r hG hr

example (S : Vol05.SphericallySymmetricSpacetime) (G r : ℝ) (hG : 0 < G) (hr : 0 < r) :
    0 < S.kretschmannInvariant G r :=
  S.kretschmann_pos G r hG hr

example : Vol05.SphericallySymmetricSpacetime.unitSchwarzschild.horizonRadius 1 = 2 :=
  Vol05.SphericallySymmetricSpacetime.unitSchwarzschild_horizon

example : Vol05.SphericallySymmetricSpacetime.unitSchwarzschild.kretschmannInvariant 1 2 = 3 / 4 :=
  Vol05.SphericallySymmetricSpacetime.unitSchwarzschild_kretschmann_at_horizon

example (a a' b b' : ℝ) (ha : |a| ≤ 1) (ha' : |a'| ≤ 1) (hb : b = 1 ∨ b = -1) (hb' : b' = 1 ∨ b' = -1) :
    |a * b + a * b' + a' * b - a' * b'| ≤ 2 :=
  Vol24.CHSHSystem.local_realism_bound a a' b b' ha ha' hb hb'

example : Vol24.CHSHSystem.singletBellSystem.chshParameter = Vol24.TsirelsonBound :=
  Vol24.CHSHSystem.singlet_chsh_attains_tsirelson

example : Vol24.CHSHSystem.singletBellSystem.chshParameter > Vol24.ClassicalCHSHBound :=
  Vol24.CHSHSystem.singlet_violates_classical_bound

example (S : Vol23.QubitState) : S.trace = 1 :=
  S.trace_eq_one

example (S : Vol23.QubitState) : S.purity ≤ 1 :=
  S.purity_le_one

example (S : Vol23.QubitState) (lam : ℝ) (hlam0 : 0 ≤ lam) (hlam1 : lam ≤ 1) :
    (S.dephase lam hlam0 hlam1).trace = S.trace :=
  S.dephase_trace_preserving lam hlam0 hlam1

example (S : Vol23.QubitState) (lam : ℝ) (hlam0 : 0 ≤ lam) (hlam1 : lam ≤ 1) :
    |(S.dephase lam hlam0 hlam1).c| ≤ |S.c| :=
  S.dephase_coherence_decay lam hlam0 hlam1

example : Vol23.QubitState.plusState.purity = 1 :=
  Vol23.QubitState.plusState_pure

example : (Vol23.QubitState.plusState.dephase 0 (by norm_num) (by norm_num)).purity = 1 / 2 :=
  Vol23.QubitState.plusState_decohered_purity

example (S : Vol27.IntegratedSystem) : 0 ≤ S.phi :=
  S.phi_nonneg

example (S : Vol27.IntegratedSystem) (h : S.H_AB < S.H_A + S.H_B) : 0 < S.phi :=
  S.phi_pos_of_strict_subadditivity h

example : 0 < Vol27.IntegratedSystem.corticalComplex.phi :=
  Vol27.IntegratedSystem.cortical_phi_pos

example : Vol27.IntegratedSystem.corticalComplex.phi = 4 / 5 :=
  Vol27.IntegratedSystem.cortical_phi_value

example : 0 < Vol27.IntegratedSystem.corticalComplex.fluxAsymmetry :=
  Vol27.IntegratedSystem.cortical_flux_asymmetry_pos

example {n : ℕ} (A : Vol33.BoundedAgent n) (i : Fin n) : 0 ≤ A.utility i :=
  A.utility_nonneg i

example {n : ℕ} (A : Vol33.BoundedAgent n) : 0 ≤ A.totalUtility :=
  A.totalUtility_nonneg

example : Vol33.BoundedAgent.dualTaskAgent.totalUtility = 25 :=
  Vol33.BoundedAgent.dualTask_totalUtility_value

example : 0 < Vol33.BoundedAgent.dualTaskAgent.totalUtility :=
  Vol33.BoundedAgent.dualTask_totalUtility_pos

/-! Computation, categories, pattern, expansion, dimensions, nothing. -/

example (c : ℕ) : Vol35.run Vol35.haltMachine c 1 = none := rfl

example (c : ℕ) : Vol35.run Vol35.loopMachine c 0 = some c := rfl

example (c : ℕ) : Vol35.Halts Vol35.haltMachine c :=
  Vol35.haltMachine_halts c

example (c : ℕ) : ¬ Vol35.Halts Vol35.loopMachine c :=
  Vol35.loopMachine_diverges c

example (m n : Vol36.SkelObj) (f : Vol36.SkelHom m n) :
    Vol36.skelComp (Vol36.skelId m) f = f :=
  Vol36.skelId_left f

example : Vol37.ReactionSystem.turingWitness.dispersion (7 / 20) = -9 / 40 :=
  Vol37.ReactionSystem.turingWitness_optimal_value

example : Vol44.DeSitterSpace.unitDeSitter.entropy = 3 * Real.pi :=
  Vol44.DeSitterSpace.unitDeSitter_entropy

example : Vol48.Compactification.unitCompact.Geff = 1 :=
  Vol48.Compactification.unitCompact_Geff

example (f g : Empty → Bool) : f = g :=
  Vol54.ex_nihilo_maps_unique Bool f g

example : Vol29.PreyRate 1 2 0 0 = 0 ∧ Vol29.PredatorRate 3 4 0 0 = 0 :=
  Vol29.extinction_equilibrium 1 2 3 4

example (c : ℕ) : Vol35.haltsWithin Vol35.haltMachine c 1 = true :=
  (Vol35.haltsWithin_correct Vol35.haltMachine c 1).mpr ⟨1, by decide, rfl⟩
