import Mathlib
import OmegaUnifiedFoundation

/-!
# Conditional pure bipartite entropy model

Joint entropy is NOT the sum of marginal entropies. We explicitly assume
nonnegative entropies and the Araki–Lieb difference bound. A pure joint state
then has equal, possibly positive marginal entropies. No density operators or
entropy functional are constructed here, and the entropy inequality is a model
hypothesis, not a derived theorem of quantum information.

API correction: `BHState.conservation` and `entropies_opposite` are removed.
`unitarity_total` now states zero JOINT entropy, not a zero sum of marginals.
The endpoint theorem retains its public signature for protocol callers.
-/
namespace OmegaProtocol.Vol25
open OmegaProtocol

structure BHState where
  bhEntropy : ℝ
  radEntropy : ℝ
  jointEntropy : ℝ
  bh_nonneg : 0 ≤ bhEntropy
  rad_nonneg : 0 ≤ radEntropy
  joint_nonneg : 0 ≤ jointEntropy
  difference_bound : |bhEntropy - radEntropy| ≤ jointEntropy
  joint_pure : jointEntropy = 0

def VonNeumannEntropy (s : BHState) : ℝ := s.bhEntropy
def RadiationEntropy (s : BHState) : ℝ := s.radEntropy

/-- The purity hypothesis concerns the joint state only. -/
theorem unitarity_total (s : BHState) : s.jointEntropy = 0 := s.joint_pure

theorem pure_marginal_entropies_equal (s : BHState) :
    VonNeumannEntropy s = RadiationEntropy s := by
  have h := s.difference_bound
  rw [s.joint_pure] at h
  have hz : |s.bhEntropy - s.radEntropy| = 0 :=
    le_antisymm h (abs_nonneg _)
  exact sub_eq_zero.mp (abs_eq_zero.mp hz)

/-- Conditional endpoint only; this does not construct a dynamical Page curve. -/
theorem page_curve_endpoint (s_final : BHState)
    (h_evaporated : VonNeumannEntropy s_final = 0) :
    RadiationEntropy s_final = 0 := by
  rw [← pure_marginal_entropies_equal s_final, h_evaporated]

/-- Reverse endpoint implication, making the conditional result exact. -/
theorem entropy_zero_iff (s : BHState) :
    VonNeumannEntropy s = 0 ↔ RadiationEntropy s = 0 := by
  rw [pure_marginal_entropies_equal]

/-- A realizable entropy-data witness with nonzero marginals and pure joint state.
    This is not a construction of its density matrix. -/
def positiveMarginalWitness : BHState :=
  { bhEntropy := 1
    radEntropy := 1
    jointEntropy := 0
    bh_nonneg := by norm_num
    rad_nonneg := by norm_num
    joint_nonneg := by norm_num
    difference_bound := by norm_num
    joint_pure := rfl }

theorem pure_joint_need_not_have_zero_marginals :
    ∃ s : BHState, s.jointEntropy = 0 ∧
      0 < VonNeumannEntropy s ∧ 0 < RadiationEntropy s := by
  exact ⟨positiveMarginalWitness, rfl, by norm_num [VonNeumannEntropy, positiveMarginalWitness],
    by norm_num [RadiationEntropy, positiveMarginalWitness]⟩

theorem bridge_vol25_entropy_nonneg (R : QRegion) : vonNeumannEntropy R ≥ 0 :=
  monotonicity_lemma R

end OmegaProtocol.Vol25
