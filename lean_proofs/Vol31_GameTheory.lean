import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-!
# A concrete prisoner's dilemma

The payoffs below are stipulated, not derived from information geometry. The
new pure-Nash predicate quantifies over every unilateral deviation; its unique
equilibrium is mutual defection, even though cooperation is Pareto-superior.
No repeated-game, mixed-strategy, or evolutionary-dynamics claim is made.
-/

namespace OmegaProtocol.Vol31
open OmegaProtocol

inductive Strategy | Cooperate | Defect

/-- Prisoner's Dilemma payoff for player 1 -/
def Payoff1 : Strategy → Strategy → ℤ
  | .Cooperate, .Cooperate => 3
  | .Cooperate, .Defect => 0
  | .Defect, .Cooperate => 5
  | .Defect, .Defect => 1

/-- Prisoner's Dilemma payoff for player 2 -/
def Payoff2 (s1 s2 : Strategy) : ℤ :=
  Payoff1 s2 s1

/-- THEOREM: Defect is a dominant strategy for Player 1 (GENUINE PROOF)
    Regardless of what Player 2 does, Player 1 gets more from Defect. -/
theorem defect_dominates_p1 (s2 : Strategy) :
  Payoff1 Strategy.Defect s2 ≥ Payoff1 Strategy.Cooperate s2 := by
  cases s2 <;> simp [Payoff1]

/-- THEOREM: Mutual Defection is a Nash Equilibrium (GENUINE PROOF) -/
theorem nash_equilibrium_defect :
  (Payoff1 Strategy.Defect Strategy.Defect ≥ Payoff1 Strategy.Cooperate Strategy.Defect) ∧
  (Payoff2 Strategy.Defect Strategy.Defect ≥ Payoff2 Strategy.Defect Strategy.Cooperate) := by
  constructor <;> simp [Payoff1, Payoff2]

/-- Legacy export name for dominance in the stipulated payoff table. -/
theorem gametheory_from_omega (s2 : Strategy) :
  Payoff1 Strategy.Defect s2 ≥ Payoff1 Strategy.Cooperate s2 := by
  exact defect_dominates_p1 s2

/-- Consistency bridge (VOL31): the mutual information is nonnegative.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `game_theory_mutual_info_nonneg` evoked. -/
theorem bridge_vol31_mutual_info_nonneg (R₁ R₂ : QRegion) :
  mutualInformation R₁ R₂ ≥ 0 := by
  exact mutualInformation_nonneg R₁ R₂

/-- Consistency bridge (VOL31): the Omega-metric `d` is reflexive (`d R R = 0`).
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `nash_from_phi` evoked. -/
theorem bridge_vol31_metric_self_zero (R : QRegion) : d R R = 0 := by
  exact qregion_self_distance_zero R

/-- No player can improve by any unilateral pure-strategy deviation. -/
def IsNash (s1 s2 : Strategy) : Prop :=
  (∀ t, Payoff1 t s2 ≤ Payoff1 s1 s2) ∧
  (∀ t, Payoff2 s1 t ≤ Payoff2 s1 s2)

theorem defect_strictly_dominates (s2 : Strategy) :
    Payoff1 Strategy.Cooperate s2 < Payoff1 Strategy.Defect s2 := by
  cases s2 <;> norm_num [Payoff1]

/-- Complete classification, stronger than checking just one candidate. -/
theorem nash_iff_mutual_defection (s1 s2 : Strategy) :
    IsNash s1 s2 ↔ s1 = Strategy.Defect ∧ s2 = Strategy.Defect := by
  constructor
  · intro h
    cases s1 with
    | Cooperate =>
        have h1 := h.1 Strategy.Defect
        have hstrict := defect_strictly_dominates s2
        omega
    | Defect =>
        cases s2 with
        | Cooperate =>
            have h2 := h.2 Strategy.Defect
            norm_num [Payoff2, Payoff1] at h2
        | Defect => exact ⟨rfl, rfl⟩
  · rintro ⟨rfl, rfl⟩
    constructor
    · intro t
      cases t <;> norm_num [Payoff1]
    · intro t
      cases t <;> norm_num [Payoff2, Payoff1]

theorem pure_nash_exists_unique :
    ∃! s : Strategy × Strategy, IsNash s.1 s.2 := by
  refine ⟨(Strategy.Defect, Strategy.Defect), ?_, ?_⟩
  · exact (nash_iff_mutual_defection _ _).mpr ⟨rfl, rfl⟩
  · intro s hs
    obtain ⟨h1, h2⟩ := (nash_iff_mutual_defection s.1 s.2).mp hs
    exact Prod.ext h1 h2

/-- Nash stability is not the same as jointly optimal welfare. -/
theorem cooperation_pareto_dominates_equilibrium :
    Payoff1 .Defect .Defect < Payoff1 .Cooperate .Cooperate ∧
    Payoff2 .Defect .Defect < Payoff2 .Cooperate .Cooperate := by
  norm_num [Payoff1, Payoff2]

end OmegaProtocol.Vol31
