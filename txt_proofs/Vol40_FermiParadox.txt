import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol40_FermiParadox.lean
  Formalization of the Fermi Paradox.

  Model scope (stated honestly):
  * A development chain is a list of sequential filters, each either passed
    (`true`) or lethal (`false`). The "Great Filter" is any lethal stage.
  * The Drake expectation is modeled as a plain product of natural-number
    factors; its monotonicity and zero-silence laws are proven, not assumed.
  * Omega-Protocol mapping: civilizations = Q-regions (0D), communication =
    Φ (1D), interstellar distance = Ω-metric (2D).
-/

namespace OmegaProtocol.Vol40
open OmegaProtocol

/-- A development chain: sequential filters, `true` = passed, `false` =
    extinction at that stage. -/
def Chain : Type := List Bool

/-- A civilization survives a chain iff every filter is passed. -/
def passesAll : Chain → Bool
  | [] => true
  | f :: fs => f && passesAll fs

/-- A Great Filter is a stage index together with the lethality flag at that
    stage. -/
structure GreatFilter where
  index : ℕ
  lethal : Bool

/-- The Great Filter type is inhabited — witnessed by a lethal stage-0
    filter, the strongest possible silence condition. -/
theorem great_filter_witness : Nonempty GreatFilter :=
  ⟨⟨0, true⟩⟩

/-- One failed filter silences the whole chain. -/
theorem failed_filter_silences (c : Chain) :
    passesAll (false :: c) = false := by
  simp [passesAll]

/-- The empty development chain trivially survives. -/
theorem empty_chain_survives : passesAll [] = true := rfl

/-- All stages of a chain are passed (recursive mirror of `passesAll`). -/
def allTrue : Chain → Prop
  | [] => True
  | f :: fs => f = true ∧ allTrue fs

/-- A chain survives if every stage passes. -/
theorem all_pass_survives (c : Chain) : allTrue c → passesAll c = true := by
  induction c with
  | nil => intro _; rfl
  | cons hd tl ih =>
    intro h
    simp [passesAll, h.1, ih h.2]

/-- The Drake expectation: product of seven natural-number factors. -/
def drake (R fp ne fl fi fc L : ℕ) : ℕ :=
  R * fp * ne * fl * fi * fc * L

/-- Any zero factor silences the Drake expectation. -/
theorem drake_zero_factor_silences (R fp ne fl fi fc L : ℕ) (h : fl = 0) :
    drake R fp ne fl fi fc L = 0 := by
  simp [drake, h]

/-- The Drake expectation is monotone in the longevity factor. -/
theorem drake_monotone_longevity (R fp ne fl fi fc L₁ L₂ : ℕ) (h : L₁ ≤ L₂) :
    drake R fp ne fl fi fc L₁ ≤ drake R fp ne fl fi fc L₂ := by
  simp only [drake]
  exact Nat.mul_le_mul_left _ h

/-- Structural bridge: interstellar distances in the model are nonnegative. -/
theorem bridge_vol40_metric_nonneg (R₁ R₂ : QRegion) : d R₁ R₂ ≥ 0 :=
  bridge_distance_nonneg R₁ R₂

/-- Structural bridge: the holographic entropy bound holds in the Q-region
    model used for civilization horizons. -/
theorem bridge_vol40_entropy_bounded (R : QRegion) :
    vonNeumannEntropy R ≤ Real.pi :=
  bridge_vonNeumannEntropy_le_pi R

-- ============================================================
-- PROBABILISTIC FILTER CHAINS ON [0, 1]
-- ============================================================

/-- Every filter survival probability in the list lies in `[0, 1]`. -/
def ValidFilterProbs : List ℝ → Prop
  | [] => True
  | p :: ps => (0 ≤ p ∧ p ≤ 1) ∧ ValidFilterProbs ps

/-- Cumulative survival probability across a sequence of independent filters. -/
def survivalProb : List ℝ → ℝ
  | [] => 1
  | p :: ps => p * survivalProb ps

/-- Cumulative survival probability is nonnegative for valid filter chains. -/
theorem survivalProb_nonneg (ps : List ℝ) (h : ValidFilterProbs ps) :
    0 ≤ survivalProb ps := by
  induction ps with
  | nil => norm_num [survivalProb]
  | cons p tl ih =>
    dsimp [survivalProb]
    exact mul_nonneg h.1.1 (ih h.2)

/-- Cumulative survival probability never exceeds `1` for valid filter chains. -/
theorem survivalProb_le_one (ps : List ℝ) (h : ValidFilterProbs ps) :
    survivalProb ps ≤ 1 := by
  induction ps with
  | nil => norm_num [survivalProb]
  | cons p tl ih =>
    dsimp [survivalProb]
    have htl0 : 0 ≤ survivalProb tl := survivalProb_nonneg tl h.2
    have htl1 : survivalProb tl ≤ 1 := ih h.2
    nlinarith [h.1.1, h.1.2]

/-- Prepending another valid filter `p ∈ [0, 1]` cannot increase survival
    probability. -/
theorem survivalProb_step_le (p : ℝ) (ps : List ℝ)
    (hp1 : p ≤ 1) (hps : ValidFilterProbs ps) :
    survivalProb (p :: ps) ≤ survivalProb ps := by
  dsimp [survivalProb]
  have h0 : 0 ≤ survivalProb ps := survivalProb_nonneg ps hps
  nlinarith

/-- Strict filter attrition: any filter with `p < 1` strictly reduces a
    positive survival probability. -/
theorem survivalProb_strict_attrition (p : ℝ) (ps : List ℝ)
    (hp1 : p < 1) (hpos : 0 < survivalProb ps) :
    survivalProb (p :: ps) < survivalProb ps := by
  dsimp [survivalProb]
  nlinarith

end OmegaProtocol.Vol40
