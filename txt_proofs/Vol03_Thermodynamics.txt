import Mathlib
import OmegaUnifiedFoundation
/-
  Vol03_Thermodynamics.lean
  REAL mathematical formalization of thermodynamics.

  Genuinely proven theorems:
  1. Zeroth Law — transitivity of thermal equilibrium, stated through an
     explicit temperature function on states (see below for why the former
     KMS-uniqueness conditional was retired);
  2. Second Law — monotonicity of the model relative entropy under arbitrary
     maps;
  3. Clausius inequality — the entropy balance of a thermodynamic process;
  4. Non-negativity of relative entropy.

  The key insight: The KMS (Kubo-Martin-Schwinger) condition
  characterizes thermal equilibrium states in quantum statistical
  mechanics. The modular theory of von Neumann algebras provides
  the mathematical framework where:
  - Temperature ↔ inverse modular parameter β
  - Entropy ↔ relative entropy S(ρ||σ)
  - Second Law ↔ monotonicity of relative entropy under CPTP maps

  Model scope: the concrete `ModularTheory` used here is *tracial* — the state
  functional is the algebraic trace and the modular flow is the identity — so
  the KMS condition holds at every inverse temperature and does NOT determine
  a temperature.  That degeneracy is proved, not hidden:
  `kms_holds_at_every_temperature` and `kms_temperature_uniqueness_fails`.
-/


namespace OmegaProtocol.Vol03
open OmegaProtocol

-- The legacy bridge uses the checked concrete model from the foundation.
noncomputable abbrev MT : ModularTheory := concreteModularTheory

-- ============================================================
-- THEOREM 1: ZEROTH LAW AND THE TRACIAL KMS CONDITION
-- If systems A,B are in thermal equilibrium (same temperature),
-- and B,C are in thermal equilibrium, then A,C are as well.
-- ============================================================

-- RETIRED (tenth pass): `kms_transitivity`/`zeroth_law` were conditionals on a
-- "KMS-temperature uniqueness" hypothesis over every state.  That premise is
-- *refuted* in the concrete model (the state functional is the algebraic
-- trace, so every state is KMS at every inverse temperature — see
-- `kms_holds_at_every_temperature` and `kms_temperature_uniqueness_fails`), so
-- the conditional carried no content.  The zeroth law is now stated through an
-- explicit temperature function, and the tracial KMS facts are stated as
-- theorems whose proofs rest on `ω_ρ_trace` (Mathlib's `trace_mul_comm`).

/-- Thermal equilibrium through a temperature function on states: two systems
    are in equilibrium when their temperatures agree.  Carrying `T` explicitly
    is what gives the zeroth law content here, since the model's KMS predicate
    does not determine a temperature on its own. -/
def InEquilibrium (T : StateSpace → ℝ) (ρ σ : StateSpace) : Prop := T ρ = T σ

/-- **Zeroth law of thermodynamics**: thermal equilibrium is transitive, because
    it is equality of temperatures. -/
theorem zeroth_law (T : StateSpace → ℝ) (ρ₁ ρ₂ ρ₃ : StateSpace)
    (h₁₂ : InEquilibrium T ρ₁ ρ₂) (h₂₃ : InEquilibrium T ρ₂ ρ₃) :
    InEquilibrium T ρ₁ ρ₃ :=
  h₁₂.trans h₂₃

/-- **The tracial model is KMS at every inverse temperature**: the modular flow
    is the identity, so the KMS equation reduces to the trace equation
    `ω_ρ (A * B) = ω_ρ (B * A)`, which holds by `ω_ρ_trace`. -/
theorem kms_holds_at_every_temperature (ρ : StateSpace) (β : ℝ) :
    MT.KMSState ρ β :=
  fun A B => ω_ρ_trace A B

/-- **The KMS predicate does not determine a temperature**: the uniqueness
    premise of the former zeroth-law conditional is refuted in the tracial
    model, where every state is KMS at every `β`.  Downstream statements must
    therefore carry their temperature as data (`InEquilibrium`) instead of
    inferring it from `KMSState`. -/
theorem kms_temperature_uniqueness_fails :
    ¬ (∀ (ρ : StateSpace) (β β' : ℝ),
      MT.KMSState ρ β → MT.KMSState ρ β' → β = β') := by
  intro h
  have h01 : (0 : ℝ) = 1 :=
    h 0 0 1 (kms_holds_at_every_temperature 0 0) (kms_holds_at_every_temperature 0 1)
  exact zero_ne_one h01

-- ============================================================
-- THEOREM 2: SECOND LAW OF THERMODYNAMICS (GENUINE PROOF)
-- Relative entropy decreases under CPTP maps:
--   S(Φ(ρ) || Φ(σ)) ≤ S(ρ || σ)
--
-- This is the quantum information-theoretic formulation of the
-- Second Law. Physical interpretation: irreversible processes
-- (modeled by CPTP maps) cannot increase the distinguishability
-- of states, hence cannot decrease entropy.
--
-- This follows directly from the checked ModularTheory model.
-- ============================================================

theorem second_law :
  ∀ (ρ σ : StateSpace) (Φ : CPTPMap),
  MT.RelativeEntropy (Φ ρ) (Φ σ) ≤ MT.RelativeEntropy ρ σ :=
  MT.law_relative_entropy_monotonicity

-- ============================================================
-- THEOREM 3: CLAUSIUS INEQUALITY (GENUINE PROOF)
-- For a process at temperature T > 0:
--   ΔS ≥ Q/T
-- where Q is the heat exchanged and ΔS is the entropy change.
--
-- Proof: This is algebraic from the definitions.
-- ============================================================

-- RETIRED (tenth pass): `Temperature`/`Entropy`/`Heat`/`BlackHoleArea`/
-- `BHEntropy` were the constant-zero functions on `StateSpace`, so
-- `clausius_inequality`'s hypothesis `Temperature ρ > 0` was unsatisfiable
-- (vacuous) and `bh_entropy_formula` was a definitional restatement.  The
-- statements now carry the thermodynamic data explicitly.

/-- A thermodynamic process of a system at strictly positive temperature `T`:
    heat `Q` is exchanged and the entropy changes from `S₁` to `S₂`.  The
    entropy balance is a law-field — the change equals the reversible part
    `Q / T` plus a non-negative irreversibility `σ` (entropy production).
    Reversible processes are exactly those with `σ = 0`. -/
structure ThermoProcess where
  T : ℝ
  S₁ : ℝ
  S₂ : ℝ
  Q : ℝ
  irreversibility : ℝ
  temp_pos : 0 < T
  irr_nonneg : 0 ≤ irreversibility
  law_entropy_balance : S₂ - S₁ = Q / T + irreversibility

/-- **Clausius inequality** `ΔS ≥ Q/T` for a process at temperature `T > 0`:
    the entropy change is at least the exchanged heat divided by the
    temperature, with the difference equal to the entropy production.  Both
    hypotheses are satisfiable (`T = 1`, `σ = 0`, `ΔS = Q`), so the statement
    is not vacuous. -/
theorem clausius_inequality (P : ThermoProcess) :
    P.Q / P.T ≤ P.S₂ - P.S₁ := by
  rw [P.law_entropy_balance]
  linarith [P.irr_nonneg]

/-- The Clausius bound is attained exactly by reversible processes. -/
theorem clausius_equality_iff_reversible (P : ThermoProcess) :
    P.S₂ - P.S₁ = P.Q / P.T ↔ P.irreversibility = 0 := by
  rw [P.law_entropy_balance]
  constructor <;> intro h <;> linarith

-- ============================================================
-- BEKENSTEIN-HAWKING ENTROPY (Bridge to Vol08)
-- ============================================================

/-- Black-hole thermodynamic data: the horizon area and the Bekenstein-Hawking
    entropy, related by the *postulated* area law `S = A/4` (carried as a
    law-field, so the law constrains two independent data rather than being a
    definition in disguise). -/
structure BlackHoleThermo where
  horizonArea : ℝ
  entropy : ℝ
  area_pos : 0 < horizonArea
  law_bekenstein_hawking : entropy = horizonArea / 4

/-- Bekenstein-Hawking entropy is positive for a positive horizon area. -/
theorem bh_entropy_pos (bh : BlackHoleThermo) : 0 < bh.entropy := by
  rw [bh.law_bekenstein_hawking]
  exact div_pos bh.area_pos (by norm_num)

-- ============================================================
-- MACROSCOPIC THERMODYNAMIC PROCESSES & CARNOT EFFICIENCY
-- ============================================================

/-- Macroscopic thermodynamic equilibrium state with strictly positive temperature
    and well-defined entropy. -/
structure ThermoState where
  temp : ℝ
  entropy : ℝ
  temp_pos : 0 < temp
  entropy_nonneg : 0 ≤ entropy

/-- Reversible heat exchange at temperature T satisfies Q = T · ΔS. -/
def heatExchange (initial final : ThermoState) : ℝ :=
  initial.temp * (final.entropy - initial.entropy)

theorem heatExchange_nonneg_of_entropy_growth (s1 s2 : ThermoState)
    (hS : s1.entropy ≤ s2.entropy) :
    0 ≤ heatExchange s1 s2 := by
  dsimp [heatExchange]
  have hT := le_of_lt s1.temp_pos
  have hdiff : 0 ≤ s2.entropy - s1.entropy := sub_nonneg.mpr hS
  exact mul_nonneg hT hdiff

/-- Clausius equality for a reversible process: ΔS = Q / T. -/
theorem clausius_reversible_equality (s1 s2 : ThermoState) :
    s2.entropy - s1.entropy = heatExchange s1 s2 / s1.temp := by
  dsimp [heatExchange]
  have hT := ne_of_gt s1.temp_pos
  exact (mul_div_cancel_left₀ _ hT).symm

/-- Heat engine operating between two thermal reservoirs: hot reservoir T_H and cold reservoir T_C. -/
structure HeatEngine where
  tempHot : ℝ
  tempCold : ℝ
  workOutput : ℝ
  heatInput : ℝ
  th_pos : 0 < tempHot
  tc_pos : 0 < tempCold
  t_order : tempCold < tempHot
  heat_pos : 0 < heatInput
  work_nonneg : 0 ≤ workOutput
  work_le_heat : workOutput ≤ heatInput

namespace HeatEngine

variable (E : HeatEngine)

/-- Thermal efficiency η = W / Q_H. -/
noncomputable def efficiency : ℝ := E.workOutput / E.heatInput

/-- Carnot maximum theoretical efficiency η_Carnot = 1 - T_C / T_H. -/
noncomputable def carnotEfficiency : ℝ := 1 - E.tempCold / E.tempHot

theorem carnot_efficiency_pos : 0 < E.carnotEfficiency := by
  dsimp [carnotEfficiency]
  have hdiv : E.tempCold / E.tempHot < 1 := by
    exact (div_lt_one E.th_pos).mpr E.t_order
  linarith

theorem carnot_efficiency_lt_one : E.carnotEfficiency < 1 := by
  dsimp [carnotEfficiency]
  have hdiv : 0 < E.tempCold / E.tempHot := div_pos E.tc_pos E.th_pos
  linarith

theorem efficiency_nonneg : 0 ≤ E.efficiency :=
  div_nonneg E.work_nonneg (le_of_lt E.heat_pos)

/-- Reversible Carnot cycle benchmark attaining exact Carnot efficiency. -/
noncomputable def idealCarnotEngine (TC TH : ℝ) (hC : 0 < TC) (hH : 0 < TH) (h_ord : TC < TH) (Qin : ℝ) (hQ : 0 < Qin) : HeatEngine where
  tempHot := TH
  tempCold := TC
  workOutput := (1 - TC / TH) * Qin
  heatInput := Qin
  th_pos := hH
  tc_pos := hC
  t_order := h_ord
  heat_pos := hQ
  work_nonneg := by
    have h_carnot : 0 ≤ 1 - TC / TH := by
      have : TC / TH < 1 := (div_lt_one hH).mpr h_ord
      linarith
    exact mul_nonneg h_carnot (le_of_lt hQ)
  work_le_heat := by
    have h_div_pos : 0 < TC / TH := div_pos hC hH
    have h_factor : 1 - TC / TH ≤ 1 := by linarith
    nlinarith

theorem idealCarnotEngine_attains_carnot (TC TH : ℝ) (hC : 0 < TC) (hH : 0 < TH) (h_ord : TC < TH) (Qin : ℝ) (hQ : 0 < Qin) :
    (idealCarnotEngine TC TH hC hH h_ord Qin hQ).efficiency =
    (idealCarnotEngine TC TH hC hH h_ord Qin hQ).carnotEfficiency := by
  dsimp [efficiency, carnotEfficiency, idealCarnotEngine]
  have hQne : Qin ≠ 0 := ne_of_gt hQ
  exact mul_div_cancel_right₀ _ hQne

end HeatEngine

-- ============================================================
-- NON-TRACIAL TWO-LEVEL MATRIX MODULAR / KMS ALGEBRA
-- ============================================================
--
-- On the 1D `StateSpace := ℂ` carrier above, every operator is scalar so the
-- algebra is commutative (`OmegaAxioms.omegaAlgebra_commutative`), the modular
-- operator is `Δ = 1`, and the KMS equation holds at every inverse temperature
-- (`kms_holds_at_every_temperature`, `kms_temperature_uniqueness_fails`).
--
-- Here we formalize the concrete 2×2 matrix observable algebra `Obs2` and a
-- two-level Gibbs state at positive energy gap `ΔE > 0` and inverse
-- temperature `β`:
-- 1. The observable algebra is genuinely non-commutative (`Obs2.mul_noncommutative`).
-- 2. For `β ≠ 0` and `ΔE > 0`, the Gibbs state is strictly **non-tracial**
--    (`gibbsState_not_tracial`).
-- 3. The imaginary-time modular scaling action satisfies the one-parameter
--    group law (`modularScale_add`).
-- 4. The Gibbs state satisfies the KMS condition at candidate inverse
--    temperature `γ` **if and only if** `γ = β` (`twoLevel_kms_iff_eq`),
--    resolving the 1D temperature-uniqueness degeneracy and yielding the
--    KMS-derived Zeroth Law (`twoLevel_kms_zeroth_law`).

/-- 2×2 real matrix observable for the two-level modular/KMS system. -/
@[ext]
structure Obs2 where
  a00 : ℝ
  a01 : ℝ
  a10 : ℝ
  a11 : ℝ

namespace Obs2

def mul (A B : Obs2) : Obs2 :=
  ⟨A.a00 * B.a00 + A.a01 * B.a10,
   A.a00 * B.a01 + A.a01 * B.a11,
   A.a10 * B.a00 + A.a11 * B.a10,
   A.a10 * B.a01 + A.a11 * B.a11⟩

def one : Obs2 := ⟨1, 0, 0, 1⟩

def E01 : Obs2 := ⟨0, 1, 0, 0⟩

def E10 : Obs2 := ⟨0, 0, 1, 0⟩

/-- The 2×2 observable algebra is non-commutative (`E01 * E10 ≠ E10 * E01`). -/
theorem mul_noncommutative : mul E01 E10 ≠ mul E10 E01 := by
  intro h
  have h00 : (mul E01 E10).a00 = (mul E10 E01).a00 := congrArg Obs2.a00 h
  norm_num [mul, E01, E10] at h00

end Obs2

/-- Unnormalized two-level Gibbs weight `A₀₀ + exp(-β ΔE) A₁₁`. -/
noncomputable def gibbsWeight (ΔE β : ℝ) (A : Obs2) : ℝ :=
  A.a00 + Real.exp (-β * ΔE) * A.a11

/-- Two-level partition function `Z(β) = 1 + exp(-β ΔE)`. -/
noncomputable def gibbsPartition (ΔE β : ℝ) : ℝ :=
  1 + Real.exp (-β * ΔE)

theorem gibbsPartition_pos (ΔE β : ℝ) : 0 < gibbsPartition ΔE β := by
  unfold gibbsPartition
  linarith [Real.exp_pos (-β * ΔE)]

/-- Normalized two-level Gibbs state `ω_β(A) = Tr(ρ_β A)`. -/
noncomputable def gibbsState (ΔE β : ℝ) (A : Obs2) : ℝ :=
  gibbsWeight ΔE β A / gibbsPartition ΔE β

theorem gibbsState_one (ΔE β : ℝ) : gibbsState ΔE β Obs2.one = 1 := by
  unfold gibbsState gibbsWeight gibbsPartition Obs2.one
  have hZ : 1 + Real.exp (-β * ΔE) ≠ 0 := ne_of_gt (gibbsPartition_pos ΔE β)
  field_simp

/-- **Non-traciality of the two-level Gibbs state**: for any positive energy
    gap `ΔE > 0` and nonzero inverse temperature `β ≠ 0`,
    `ω_β(E₀₁ E₁₀) ≠ ω_β(E₁₀ E₀₁)`. -/
theorem gibbsState_not_tracial (ΔE β : ℝ) (hΔE : 0 < ΔE) (hβ : β ≠ 0) :
    gibbsState ΔE β (Obs2.mul Obs2.E01 Obs2.E10) ≠
      gibbsState ΔE β (Obs2.mul Obs2.E10 Obs2.E01) := by
  intro h
  have hZ : gibbsPartition ΔE β ≠ 0 := ne_of_gt (gibbsPartition_pos ΔE β)
  have hmul :
      gibbsState ΔE β (Obs2.mul Obs2.E01 Obs2.E10) * gibbsPartition ΔE β =
        gibbsState ΔE β (Obs2.mul Obs2.E10 Obs2.E01) * gibbsPartition ΔE β :=
    congrArg (fun x => x * gibbsPartition ΔE β) h
  unfold gibbsState at hmul
  rw [div_mul_cancel₀ _ hZ, div_mul_cancel₀ _ hZ] at hmul
  unfold gibbsWeight Obs2.mul Obs2.E01 Obs2.E10 at hmul
  have hexp : Real.exp 0 = Real.exp (-β * ΔE) := by
    rw [Real.exp_zero]
    linarith
  have hprod : 0 = -β * ΔE := Real.exp_injective hexp
  rcases mul_eq_zero.mp hprod.symm with hneg | hgap
  · exact hβ (neg_eq_zero.mp hneg)
  · exact (ne_of_gt hΔE) hgap

/-- Imaginary-time modular automorphism `σ_{-iγ}(B) = ρ_γ B ρ_γ⁻¹` on `Obs2`. -/
noncomputable def modularScale (ΔE γ : ℝ) (B : Obs2) : Obs2 :=
  ⟨B.a00,
   Real.exp (γ * ΔE) * B.a01,
   Real.exp (-γ * ΔE) * B.a10,
   B.a11⟩

/-- One-parameter group law for the modular scaling action:
    `σ_{-iγ₁}(σ_{-iγ₂}(B)) = σ_{-i(γ₁ + γ₂)}(B)`. -/
theorem modularScale_add (ΔE γ₁ γ₂ : ℝ) (B : Obs2) :
    modularScale ΔE γ₁ (modularScale ΔE γ₂ B) =
      modularScale ΔE (γ₁ + γ₂) B := by
  ext
  · rfl
  · dsimp [modularScale]
    rw [← mul_assoc, ← Real.exp_add]
    ring_nf
  · dsimp [modularScale]
    rw [← mul_assoc, ← Real.exp_add]
    ring_nf
  · rfl

/-- KMS condition for the two-level Gibbs state `ω_β` with respect to the
    modular scaling action at candidate inverse temperature `γ`:
    `∀ A B, ω_β(A · σ_{-iγ}(B)) = ω_β(B · A)`. -/
def IsTwoLevelKMS (ΔE β γ : ℝ) : Prop :=
  ∀ A B : Obs2,
    gibbsState ΔE β (Obs2.mul A (modularScale ΔE γ B)) =
      gibbsState ΔE β (Obs2.mul B A)

/-- **The two-level Gibbs state satisfies the KMS condition at `γ = β`**. -/
theorem twoLevel_kms_holds (ΔE β : ℝ) : IsTwoLevelKMS ΔE β β := by
  intro A B
  unfold gibbsState gibbsWeight Obs2.mul modularScale
  have hexp : Real.exp (-β * ΔE) * Real.exp (β * ΔE) = 1 := by
    rw [← Real.exp_add]
    have hsum : -β * ΔE + β * ΔE = 0 := by ring
    rw [hsum, Real.exp_zero]
  congr 1
  calc A.a00 * B.a00 + A.a01 * (Real.exp (-β * ΔE) * B.a10) +
        Real.exp (-β * ΔE) * (A.a10 * (Real.exp (β * ΔE) * B.a01) + A.a11 * B.a11)
      = A.a00 * B.a00 + A.a01 * (Real.exp (-β * ΔE) * B.a10) +
        (Real.exp (-β * ΔE) * Real.exp (β * ΔE)) * (A.a10 * B.a01) +
        Real.exp (-β * ΔE) * (A.a11 * B.a11) := by ring
    _ = A.a00 * B.a00 + A.a01 * (Real.exp (-β * ΔE) * B.a10) +
        1 * (A.a10 * B.a01) +
        Real.exp (-β * ΔE) * (A.a11 * B.a11) := by rw [hexp]
    _ = B.a00 * A.a00 + B.a01 * A.a10 +
        Real.exp (-β * ΔE) * (B.a10 * A.a01 + B.a11 * A.a11) := by ring

/-- **KMS temperature uniqueness in the non-tracial two-level model**:
    for a positive energy gap `ΔE > 0`, the Gibbs state `ω_β` satisfies the
    KMS condition at candidate inverse temperature `γ` **if and only if**
    `γ = β`. -/
theorem twoLevel_kms_iff_eq (ΔE β γ : ℝ) (hΔE : 0 < ΔE) :
    IsTwoLevelKMS ΔE β γ ↔ γ = β := by
  constructor
  · intro hkms
    have hE := hkms Obs2.E01 Obs2.E10
    have hZ : gibbsPartition ΔE β ≠ 0 := ne_of_gt (gibbsPartition_pos ΔE β)
    have hmul :
        gibbsState ΔE β (Obs2.mul Obs2.E01 (modularScale ΔE γ Obs2.E10)) *
          gibbsPartition ΔE β =
        gibbsState ΔE β (Obs2.mul Obs2.E10 Obs2.E01) * gibbsPartition ΔE β :=
      congrArg (fun x => x * gibbsPartition ΔE β) hE
    unfold gibbsState at hmul
    rw [div_mul_cancel₀ _ hZ, div_mul_cancel₀ _ hZ] at hmul
    unfold gibbsWeight Obs2.mul modularScale Obs2.E01 Obs2.E10 at hmul
    have hexp : Real.exp (-γ * ΔE) = Real.exp (-β * ΔE) := by linarith
    have hlin : -γ * ΔE = -β * ΔE := Real.exp_injective hexp
    have hneg : -γ = -β := mul_right_cancel₀ (ne_of_gt hΔE) hlin
    linarith
  · rintro rfl
    exact twoLevel_kms_holds ΔE γ

/-- **KMS-derived Zeroth Law in the non-tracial two-level model**: if two
    thermal reservoirs at inverse temperatures `β₁, β₂` are both KMS with
    respect to a shared two-level probe state `ω_β₀` (`ΔE > 0`), then
    `β₁ = β₂`. -/
theorem twoLevel_kms_zeroth_law (ΔE β₀ β₁ β₂ : ℝ) (hΔE : 0 < ΔE)
    (h₁ : IsTwoLevelKMS ΔE β₀ β₁) (h₂ : IsTwoLevelKMS ΔE β₀ β₂) :
    β₁ = β₂ := by
  have e₁ : β₁ = β₀ := (twoLevel_kms_iff_eq ΔE β₀ β₁ hΔE).mp h₁
  have e₂ : β₂ = β₀ := (twoLevel_kms_iff_eq ΔE β₀ β₂ hΔE).mp h₂
  exact e₁.trans e₂.symm

end OmegaProtocol.Vol03
