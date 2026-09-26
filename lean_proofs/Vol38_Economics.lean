import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-!
# A non-degenerate linear market

Supply increases from zero; demand decreases from a positive intercept. Their
slopes are explicitly positive. We derive a unique positive equilibrium and
positive traded quantity. These are results about chosen linear curves, not
general-equilibrium theory or economic laws derived from Omega.

Curves are total on real prices. Negative prices and demand past the choke price
are algebraic extensions, not physical markets. Equilibrium lies strictly between
zero and the choke price. API change: all market quantities take `LinearMarket`.
-/
namespace OmegaProtocol.Vol38
open OmegaProtocol

structure LinearMarket where
  demandIntercept : ℝ
  supplySlope : ℝ
  demandSlope : ℝ
  demandIntercept_pos : 0 < demandIntercept
  supplySlope_pos : 0 < supplySlope
  demandSlope_pos : 0 < demandSlope

def Supply (market : LinearMarket) (p : ℝ) : ℝ := market.supplySlope * p
def Demand (market : LinearMarket) (p : ℝ) : ℝ :=
  market.demandIntercept - market.demandSlope * p

def ExcessDemand (market : LinearMarket) (p : ℝ) : ℝ :=
  Demand market p - Supply market p

noncomputable def equilibrium_price (market : LinearMarket) : ℝ :=
  market.demandIntercept / (market.supplySlope + market.demandSlope)

theorem total_slope_pos (market : LinearMarket) :
    0 < market.supplySlope + market.demandSlope :=
  add_pos market.supplySlope_pos market.demandSlope_pos

theorem equilibrium_price_pos (market : LinearMarket) : 0 < equilibrium_price market :=
  div_pos market.demandIntercept_pos (total_slope_pos market)

theorem equilibrium_price_balance (market : LinearMarket) :
    (market.supplySlope + market.demandSlope) * equilibrium_price market =
      market.demandIntercept := by
  unfold equilibrium_price
  field_simp [ne_of_gt (total_slope_pos market)] <;> ring

theorem market_clearing (market : LinearMarket) :
    Supply market (equilibrium_price market) = Demand market (equilibrium_price market) := by
  have h := equilibrium_price_balance market
  unfold Supply Demand
  linarith

theorem equilibrium_excess_demand_zero (market : LinearMarket) :
    ExcessDemand market (equilibrium_price market) = 0 := by
  unfold ExcessDemand
  rw [← market_clearing]
  exact sub_self _

theorem excessDemand_strictAnti (market : LinearMarket) :
    StrictAnti (ExcessDemand market) := by
  intro a b hab
  have hs := mul_lt_mul_of_pos_left hab market.supplySlope_pos
  have hd := mul_lt_mul_of_pos_left hab market.demandSlope_pos
  unfold ExcessDemand Demand Supply
  linarith

/-- Clearing is equivalent to the computed equilibrium, not true at every price. -/
theorem market_clearing_iff (market : LinearMarket) (p : ℝ) :
    Supply market p = Demand market p ↔ p = equilibrium_price market := by
  constructor
  · intro h
    have hz : ExcessDemand market p = 0 := by
      unfold ExcessDemand
      rw [← h]
      exact sub_self _
    exact (excessDemand_strictAnti market).injective
      (hz.trans (equilibrium_excess_demand_zero market).symm)
  · rintro rfl
    exact market_clearing market

theorem market_equilibrium_exists_unique (market : LinearMarket) :
    ∃! p : ℝ, Supply market p = Demand market p :=
  ⟨equilibrium_price market, market_clearing market,
    fun p hp => (market_clearing_iff market p).mp hp⟩

theorem equilibrium_quantity_pos (market : LinearMarket) :
    0 < Supply market (equilibrium_price market) ∧
    0 < Demand market (equilibrium_price market) := by
  have h : 0 < Supply market (equilibrium_price market) :=
    mul_pos market.supplySlope_pos (equilibrium_price_pos market)
  exact ⟨h, (market_clearing market) ▸ h⟩

/-- The equilibrium occurs before the linear demand curve becomes negative. -/
theorem equilibrium_below_choke_price (market : LinearMarket) :
    equilibrium_price market < market.demandIntercept / market.demandSlope := by
  exact div_lt_div_of_pos_left market.demandIntercept_pos market.demandSlope_pos
    (by linarith [market.supplySlope_pos])

/-- Prices below/above equilibrium have positive/negative excess demand. -/
theorem excess_demand_sign (market : LinearMarket) (p : ℝ) :
    (p < equilibrium_price market → 0 < ExcessDemand market p) ∧
    (equilibrium_price market < p → ExcessDemand market p < 0) := by
  constructor
  · intro hp
    have h := excessDemand_strictAnti market hp
    rwa [equilibrium_excess_demand_zero] at h
  · intro hp
    have h := excessDemand_strictAnti market hp
    rwa [equilibrium_excess_demand_zero] at h

/-- Legacy export name; only a consequence of the linear-market definitions. -/
theorem economics_from_omega (market : LinearMarket) :
    ExcessDemand market (equilibrium_price market) = 0 :=
  equilibrium_excess_demand_zero market

/-- Structural consistency of the separate zero-information model only. -/
theorem bridge_vol38_entropy_nonneg (R : QRegion) : vonNeumannEntropy R ≥ 0 :=
  monotonicity_lemma R

/-- Structural consistency of the separate zero-information model only. -/
theorem bridge_vol38_coupling_symm (R₁ R₂ : QRegion) : Φ R₁ R₂ = Φ R₂ R₁ :=
  Φ_symm R₁ R₂

end OmegaProtocol.Vol38
