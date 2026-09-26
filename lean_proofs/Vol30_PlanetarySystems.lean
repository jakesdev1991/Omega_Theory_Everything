import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-!
# A positive-parameter Kepler period model

The period-radius relation is a chosen Newtonian two-body formula, not a
consequence of the zero-information Q-region model or a derivation from an ODE.
Positive gravitational constant and central mass exclude the former zero-denominator
model. Radius is a real argument, but physical claims require nonnegative or
positive radius explicitly. Lean's total square root also defines the function
outside that domain; no orbital interpretation is assigned there.

API change: period laws now take a `KeplerSystem` and radius sign hypotheses.
-/
namespace OmegaProtocol.Vol30
open OmegaProtocol

structure KeplerSystem where
  gravitationalConstant : ℝ
  centralMass : ℝ
  gravitationalConstant_pos : 0 < gravitationalConstant
  centralMass_pos : 0 < centralMass

noncomputable def keplerCoefficient (sys : KeplerSystem) : ℝ :=
  4 * Real.pi ^ 2 / (sys.gravitationalConstant * sys.centralMass)

theorem keplerCoefficient_pos (sys : KeplerSystem) : 0 < keplerCoefficient sys := by
  unfold keplerCoefficient
  exact div_pos (mul_pos (by norm_num) (pow_pos Real.pi_pos 2))
    (mul_pos sys.gravitationalConstant_pos sys.centralMass_pos)

noncomputable def OrbitalPeriod (sys : KeplerSystem) (a : ℝ) : ℝ :=
  Real.sqrt (keplerCoefficient sys * a ^ 3)

/-- The chosen formula satisfies the squared law on its physical domain. -/
theorem kepler_third_law (sys : KeplerSystem) (a : ℝ) (ha : 0 ≤ a) :
    OrbitalPeriod sys a ^ 2 = keplerCoefficient sys * a ^ 3 := by
  exact Real.sq_sqrt (mul_nonneg (keplerCoefficient_pos sys).le (pow_nonneg ha 3))

theorem orbital_period_pos (sys : KeplerSystem) (a : ℝ) (ha : 0 < a) :
    0 < OrbitalPeriod sys a :=
  Real.sqrt_pos.mpr (mul_pos (keplerCoefficient_pos sys) (pow_pos ha 3))

theorem orbital_period_zero (sys : KeplerSystem) : OrbitalPeriod sys 0 = 0 := by
  simp [OrbitalPeriod]

theorem orbital_period_zero_iff (sys : KeplerSystem) (a : ℝ) (ha : 0 ≤ a) :
    OrbitalPeriod sys a = 0 ↔ a = 0 := by
  constructor
  · intro hz
    rcases lt_or_eq_of_le ha with hp | he
    · have hpos := orbital_period_pos sys a hp
      rw [hz] at hpos
      exact False.elim (lt_irrefl _ hpos)
    · exact he.symm
  · rintro rfl
    exact orbital_period_zero sys

/-- Positive radii ensure the ratio has a nonzero denominator. -/
theorem kepler_ratio_constant (sys : KeplerSystem) (a₁ a₂ : ℝ)
    (ha1 : 0 < a₁) (ha2 : 0 < a₂) :
    OrbitalPeriod sys a₁ ^ 2 / a₁ ^ 3 = OrbitalPeriod sys a₂ ^ 2 / a₂ ^ 3 := by
  rw [kepler_third_law sys a₁ ha1.le, kepler_third_law sys a₂ ha2.le]
  field_simp [ne_of_gt ha1, ne_of_gt ha2] <;> ring

/-- A strictly larger nonnegative radius has a strictly longer period. -/
theorem orbital_period_strictMonoOn (sys : KeplerSystem) :
    StrictMonoOn (OrbitalPeriod sys) (Set.Ici 0) := by
  intro a ha b hb hab
  have ha0 : 0 ≤ a := ha
  have hb0 : 0 < b := lt_of_le_of_lt ha0 hab
  have hfactor : 0 < b ^ 2 + b * a + a ^ 2 :=
    add_pos_of_pos_of_nonneg
      (add_pos_of_pos_of_nonneg (pow_pos hb0 2) (mul_nonneg hb0.le ha0))
      (sq_nonneg a)
  have hdiff : 0 < b ^ 3 - a ^ 3 := by
    calc 0 < (b - a) * (b ^ 2 + b * a + a ^ 2) :=
          mul_pos (sub_pos.mpr hab) hfactor
      _ = b ^ 3 - a ^ 3 := by ring
  exact Real.sqrt_lt_sqrt
    (mul_nonneg (keplerCoefficient_pos sys).le (pow_nonneg ha0 3))
    (mul_lt_mul_of_pos_left (sub_pos.mp hdiff) (keplerCoefficient_pos sys))

theorem orbital_period_sq_nonneg (sys : KeplerSystem) (a : ℝ) :
    0 ≤ OrbitalPeriod sys a ^ 2 := sq_nonneg _

/-- Structural consistency of the separate legacy Q-region model only. -/
theorem bridge_vol30_metric_self_zero (R : QRegion) : d R R = 0 :=
  qregion_self_distance_zero R

/-- Structural consistency of the separate legacy Q-region model only. -/
theorem bridge_vol30_metric_nonneg (R₁ R₂ : QRegion) : d R₁ R₂ ≥ 0 :=
  distance_nonneg R₁ R₂

end OmegaProtocol.Vol30
