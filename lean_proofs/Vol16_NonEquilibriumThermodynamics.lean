import Mathlib
import OmegaUnifiedFoundation

/-!
# Conditional entropy-balance dynamics

Entropy, flux and production are real time-dependent functions. The balance law
is an explicit `HasDerivAt` hypothesis, not a zero-valued derivative definition.
Nonnegative production alone does not imply growing system entropy: an open
system can export entropy. Isolation at every time supplies the missing condition
for global monotonicity; strictly positive production gives strict monotonicity.

The balance law and production sign are model assumptions, not a microscopic
thermodynamic derivation. Absolute entropy nonnegativity requires initial data
and a time direction; it is not assumed for every real-time trajectory.
-/
namespace OmegaProtocol.Vol16
open OmegaProtocol

abbrev ThermoTime := ℝ

structure EntropyProcess where
  entropy : ℝ → ℝ
  flux : ℝ → ℝ
  production : ℝ → ℝ
  entropy_hasDerivAt : ∀ t, HasDerivAt entropy (flux t + production t) t
  production_nonneg : ∀ t, 0 ≤ production t

def Entropy (sys : EntropyProcess) : ThermoTime → ℝ := sys.entropy
def EntropyFlux (sys : EntropyProcess) : ThermoTime → ℝ := sys.flux
def EntropyProduction (sys : EntropyProcess) : ThermoTime → ℝ := sys.production
noncomputable def TimeDeriv (f : ThermoTime → ℝ) (t : ThermoTime) : ℝ := deriv f t

def IsIsolated (sys : EntropyProcess) : Prop := ∀ t, EntropyFlux sys t = 0

theorem entropy_balance (sys : EntropyProcess) (t : ThermoTime) :
    TimeDeriv (Entropy sys) t = EntropyFlux sys t + EntropyProduction sys t :=
  (sys.entropy_hasDerivAt t).deriv

theorem entropy_production_nonneg (sys : EntropyProcess) (t : ThermoTime) :
    0 ≤ EntropyProduction sys t := sys.production_nonneg t

/-- Pointwise rate statement; global monotonicity additionally needs isolation
    at every time, as stated in `isolated_entropy_monotone`. -/
theorem isolated_entropy_nondecreasing (sys : EntropyProcess) (t : ThermoTime)
    (h_isolated : EntropyFlux sys t = 0) : 0 ≤ TimeDeriv (Entropy sys) t := by
  rw [entropy_balance, h_isolated, zero_add]
  exact entropy_production_nonneg sys t

theorem isolated_entropy_monotone (sys : EntropyProcess) (h : IsIsolated sys) :
    Monotone (Entropy sys) := by
  apply monotone_of_deriv_nonneg (fun t => (sys.entropy_hasDerivAt t).differentiableAt)
  intro t
  exact isolated_entropy_nondecreasing sys t (h t)

/-- Strict entropy growth is stronger than the nonnegative-production assumption. -/
theorem isolated_entropy_strictMono (sys : EntropyProcess) (h : IsIsolated sys)
    (hp : ∀ t, 0 < EntropyProduction sys t) : StrictMono (Entropy sys) := by
  apply strictMono_of_deriv_pos
  intro t
  change 0 < TimeDeriv (Entropy sys) t
  rw [entropy_balance, h t, zero_add]
  exact hp t

theorem entropy_nonneg_after_initial (sys : EntropyProcess) (h : IsIsolated sys)
    (t₀ t : ThermoTime) (h₀ : 0 ≤ Entropy sys t₀) (ht : t₀ ≤ t) :
    0 ≤ Entropy sys t := le_trans h₀ (isolated_entropy_monotone sys h ht)

def constantProcess (S : ℝ) : EntropyProcess :=
  { entropy := fun _ => S
    flux := fun _ => 0
    production := fun _ => 0
    entropy_hasDerivAt := fun t => by simpa using hasDerivAt_const t S
    production_nonneg := fun _ => le_rfl }

noncomputable def growingProcess : EntropyProcess :=
  { entropy := Real.exp
    flux := fun _ => 0
    production := Real.exp
    entropy_hasDerivAt := fun t => by simpa using Real.hasDerivAt_exp t
    production_nonneg := fun t => (Real.exp_pos t).le }

/-- A globally positive entropy history that decreases by exporting entropy. -/
noncomputable def exportingProcess : EntropyProcess :=
  { entropy := fun t => Real.exp (-t)
    flux := fun t => -Real.exp (-t)
    production := fun _ => 0
    entropy_hasDerivAt := fun t => by simpa using ((hasDerivAt_id t).neg).exp
    production_nonneg := fun _ => le_rfl }

theorem growingProcess_isolated : IsIsolated growingProcess := fun _ => rfl

theorem growingProcess_strictMono : StrictMono (Entropy growingProcess) :=
  isolated_entropy_strictMono growingProcess growingProcess_isolated Real.exp_pos

theorem exportingProcess_decreases (s t : ℝ) (h : s < t) :
    Entropy exportingProcess t < Entropy exportingProcess s :=
  Real.exp_lt_exp.mpr (neg_lt_neg h)

/-- A realizable counterexample: nonnegative production does not suffice for
    monotonic system entropy unless flux is controlled. -/
theorem nonnegative_production_not_enough : ¬ Monotone (Entropy exportingProcess) := by
  intro h
  have hle := h (show (0 : ℝ) ≤ 1 by norm_num)
  have hlt := exportingProcess_decreases 0 1 (by norm_num)
  linarith

end OmegaProtocol.Vol16
