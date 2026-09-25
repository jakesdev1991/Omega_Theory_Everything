import Mathlib

/-!
# Information Physics

Formal bridge between Landauer's principle (information thermodynamics) and
Einstein's mass-energy equivalence.

Genuinely proven theorems:
1. **Information-Mass-Energy Equivalence** — converting the mass equivalent of
   `I` bits back through Einstein's equation yields the exact Landauer erasure
   energy.
2. **Conservation of Total State Energy** — when bound mass converts into
   propagating information `ΔI`, the total energy remains strictly conserved.

Model assumptions:
- `E_landauer = I · k_B · T · ln 2` (thermal cost of processing/erasing `I`
  bits at local temperature `T`).
- `E = m · c²` (Einstein mass-energy equivalence for bound mass `m`).
- Information mass is defined by the equivalence
  `m_I = E_landauer / c² = I · k_B · T · ln 2 / c²`.
-/

namespace InformationPhysics

/-- Environmental parameters and physical constants -/
structure Environment where
  c : ℝ -- Speed of light (c > 0)
  kB : ℝ -- Boltzmann constant (k_B > 0)
  T : ℝ -- Local temperature in Kelvin (T > 0)
  hc : c > 0
  hkB : kB > 0
  hT : T > 0

/-- Landauer's Principle: Thermal energy required to process/erase I bits of information -/
noncomputable def landauerEnergy (env : Environment) (I : ℝ) : ℝ :=
  I * env.kB * env.T * Real.log 2

/-- Einstein's Mass-Energy Equivalence: Energy of bound mass m -/
def einsteinEnergy (env : Environment) (m : ℝ) : ℝ :=
  m * env.c ^ 2

/-- Equivalent mass of bound information I at local temperature T -/
noncomputable def informationMass (env : Environment) (I : ℝ) : ℝ :=
  landauerEnergy env I / env.c ^ 2

/-- **Theorem 1: Information-Mass-Energy Equivalence**
    Converting the mass equivalent of I bits back through Einstein's equation
    yields the exact Landauer erasure energy. -/
theorem information_mass_energy_equivalence (env : Environment) (I : ℝ) :
    einsteinEnergy env (informationMass env I) = landauerEnergy env I := by
  dsimp [einsteinEnergy, informationMass]
  have hc2 : env.c ^ 2 ≠ 0 := by
    apply ne_of_gt
    exact pow_pos env.hc 2
  exact div_mul_cancel₀ (landauerEnergy env I) hc2

/-- Total System Energy: Sum of bound mass energy and unbound propagating information energy -/
noncomputable def totalSystemEnergy (env : Environment) (m_bound : ℝ) (I_propagating : ℝ) : ℝ :=
  einsteinEnergy env m_bound + landauerEnergy env I_propagating

/-- **Theorem 2: Conservation of Total State Energy**
    When bound mass converts into propagating information ΔI, total energy
    remains strictly conserved. -/
theorem energy_conservation_during_state_conversion (env : Environment) (m_initial : ℝ) (ΔI : ℝ) :
    totalSystemEnergy env (m_initial - informationMass env ΔI) ΔI =
      einsteinEnergy env m_initial := by
  dsimp [totalSystemEnergy, einsteinEnergy]
  rw [sub_mul]
  have hc2 : env.c ^ 2 ≠ 0 := by
    apply ne_of_gt
    exact pow_pos env.hc 2
  dsimp [informationMass]
  rw [div_mul_cancel₀ (landauerEnergy env ΔI) hc2]
  exact sub_add_cancel (m_initial * env.c ^ 2) (landauerEnergy env ΔI)

end InformationPhysics
