import Mathlib

/-!
# Information Physics

Algebraic energy bookkeeping under explicitly chosen Landauer-limit and
mass–energy formulas. This does not derive an intrinsic mass of information
or assert that every physical erasure process attains the Landauer bound.
The non-ideal erasure section adds explicit excess costs and aggregate
feasibility. It does not change the ideal `informationMass` used by the
phenomenological density/scale pipeline.

Genuinely proven theorems:
1. **Information-Mass-Energy Equivalence** — converting the mass equivalent of
   `I` bits back through Einstein's equation yields the exact Landauer erasure
   energy.
2. **Conservation of Total State Energy** — when bound mass converts into
   propagating information `ΔI`, the total energy remains strictly conserved.
3. **Positivity of Information Mass** — a positive number of bits at positive
   temperature carries strictly positive equivalent mass. This is the single
   fact that `DynamicPlanckScale` consumes from this module.

Model assumptions:
- `E_landauer = I · k_B · T · ln 2` is the idealized Landauer-limit energy
  assigned to erasing `I` bits at temperature `T`; general erasure costs need
  not equal this lower-limit value.
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

/-- Chosen Landauer-limit energy for erasing `I` bits. -/
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

/-- **Theorem 3: Positivity of Information Mass**
    A positive number of bits `I` at positive temperature has strictly positive
    equivalent mass `m_I = I · k_B · T · ln 2 / c²`. -/
theorem informationMass_pos (env : Environment) {I : ℝ} (hI : 0 < I) :
    0 < informationMass env I := by
  unfold informationMass landauerEnergy
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  exact div_pos (mul_pos (mul_pos (mul_pos hI env.hkB) env.hT) hlog) (pow_pos env.hc 2)

/-- Fixed-environment information mass is exactly a positive linear scale. -/
theorem informationMass_linear (env : Environment) (I : ℝ) :
    informationMass env I = I * informationMass env 1 := by
  unfold informationMass landauerEnergy
  ring

theorem informationMass_zero (env : Environment) : informationMass env 0 = 0 := by
  simp [informationMass, landauerEnergy]

theorem informationMass_add (env : Environment) (I J : ℝ) :
    informationMass env (I + J) = informationMass env I + informationMass env J := by
  unfold informationMass landauerEnergy
  ring

theorem informationMass_strictMono (env : Environment) :
    StrictMono (informationMass env) := by
  intro I J h
  rw [informationMass_linear env I, informationMass_linear env J]
  exact mul_lt_mul_of_pos_right h (informationMass_pos env zero_lt_one)

/-- No nonzero information quantity has zero equivalent mass in this model. -/
theorem informationMass_eq_zero_iff (env : Environment) (I : ℝ) :
    informationMass env I = 0 ↔ I = 0 := by
  constructor
  · intro h
    apply (informationMass_strictMono env).injective
    rw [h, informationMass_zero]
  · rintro rfl
    exact informationMass_zero env

theorem informationMass_nonneg (env : Environment) {I : ℝ} (hI : 0 ≤ I) :
    0 ≤ informationMass env I := by
  have h := (informationMass_strictMono env).monotone hI
  rwa [informationMass_zero] at h

/-- Conservation also holds with information already present, not just an
    initially empty information sector. -/
theorem conversion_preserves_existing_energy (env : Environment) (m I ΔI : ℝ) :
    totalSystemEnergy env (m - informationMass env ΔI) (I + ΔI) =
      totalSystemEnergy env m I := by
  have hmass : informationMass env ΔI * env.c ^ 2 = landauerEnergy env ΔI :=
    information_mass_energy_equivalence env ΔI
  have hadd : landauerEnergy env (I + ΔI) =
      landauerEnergy env I + landauerEnergy env ΔI := by
    unfold landauerEnergy
    ring
  unfold totalSystemEnergy einsteinEnergy
  rw [sub_mul, hmass, hadd]
  ring

/-- Conservation alone permits negative remaining mass. This extra condition
    is precisely what is needed for both output sectors to be nonnegative. -/
theorem conversion_outputs_nonneg_iff (env : Environment) (m I ΔI : ℝ)
    (hI : 0 ≤ I) (hΔI : 0 ≤ ΔI) :
    (0 ≤ m - informationMass env ΔI ∧ 0 ≤ I + ΔI) ↔
      informationMass env ΔI ≤ m := by
  constructor
  · intro h
    exact sub_nonneg.mp h.1
  · intro h
    exact ⟨sub_nonneg.mpr h, add_nonneg hI hΔI⟩

/-! ## Non-ideal erasure accounting

The chosen Landauer formula remains an ideal lower-limit model. An `Erasure`
adds explicit nonnegative excess energy; this assumes, rather than derives,
the lower bound. Quantities of information are nonnegative reals, not bitstrings.
A batch shares one fixed environment. No microscopic protocol or heat reservoir
is modeled. In particular, zero erased information need not imply zero overhead.
-/

structure Erasure where
  bits : ℝ
  excess : ℝ
  bits_nonneg : 0 ≤ bits
  excess_nonneg : 0 ≤ excess

noncomputable def erasureCost (env : Environment) (job : Erasure) : ℝ :=
  landauerEnergy env job.bits + job.excess

def idealErasure (bits : ℝ) (h : 0 ≤ bits) : Erasure :=
  ⟨bits, 0, h, le_rfl⟩

theorem landauerEnergy_pos (env : Environment) {bits : ℝ} (h : 0 < bits) :
    0 < landauerEnergy env bits := by
  exact mul_pos (mul_pos (mul_pos h env.hkB) env.hT) (Real.log_pos (by norm_num))

theorem landauerEnergy_nonneg (env : Environment) {bits : ℝ} (h : 0 ≤ bits) :
    0 ≤ landauerEnergy env bits := by
  exact mul_nonneg (mul_nonneg (mul_nonneg h env.hkB.le) env.hT.le)
    (Real.log_pos (by norm_num)).le

theorem landauerEnergy_eq_zero_iff (env : Environment) (bits : ℝ) :
    landauerEnergy env bits = 0 ↔ bits = 0 := by
  have hscale : 0 < env.kB * env.T * Real.log 2 :=
    mul_pos (mul_pos env.hkB env.hT) (Real.log_pos (by norm_num))
  have hlinear : landauerEnergy env bits = bits * (env.kB * env.T * Real.log 2) := by
    unfold landauerEnergy
    ring
  rw [hlinear]
  constructor
  · intro h
    exact (mul_eq_zero.mp h).resolve_right (ne_of_gt hscale)
  · rintro rfl
    simp

theorem erasureCost_lower_bound (env : Environment) (job : Erasure) :
    landauerEnergy env job.bits ≤ erasureCost env job := by
  unfold erasureCost
  linarith [job.excess_nonneg]

theorem erasureCost_nonneg (env : Environment) (job : Erasure) :
    0 ≤ erasureCost env job :=
  add_nonneg (landauerEnergy_nonneg env job.bits_nonneg) job.excess_nonneg

/-- Saturation is a special case, not a property of all admitted processes. -/
theorem erasureCost_saturates_iff (env : Environment) (job : Erasure) :
    erasureCost env job = landauerEnergy env job.bits ↔ job.excess = 0 := by
  unfold erasureCost
  constructor <;> intro h <;> linarith

theorem erasureCost_strict_iff (env : Environment) (job : Erasure) :
    landauerEnergy env job.bits < erasureCost env job ↔ 0 < job.excess := by
  unfold erasureCost
  constructor <;> intro h <;> linarith

theorem erasureCost_eq_zero_iff (env : Environment) (job : Erasure) :
    erasureCost env job = 0 ↔ job.bits = 0 ∧ job.excess = 0 := by
  constructor
  · intro h
    have hn := landauerEnergy_nonneg env job.bits_nonneg
    have he := job.excess_nonneg
    unfold erasureCost at h
    have hz : landauerEnergy env job.bits = 0 := by linarith
    exact ⟨(landauerEnergy_eq_zero_iff env job.bits).mp hz, by linarith⟩
  · rintro ⟨hb, he⟩
    simp [erasureCost, landauerEnergy, hb, he]

theorem idealErasure_cost (env : Environment) (bits : ℝ) (h : 0 ≤ bits) :
    erasureCost env (idealErasure bits h) = landauerEnergy env bits := by
  simp [erasureCost, idealErasure]

/-- One erased bit with a positive unit of excess cost: a non-saturating witness. -/
def lossyBit : Erasure := ⟨1, 1, by norm_num, by norm_num⟩

theorem not_every_erasure_is_ideal (env : Environment) :
    ¬ (∀ job : Erasure, erasureCost env job = landauerEnergy env job.bits) := by
  intro h
  have hz := (erasureCost_saturates_iff env lossyBit).mp (h lossyBit)
  norm_num [lossyBit] at hz

def batchBits : List Erasure → ℝ
  | [] => 0
  | job :: jobs => job.bits + batchBits jobs

def batchExcess : List Erasure → ℝ
  | [] => 0
  | job :: jobs => job.excess + batchExcess jobs

noncomputable def batchCost (env : Environment) : List Erasure → ℝ
  | [] => 0
  | job :: jobs => erasureCost env job + batchCost env jobs

theorem batchExcess_nonneg (jobs : List Erasure) : 0 ≤ batchExcess jobs := by
  induction jobs with
  | nil => exact le_rfl
  | cons job jobs ih => exact add_nonneg job.excess_nonneg ih

theorem batchCost_nonneg (env : Environment) (jobs : List Erasure) :
    0 ≤ batchCost env jobs := by
  induction jobs with
  | nil => exact le_rfl
  | cons job jobs ih => exact add_nonneg (erasureCost_nonneg env job) ih

/-- All batch overhead is explicit; it cannot disappear by grouping erasures. -/
theorem batchCost_decomposition (env : Environment) (jobs : List Erasure) :
    batchCost env jobs = landauerEnergy env (batchBits jobs) + batchExcess jobs := by
  induction jobs with
  | nil => simp [batchCost, batchBits, batchExcess, landauerEnergy]
  | cons job jobs ih =>
      simp only [batchCost, batchBits, batchExcess, ih, erasureCost, landauerEnergy]
      ring

theorem batchExcess_eq_zero_iff (jobs : List Erasure) :
    batchExcess jobs = 0 ↔ ∀ job ∈ jobs, job.excess = 0 := by
  induction jobs with
  | nil => simp [batchExcess]
  | cons job jobs ih =>
      change job.excess + batchExcess jobs = 0 ↔ _
      constructor
      · intro h
        have hp := job.excess_nonneg
        have ht := batchExcess_nonneg jobs
        have hjob : job.excess = 0 := by linarith
        have htail : batchExcess jobs = 0 := by linarith
        intro other hm
        rcases List.mem_cons.mp hm with heq | hin
        · subst other
          exact hjob
        · exact ih.mp htail other hin
      · intro h
        have hjob : job.excess = 0 := h job (by simp)
        have htail : batchExcess jobs = 0 := ih.mpr (by
          intro other hm
          exact h other (by simp only [List.mem_cons]; exact Or.inr hm))
        rw [hjob, htail]
        norm_num

theorem batchCost_lower_bound (env : Environment) (jobs : List Erasure) :
    landauerEnergy env (batchBits jobs) ≤ batchCost env jobs := by
  rw [batchCost_decomposition]
  linarith [batchExcess_nonneg jobs]

/-- A batch attains the ideal bound exactly when every member does. -/
theorem batchCost_saturates_iff (env : Environment) (jobs : List Erasure) :
    batchCost env jobs = landauerEnergy env (batchBits jobs) ↔
      ∀ job ∈ jobs, job.excess = 0 := by
  rw [batchCost_decomposition, ← batchExcess_eq_zero_iff]
  constructor <;> intro h <;> linarith

theorem batchCost_append (env : Environment) (xs ys : List Erasure) :
    batchCost env (xs ++ ys) = batchCost env xs + batchCost env ys := by
  induction xs with
  | nil => simp [batchCost]
  | cons job jobs ih =>
      simp only [List.cons_append, batchCost, ih]
      ring

/-- Algebraic residual after paying ALL ideal and excess costs from bound mass.
    This is not a physical conversion mechanism. Negative residuals are possible
    unless the affordability condition below holds. -/
noncomputable def remainingMass (env : Environment) (mass : ℝ) (jobs : List Erasure) : ℝ :=
  mass - batchCost env jobs / env.c ^ 2

/-- Conservation includes spent erasure energy, not just the surviving mass. -/
theorem erasure_energy_bookkeeping (env : Environment) (mass : ℝ) (jobs : List Erasure) :
    einsteinEnergy env (remainingMass env mass jobs) + batchCost env jobs =
      einsteinEnergy env mass := by
  unfold remainingMass einsteinEnergy
  rw [sub_mul, div_mul_cancel₀ _ (ne_of_gt (pow_pos env.hc 2))]
  ring

/-- Exact aggregate affordability; conservation alone is not feasibility. -/
theorem batch_affordable_iff (env : Environment) (mass : ℝ) (jobs : List Erasure) :
    0 ≤ remainingMass env mass jobs ↔ batchCost env jobs ≤ einsteinEnergy env mass := by
  unfold remainingMass einsteinEnergy
  rw [sub_nonneg, div_le_iff₀ (pow_pos env.hc 2)]

/-- Paying concatenated batches agrees with threading the residual mass. -/
theorem remainingMass_append (env : Environment) (mass : ℝ) (xs ys : List Erasure) :
    remainingMass env mass (xs ++ ys) = remainingMass env (remainingMass env mass xs) ys := by
  unfold remainingMass
  rw [batchCost_append]
  ring

/-- No intermediate prefix runs out of mass if the complete batch is affordable.
    Nonnegative costs are essential: this does not allow future rebates to fund
    an earlier unaffordable operation. -/
theorem affordable_batch_has_affordable_prefix (env : Environment) (mass : ℝ)
    (xs ys : List Erasure) (h : 0 ≤ remainingMass env mass (xs ++ ys)) :
    0 ≤ remainingMass env mass xs := by
  apply (batch_affordable_iff env mass xs).mpr
  have htotal := (batch_affordable_iff env mass (xs ++ ys)).mp h
  rw [batchCost_append] at htotal
  have htail := batchCost_nonneg env ys
  linarith

end InformationPhysics
