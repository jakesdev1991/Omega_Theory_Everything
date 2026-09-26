import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol35_TheoryOfComputation.lean
  Formalization of Theory of Computation.

  Model scope (stated honestly):
  * Cantor's diagonal argument (no surjection `ℕ → (ℕ → Bool)`) is a
    genuine proof, kept under its legacy name for `OmegaProtocol.lean`.
  * `Machine` is an explicit toy computability semantics: a total step
    function over ℕ configurations. Bounded halting is decidable and a
    halting diagonal theorem holds: no enumeration of machines captures
    all halting behavior. This is NOT a Turing-machine model and proves
    no equivalence with any standard machine class.
  * Omega-Protocol mapping: machine states = Q-regions (0D), tape
    transitions = Φ (1D), state-space distance = Ω-metric (2D).
-/

namespace OmegaProtocol.Vol35
open OmegaProtocol

/-- THEOREM: Cantor's Diagonal Argument (GENUINE PROOF)
    There is no surjection from ℕ to (ℕ → Bool). -/
theorem cantor_diagonal : ¬ Function.Surjective (f : ℕ → ℕ → Bool) := by
  intro h_surj
  -- Construct the diagonal sequence d(n) = ¬f(n)(n)
  let d : ℕ → Bool := fun n => !f n n
  -- By surjectivity, there exists k such that f(k) = d
  obtain ⟨k, hk⟩ := h_surj d
  -- Evaluate at k: f(k)(k) = d(k) = ¬f(k)(k)
  have h_contra : f k k = !f k k := by
    have := congr_fun hk k
    exact this
  -- This is a contradiction: x ≠ ¬x
  cases h : f k k <;> simp_all

/-- Consistency bridge (VOL35): the Omega-metric `d` is reflexive (`d R R = 0`).
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `computation_from_omega` evoked. -/
theorem bridge_vol35_metric_self_zero (R : QRegion) : d R R = 0 := by
  exact qregion_self_distance_zero R

/-- Consistency bridge (VOL35): the von Neumann entropy is nonnegative.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `computation_entropy_nonneg` evoked. -/
theorem bridge_vol35_entropy_nonneg (R : QRegion) : vonNeumannEntropy R ≥ 0 := by
  exact monotonicity_lemma R

theorem cantor_diagonal_nontrivial :
    ∀ e : ℕ → (ℕ → Bool), ¬ Function.Surjective e := by
  intro e hs
  obtain ⟨k, hk⟩ := hs (fun n => !(e n n))
  have hkk : e k k = !(e k k) := congrFun hk k
  exact (Bool.eq_not_self (e k k)).mp hkk

-- ============================================================
-- EXPLICIT COMPUTABILITY SEMANTICS
-- A total step function over ℕ configurations, bounded-halt
-- decidability, and a halting diagonal theorem. This is a toy
-- semantics, not a Turing-machine equivalence proof.
-- ============================================================

/-- A machine is a total step function on ℕ configurations.
    `none` means "halted here"; `some c'` means "continue at c'". -/
structure Machine where
  step : ℕ → Option ℕ

/-- Run a machine from `c` for `k` steps. Halting propagates:
    once `none`, always `none`. -/
def run (M : Machine) (c : ℕ) : ℕ → Option ℕ
  | 0 => some c
  | k + 1 =>
    match run M c k with
    | none => none
    | some c' => M.step c'

theorem run_zero (M : Machine) (c : ℕ) : run M c 0 = some c := rfl

theorem run_succ (M : Machine) (c : ℕ) (k : ℕ) :
    run M c (k + 1) =
      match run M c k with
      | none => none
      | some c' => M.step c' := rfl

/-- One step from a live configuration applies the step function. -/
theorem run_succ_some (M : Machine) (c c' : ℕ) (k : ℕ)
    (h : run M c k = some c') :
    run M c (k + 1) = M.step c' := by
  rw [run_succ, h]

/-- Halting propagates forward: a halted run stays halted. -/
theorem run_none_stable (M : Machine) (c k : ℕ) (h : run M c k = none) :
    run M c (k + 1) = none := by
  rw [run_succ, h]

/-- A machine halts on `c` if some finite run reaches `none`. -/
def Halts (M : Machine) (c : ℕ) : Prop :=
  ∃ k, run M c k = none

/-- Halting, once achieved, persists at all later step counts. -/
theorem halts_stable (M : Machine) (c k : ℕ) (h : run M c k = none) :
    ∀ j, run M c (k + j) = none := by
  intro j
  induction j with
  | zero => simpa using h
  | succ j ih =>
    have hstep : k + (j + 1) = (k + j) + 1 := by omega
    rw [hstep]
    exact run_none_stable M c (k + j) ih

/-- Bounded halting is decidable: it is a bounded search over a
    decidable predicate. -/
theorem bounded_halting_decidable (M : Machine) (c K : ℕ) :
    Decidable (∃ k ∈ Finset.range (K + 1), run M c k = none) :=
  inferInstance

/-- The machine that halts immediately on every input. -/
def haltMachine : Machine := ⟨fun _ => none⟩

/-- The machine that loops forever on every input. -/
def loopMachine : Machine := ⟨fun c => some c⟩

theorem haltMachine_halts (c : ℕ) : Halts haltMachine c :=
  ⟨1, rfl⟩

theorem loopMachine_diverges (c : ℕ) : ¬ Halts loopMachine c := by
  rintro ⟨k, hk⟩
  have hstay : ∀ k, run loopMachine c k = some c := by
    intro k
    induction k with
    | zero => rfl
    | succ k ih =>
      rw [run_succ_some _ _ _ _ ih]
  rw [hstay k] at hk
  exact Option.noConfusion hk

/-- Diagonal machine: on input `c`, loop forever if `e c` halts on `c`,
    else halt immediately. Classical logic is used only to form the
    step function; the kernel still checks the proof. -/
noncomputable def diagMachine (e : ℕ → Machine) : Machine :=
  ⟨fun c => if Halts (e c) c then some c else none⟩

/-- Halting diagonal theorem: no enumeration of machines captures all
    halting behavior — the diagonal machine disagrees with every row
    on its own index. -/
theorem halts_diagonal (e : ℕ → Machine) :
    ∃ M : Machine, ∀ n, (Halts M n ↔ ¬ Halts (e n) n) := by
  refine ⟨diagMachine e, fun n => ?_⟩
  constructor
  · intro hM hn
    have hstay : ∀ k, run (diagMachine e) n k = some n := by
      intro k
      induction k with
      | zero => rfl
      | succ k ih =>
        rw [run_succ_some _ _ _ _ ih]
        show (if Halts (e n) n then some n else none) = some n
        exact if_pos hn
    obtain ⟨k, hk⟩ := hM
    rw [hstay k] at hk
    exact Option.noConfusion hk
  · intro hn
    refine ⟨1, ?_⟩
    have hone : (1 : ℕ) = 0 + 1 := rfl
    rw [hone, run_succ_some _ _ _ _ (run_zero _ _)]
    show (if Halts (e n) n then some n else none) = none
    exact if_neg hn

end OmegaProtocol.Vol35
