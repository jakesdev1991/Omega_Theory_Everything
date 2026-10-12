import Mathlib

/-!
# CBwK budget pacer

Compatibility statements for the tactic harness and a bounded aggregate ledger.
Dimensions record actual allocation against budget. Ledger transitions construct
checked balance proofs; the legacy headroom-only API remains arithmetic rather
than refund authorization. No claim is made about concurrent host execution.
-/

namespace CBwK

structure Dimension where
  budget : Nat
  allocated : Nat
  inv : allocated ≤ budget

structure System where
  dimensions : Nat → Dimension

/-- A system's dimension invariant is preserved when the dimension is read. -/
theorem system_invariant_preservation (sys : System) (i : Nat) :
    (sys.dimensions i).allocated ≤ (sys.dimensions i).budget := by
  exact (sys.dimensions i).inv

structure PacerState where
  headroom : Nat

structure Reserved (state : PacerState) (requested : Nat)
    (admissible : requested ≤ state.headroom) where
  cost : Nat
  bound : cost ≤ state.headroom
  matches_request : cost = requested

/-- Reserve only an admissible amount of headroom. -/
def reserve (state : PacerState) (cost : Nat)
    (admissible : cost ≤ state.headroom) : Reserved state cost admissible :=
  { cost := cost, bound := admissible, matches_request := rfl }

theorem no_budget_overrun (state : PacerState) (cost : Nat)
    (admissible : cost ≤ state.headroom) :
    (reserve state cost admissible).cost ≤ state.headroom := by
  exact (reserve state cost admissible).bound

/-- Legacy arithmetic helper with no refund bound. Use `settleLedger` for
    capacity-bounded accounting; this helper does not implement authorization. -/
def settle (state : PacerState) (reclaimed : Nat) : PacerState :=
  { headroom := state.headroom + reclaimed }

theorem settle_headroom_nondecreasing (state : PacerState) (reclaimed : Nat) :
    state.headroom ≤ (settle state reclaimed).headroom := by
  dsimp [settle]
  omega

/-- Spending changes the state; `reserve` alone is only a certificate. -/
def spend (state : PacerState) (cost : Nat) (_ : cost ≤ state.headroom) : PacerState :=
  ⟨state.headroom - cost⟩

theorem spend_conserves_headroom (state : PacerState) (cost : Nat)
    (h : cost ≤ state.headroom) :
    (spend state cost h).headroom + cost = state.headroom := by
  dsimp [spend]
  omega

/-- Full reclamation restores exactly the previous balance. This is an
    accounting equation, not a guarantee that reclamation is authorized. -/
theorem settle_after_spend (state : PacerState) (cost : Nat)
    (h : cost ≤ state.headroom) :
    settle (spend state cost h) cost = state := by
  cases state with
  | mk headroom =>
      have he : headroom - cost + cost = headroom := Nat.sub_add_cancel h
      simpa only [settle, spend, he]

/-- A run rejects the first unaffordable cost, rather than using saturating
    subtraction to hide an overrun. Each later cost uses the updated balance. -/
def runCosts (headroom : Nat) : List Nat → Option Nat
  | [] => some headroom
  | cost :: costs =>
      if cost ≤ headroom then runCosts (headroom - cost) costs else none

/-- Exact characterization of success: cumulative cost fits AND the remaining
    balance is the original balance minus the entire cost. -/
theorem runCosts_eq_some_iff (costs : List Nat) (headroom remaining : Nat) :
    runCosts headroom costs = some remaining ↔
      costs.sum ≤ headroom ∧ remaining = headroom - costs.sum := by
  induction costs generalizing headroom with
  | nil => simp [runCosts, eq_comm]
  | cons cost costs ih =>
      simp only [runCosts, List.sum_cons]
      split_ifs with h
      · rw [ih]
        omega
      · constructor
        · intro hbad
          cases hbad
        · rintro ⟨hbound, _⟩
          omega

/-- Safety over arbitrarily many reservations, not just one call. -/
theorem runCosts_conservation (costs : List Nat) (headroom remaining : Nat)
    (h : runCosts headroom costs = some remaining) :
    costs.sum + remaining = headroom := by
  obtain ⟨hbound, hremaining⟩ := (runCosts_eq_some_iff costs headroom remaining).mp h
  omega

/-- Regression: each request fits the original balance, but both do not. -/
example : runCosts 10 [6, 6] = none := by decide
example : runCosts 10 [6, 4] = some 0 := by decide
example : runCosts 0 [0, 0] = some 0 := by decide

/-! ## Capacity-bounded aggregate accounting

The legacy `PacerState`/`settle` API above is only arithmetic and cannot bound
refunds. New integrations should use this ledger: funds are available, reserved,
or spent, and transitions preserve capacity. Settlement may release or charge
only outstanding reserves, so spent funds cannot be refunded through this API.

This is aggregate, sequential accounting, not ticket ownership, authentication,
or concurrent storage. A caller must thread the returned state; copying or
rolling back an old state is outside the model. Reservations are not individually
identified, so a release is not an authorization proof for a particular request.
-/

structure Ledger where
  capacity : Nat
  available : Nat
  reserved : Nat
  spent : Nat
  balance : available + reserved + spent = capacity

def startLedger (capacity : Nat) : Ledger :=
  { capacity := capacity
    available := capacity
    reserved := 0
    spent := 0
    balance := by omega }

/-- The dimension view counts held plus spent funds as allocated, rather than
    proving budget ≤ itself. This projection is constructed from ledger balance. -/
def dimensionOfLedger (state : Ledger) : Dimension :=
  { budget := state.capacity
    allocated := state.reserved + state.spent
    inv := by have hs := state.balance; omega }

/-- Move affordable available funds into the outstanding-reserve bucket. -/
def reserveLedger (state : Ledger) (cost : Nat) : Option Ledger :=
  if h : cost ≤ state.available then
    some { capacity := state.capacity
           available := state.available - cost
           reserved := state.reserved + cost
           spent := state.spent
           balance := by have hs := state.balance; omega }
  else none

/-- Settle part or all of outstanding reserves; reject aggregate over-settlement.
    `charged` becomes spent and `released` returns to available funds. -/
def settleLedger (state : Ledger) (charged released : Nat) : Option Ledger :=
  if h : charged + released ≤ state.reserved then
    some { capacity := state.capacity
           available := state.available + released
           reserved := state.reserved - (charged + released)
           spent := state.spent + charged
           balance := by have hs := state.balance; omega }
  else none

theorem ledger_available_bounded (state : Ledger) : state.available ≤ state.capacity := by
  have h := state.balance
  omega

theorem reserveLedger_accepts_iff (state : Ledger) (cost : Nat) :
    (∃ out, reserveLedger state cost = some out) ↔ cost ≤ state.available := by
  by_cases h : cost ≤ state.available <;> simp [reserveLedger, h]

theorem settleLedger_accepts_iff (state : Ledger) (charged released : Nat) :
    (∃ out, settleLedger state charged released = some out) ↔
      charged + released ≤ state.reserved := by
  by_cases h : charged + released ≤ state.reserved <;> simp [settleLedger, h]

/-- Exact transition effects, including preservation of the capacity parameter. -/
theorem reserveLedger_effect (state out : Ledger) (cost : Nat)
    (h : reserveLedger state cost = some out) :
    out.capacity = state.capacity ∧ out.available + cost = state.available ∧
      out.reserved = state.reserved + cost ∧ out.spent = state.spent := by
  unfold reserveLedger at h
  split_ifs at h with hc
  · simp only [Option.some.injEq] at h
    subst out
    exact ⟨rfl, by dsimp; omega, rfl, rfl⟩

theorem settleLedger_effect (state out : Ledger) (charged released : Nat)
    (h : settleLedger state charged released = some out) :
    out.capacity = state.capacity ∧ out.available = state.available + released ∧
      out.spent = state.spent + charged ∧ out.reserved + charged + released = state.reserved := by
  unfold settleLedger at h
  split_ifs at h with hc
  · simp only [Option.some.injEq] at h
    subst out
    exact ⟨rfl, rfl, rfl, by dsimp; omega⟩

theorem settleLedger_rejects_overdraw (state : Ledger) (charged released : Nat)
    (h : state.reserved < charged + released) : settleLedger state charged released = none := by
  simp [settleLedger, not_le.mpr h]

/-- A successful release is bounded by pre-existing outstanding reserves. -/
theorem ledger_no_over_refund (state out : Ledger) (charged released : Nat)
    (h : settleLedger state charged released = some out) : released ≤ state.reserved := by
  have hb := (settleLedger_accepts_iff state charged released).mp ⟨out, h⟩
  omega

/-- After all reserves have been released, another positive release is rejected.
    This concerns the returned state, not a reused or rolled-back snapshot. -/
theorem ledger_cannot_refund_after_full_release (state out : Ledger)
    (h : settleLedger state 0 state.reserved = some out) (released : Nat) (hr : 0 < released) :
    settleLedger out 0 released = none := by
  have he := (settleLedger_effect state out 0 state.reserved h).2.2.2
  apply settleLedger_rejects_overdraw
  omega

/-- Spent funds only increase during settlement; releasing reserves cannot undo a charge. -/
theorem ledger_spent_nondecreasing (state out : Ledger) (charged released : Nat)
    (h : settleLedger state charged released = some out) : state.spent ≤ out.spent := by
  have he := (settleLedger_effect state out charged released h).2.2.1
  omega

/-- Operations accepted by the bounded accounting interface. -/
inductive LedgerAction
  | reserve (cost : Nat)
  | settle (charged released : Nat)
  deriving Repr

def applyLedgerAction (state : Ledger) : LedgerAction → Option Ledger
  | .reserve cost => reserveLedger state cost
  | .settle charged released => settleLedger state charged released

/-- Pure sequential execution, aborting on the first rejected operation.
    This does not implement transactional rollback or persistent storage. -/
def runLedger (state : Ledger) : List LedgerAction → Option Ledger
  | [] => some state
  | action :: actions =>
      (applyLedgerAction state action).bind fun next => runLedger next actions

theorem ledger_action_preserves_accounting (state out : Ledger) (action : LedgerAction)
    (h : applyLedgerAction state action = some out) :
    out.capacity = state.capacity ∧ state.spent ≤ out.spent := by
  cases action with
  | reserve cost =>
      have he := reserveLedger_effect state out cost h
      exact ⟨he.1, by simpa only [he.2.2.2] using (le_refl state.spent)⟩
  | settle charged released =>
      exact ⟨(settleLedger_effect state out charged released h).1,
        ledger_spent_nondecreasing state out charged released h⟩

/-- Any successful finite sequence preserves capacity and never unspends charges. -/
theorem ledger_trace_preserves_accounting (actions : List LedgerAction) (state out : Ledger)
    (h : runLedger state actions = some out) :
    out.capacity = state.capacity ∧ state.spent ≤ out.spent := by
  induction actions generalizing state with
  | nil =>
      simp only [runLedger, Option.some.injEq] at h
      subst out
      exact ⟨rfl, le_refl _⟩
  | cons action actions ih =>
      simp only [runLedger] at h
      cases hs : applyLedgerAction state action with
      | none => simp [hs] at h
      | some next =>
          have hr : runLedger next actions = some out := by simpa [hs] using h
          have htail := ih next hr
          have hstep := ledger_action_preserves_accounting state next action hs
          exact ⟨htail.1.trans hstep.1, le_trans hstep.2 htail.2⟩

/-- Exact conservation against the original capacity, across reserves and settlements. -/
theorem ledger_trace_conservation (capacity : Nat) (actions : List LedgerAction) (out : Ledger)
    (h : runLedger (startLedger capacity) actions = some out) :
    out.available + out.reserved + out.spent = capacity := by
  have hc := (ledger_trace_preserves_accounting actions (startLedger capacity) out h).1
  exact out.balance.trans hc

end CBwK
