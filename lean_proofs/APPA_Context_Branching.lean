import Mathlib

/-!
# APPA context branching and declassification gate

This file is deliberately self-contained.  The security properties below are
proved from the data types and definitions; there are no trusted declarations or
untrusted proof-producing functions in this module.

The model is intentionally small: an orchestrator keeps its parent context,
while an untrusted read is evaluated in a child context with write and execute
capabilities disabled.  A derivative can return to the parent only after its
schema and provenance checks have been supplied.
-/

namespace APPA

/-- The integrity lattice used by the gate. -/
inductive IntegrityLabel
  | low
  | high
  deriving DecidableEq, Repr

namespace IntegrityLabel

instance : LE IntegrityLabel where
  le a b := match a, b with
    | .low, _ => True
    | .high, .high => True
    | .high, .low => False

theorem low_le_all (label : IntegrityLabel) : IntegrityLabel.low ≤ label := by
  cases label <;> trivial

theorem high_le_high : IntegrityLabel.high ≤ IntegrityLabel.high := by
  trivial

end IntegrityLabel

/-- Public gate-lattice names used by the REPL harness. -/
theorem low_le_all (label : IntegrityLabel) : IntegrityLabel.low ≤ label :=
  IntegrityLabel.low_le_all label

theorem high_le_high : IntegrityLabel.high ≤ IntegrityLabel.high :=
  IntegrityLabel.high_le_high

/-- Capabilities that can be carried by a context branch. -/
structure Capability where
  can_write : Bool
  can_execute : Bool
  deriving Repr

/-- The minimum context metadata needed by this proof. -/
structure Context where
  label : IntegrityLabel
  capability : Capability
  deriving Repr

/-- A read from an untrusted source is represented as data only. -/
structure UntrustedRead where
  parent : Context
  payload : String
  deriving Repr

/-- The branch created for an untrusted read.  The parent is copied, not
    mutated, and the child receives a least-privilege capability set. -/
structure ContextBranch where
  parent : Context
  child : Context
  read : UntrustedRead
  deriving Repr

def spawn_isolated (read : UntrustedRead) : ContextBranch :=
  { parent := read.parent
    child :=
      { label := IntegrityLabel.low
        capability := { can_write := false, can_execute := false } }
    read := read }

/-- A parent trajectory retains its original label after branching. -/
theorem parent_label_untainted (read : UntrustedRead) :
    (spawn_isolated read).parent.label = read.parent.label := by
  rfl

/-- Child branches cannot write to the host context. -/
theorem child_write_isolated (read : UntrustedRead) :
    (spawn_isolated read).child.capability.can_write = false := by
  rfl

/-- Child branches cannot execute capabilities. -/
theorem child_execute_isolated (read : UntrustedRead) :
    (spawn_isolated read).child.capability.can_execute = false := by
  rfl

/-- A derivative carries the checks made by the sanitizer. -/
structure SanitizedDerivative where
  payload : String
  schema_valid : Bool
  provenance_valid : Bool
  deriving Repr

/-- Conditions required before a derivative can cross the declassification
    boundary.  Keeping the conditions as a hypothesis makes the gate's trust
    boundary explicit instead of assuming it globally. -/
def GateCondition (derivative : SanitizedDerivative) : Prop :=
  derivative.schema_valid = true ∧ derivative.provenance_valid = true

structure DeclassifiedDerivative where
  payload : String
  schema_valid : Bool
  provenance_valid : Bool
  deriving Repr

/-- The gate copies only an already-validated derivative. -/
def declassification_gate
    (derivative : SanitizedDerivative)
    (condition : GateCondition derivative) : DeclassifiedDerivative :=
  { payload := derivative.payload
    schema_valid := derivative.schema_valid
    provenance_valid := derivative.provenance_valid }

/-- The gate preserves both structural checks. -/
theorem declassification_gate_bounded
    (derivative : SanitizedDerivative)
    (h_cond : GateCondition derivative) :
    let result := declassification_gate derivative h_cond
    result.schema_valid = true ∧ result.provenance_valid = true := by
  dsimp [declassification_gate]
  exact h_cond

/-- Complete branching security guarantee. -/
theorem appa_branching_security_guarantee
    (read : UntrustedRead)
    (derivative : SanitizedDerivative)
    (h_cond : GateCondition derivative) :
    (spawn_isolated read).parent.label = read.parent.label ∧
    (spawn_isolated read).child.capability.can_write = false ∧
    (spawn_isolated read).child.capability.can_execute = false ∧
    (declassification_gate derivative h_cond).schema_valid = true ∧
    (declassification_gate derivative h_cond).provenance_valid = true := by
  refine ⟨parent_label_untainted read, child_write_isolated read,
    child_execute_isolated read, ?_⟩
  exact declassification_gate_bounded derivative h_cond

end APPA

/-- Compatibility namespace for clients that keep all Omega protocol modules
    under the OmegaProtocol naming convention. -/
namespace OmegaProtocol.APPA

abbrev IntegrityLabel := _root_.APPA.IntegrityLabel
abbrev Capability := _root_.APPA.Capability
abbrev Context := _root_.APPA.Context
abbrev UntrustedRead := _root_.APPA.UntrustedRead
abbrev ContextBranch := _root_.APPA.ContextBranch
abbrev SanitizedDerivative := _root_.APPA.SanitizedDerivative
abbrev DeclassifiedDerivative := _root_.APPA.DeclassifiedDerivative
abbrev GateCondition := _root_.APPA.GateCondition

abbrev spawn_isolated := _root_.APPA.spawn_isolated
abbrev parent_label_untainted := _root_.APPA.parent_label_untainted
abbrev child_write_isolated := _root_.APPA.child_write_isolated
abbrev child_execute_isolated := _root_.APPA.child_execute_isolated
abbrev declassification_gate := _root_.APPA.declassification_gate
abbrev declassification_gate_bounded := _root_.APPA.declassification_gate_bounded
abbrev appa_branching_security_guarantee := _root_.APPA.appa_branching_security_guarantee

end OmegaProtocol.APPA
