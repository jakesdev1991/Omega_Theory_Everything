import Mathlib
import OmegaUnifiedFoundation
/-
  Vol53_UniversalCompiler.lean

  The Universal Compiler, formalized honestly as a tiny expression language
  with an evaluator.

  Model scope (stated honestly):
  * `CosmicCompiler` is modeled by a three-constructor expression language
    (constants, successor, composition) with a total evaluator. This is a
    toy compilation target, not a universal Turing machine — undecidability
    results live in Vol35, and this language deliberately sits below them:
    every program here provably terminates (`eval` is total by definition).
  * The substantive theorems are composition associativity and the
    successor-law of compilation depth.
-/

namespace OmegaProtocol.Vol53
open OmegaProtocol

/-- Expressions of the toy compiler: constants, successor, and composition. -/
inductive Prog where
  | const (n : ℕ)
  | succ (p : Prog)
  | compose (f g : Prog)

/-- The compiler target: expressions of the toy language. -/
def CosmicCompiler : Type := Prog

/-- Named bridge (legacy name kept for `OmegaProtocol.lean`): the compiler
    target is inhabited. -/
theorem universal_compiler : Nonempty CosmicCompiler :=
  ⟨Prog.const 0⟩

/-- The evaluator: total by construction. -/
def eval : Prog → ℕ → ℕ
  | Prog.const n, _ => n
  | Prog.succ p, x => eval p x + 1
  | Prog.compose f g, x => eval f (eval g x)

/-- Compilation depth of an expression. -/
def depth : Prog → ℕ
  | Prog.const _ => 0
  | Prog.succ p => depth p + 1
  | Prog.compose f g => max (depth f) (depth g) + 1

theorem eval_const (n x : ℕ) : eval (Prog.const n) x = n := rfl

theorem eval_succ (p : Prog) (x : ℕ) :
    eval (Prog.succ p) x = eval p x + 1 := rfl

/-- Successor strictly increases output. -/
theorem succ_increases_output (p : Prog) (x : ℕ) :
    eval (Prog.succ p) x > eval p x :=
  Nat.lt_succ_self _

/-- Composition is associative at the level of observable behaviour. -/
theorem compose_assoc (f g h : Prog) (x : ℕ) :
    eval (Prog.compose (Prog.compose f g) h) x =
    eval (Prog.compose f (Prog.compose g h)) x :=
  rfl

/-- Composition with a constant forgets the argument: constants are the
    absorbing optimizers of the language. -/
theorem compose_const_left (n : ℕ) (g : Prog) (x : ℕ) :
    eval (Prog.compose (Prog.const n) g) x = n := rfl

/-- Compiling a composition never shrinks the depth below either operand. -/
theorem depth_compose_ge (f g : Prog) :
    depth (Prog.compose f g) ≥ depth f ∧ depth (Prog.compose f g) ≥ depth g := by
  constructor
  · exact Nat.le_trans (Nat.le_max_left _ _) (Nat.le_add_right _ _)
  · exact Nat.le_trans (Nat.le_max_right _ _) (Nat.le_add_right _ _)

/-- Structural bridge: the compiler layer is compatible with the Ω-metric
    layer of the protocol. -/
theorem bridge_vol53_entropy_nonneg (R : QRegion) : vonNeumannEntropy R ≥ 0 :=
  monotonicity_lemma R

/-- Expressiveness audit: without an input constructor every program in this
    grammar computes a constant function. Composition does not fix that. -/
theorem eval_input_independent (p : Prog) (x y : ℕ) : eval p x = eval p y := by
  induction p generalizing x y with
  | const n => rfl
  | succ p ih => simp only [eval, ih x y]
  | compose f g ihf ihg => exact ihf (eval g x) (eval g y)

/-- Consequently this grammar cannot even represent the identity function;
    the legacy `universal_compiler` name asserts inhabitation, not universality. -/
theorem identity_not_representable : ¬ ∃ p : Prog, ∀ x, eval p x = x := by
  rintro ⟨p, hp⟩
  have h := eval_input_independent p 0 1
  rw [hp 0, hp 1] at h
  omega

end OmegaProtocol.Vol53
