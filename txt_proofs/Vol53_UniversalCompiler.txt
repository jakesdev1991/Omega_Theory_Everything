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
  bridge_vonNeumannEntropy_nonneg R

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

-- ============================================================
-- INPUT-AWARE EXPRESSION LANGUAGE & VERIFIED STACK COMPILER
-- ============================================================
--
-- Above, `Prog` lacks an input constructor (`identity_not_representable`).
-- Here we extend the module with:
-- 1. An input-aware arithmetic expression language `Expr` (`var`, `const`,
--    `add`, `mul`) with total denotational semantics `evalExpr`, proving
--    constructive representability of the identity and all affine functions,
--    plus syntactic composition `subst` (`evalExpr_subst`, `subst_assoc`).
-- 2. A stack virtual machine (`Instr`, `execInstr`, `execCode`) and a
--    bytecode compiler `compile : Expr → List Instr`, proving the
--    McCarthy–Painter compiler correctness theorem (`compile_correct_stack`,
--    `compile_correct`).

/-- Input-aware arithmetic expression AST (`var` reads the runtime input `x`). -/
inductive Expr where
  | var
  | const (n : ℕ)
  | add (e₁ e₂ : Expr)
  | mul (e₁ e₂ : Expr)

/-- Total denotational semantics of `Expr` at input `x : ℕ`. -/
def evalExpr : Expr → ℕ → ℕ
  | .var, x => x
  | .const n, _ => n
  | .add e₁ e₂, x => evalExpr e₁ x + evalExpr e₂ x
  | .mul e₁ e₂, x => evalExpr e₁ x * evalExpr e₂ x

/-- **The identity function is representable** in `Expr` (witnessed by `.var`),
    resolving the expressiveness limitation of `Prog`. -/
theorem identity_representable : ∃ e : Expr, ∀ x, evalExpr e x = x :=
  ⟨Expr.var, fun _ => rfl⟩

/-- Every affine function `a * x + b` is representable in `Expr`. -/
theorem affine_representable (a b : ℕ) :
    ∃ e : Expr, ∀ x, evalExpr e x = a * x + b :=
  ⟨Expr.add (Expr.mul (Expr.const a) Expr.var) (Expr.const b), fun _ => rfl⟩

/-- Syntactic composition (substitution of `g` for `.var` in `f`). -/
def subst : Expr → Expr → Expr
  | .var, g => g
  | .const n, _ => .const n
  | .add e₁ e₂, g => .add (subst e₁ g) (subst e₂ g)
  | .mul e₁ e₂, g => .mul (subst e₁ g) (subst e₂ g)

/-- **Substitution semantics theorem**: evaluating `subst f g` at `x` equals
    evaluating `f` at `evalExpr g x`. -/
theorem evalExpr_subst (f g : Expr) (x : ℕ) :
    evalExpr (subst f g) x = evalExpr f (evalExpr g x) := by
  induction f with
  | var => rfl
  | const _ => rfl
  | add _ _ ih₁ ih₂ => simp only [subst, evalExpr, ih₁, ih₂]
  | mul _ _ ih₁ ih₂ => simp only [subst, evalExpr, ih₁, ih₂]

/-- Syntactic substitution is associative. -/
theorem subst_assoc (f g h : Expr) :
    subst (subst f g) h = subst f (subst g h) := by
  induction f with
  | var => rfl
  | const _ => rfl
  | add _ _ ih₁ ih₂ => simp only [subst, ih₁, ih₂]
  | mul _ _ ih₁ ih₂ => simp only [subst, ih₁, ih₂]

/-- Stack-machine instruction set for the target virtual machine. -/
inductive Instr where
  | pushVar
  | pushConst (n : ℕ)
  | add
  | mul

/-- Single-instruction step on `(input, stack)`. -/
def execInstr : Instr → ℕ → List ℕ → List ℕ
  | .pushVar, x, s => x :: s
  | .pushConst n, _, s => n :: s
  | .add, _, b :: a :: s => (a + b) :: s
  | .add, _, s => s
  | .mul, _, b :: a :: s => (a * b) :: s
  | .mul, _, s => s

/-- Multi-instruction execution over a bytecode sequence. -/
def execCode : List Instr → ℕ → List ℕ → List ℕ
  | [], _, s => s
  | i :: is, x, s => execCode is x (execInstr i x s)

/-- Code concatenation threads the stack sequentially through both segments. -/
theorem execCode_append (c₁ c₂ : List Instr) (x : ℕ) (s : List ℕ) :
    execCode (c₁ ++ c₂) x s = execCode c₂ x (execCode c₁ x s) := by
  induction c₁ generalizing s with
  | nil => rfl
  | cons i is ih =>
    simp only [List.cons_append, execCode]
    exact ih (execInstr i x s)

/-- Compiler from `Expr` to stack-machine bytecode `List Instr`. -/
def compile : Expr → List Instr
  | .var => [.pushVar]
  | .const n => [.pushConst n]
  | .add e₁ e₂ => compile e₁ ++ compile e₂ ++ [.add]
  | .mul e₁ e₂ => compile e₁ ++ compile e₂ ++ [.mul]

/-- **McCarthy–Painter compiler correctness theorem (general stack)**:
    executing `compile e` on input `x` and initial stack `s` pushes
    `evalExpr e x` onto `s` while preserving the caller's stack frame `s`. -/
theorem compile_correct_stack (e : Expr) (x : ℕ) (s : List ℕ) :
    execCode (compile e) x s = evalExpr e x :: s := by
  induction e generalizing s with
  | var => rfl
  | const _ => rfl
  | add e₁ _ ih₁ ih₂ =>
    simp only [compile, evalExpr, execCode_append]
    rw [ih₁ s, ih₂ (evalExpr e₁ x :: s)]
    rfl
  | mul e₁ _ ih₁ ih₂ =>
    simp only [compile, evalExpr, execCode_append]
    rw [ih₁ s, ih₂ (evalExpr e₁ x :: s)]
    rfl

/-- **Top-level compiler correctness**: executing `compile e` from the empty
    stack produces the singleton stack `[evalExpr e x]`. -/
theorem compile_correct (e : Expr) (x : ℕ) :
    execCode (compile e) x [] = [evalExpr e x] :=
  compile_correct_stack e x []

end OmegaProtocol.Vol53
