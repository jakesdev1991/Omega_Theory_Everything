import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol34_QuantumComputing.lean
  Formalization of Quantum Computing.

  Model scope (stated honestly):
  * Gates are modeled by the two-element Clifford fragment {I, H} with
    composition and adjoint given by explicit tables: H is self-adjoint and
    involutive, I is the identity. This is a genuine (toy) group table —
    unlike a `Unit`-valued model, each equation here is a case-checked fact
    about an explicit algebra.
  * Substantive theorems: unitarity of H (H†H = I), associativity of gate
    composition, and existence of inverses for every gate.
  * Omega-Protocol mapping: qubits = Q-regions (0D), gates = Φ operations
    (1D), circuit depth = Ω-metric (2D).
-/

namespace OmegaProtocol.Vol34
open OmegaProtocol

/-- The toy single-qubit gate set {I, H}. -/
inductive QGate
  | I
  | H

/-- Legacy carrier name kept for `OmegaProtocol.lean`. -/
def QuantumGate : Type := QGate

/-- Gate adjoint: both I and H are self-adjoint. -/
def GateAdjoint : QuantumGate → QuantumGate
  | .I => .I
  | .H => .H

/-- Gate composition: I is the unit and H is involutive. -/
def GateCompose : QuantumGate → QuantumGate → QuantumGate
  | .I, g => g
  | g, .I => g
  | .H, .H => .I

def GateIdentity : QuantumGate := QGate.I

def HadamardGate : QuantumGate := QGate.H

theorem hadamard_self_adjoint : GateAdjoint HadamardGate = HadamardGate :=
  rfl

theorem hadamard_involution :
    GateCompose HadamardGate HadamardGate = GateIdentity :=
  rfl

/-- H is unitary in the model: H†H = I. -/
theorem hadamard_unitary :
    GateCompose (GateAdjoint HadamardGate) HadamardGate = GateIdentity := by
  rw [hadamard_self_adjoint]
  exact hadamard_involution

/-- Composition in the gate fragment is associative. -/
theorem gate_compose_assoc (a b c : QuantumGate) :
    GateCompose (GateCompose a b) c = GateCompose a (GateCompose b c) := by
  cases a <;> cases b <;> cases c <;> rfl

/-- Every gate has a right inverse under composition. -/
theorem gate_inverses_exist (g : QuantumGate) :
    ∃ h : QuantumGate, GateCompose g h = GateIdentity := by
  cases g
  · exact ⟨QGate.I, rfl⟩
  · exact ⟨QGate.H, rfl⟩

/-- COROLLARY bridge for `OmegaProtocol.lean` (legacy name kept): restates
    Hadamard unitarity at protocol level. -/
theorem quantum_computing_from_omega :
    GateCompose (GateAdjoint HadamardGate) HadamardGate = GateIdentity :=
  hadamard_unitary

/-- Structural bridge: the Ω-metric is reflexive on the Q-region model used
    for circuit-depth distances. -/
theorem bridge_vol34_metric_self_zero (R : QRegion) : d R R = 0 :=
  bridge_qregion_self_distance_zero R

-- ============================================================
-- FAITHFUL 2×2 UNITARY MATRIX REPRESENTATION & STATE AMPLITUDES
-- ============================================================
--
-- Above, `QGate = {I, H}` is a syntactic two-element algebra.  Here we
-- construct the concrete 2×2 real/orthogonal unitary matrix representation
-- `GateMatrix` (`I`, Hadamard `H = (1/√2)[[1,1],[1,-1]]`, Pauli `X`, Pauli `Z`)
-- together with 2-component state vectors `QubitVec`, and prove:
-- 1. `H`, `X`, and `Z` are self-adjoint and unitary (`Mᵀ M = I`).
-- 2. Hadamard conjugation exchanges bit-flip and phase-flip gates
--    (`H X H = Z` and `H Z H = X`), while `X` and `Z` do not commute.
-- 3. `interpGate : QuantumGate → GateMatrix` is a faithful (`Injective`)
--    dagger-monoid homomorphism (`interpGate_compose`, `interpGate_adjoint`,
--    `interpGate_injective`).
-- 4. `matH` preserves the Euclidean norm squared `v₀² + v₁²` of any state
--    vector (`matH_preserves_norm`).

/-- 2×2 gate matrix on a single-qubit state space. -/
@[ext]
structure GateMatrix where
  a00 : ℝ
  a01 : ℝ
  a10 : ℝ
  a11 : ℝ

/-- Matrix multiplication of 2×2 gates. -/
def matMul (M N : GateMatrix) : GateMatrix :=
  ⟨M.a00 * N.a00 + M.a01 * N.a10,
   M.a00 * N.a01 + M.a01 * N.a11,
   M.a10 * N.a00 + M.a11 * N.a10,
   M.a10 * N.a01 + M.a11 * N.a11⟩

/-- Transpose (adjoint on real gate matrices). -/
def matTranspose (M : GateMatrix) : GateMatrix :=
  ⟨M.a00, M.a10, M.a01, M.a11⟩

/-- 2×2 identity matrix. -/
def matId : GateMatrix := ⟨1, 0, 0, 1⟩

/-- Pauli-X (bit-flip / NOT) gate matrix. -/
def matX : GateMatrix := ⟨0, 1, 1, 0⟩

/-- Pauli-Z (phase-flip) gate matrix. -/
def matZ : GateMatrix := ⟨1, 0, 0, -1⟩

/-- Normalization factor `1 / √2` of the Hadamard gate. -/
noncomputable def invSqrt2 : ℝ := 1 / Real.sqrt 2

theorem invSqrt2_sq : invSqrt2 * invSqrt2 = 1 / 2 := by
  unfold invSqrt2
  have hs : (Real.sqrt 2) ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  calc (1 / Real.sqrt 2) * (1 / Real.sqrt 2)
      = 1 / ((Real.sqrt 2) ^ 2) := by ring
    _ = 1 / 2 := by rw [hs]

/-- Concrete 2×2 Hadamard matrix `(1/√2) [[1, 1], [1, -1]]`. -/
noncomputable def matH : GateMatrix :=
  ⟨invSqrt2, invSqrt2, invSqrt2, -invSqrt2⟩

theorem matH_self_adjoint : matTranspose matH = matH := by
  ext <;> rfl

theorem matH_involution : matMul matH matH = matId := by
  ext
  · dsimp [matMul, matH, matId]
    calc invSqrt2 * invSqrt2 + invSqrt2 * invSqrt2
        = 2 * (invSqrt2 * invSqrt2) := by ring
      _ = 1 := by rw [invSqrt2_sq]; norm_num
  · dsimp [matMul, matH, matId]; ring
  · dsimp [matMul, matH, matId]; ring
  · dsimp [matMul, matH, matId]
    calc invSqrt2 * invSqrt2 + -invSqrt2 * -invSqrt2
        = 2 * (invSqrt2 * invSqrt2) := by ring
      _ = 1 := by rw [invSqrt2_sq]; norm_num

/-- **Matrix unitarity of the 2×2 Hadamard gate**: `Hᵀ H = I`. -/
theorem matH_unitary : matMul (matTranspose matH) matH = matId := by
  rw [matH_self_adjoint]
  exact matH_involution

theorem matX_unitary : matMul (matTranspose matX) matX = matId := by
  ext <;> dsimp [matMul, matTranspose, matX, matId] <;> norm_num

theorem matZ_unitary : matMul (matTranspose matZ) matZ = matId := by
  ext <;> dsimp [matMul, matTranspose, matZ, matId] <;> norm_num

/-- **Hadamard conjugation turns bit-flip into phase-flip**: `H X H = Z`. -/
theorem hadamard_conjugates_X_to_Z :
    matMul (matMul matH matX) matH = matZ := by
  ext
  · dsimp [matMul, matH, matX, matZ]
    calc (invSqrt2 * 0 + invSqrt2 * 1) * invSqrt2 +
          (invSqrt2 * 1 + invSqrt2 * 0) * invSqrt2
        = 2 * (invSqrt2 * invSqrt2) := by ring
      _ = 1 := by rw [invSqrt2_sq]; norm_num
  · dsimp [matMul, matH, matX, matZ]; ring
  · dsimp [matMul, matH, matX, matZ]; ring
  · dsimp [matMul, matH, matX, matZ]
    calc (invSqrt2 * 1 + -invSqrt2 * 0) * invSqrt2 +
          (invSqrt2 * 0 + -invSqrt2 * 1) * -invSqrt2
        = -2 * (invSqrt2 * invSqrt2) := by ring
      _ = -1 := by rw [invSqrt2_sq]; norm_num

/-- Pauli-X and Pauli-Z do not commute (`X Z ≠ Z X`). -/
theorem pauli_XZ_noncommutative : matMul matX matZ ≠ matMul matZ matX := by
  intro h
  have h01 : (matMul matX matZ).a01 = (matMul matZ matX).a01 :=
    congrArg GateMatrix.a01 h
  norm_num [matMul, matX, matZ] at h01

/-- Matrix interpretation of the syntactic gate set `QuantumGate`. -/
noncomputable def interpGate : QuantumGate → GateMatrix
  | .I => matId
  | .H => matH

theorem interpGate_adjoint (g : QuantumGate) :
    interpGate (GateAdjoint g) = matTranspose (interpGate g) := by
  cases g
  · ext <;> rfl
  · exact matH_self_adjoint.symm

/-- `interpGate` is a monoid homomorphism from `GateCompose` to `matMul`. -/
theorem interpGate_compose (g₁ g₂ : QuantumGate) :
    interpGate (GateCompose g₁ g₂) = matMul (interpGate g₁) (interpGate g₂) := by
  cases g₁ <;> cases g₂
  · ext <;> dsimp [GateCompose, interpGate, matMul, matId] <;> ring
  · ext <;> dsimp [GateCompose, interpGate, matMul, matId, matH] <;> ring
  · ext <;> dsimp [GateCompose, interpGate, matMul, matId, matH] <;> ring
  · exact matH_involution.symm

/-- **Faithfulness of the matrix representation**: distinct syntactic gates map
    to distinct 2×2 matrices. -/
theorem interpGate_injective : Function.Injective interpGate := by
  intro g₁ g₂ h
  cases g₁ <;> cases g₂
  · rfl
  · exfalso
    have h11 : (interpGate .I).a11 = (interpGate .H).a11 :=
      congrArg GateMatrix.a11 h
    dsimp [interpGate, matId, matH, invSqrt2] at h11
    have hnonneg : 0 ≤ 1 / Real.sqrt 2 :=
      div_nonneg zero_le_one (Real.sqrt_nonneg 2)
    linarith
  · exfalso
    have h11 : (interpGate .H).a11 = (interpGate .I).a11 :=
      congrArg GateMatrix.a11 h
    dsimp [interpGate, matId, matH, invSqrt2] at h11
    have hnonneg : 0 ≤ 1 / Real.sqrt 2 :=
      div_nonneg zero_le_one (Real.sqrt_nonneg 2)
    linarith
  · rfl

/-- Two-component qubit state vector. -/
@[ext]
structure QubitVec where
  v0 : ℝ
  v1 : ℝ

def applyGate (M : GateMatrix) (v : QubitVec) : QubitVec :=
  ⟨M.a00 * v.v0 + M.a01 * v.v1,
   M.a10 * v.v0 + M.a11 * v.v1⟩

def normSq (v : QubitVec) : ℝ := v.v0 ^ 2 + v.v1 ^ 2

/-- **The 2×2 Hadamard matrix preserves state norm squared**:
    `‖H v‖² = ‖v‖²` for every `v : QubitVec`. -/
theorem matH_preserves_norm (v : QubitVec) :
    normSq (applyGate matH v) = normSq v := by
  unfold normSq applyGate matH
  dsimp
  calc (invSqrt2 * v.v0 + invSqrt2 * v.v1) ^ 2 +
        (invSqrt2 * v.v0 + -invSqrt2 * v.v1) ^ 2
      = 2 * (invSqrt2 * invSqrt2) * (v.v0 ^ 2 + v.v1 ^ 2) := by ring
    _ = 2 * (1 / 2) * (v.v0 ^ 2 + v.v1 ^ 2) := by rw [invSqrt2_sq]
    _ = v.v0 ^ 2 + v.v1 ^ 2 := by ring

end OmegaProtocol.Vol34
