import Mathlib
import OmegaUnifiedFoundation
/-
  Vol04_QuantumMechanics.lean
  REAL mathematical formalization of quantum mechanics.

  Genuinely proven theorems:
  1. Born Rule — from projection idempotence + self-adjointness
  2. Cauchy-Schwarz Uncertainty — Var(A)·Var(B) ≥ |⟨u,v⟩|²
  3. Robertson Uncertainty (GENERAL complex inner product space, the
     `Robertson` section): Var(A)·Var(B) ≥ (|⟨ψ,[A,B]ψ⟩|/2)² for
     self-adjoint A, B and a normalized state ψ — the full commutator
     form via the self-adjoint transfer (`inner_dev_eq`) and the
     commutator link (`comm_link`), the two steps formerly marked
     MISSING. `norm_sq_ge_im_sq` below is the auxiliary complex-number
     lemma kept from the partial proof.
  4. Expectation value is real for self-adjoint operators
  5. Projection probabilities are non-negative and ≤ 1
  6. Orthogonal projections give exclusive outcomes
  7. Schrödinger dynamics (one-dimensional model): the explicit
     trajectory ψ(t) = exp(−iωt)·ψ₀ has a genuine derivative
     (`HasDerivAt`) and solves iℏ·dψ/dt = H_{ℏω} ψ for the
     self-adjoint Hamiltonian z ↦ ℏω·z.

  Model assumptions (physical postulates, not provable from math alone):
  - ℏ > 0 (physical constant; here hbar = 1)
  - Which self-adjoint operators generate which unitary dynamics on a
    general H is not derived; the one-dimensional solution above is
    constructed and verified. `time_derivative` below remains the zero
    map of the legacy minimal model and is not used by the dynamics.
-/


namespace OmegaProtocol.Vol04
open OmegaProtocol

-- ============================================================
-- DEFINITIONS
-- ============================================================

/-- Self-adjoint operator: ⟨Aψ, φ⟩ = ⟨ψ, Aφ⟩ for all ψ, φ -/
def IsSelfAdjoint (A : Operator) : Prop :=
  ∀ (ψ φ : StateSpace), @inner ℂ StateSpace _ (A ψ) φ = @inner ℂ StateSpace _ ψ (A φ)

/-- Expectation value: ⟨ψ|A|ψ⟩ -/
noncomputable def Expectation (A : Operator) (ψ : StateSpace) : ℂ :=
  @inner ℂ StateSpace _ ψ (A ψ)

/-- Variance: ‖(A - ⟨A⟩)ψ‖² -/
noncomputable def Variance (A : Operator) (ψ : StateSpace) : ℝ :=
  ‖A ψ - (Expectation A ψ) • ψ‖ ^ 2

/-- Commutator: [A,B] = AB - BA -/
noncomputable def commutator_op (A B : Operator) : Operator :=
  A * B - B * A

/-- The deviation vector (A - ⟨A⟩)ψ -/
noncomputable def deviation (A : Operator) (ψ : StateSpace) : StateSpace :=
  A ψ - (Expectation A ψ) • ψ

-- ============================================================
-- THEOREM 1: Born Rule (GENUINE PROOF)
-- For a projection P (P² = P, P* = P):
--   ⟨ψ|P|ψ⟩ = ‖Pψ‖²
-- ============================================================

theorem born_rule (P : Operator) (hP_proj : P * P = P ∧ IsSelfAdjoint P) (ψ : StateSpace) :
  @inner ℂ StateSpace _ ψ (P ψ) = (‖P ψ‖ ^ 2 : ℂ) := by
  have h_idem : P * P = P := hP_proj.1
  have h_sa : IsSelfAdjoint P := hP_proj.2
  have h_eq : P (P ψ) = P ψ := by
    calc P (P ψ) = (P * P) ψ := rfl
    _ = P ψ := by rw [h_idem]
  calc @inner ℂ StateSpace _ ψ (P ψ)
      = @inner ℂ StateSpace _ ψ (P (P ψ)) := by rw [h_eq]
    _ = @inner ℂ StateSpace _ (P ψ) (P ψ) := by rw [h_sa]
    _ = (‖P ψ‖ ^ 2 : ℂ) := inner_self_eq_norm_sq_to_K (P ψ)

-- ============================================================
-- THEOREM 2: Born Rule Probability (GENUINE PROOF)
-- The measurement probability ‖Pψ‖² is non-negative.
-- ============================================================

theorem born_rule_nonneg (P : Operator) (ψ : StateSpace) :
  ‖P ψ‖ ^ 2 ≥ 0 := sq_nonneg _

-- ============================================================
-- THEOREM 3: Projection Norm Bound (GENUINE PROOF)
-- For a projection on a unit vector: ‖Pψ‖ ≤ ‖ψ‖
-- Equivalently, measurement probability ≤ 1 for normalized states.
-- ============================================================

theorem projection_norm_le (P : Operator) (hP : P * P = P ∧ IsSelfAdjoint P) (ψ : StateSpace) :
  ‖P ψ‖ ^ 2 ≤ ‖ψ‖ ^ 2 := by
  -- ‖Pψ‖² = re⟨Pψ,Pψ⟩ (Mathlib: inner_self_eq_norm_sq)
  -- = re⟨ψ,P(Pψ)⟩ (self-adjointness of P)
  -- = re⟨ψ,Pψ⟩ (P² = P)
  -- ≤ ‖⟨ψ,Pψ⟩‖ (re z ≤ ‖z‖)
  -- ≤ ‖ψ‖ * ‖Pψ‖ (Cauchy-Schwarz)
  -- So ‖Pψ‖² ≤ ‖ψ‖ * ‖Pψ‖, giving ‖Pψ‖ ≤ ‖ψ‖
  have h_cs := norm_inner_le_norm (𝕜 := ℂ) ψ (P ψ)
  -- ‖Pψ‖² = re⟨Pψ,Pψ⟩
  have h_norm := inner_self_eq_norm_sq (𝕜 := ℂ) (P ψ)
  -- re⟨Pψ,Pψ⟩ = re⟨ψ,Pψ⟩  (via self-adj + idempotence)
  have h_eq : @inner ℂ StateSpace _ (P ψ) (P ψ) = @inner ℂ StateSpace _ ψ (P ψ) := by
    have h_pp : P (P ψ) = P ψ := by
      show (P * P) ψ = P ψ; rw [hP.1]
    calc @inner ℂ StateSpace _ (P ψ) (P ψ)
        = @inner ℂ StateSpace _ ψ (P (P ψ)) := hP.2 ψ (P ψ)
      _ = @inner ℂ StateSpace _ ψ (P ψ) := by rw [h_pp]
  -- re⟨ψ,Pψ⟩ ≤ ‖⟨ψ,Pψ⟩‖
  have h_re_le : RCLike.re (@inner ℂ StateSpace _ ψ (P ψ)) ≤
    ‖@inner ℂ StateSpace _ ψ (P ψ)‖ := RCLike.re_le_norm _
  -- Chain: ‖Pψ‖² = re⟨Pψ,Pψ⟩ = re⟨ψ,Pψ⟩ ≤ ‖⟨ψ,Pψ⟩‖ ≤ ‖ψ‖·‖Pψ‖
  have h_chain : ‖P ψ‖ ^ 2 ≤ ‖ψ‖ * ‖P ψ‖ := by
    calc (‖P ψ‖ ^ 2 : ℝ)
        = RCLike.re (@inner ℂ StateSpace _ (P ψ) (P ψ)) := h_norm.symm
      _ = RCLike.re (@inner ℂ StateSpace _ ψ (P ψ)) := by rw [h_eq]
      _ ≤ ‖@inner ℂ StateSpace _ ψ (P ψ)‖ := h_re_le
      _ ≤ ‖ψ‖ * ‖P ψ‖ := h_cs
  nlinarith [sq_nonneg (‖ψ‖ - ‖P ψ‖)]

-- ============================================================
-- THEOREM 4: Cauchy-Schwarz Uncertainty (GENUINE PROOF)
-- Var(A)·Var(B) ≥ |⟨u,v⟩|² where u,v are deviation vectors
-- ============================================================

theorem variance_eq_norm_sq (A : Operator) (ψ : StateSpace) :
  Variance A ψ = ‖deviation A ψ‖ ^ 2 := rfl

theorem cauchy_schwarz_variance (A B : Operator) (ψ : StateSpace) :
  Variance A ψ * Variance B ψ ≥ ‖@inner ℂ StateSpace _ (deviation A ψ) (deviation B ψ)‖ ^ 2 := by
  rw [variance_eq_norm_sq, variance_eq_norm_sq]
  have h_cs := norm_inner_le_norm (𝕜 := ℂ) (deviation A ψ) (deviation B ψ)
  have h_sq : ‖@inner ℂ StateSpace _ (deviation A ψ) (deviation B ψ)‖ ^ 2
    ≤ (‖deviation A ψ‖ * ‖deviation B ψ‖) ^ 2 := by
    apply sq_le_sq'
    · linarith [norm_nonneg (@inner ℂ StateSpace _ (deviation A ψ) (deviation B ψ))]
    · exact h_cs
  rw [mul_pow] at h_sq
  linarith

-- ============================================================
-- THEOREM 5: auxiliary lemma toward Robertson
-- The full uncertainty principle connecting to the commutator,
--   Var(A)·Var(B) ≥ |⟨ψ,[A,B]ψ⟩|²/4,
-- IS proven in this file — in the general `Robertson` section below,
-- for an arbitrary complex inner product space. Its proof chain:
-- (a) Cauchy-Schwarz: Var(A)·Var(B) ≥ |⟨u,v⟩|²  [above]
-- (b) Self-adjoint transfer of deviations  [`Robertson.inner_dev_eq`]
-- (c) Commutator link ⟨u,v⟩ − ⟨v,u⟩ = ⟨ψ,[A,B]ψ⟩
--     [`Robertson.comm_link`]
-- (d) Triangle bound ‖z − conj z‖ ≤ 2‖z‖ (replacing the earlier
--     |⟨u,v⟩|² ≥ Im⟨u,v⟩² route; the lemma below remains as the
--     complex-number fact of that route)
-- ============================================================

/-- Norm squared dominates imaginary part squared.
    For any complex number z: ‖z‖² ≥ (Im z)².
    This is the key step connecting Cauchy-Schwarz to the commutator. -/
theorem norm_sq_ge_im_sq (z : ℂ) : ‖z‖ ^ 2 ≥ z.im ^ 2 := by
  have h := RCLike.norm_sq_eq_def (K := ℂ) (z := z)
  -- h : ‖z‖² = re(z) * re(z) + im(z) * im(z)
  simp only [RCLike.re_to_complex, RCLike.im_to_complex] at h
  rw [h]; nlinarith [sq_nonneg z.re]

/-- For self-adjoint A, the inner product ⟨Aψ,φ⟩ = ⟨ψ,Aφ⟩
    lets us transfer A across the inner product. This is
    the key property that connects deviation inner products
    to operator products. -/
theorem self_adjoint_deviation_transfer (A : Operator) (hA : IsSelfAdjoint A) (ψ φ : StateSpace) :
  @inner ℂ StateSpace _ (A ψ) φ = @inner ℂ StateSpace _ ψ (A φ) := hA ψ φ

-- ============================================================
-- THEOREM 6: Expectation Value Properties (GENUINE PROOFS)
-- ============================================================

/-- For a self-adjoint operator, the expectation value is real -/
theorem expectation_real (A : Operator) (hA : IsSelfAdjoint A) (ψ : StateSpace) :
  starRingEnd ℂ (Expectation A ψ) = Expectation A ψ := by
  unfold Expectation
  rw [inner_conj_symm]
  exact hA ψ ψ ▸ rfl

-- ============================================================
-- THEOREM 7: Orthogonal Projections (GENUINE PROOF)
-- If P and Q are orthogonal projections (PQ = 0),
-- their measurement outcomes are exclusive.
-- ============================================================

theorem orthogonal_exclusive (P Q : Operator)
  (hPQ : P * Q = 0) (ψ : StateSpace) :
  P (Q ψ) = 0 := by
  calc P (Q ψ) = (P * Q) ψ := rfl
    _ = (0 : Operator) ψ := by rw [hPQ]
    _ = 0 := by simp

-- ============================================================
-- GENERAL HILBERT-SPACE SECTION: THE ROBERTSON UNCERTAINTY RELATION
--
-- The theorems above live on the one-dimensional minimal model, where
-- every commutator vanishes. The section below is stated for an
-- ARBITRARY complex inner product space H, so the commutator
-- [A,B] = A∘B − B∘A is genuinely nontrivial. It closes the two gaps
-- marked MISSING in the old partial proof chain:
--   (b) self-adjoint transfer of deviations  → `inner_dev_eq`
--   (c) commutator link Im⟨A'ψ,B'ψ⟩         → `comm_link`
-- and finishes with the triangle bound ‖z − conj z‖ ≤ 2‖z‖ (in place
-- of the earlier |z|² ≥ Im(z)² route). Together with Cauchy–Schwarz
-- this yields the full Robertson inequality.
-- ============================================================

namespace Robertson
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]

/-- Self-adjoint operator on a general complex inner product space:
    ⟨Aψ, φ⟩ = ⟨ψ, Aφ⟩ for all ψ, φ. -/
def SelfAdjoint (A : H →L[ℂ] H) : Prop :=
  ∀ (ψ φ : H), @inner ℂ H _ (A ψ) φ = @inner ℂ H _ ψ (A φ)

/-- Expectation value ⟨ψ, Aψ⟩ on a general space. -/
noncomputable def expval (A : H →L[ℂ] H) (ψ : H) : ℂ :=
  @inner ℂ H _ ψ (A ψ)

/-- Deviation vector (A − ⟨A⟩)ψ on a general space. -/
noncomputable def deviation (A : H →L[ℂ] H) (ψ : H) : H :=
  A ψ - (expval A ψ) • ψ

/-- Variance ‖(A − ⟨A⟩)ψ‖² on a general space. -/
noncomputable def variance (A : H →L[ℂ] H) (ψ : H) : ℝ :=
  ‖deviation A ψ‖ ^ 2

/-- Commutator [A,B] = A∘B − B∘A as an operator on a general space. -/
def ccomm (A B : H →L[ℂ] H) : H →L[ℂ] H := A.comp B - B.comp A

/-- Self-adjointness transfers an operator across the inner product;
    this is the transfer step (b) of the Robertson proof chain. -/
theorem self_adjoint_transfer (A : H →L[ℂ] H) (hA : SelfAdjoint A) (ψ φ : H) :
    @inner ℂ H _ (A ψ) φ = @inner ℂ H _ ψ (A φ) :=
  hA ψ φ

/-- THE TRANSFER STEP (b): for self-adjoint A and a NORMALIZED state ψ
    (⟨ψ,ψ⟩ = 1), the deviation inner product reduces to an operator
    expression ⟨Aψ, Bψ⟩ minus the product of expectations. The scalar
    corrections from the deviations cancel exactly because
    ⟨Aψ, ψ⟩ = ⟨ψ, Aψ⟩ and ⟨ψ,ψ⟩ = 1. -/
theorem inner_dev_eq (A B : H →L[ℂ] H) (hA : SelfAdjoint A) (ψ : H)
    (hψ : @inner ℂ H _ ψ ψ = 1) :
    @inner ℂ H _ (deviation A ψ) (deviation B ψ)
      = @inner ℂ H _ ψ ((A.comp B) ψ) - expval B ψ * expval A ψ := by
  have h1 : @inner ℂ H _ (A ψ) ψ = expval A ψ := hA ψ ψ
  have h2 : @inner ℂ H _ (A ψ) (B ψ) = @inner ℂ H _ ψ ((A.comp B) ψ) :=
    hA ψ (B ψ)
  have hb : @inner ℂ H _ ψ (B ψ) = expval B ψ := rfl
  show @inner ℂ H _ (A ψ - expval A ψ • ψ) (B ψ - expval B ψ • ψ) = _
  rw [inner_sub_right, inner_sub_left, inner_sub_left,
    inner_smul_left, inner_smul_left, inner_smul_right, inner_smul_right,
    h1, hb, hψ, h2]
  ring

/-- THE COMMUTATOR LINK (c): for self-adjoint A, B and a normalized
    state, the antisymmetrized deviation inner product equals the
    expectation of the commutator. The expectation products cancel by
    commutativity of ℂ. -/
theorem comm_link (A B : H →L[ℂ] H) (hA : SelfAdjoint A) (hB : SelfAdjoint B)
    (ψ : H) (hψ : @inner ℂ H _ ψ ψ = 1) :
    @inner ℂ H _ (deviation A ψ) (deviation B ψ)
      - @inner ℂ H _ (deviation B ψ) (deviation A ψ)
      = @inner ℂ H _ ψ ((ccomm A B) ψ) := by
  have e1 := inner_dev_eq A B hA ψ hψ
  have e2 := inner_dev_eq B A hB ψ hψ
  rw [e1, e2]
  simp only [ccomm, ContinuousLinearMap.sub_apply, inner_sub_right]
  ring

/-- THE ROBERTSON UNCERTAINTY RELATION (general complex inner product
    space): for self-adjoint operators A, B and a normalized state ψ,
    Var(A)·Var(B) ≥ (|⟨ψ, [A,B] ψ⟩| / 2)².
    Proof: Cauchy–Schwarz gives Var(A)·Var(B) ≥ |⟨A'ψ, B'ψ⟩|²; the
    commutator link identifies the antisymmetric part of that inner
    product with ⟨ψ,[A,B]ψ⟩; and the triangle inequality bounds
    ‖z − conj z‖ ≤ 2‖z‖. This is the full commutator form of the
    uncertainty principle — nontrivial whenever A and B fail to
    commute. -/
theorem robertson_uncertainty (A B : H →L[ℂ] H) (hA : SelfAdjoint A)
    (hB : SelfAdjoint B) (ψ : H) (hψ : @inner ℂ H _ ψ ψ = 1) :
    variance A ψ * variance B ψ
      ≥ (‖@inner ℂ H _ ψ ((ccomm A B) ψ)‖ / 2) ^ 2 := by
  have hcs := norm_inner_le_norm (𝕜 := ℂ) (deviation A ψ) (deviation B ψ)
  have h1 : ‖@inner ℂ H _ (deviation A ψ) (deviation B ψ)‖ ^ 2
      ≤ (‖deviation A ψ‖ * ‖deviation B ψ‖) ^ 2 := by
    apply sq_le_sq'
    · linarith [norm_nonneg (@inner ℂ H _ (deviation A ψ) (deviation B ψ))]
    · exact hcs
  rw [mul_pow] at h1
  have hlink := comm_link A B hA hB ψ hψ
  have hconj : starRingEnd ℂ (@inner ℂ H _ (deviation A ψ) (deviation B ψ))
      = @inner ℂ H _ (deviation B ψ) (deviation A ψ) :=
    inner_conj_symm (deviation B ψ) (deviation A ψ)
  rw [← hconj] at hlink
  have htri : ‖@inner ℂ H _ ψ ((ccomm A B) ψ)‖
      ≤ ‖@inner ℂ H _ (deviation A ψ) (deviation B ψ)‖
        + ‖starRingEnd ℂ (@inner ℂ H _ (deviation A ψ) (deviation B ψ))‖ := by
    rw [← hlink]
    exact norm_sub_le _ _
  have hcj : ‖starRingEnd ℂ (@inner ℂ H _ (deviation A ψ) (deviation B ψ))‖
      = ‖@inner ℂ H _ (deviation A ψ) (deviation B ψ)‖ :=
    Complex.norm_conj _
  have hdiv : ‖@inner ℂ H _ ψ ((ccomm A B) ψ)‖ / 2
      ≤ ‖@inner ℂ H _ (deviation A ψ) (deviation B ψ)‖ := by
    rw [div_le_iff₀ two_pos]
    linarith
  have hfin : (‖@inner ℂ H _ ψ ((ccomm A B) ψ)‖ / 2) ^ 2
      ≤ ‖@inner ℂ H _ (deviation A ψ) (deviation B ψ)‖ ^ 2 := by
    apply sq_le_sq'
    · linarith [norm_nonneg (@inner ℂ H _ ψ ((ccomm A B) ψ))]
    · exact hdiv
  show ‖deviation A ψ‖ ^ 2 * ‖deviation B ψ‖ ^ 2 ≥ _
  linarith

end Robertson

-- ============================================================
-- PHYSICAL POSTULATES (model constants; nothing here is a Lean `axiom`)
-- ============================================================

def hbar : ℝ := 1
theorem hbar_pos : hbar > 0 := by
  norm_num [hbar]
noncomputable def time_derivative (_ : ℝ → StateSpace) (_ : ℝ) : StateSpace := 0

-- The retired theorem `schrodinger_equation` restated its hypothesis
-- `h_dynamics` verbatim as its conclusion (a P→P tautology); it was
-- removed in the eighth pass. The block below replaces it with genuine
-- dynamics: an explicit trajectory with a REAL derivative
-- (`HasDerivAt`, not the zero `time_derivative` of the minimal model)
-- that provably solves the equation for a self-adjoint Hamiltonian.
-- What remains a postulate is the general theory (which self-adjoint
-- H generate which unitary groups on a general H); the
-- one-dimensional solution here is constructed and verified.

/-- The Hamiltonian of the one-dimensional model: multiplication by the
    real energy E. Stated on the ℂ carrier of the one-dimensional state
    space (a bundled operator construction is unnecessary here, and the
    `StateSpace` definition is not reducible enough to reuse the ℂ
    instances syntactically). -/
noncomputable def hamiltonian (E : ℝ) : ℂ → ℂ :=
  fun z => (E : ℂ) * z

/-- The Hamiltonian of the one-dimensional model is self-adjoint for
    real energies: ⟨Hψ, φ⟩ = ⟨ψ, Hφ⟩ on the ℂ carrier. -/
theorem hamiltonian_self_adjoint (E : ℝ) (ψ φ : ℂ) :
    @inner ℂ ℂ _ (hamiltonian E ψ) φ = @inner ℂ ℂ _ ψ (hamiltonian E φ) := by
  simp only [hamiltonian, RCLike.inner_apply, map_mul, Complex.conj_ofReal]
  ring

/-- Schrödinger trajectory for angular frequency ω and initial state
    ψ₀: ψ(t) = exp(−iωt)·ψ₀. -/
noncomputable def schrodingerTrajectory (ω : ℝ) (ψ₀ : ℂ) : ℝ → ℂ :=
  fun t => Complex.exp (-(t * (Complex.I * (ω : ℂ)))) * ψ₀

/-- The trajectory is genuinely differentiable, with derivative
    ψ'(t) = −iω·ψ(t). -/
theorem hasDerivAt_schrodingerTrajectory (ω : ℝ) (ψ₀ : ℂ) (t : ℝ) :
    HasDerivAt (schrodingerTrajectory ω ψ₀)
      (Complex.exp (-(t * (Complex.I * (ω : ℂ)))) * ψ₀
        * -(Complex.I * (ω : ℂ))) t := by
  have h1 : HasDerivAt (fun t : ℝ => (t : ℂ) * (Complex.I * (ω : ℂ)))
      ((1 : ℂ) * (Complex.I * (ω : ℂ))) t := by
    simpa only [id_eq, Complex.ofReal_one] using
      (HasDerivAt.ofReal_comp (hasDerivAt_id t)).mul_const (Complex.I * (ω : ℂ))
  have h2 : HasDerivAt (fun t : ℝ => -(t * (Complex.I * (ω : ℂ))))
      (-(1 * (Complex.I * (ω : ℂ)))) t := h1.neg
  have h3 := h2.cexp
  have h4 : HasDerivAt (fun t : ℝ => Complex.exp (-(t * (Complex.I * (ω : ℂ)))) * ψ₀)
      (Complex.exp (-(t * (Complex.I * (ω : ℂ))))
        * -(1 * (Complex.I * (ω : ℂ))) * ψ₀) t := h3.mul_const ψ₀
  exact h4.congr_deriv (by ring)

/-- THE SCHRÖDINGER EQUATION (genuine solution, one-dimensional model):
    for angular frequency ω and energy E = ℏω, the explicit trajectory
    above satisfies iℏ·dψ/dt = H_E ψ with a genuine derivative. The
    identity holds for the model's hbar without unfolding it. This is
    the first Schrödinger theorem in this corpus that is proved rather
    than postulated: the dynamics are constructed and verified. -/
theorem schrodinger_equation (ω : ℝ) (ψ₀ : ℂ) (t : ℝ) :
    Complex.I * (hbar : ℂ) * (deriv (schrodingerTrajectory ω ψ₀) t)
      = hamiltonian (hbar * ω) (schrodingerTrajectory ω ψ₀ t) := by
  rw [(hasDerivAt_schrodingerTrajectory ω ψ₀ t).deriv]
  simp only [schrodingerTrajectory, hamiltonian, Complex.ofReal_mul]
  have key : -(Complex.I * (Complex.I * (ω : ℂ))) = (ω : ℂ) := by
    rw [← mul_assoc, Complex.I_mul_I, neg_one_mul, neg_neg]
  linear_combination
    (Complex.exp (-(t * (Complex.I * (ω : ℂ)))) * ψ₀ * (hbar : ℂ)) * key

end OmegaProtocol.Vol04
