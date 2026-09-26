import Mathlib
import OmegaUnifiedFoundation

/-!
# Unit-phase reference frames

A small nontrivial coordinate model: frames are complex phases of norm one and
changes of frame act by the ratio of phases. These transformations are unitary
on the legacy one-dimensional complex state space. This is not a theory of
relational observers or general quantum reference frames.
-/
namespace OmegaProtocol.Vol49
open OmegaProtocol

def ReferenceFrame : Type := {z : ℂ // ‖z‖ = 1}

theorem frame_phase_ne_zero (A : ReferenceFrame) : A.1 ≠ 0 := by
  intro h
  have hn := A.2
  rw [h, norm_zero] at hn
  norm_num at hn

noncomputable def FrameTransform (A B : ReferenceFrame) : Operator :=
  (B.1 / A.1) • ContinuousLinearMap.id ℂ StateSpace

theorem frame_compose (A B C : ReferenceFrame) :
    FrameTransform A C = FrameTransform B C ∘L FrameTransform A B := by
  apply ContinuousLinearMap.ext
  intro ψ
  change (C.1 / A.1) • ψ = (C.1 / B.1) • ((B.1 / A.1) • ψ)
  rw [smul_smul]
  congr 1
  field_simp [frame_phase_ne_zero A, frame_phase_ne_zero B] <;> ring

theorem frame_identity (A : ReferenceFrame) :
    FrameTransform A A = ContinuousLinearMap.id ℂ StateSpace := by
  unfold FrameTransform
  rw [div_self (frame_phase_ne_zero A), one_smul]

theorem frame_self_identity (A : ReferenceFrame) (ψ : StateSpace) :
    FrameTransform A A ψ = ψ := by
  rw [frame_identity]
  simp

/-- Both directions are genuine inverse transformations. -/
theorem frame_round_trip (A B : ReferenceFrame) (ψ : StateSpace) :
    FrameTransform B A (FrameTransform A B ψ) = ψ := by
  change (FrameTransform B A ∘L FrameTransform A B) ψ = ψ
  rw [← frame_compose, frame_identity]
  rfl

theorem frame_preserves_norm (A B : ReferenceFrame) (ψ : StateSpace) :
    ‖FrameTransform A B ψ‖ = ‖ψ‖ := by
  change ‖(B.1 / A.1) • ψ‖ = ‖ψ‖
  rw [norm_smul, norm_div, A.2, B.2]
  norm_num

noncomputable def positiveFrame : ReferenceFrame := ⟨1, by simp⟩
noncomputable def negativeFrame : ReferenceFrame := ⟨-1, by simp⟩

def unitState : StateSpace := (1 : ℂ)

/-- Unlike the old singleton model, a frame change can change a state. -/
theorem frame_change_nontrivial :
    FrameTransform positiveFrame negativeFrame unitState = -unitState := by
  change ((-1 : ℂ) / 1) * 1 = -1
  norm_num

end OmegaProtocol.Vol49
