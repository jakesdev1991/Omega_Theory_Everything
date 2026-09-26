import Mathlib
import OmegaAxioms
import OmegaUnifiedFoundation

/-
  Vol36_ToposTheory.lean
  Formalization of Topos Theory.

  Model scope (stated honestly):
  * The excluded-middle theorem is plain classical logic, kept under its
    legacy name for `OmegaProtocol.lean`; it is not topos theory.
  * The categorical content is a concrete skeletal category of finite
    sets: objects are natural numbers (standing for `Fin n`), maps are
    all functions. The category laws, terminal object, binary products,
    equalizers, exponentials, and characteristic maps for decidable
    subsets are genuinely proven below.
  * This is NOT a full elementary topos: pullback-based subobject
    classification is not formalized. The classifier section proves the
    characteristic-map universal property for subsets only.
  * Omega-Protocol mapping: subobjects = Q-regions (0D), morphisms = Φ
    (1D), categorical distance = Ω-metric (2D).
-/

namespace OmegaProtocol.Vol36
open OmegaProtocol

/-- THEOREM: Law of Excluded Middle in a Boolean Algebra (GENUINE PROOF) -/
theorem boolean_excluded_middle (a : Prop) [Decidable a] : a ∨ ¬a :=
  em a

/-- COROLLARY: Topos Theory from Omega Protocol
    Subobjects = Q-Regions (0D)
    Morphisms = Φ (1D)
    Categorical distance = Ω-Metric (2D)
    Computation = Informational Viscosity (3D)
    Logic = RCOD Asymmetry (4D) -/
theorem topos_from_omega (a : Prop) [Decidable a] : a ∨ ¬a := by
  exact em a

/-- Consistency bridge (VOL36): the von Neumann entropy satisfies the bound `S ≤ π`.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `topos_entropy_bound` evoked. -/
theorem bridge_vol36_entropy_bounded (R : QRegion) : vonNeumannEntropy R ≤ Real.pi := by
  exact entropy_bounded R

/-- Consistency bridge (VOL36): the coupling `Φ` is symmetric.
    Proven against the concrete Q-region model fixed in `OmegaAxioms.lean`;
    it certifies internal coherence of the formalization and is NOT a
    derivation of the physical law the legacy name `topos_phi_symm` evoked. -/
theorem bridge_vol36_coupling_symm (R₁ R₂ : QRegion) : Φ R₁ R₂ = Φ R₂ R₁ := by
  exact Φ_symm R₁ R₂

-- ============================================================
-- A CONCRETE SKELETAL CATEGORY: FINITE SETS AS `Fin n`
-- Objects are natural numbers; maps `m → n` are all functions
-- `Fin m → Fin n`. Category laws, terminal object, binary
-- products, equalizers, exponentials, and characteristic maps.
-- Full pullback-based subobject classification is NOT claimed.
-- ============================================================

/-- Objects: `m : ℕ` stands for the set `Fin m`. -/
abbrev SkelObj := ℕ

/-- Morphisms: all functions between the representing finite types. -/
def SkelHom (m n : SkelObj) : Type := Fin m → Fin n

/-- Identity morphism. -/
def skelId (m : SkelObj) : SkelHom m m := id

/-- Composition (diagrammatic order: `f` then `g`). -/
def skelComp {m n k : SkelObj} (f : SkelHom m n) (g : SkelHom n k) :
    SkelHom m k :=
  g ∘ f

theorem skelComp_assoc {m n k l : SkelObj} (f : SkelHom m n)
    (g : SkelHom n k) (h : SkelHom k l) :
    skelComp (skelComp f g) h = skelComp f (skelComp g h) := rfl

theorem skelId_left {m n : SkelObj} (f : SkelHom m n) :
    skelComp (skelId m) f = f := rfl

theorem skelId_right {m n : SkelObj} (f : SkelHom m n) :
    skelComp f (skelId n) = f := rfl

-- ------------------------------------------------------------
-- Terminal object
-- ------------------------------------------------------------

/-- The terminal object: the one-point set. -/
abbrev skelTerminal : SkelObj := 1

/-- Every object has a unique map into the terminal object. -/
theorem skel_terminal (m : SkelObj) :
    ∃! _ : SkelHom m skelTerminal, True := by
  refine ⟨fun _ => (⟨0, by decide⟩ : Fin 1), True.intro, fun g _ => ?_⟩
  funext x
  have hlt : (g x).val < 1 := (g x).isLt
  have h0 : (g x).val = 0 := by omega
  exact (Fin.ext h0).symm

-- ------------------------------------------------------------
-- Binary products
-- ------------------------------------------------------------

/-- Binary product of objects. -/
def skelProd (m n : SkelObj) : SkelObj := m * n

/-- First and second projections. -/
noncomputable def skelFst {m n : SkelObj} : SkelHom (skelProd m n) m :=
  fun p => (finProdFinEquiv.symm p).1

noncomputable def skelSnd {m n : SkelObj} : SkelHom (skelProd m n) n :=
  fun p => (finProdFinEquiv.symm p).2

/-- Pairing of two maps. -/
noncomputable def skelPair {z m n : SkelObj} (f : SkelHom z m)
    (g : SkelHom z n) : SkelHom z (skelProd m n) :=
  fun x => finProdFinEquiv (f x, g x)

theorem skelProd_beta_fst {z m n : SkelObj} (f : SkelHom z m)
    (g : SkelHom z n) :
    skelComp (skelPair f g) skelFst = f := by
  funext x
  show (finProdFinEquiv.symm (finProdFinEquiv (f x, g x))).1 = f x
  rw [Equiv.symm_apply_apply]

theorem skelProd_beta_snd {z m n : SkelObj} (f : SkelHom z m)
    (g : SkelHom z n) :
    skelComp (skelPair f g) skelSnd = g := by
  funext x
  show (finProdFinEquiv.symm (finProdFinEquiv (f x, g x))).2 = g x
  rw [Equiv.symm_apply_apply]

theorem skelProd_unique {z m n : SkelObj} (f : SkelHom z m)
    (g : SkelHom z n) (h : SkelHom z (skelProd m n))
    (h1 : skelComp h skelFst = f) (h2 : skelComp h skelSnd = g) :
    h = skelPair f g := by
  funext x
  have hpair : finProdFinEquiv.symm (h x) = (f x, g x) := by
    apply Prod.ext
    · have h1x := congrFun h1 x
      exact h1x
    · have h2x := congrFun h2 x
      exact h2x
  calc h x = finProdFinEquiv (finProdFinEquiv.symm (h x)) :=
        (Equiv.apply_symm_apply _ _).symm
    _ = finProdFinEquiv (f x, g x) := by rw [hpair]
    _ = skelPair f g x := rfl

/-- Products exist with the usual universal property. -/
theorem skel_has_products (z m n : SkelObj) (f : SkelHom z m)
    (g : SkelHom z n) :
    ∃! h : SkelHom z (skelProd m n),
      skelComp h skelFst = f ∧ skelComp h skelSnd = g := by
  refine ⟨skelPair f g,
    ⟨skelProd_beta_fst f g, skelProd_beta_snd f g⟩, fun h hh => ?_⟩
  exact skelProd_unique f g h hh.1 hh.2

-- ------------------------------------------------------------
-- Equalizers
-- ------------------------------------------------------------

/-- Equalizer object: the cardinality of the subtype where `f` and `g`
    agree. -/
noncomputable def skelEq (m n : SkelObj) (f g : SkelHom m n) : SkelObj :=
  Fintype.card { x : Fin m // f x = g x }

/-- The equalizer inclusion. -/
noncomputable def skelEqIncl {m n : SkelObj} (f g : SkelHom m n) :
    SkelHom (skelEq m n f g) m :=
  fun i => (Fintype.equivFin { x : Fin m // f x = g x }.symm i).1

/-- The inclusion equalizes the pair. -/
theorem skelEq_fork {m n : SkelObj} (f g : SkelHom m n) :
    skelComp (skelEqIncl f g) f = skelComp (skelEqIncl f g) g := by
  funext i
  exact (Fintype.equivFin { x : Fin m // f x = g x }.symm i).2

/-- The universal factorization through the equalizer. -/
noncomputable def skelEqFactor {z m n : SkelObj} {f g : SkelHom m n}
    (h : SkelHom z m) (hh : skelComp h f = skelComp h g) :
    SkelHom z (skelEq m n f g) :=
  fun x => Fintype.equivFin { x : Fin m // f x = g x } ⟨h x, congrFun hh x⟩

theorem skelEq_factor_triangle {z m n : SkelObj} {f g : SkelHom m n}
    (h : SkelHom z m) (hh : skelComp h f = skelComp h g) :
    skelComp (skelEqFactor h hh) (skelEqIncl f g) = h := by
  funext x
  show ((Fintype.equivFin { x : Fin m // f x = g x }).symm
      ((Fintype.equivFin { x : Fin m // f x = g x }) ⟨h x, congrFun hh x⟩)).1
    = h x
  rw [Equiv.symm_apply_apply]

theorem skelEq_unique {z m n : SkelObj} {f g : SkelHom m n}
    (h : SkelHom z m) (hh : skelComp h f = skelComp h g)
    (k : SkelHom z (skelEq m n f g))
    (hk : skelComp k (skelEqIncl f g) = h) :
    k = skelEqFactor h hh := by
  funext x
  have hsymm : (Fintype.equivFin { x : Fin m // f x = g x }).symm (k x) =
      ⟨h x, congrFun hh x⟩ := by
    apply Subtype.ext
    have hkx := congrFun hk x
    exact hkx
  have hcongr := congrArg
    (fun s : { x : Fin m // f x = g x } => (Fintype.equivFin _) s) hsymm
  rw [Equiv.apply_symm_apply] at hcongr
  have hfactor : skelEqFactor h hh x =
      (Fintype.equivFin { x : Fin m // f x = g x }) ⟨h x, congrFun hh x⟩ := rfl
  rw [hfactor]
  exact hcongr

/-- Equalizers exist with the usual universal property. -/
theorem skel_has_equalizers (z m n : SkelObj) (f g : SkelHom m n)
    (h : SkelHom z m) (hh : skelComp h f = skelComp h g) :
    ∃! k : SkelHom z (skelEq m n f g),
      skelComp k (skelEqIncl f g) = h := by
  refine ⟨skelEqFactor h hh, skelEq_factor_triangle h hh,
    fun k hk => skelEq_unique h hh k hk⟩

-- ------------------------------------------------------------
-- Exponentials
-- ------------------------------------------------------------

/-- Exponential object: the cardinality of the function type. -/
noncomputable def skelExp (m n : SkelObj) : SkelObj :=
  Fintype.card (Fin m → Fin n)

/-- Evaluation map. -/
noncomputable def skelEval {m n : SkelObj} :
    SkelHom (skelProd (skelExp m n) m) n :=
  fun p => (Fintype.equivFin (Fin m → Fin n)).symm
    (finProdFinEquiv.symm p).1 (finProdFinEquiv.symm p).2

/-- Currying. -/
noncomputable def skelCurry {z m n : SkelObj}
    (f : SkelHom (skelProd z m) n) : SkelHom z (skelExp m n) :=
  fun x => Fintype.equivFin (Fin m → Fin n)
    (fun y => f (finProdFinEquiv (x, y)))

/-- Functorial action on products in the first argument. -/
noncomputable def skelProdMap {z m E : SkelObj} (a : SkelHom z E) :
    SkelHom (skelProd z m) (skelProd E m) :=
  fun p => finProdFinEquiv (a (finProdFinEquiv.symm p).1,
    (finProdFinEquiv.symm p).2)

theorem skelProdMap_apply {z m E : SkelObj} (u : SkelHom z E)
    (x : Fin z) (y : Fin m) :
    skelProdMap u (finProdFinEquiv (x, y)) =
      finProdFinEquiv (u x, y) := by
  show finProdFinEquiv (u (finProdFinEquiv.symm
      (finProdFinEquiv (x, y))).1,
    (finProdFinEquiv.symm (finProdFinEquiv (x, y))).2) = _
  rw [Equiv.symm_apply_apply finProdFinEquiv]

theorem skelEval_apply_pair {m n : SkelObj} (t : Fin (skelExp m n))
    (y : Fin m) :
    skelEval (finProdFinEquiv (t, y)) =
      (Fintype.equivFin (Fin m → Fin n)).symm t y := by
  show (Fintype.equivFin (Fin m → Fin n)).symm
      (finProdFinEquiv.symm (finProdFinEquiv (t, y))).1
      (finProdFinEquiv.symm (finProdFinEquiv (t, y))).2 = _
  rw [Equiv.symm_apply_apply finProdFinEquiv]

/-- Beta law, pointwise: evaluation after currying. -/
theorem skelExp_beta_pt {z m n : SkelObj} (f : SkelHom (skelProd z m) n)
    (x : Fin z) (y : Fin m) :
    skelEval (finProdFinEquiv (skelCurry f x, y)) =
      f (finProdFinEquiv (x, y)) := by
  rw [skelEval_apply_pair]
  have hcurry : skelCurry f x = (Fintype.equivFin (Fin m → Fin n))
      (fun w => f (finProdFinEquiv (x, w))) := rfl
  rw [hcurry, Equiv.symm_apply_apply]

/-- Beta law as an equation of morphisms. -/
theorem skelExp_beta {z m n : SkelObj} (f : SkelHom (skelProd z m) n) :
    skelComp (skelProdMap (skelCurry f)) skelEval = f := by
  funext p
  have hpair : finProdFinEquiv ((finProdFinEquiv.symm p).1,
      (finProdFinEquiv.symm p).2) = p := by
    rw [Prod.mk.eta (finProdFinEquiv.symm p)]
    exact Equiv.apply_symm_apply _ _
  have hcomp : (skelComp (skelProdMap (skelCurry f)) skelEval) p =
      skelEval (skelProdMap (skelCurry f) p) := rfl
  rw [hcomp, ← hpair, skelProdMap_apply, skelExp_beta_pt]

/-- Eta law: currying is the unique factorization through evaluation. -/
theorem skelExp_unique {z m n : SkelObj} (f : SkelHom (skelProd z m) n)
    (u : SkelHom z (skelExp m n))
    (hu : ∀ x y, skelEval (finProdFinEquiv (u x, y)) =
      f (finProdFinEquiv (x, y))) :
    u = skelCurry f := by
  funext x
  have hfun : (Fintype.equivFin (Fin m → Fin n)).symm (u x) =
      (fun y => f (finProdFinEquiv (x, y))) := by
    funext y
    have hxy := hu x y
    rw [skelEval_apply_pair] at hxy
    exact hxy
  have hcongr := congrArg
    (fun g : Fin m → Fin n => (Fintype.equivFin _) g) hfun
  rw [Equiv.apply_symm_apply] at hcongr
  have hcurry : skelCurry f x = (Fintype.equivFin (Fin m → Fin n))
      (fun y => f (finProdFinEquiv (x, y))) := rfl
  rw [hcurry]
  exact hcongr

-- ------------------------------------------------------------
-- Characteristic maps (subobject-classifier property for subsets)
-- ------------------------------------------------------------

/-- The truth-value object: booleans as `Fin 2`. -/
abbrev skelOmega : SkelObj := 2

/-- The `true` point. -/
def skelTrue : SkelHom skelTerminal skelOmega :=
  fun _ => (⟨1, by decide⟩ : Fin 2)

/-- Characteristic map of a decidable subset of `Fin n`. -/
def skelChar {n : SkelObj} (S : Fin n → Prop) [DecidablePred S] :
    SkelHom n skelOmega :=
  fun x => if S x then (⟨1, by decide⟩ : Fin 2) else (⟨0, by decide⟩ : Fin 2)

theorem skelChar_mem {n : SkelObj} (S : Fin n → Prop) [DecidablePred S]
    (x : Fin n) :
    skelChar S x = (⟨1, by decide⟩ : Fin 2) ↔ S x := by
  unfold skelChar
  by_cases h : S x
  · rw [if_pos h]
    exact ⟨fun _ => h, fun _ => rfl⟩
  · rw [if_neg h]
    constructor
    · intro h01
      have hcon := congrArg Fin.val h01
      have hne : ¬ (Fin.val (⟨0, by decide⟩ : Fin 2)) =
        (Fin.val (⟨1, by decide⟩ : Fin 2)) := by decide
      exact absurd hcon hne
    · intro hs
      exact absurd hs h

/-- Characteristic maps are the unique classifiers of subsets. -/
theorem skelChar_unique {n : SkelObj} (S : Fin n → Prop) [DecidablePred S]
    (χ : SkelHom n skelOmega)
    (hχ : ∀ x, (χ x = (⟨1, by decide⟩ : Fin 2) ↔ S x)) :
    χ = skelChar S := by
  funext x
  by_cases h : S x
  · have e1 : χ x = (⟨1, by decide⟩ : Fin 2) := (hχ x).mpr h
    have e2 : skelChar S x = (⟨1, by decide⟩ : Fin 2) :=
      (skelChar_mem S x).mpr h
    rw [e1, e2]
  · have n1 : χ x ≠ (⟨1, by decide⟩ : Fin 2) :=
      fun he => h ((hχ x).mp he)
    have n2 : skelChar S x ≠ (⟨1, by decide⟩ : Fin 2) :=
      fun he => h ((skelChar_mem S x).mp he)
    have z1 : χ x = (⟨0, by decide⟩ : Fin 2) := by
      have hlt : (χ x).val < 2 := (χ x).isLt
      match hval : (χ x).val with
      | 0 => exact Fin.ext hval
      | 1 =>
        exfalso
        apply n1
        exact Fin.ext hval
      | _ + 2 => omega
    have z2 : skelChar S x = (⟨0, by decide⟩ : Fin 2) := by
      have hlt : (skelChar S x).val < 2 := (skelChar S x).isLt
      match hval : (skelChar S x).val with
      | 0 => exact Fin.ext hval
      | 1 =>
        exfalso
        apply n2
        exact Fin.ext hval
      | _ + 2 => omega
    rw [z1, z2]

end OmegaProtocol.Vol36
