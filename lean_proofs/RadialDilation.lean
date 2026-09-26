import DynamicCODScale
import Mathlib

/-!
# Macroscopic dilation from the two-factor law: obstruction and reconciliation

Companion to `Omega_Theory_v4.0_Radial_Metric.md` (§§3–5) and
`Sim7_Radial_Metric.py`. `RadialMetric.lean` proved the finite-chain
two-factor law (`chainLength_const_scale`: proper length is scale × weight per
link) and the algebraic `Φ`-form Schwarzschild identities. This module proves
what those two facts *imply* for the macroscopic radial metric — the content
Sim7 checks numerically:

1. **Obstruction** (§4 of the note). If `ℓ_P(Φ)` is the only `Φ`-dependence of
   the spatial metric (uniform correlation decay, flat information metric),
   the areal component is `g_RR = (1 - Φ²)² ≤ 1` on the physical range
   (`gRR_lpOnly_le_one`), with equality only at the vacuum
   (`gRR_lpOnly_eq_one_imp`), and the Misner–Sharp mass is strictly negative
   inside (`misnerSharpLPOnly_neg`). The deviation of the conformal factor
   from flat is explicitly second order in `Φ`
   (`lpRatio_quadratic_deviation`), which is the Lean content of the note's
   `f'(0) = 0 ⟹ M_ADM = 0` remark: no linear term, no Newtonian far field.
2. **Reconciliation** (§§3, 5). The correlation-decay rate required by
   Schwarzschild, `κ/κ₀ = (1+Φ)²/√(1-Φ²)` (`kappaRequired`), reproduces the
   isotropic conformal factor exactly through the two-factor product
   (`kappa_lp_product`, `isotropic_conformal_reproduction`), outruns the
   inverse rendering scale in integrated form (`kappa_outruns_inverse_lp`:
   `κ₂ℓ₂ > κ₁ℓ₁` with explicit margin `(1+Φ₂)² > (1+Φ₁)²`, the integral of the
   note's `d ln κ/dΦ > Φ/(1-Φ²)` criterion), and yields areal dilation
   (`gRR_schwarzschild_gt_one`) with areal reciprocity (`areal_reciprocity`,
   i.e. `(-g_tt)·g_RR = 1`).
3. **Bridge** (`dilation_reconciliation`). One conjunction stating that on
   `(0, 1)`, `ℓ_P` alone contracts while `ℓ_P·κ` dilates — microscopic
   contraction and macroscopic dilation are compatible.

## Model assumptions (stated, not derived)
* The `Φ` convention and canonical profile are those of `DynamicCODScale`
  (`Φ = 0` vacuum, `ℓ_P(Φ) = ℓ_P0·√(1-Φ²)`); `lpRatio_eq_cod_ratio` ties the
  ratio used here to that kernel-checked module.
* The `Φ`-form Schwarzschild dictionary of §5 (`Φ := ψ - 1`, isotropic
  conformal factor minus one) is identified to match known physics, not
  derived from the Omega postulates. Only its algebraic consequences are
  proved here.
* The infinitesimal dilation criterion `d ln κ/dΦ > Φ/(1-Φ²)` is stated in the
  note; what is proved here is its integrated form
  (`kappa_outruns_inverse_lp`), which carries the same physical content
  without a derivative API.
-/

namespace RadialDilation

/-! ## Definitions -/

/-- Rendering-scale ratio `ℓ_P(Φ)/ℓ_P0 = √(1-Φ²)` from the canonical COD
profile of `DynamicCODScale`. -/
noncomputable def lpRatio (Φ : ℝ) : ℝ := Real.sqrt (1 - Φ ^ 2)

/-- Areal-gauge `g_RR` when `ℓ_P(Φ)` is the only `Φ`-dependence (flat
information metric): `(1-Φ²)²`. See §4 of the note. -/
noncomputable def gRR_lpOnly (Φ : ℝ) : ℝ := (1 - Φ ^ 2) ^ 2

/-- Quasi-local (Misner–Sharp) mass of time-symmetric spherical data,
`(R/2)(1 - 1/g_RR)`, evaluated on the `ℓ_P`-only metric. -/
noncomputable def misnerSharpLPOnly (R Φ : ℝ) : ℝ :=
  R / 2 * (1 - 1 / gRR_lpOnly Φ)

/-- Correlation-decay rate (in units of `κ₀`) required to reproduce
Schwarzschild through the two-factor law: `(1+Φ)²/√(1-Φ²)`. See §5. -/
noncomputable def kappaRequired (Φ : ℝ) : ℝ :=
  (1 + Φ) ^ 2 / Real.sqrt (1 - Φ ^ 2)

/-- Isotropic conformal factor `(1+Φ)²`: the two-factor product `ℓ_P·κ`
reproduces exactly this (`kappa_lp_product`). -/
noncomputable def conformalFactor (Φ : ℝ) : ℝ := (1 + Φ) ^ 2

/-- Areal-gauge `g_RR` of Schwarzschild written in `Φ = ψ - 1`:
`((1+Φ)/(1-Φ))²`. See §5. -/
noncomputable def gRR_schwarzschild (Φ : ℝ) : ℝ := ((1 + Φ) / (1 - Φ)) ^ 2

/-- Squared lapse `N² = ((1-Φ)/(1+Φ))²`, i.e. `-g_tt` in the `Φ` dictionary. -/
noncomputable def lapseSq (Φ : ℝ) : ℝ := ((1 - Φ) / (1 + Φ)) ^ 2

/-! ## Helpers -/

/-- On `[0, 1]`, `Φ² ≤ 1`. The product `Φ·(1-Φ) ≥ 0` gives `Φ² ≤ Φ ≤ 1`. -/
theorem sq_le_one_of_Icc (Φ : ℝ) (h0 : 0 ≤ Φ) (h1 : Φ ≤ 1) : Φ ^ 2 ≤ 1 := by
  nlinarith [mul_nonneg h0 (show (0 : ℝ) ≤ 1 - Φ by linarith)]

/-- `Φ² < 1` forces `-1 < Φ`: from `(Φ+1)² ≥ 0`, `2(Φ+1) ≥ 1-Φ² > 0`. -/
theorem neg_one_lt_of_sq_lt_one (Φ : ℝ) (h : Φ ^ 2 < 1) : -1 < Φ := by
  nlinarith [sq_nonneg (Φ + 1), h]

/-! ## 1. Obstruction: `ℓ_P` alone can only contract -/

/-- **§4 obstruction, part 1.** With uniform correlation decay the areal
component `(1-Φ²)²` never exceeds the flat value on the physical range. -/
theorem gRR_lpOnly_le_one (Φ : ℝ) (h0 : 0 ≤ Φ) (h1 : Φ ≤ 1) :
    gRR_lpOnly Φ ≤ 1 := by
  unfold gRR_lpOnly
  have hsq0 : 0 ≤ Φ ^ 2 := sq_nonneg Φ
  have hsq1 : Φ ^ 2 ≤ 1 := sq_le_one_of_Icc Φ h0 h1
  nlinarith [mul_nonneg hsq0 (show (0 : ℝ) ≤ 2 - Φ ^ 2 by linarith)]

/-- **§4 obstruction, part 2.** Equality holds only at the vacuum: from
`(1-Φ²)² = 1`, i.e. `Φ²(2-Φ²) = 0`, the `Φ² = 2` branch contradicts
`Φ² ≤ 1`, leaving `Φ = 0`. -/
theorem gRR_lpOnly_eq_one_imp (Φ : ℝ) (h0 : 0 ≤ Φ) (h1 : Φ ≤ 1)
    (h : gRR_lpOnly Φ = 1) : Φ = 0 := by
  unfold gRR_lpOnly at h
  have hsq1 : Φ ^ 2 ≤ 1 := sq_le_one_of_Icc Φ h0 h1
  have hfact : Φ ^ 2 * (2 - Φ ^ 2) = 0 := by
    have hring : Φ ^ 2 * (2 - Φ ^ 2) = 1 - (1 - Φ ^ 2) ^ 2 := by ring
    rw [hring, h, sub_self]
  rcases mul_eq_zero.mp hfact with hzero | htwo
  · have hmul : Φ * Φ = 0 := by
      rw [← pow_two]
      exact hzero
    rcases mul_eq_zero.mp hmul with h | h
    · exact h
    · exact h
  · linarith

/-- At the vacuum the `ℓ_P`-only metric is exactly flat. -/
theorem gRR_lpOnly_zero : gRR_lpOnly 0 = 1 := by
  unfold gRR_lpOnly
  norm_num

/-- **§4 obstruction, part 3.** The quasi-local mass of the `ℓ_P`-only metric
is strictly negative inside: `0 < g < 1` gives `1/g > 1`, so with `R > 0`,
`(R/2)(1-1/g) < 0`. -/
theorem misnerSharpLPOnly_neg (R Φ : ℝ) (hR : 0 < R)
    (h0 : 0 < Φ) (h1 : Φ < 1) : misnerSharpLPOnly R Φ < 0 := by
  unfold misnerSharpLPOnly
  have hsq1 : Φ ^ 2 < 1 := by
    nlinarith [mul_pos (show (0 : ℝ) < 1 - Φ by linarith)
      (show (0 : ℝ) < 1 + Φ by linarith)]
  have hgpos : 0 < gRR_lpOnly Φ := by
    unfold gRR_lpOnly
    have hpos : (0 : ℝ) < 1 - Φ ^ 2 := by linarith
    exact pow_pos hpos 2
  have hglt : gRR_lpOnly Φ < 1 := by
    have hle := gRR_lpOnly_le_one Φ (le_of_lt h0) (le_of_lt h1)
    have hne : gRR_lpOnly Φ ≠ 1 := by
      intro heq
      have hΦ := gRR_lpOnly_eq_one_imp Φ (le_of_lt h0) (le_of_lt h1) heq
      linarith
    exact lt_of_le_of_ne hle hne
  have hgne : gRR_lpOnly Φ ≠ 0 := ne_of_gt hgpos
  have hinv : (1 : ℝ) < 1 / gRR_lpOnly Φ := by
    field_simp
    linarith
  have hR2 : (0 : ℝ) < R / 2 := by linarith
  have hneg : (1 : ℝ) - 1 / gRR_lpOnly Φ < 0 := by linarith
  exact mul_neg_of_pos_of_neg hR2 hneg

/-- At the vacuum the quasi-local mass vanishes, as it must for flat data. -/
theorem misnerSharpLPOnly_zero (R : ℝ) : misnerSharpLPOnly R 0 = 0 := by
  unfold misnerSharpLPOnly
  rw [gRR_lpOnly_zero]
  norm_num

/-- **No linear term.** `(1-Ω)(1+Ω) = Φ²` where `Ω = ℓ_P/ℓ_P0`, so the
deviation `1-Ω = Φ²/(1+Ω)` is explicitly second order in `Φ`. This is the
Lean content of `f'(0) = 0 ⟹ M_ADM = 0`: with a harmonic profile `Φ = A/ρ`
there is no `1/ρ` term, hence no Newtonian far field. -/
theorem lpRatio_quadratic_deviation (Φ : ℝ) (hΦ : Φ ^ 2 < 1) :
    (1 - lpRatio Φ) * (1 + lpRatio Φ) = Φ ^ 2 := by
  unfold lpRatio
  have hsq : (Real.sqrt (1 - Φ ^ 2)) ^ 2 = 1 - Φ ^ 2 :=
    Real.sq_sqrt (by linarith)
  have hexpand : (1 - Real.sqrt (1 - Φ ^ 2)) * (1 + Real.sqrt (1 - Φ ^ 2))
      = 1 - (Real.sqrt (1 - Φ ^ 2)) ^ 2 := by ring
  rw [hexpand, hsq]
  ring

/-! ## 2. Reconciliation: the required `κ` dilates -/

/-- The required decay rate is strictly positive on `Φ² < 1`. -/
theorem kappaRequired_pos (Φ : ℝ) (hΦ : Φ ^ 2 < 1) :
    0 < kappaRequired Φ := by
  unfold kappaRequired
  have hsqrt : (0 : ℝ) < Real.sqrt (1 - Φ ^ 2) :=
    Real.sqrt_pos.mpr (by linarith)
  have hbase : (0 : ℝ) < 1 + Φ := by
    have hlo := neg_one_lt_of_sq_lt_one Φ hΦ
    linarith
  exact div_pos (pow_pos hbase 2) hsqrt

/-- At the vacuum the required rate is the baseline rate, `κ/κ₀ = 1`. -/
theorem kappaRequired_zero : kappaRequired 0 = 1 := by
  have h1 : ((1 : ℝ) - 0 ^ 2) = 1 := by norm_num
  have h2 : Real.sqrt ((1 : ℝ) - 0 ^ 2) = 1 := by rw [h1, Real.sqrt_one]
  unfold kappaRequired
  rw [h2]
  norm_num

/-- **The two-factor product.** `κ·ℓ_P = (1+Φ)²` exactly: the square roots
cancel, leaving the isotropic conformal factor. This is the discrete content
of `g_rr = (ℓ_P·κ)²` for the Schwarzschild dictionary. -/
theorem kappa_lp_product (Φ : ℝ) (hΦ : Φ ^ 2 < 1) :
    kappaRequired Φ * lpRatio Φ = conformalFactor Φ := by
  unfold kappaRequired lpRatio conformalFactor
  have hsne : Real.sqrt (1 - Φ ^ 2) ≠ 0 :=
    ne_of_gt (Real.sqrt_pos.mpr (by linarith))
  exact div_mul_cancel₀ _ hsne

/-- **Isotropic reproduction.** `(ℓ_P·κ)²` is the conformal factor squared,
i.e. the isotropic `g_ρρ = (1+Φ)⁴` of §5. -/
theorem isotropic_conformal_reproduction (Φ : ℝ) (hΦ : Φ ^ 2 < 1) :
    (lpRatio Φ * kappaRequired Φ) ^ 2 = (conformalFactor Φ) ^ 2 := by
  have e := kappa_lp_product Φ hΦ
  have e' : lpRatio Φ * kappaRequired Φ = conformalFactor Φ := by
    rw [mul_comm (lpRatio Φ) (kappaRequired Φ)]
    exact e
  rw [e']

/-- The conformal factor is strictly increasing in `Φ` (on `Φ² < 1`, where
`1+Φ > 0`): `a²-b² = (a-b)(a+b)` with both factors positive. -/
theorem conformalFactor_strictMono (Φ₁ Φ₂ : ℝ)
    (h1 : Φ₁ ^ 2 < 1) (hlt : Φ₁ < Φ₂) :
    conformalFactor Φ₁ < conformalFactor Φ₂ := by
  unfold conformalFactor
  have h1lo : (-1 : ℝ) < Φ₁ := neg_one_lt_of_sq_lt_one Φ₁ h1
  have e : (1 + Φ₂) ^ 2 - (1 + Φ₁) ^ 2
      = ((1 + Φ₂) - (1 + Φ₁)) * ((1 + Φ₂) + (1 + Φ₁)) := by ring
  have hdiff : (0 : ℝ) < (1 + Φ₂) - (1 + Φ₁) := by linarith
  have hsum : (0 : ℝ) < (1 + Φ₂) + (1 + Φ₁) := by linarith
  have hprod : (0 : ℝ) < (1 + Φ₂) ^ 2 - (1 + Φ₁) ^ 2 := by
    rw [e]
    exact mul_pos hdiff hsum
  linarith

/-- **Integrated dilation criterion.** `κ` outruns the inverse rendering
scale: `κ₂ℓ₂ > κ₁ℓ₁` with explicit margin `(1+Φ₂)² > (1+Φ₁)²`. This is the
integral of the note's `d ln κ/dΦ > Φ/(1-Φ²)` over `[Φ₁, Φ₂]`. -/
theorem kappa_outruns_inverse_lp (Φ₁ Φ₂ : ℝ)
    (h1 : Φ₁ ^ 2 < 1) (h2 : Φ₂ ^ 2 < 1) (hlt : Φ₁ < Φ₂) :
    kappaRequired Φ₁ * lpRatio Φ₁ < kappaRequired Φ₂ * lpRatio Φ₂ := by
  rw [kappa_lp_product Φ₁ h1, kappa_lp_product Φ₂ h2]
  exact conformalFactor_strictMono Φ₁ Φ₂ h1 hlt

/-- **Areal dilation.** The Schwarzschild `g_RR = ((1+Φ)/(1-Φ))²` strictly
exceeds the flat value on `(0, 1)`: the ratio exceeds 1, and squaring a
number above 1 stays above 1. -/
theorem gRR_schwarzschild_gt_one (Φ : ℝ) (h0 : 0 < Φ) (h1 : Φ < 1) :
    1 < gRR_schwarzschild Φ := by
  unfold gRR_schwarzschild
  have hden : (0 : ℝ) < 1 - Φ := by linarith
  have hden_ne : (1 : ℝ) - Φ ≠ 0 := ne_of_gt hden
  have h3 : (1 : ℝ) < (1 + Φ) / (1 - Φ) := by
    field_simp
    linarith
  have ht_pos : (0 : ℝ) < (1 + Φ) / (1 - Φ) := by linarith
  have hmul := mul_lt_mul_of_pos_left h3 ht_pos
  rw [mul_one] at hmul
  have hpow : ((1 + Φ) / (1 - Φ)) ^ 2
      = ((1 + Φ) / (1 - Φ)) * ((1 + Φ) / (1 - Φ)) := pow_two _
  rw [hpow]
  linarith

/-- At the vacuum the Schwarzschild areal component is exactly flat. -/
theorem gRR_schwarzschild_zero : gRR_schwarzschild 0 = 1 := by
  unfold gRR_schwarzschild
  norm_num

/-- Reciprocal powers cancel: `(a/b)²·(b/a)² = 1` for nonzero `a, b`. -/
theorem div_pow_mul_inv_pow (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
    (a / b) ^ 2 * (b / a) ^ 2 = 1 := by
  rw [pow_two, pow_two, div_mul_div_comm, div_mul_div_comm, div_mul_div_comm]
  have hnum : (a * a) * (b * b) = (b * b) * (a * a) := by ring
  rw [hnum]
  exact div_self (mul_ne_zero (mul_ne_zero hb hb) (mul_ne_zero ha ha))

/-- **Areal reciprocity.** `N²·g_RR = 1`, i.e. `(-g_tt)·g_RR = 1` in the `Φ`
dictionary: the lapse and the areal component are exact reciprocals. -/
theorem areal_reciprocity (Φ : ℝ) (h1 : (1 : ℝ) + Φ ≠ 0)
    (h2 : (1 : ℝ) - Φ ≠ 0) :
    lapseSq Φ * gRR_schwarzschild Φ = 1 := by
  unfold lapseSq gRR_schwarzschild
  exact div_pow_mul_inv_pow _ _ h2 h1

/-- The ratio used here is exactly the kernel-checked COD profile divided by
its baseline: `lpRatio Φ = ℓ_P(Φ)/ℓ_P0`. -/
theorem lpRatio_eq_cod_ratio (env : DynamicCODScale.CODEnvironment) (Φ : ℝ) :
    lpRatio Φ = DynamicCODScale.dynamicPlanckCOD env Φ / env.lP0 := by
  have hne : env.lP0 ≠ 0 := ne_of_gt env.hlP0
  unfold lpRatio DynamicCODScale.dynamicPlanckCOD
  rw [mul_div_cancel_left₀ _ hne]

/-! ## 3. Bridge -/

/-- **Dilation reconciliation.** On `(0, 1)`: `ℓ_P` alone contracts
(`g_RR ≤ 1`), while the two-factor product dilates (`g_RR > 1`) with the
conformal factor `(1+Φ)²` behind it and exact lapse reciprocity. Microscopic
contraction and macroscopic dilation are compatible. -/
theorem dilation_reconciliation (Φ : ℝ) (h0 : 0 < Φ) (h1 : Φ < 1) :
    gRR_lpOnly Φ ≤ 1
      ∧ 1 < gRR_schwarzschild Φ
      ∧ kappaRequired Φ * lpRatio Φ = conformalFactor Φ
      ∧ lapseSq Φ * gRR_schwarzschild Φ = 1 := by
  have hΦ : Φ ^ 2 < 1 := by
    nlinarith [mul_pos (show (0 : ℝ) < 1 - Φ by linarith)
      (show (0 : ℝ) < 1 + Φ by linarith)]
  have h1' : (1 : ℝ) + Φ ≠ 0 := by
    have hlo := neg_one_lt_of_sq_lt_one Φ hΦ
    exact ne_of_gt (by linarith)
  have h2' : (1 : ℝ) - Φ ≠ 0 := ne_of_gt (by linarith)
  exact ⟨gRR_lpOnly_le_one Φ (le_of_lt h0) (le_of_lt h1),
    gRR_schwarzschild_gt_one Φ h0 h1,
    kappa_lp_product Φ hΦ,
    areal_reciprocity Φ h1' h2'⟩

end RadialDilation
