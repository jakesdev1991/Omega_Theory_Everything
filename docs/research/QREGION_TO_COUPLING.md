# Q-region → scalar-Gauss-Bonnet coupling: derivation record

**Status (2026-11-06):** first machine-checked bridge from the Q-region overlap
`Φ` to the SGB coupling `η`. Committed as `lean_proofs/QRegionCoupling.lean`
(build green, 0 sorry, 0 axioms; ToE = 8731 jobs).

## The chain

All steps except the last are definitions already in the corpus
(`OmegaAxioms`, `DynamicCODScale`):

1. Pair overlap: `Φ R₁ R₂ = min d₁ d₂` for distinct regions, `1` for self
2. Mutual information: `I = Φ(1-Φ)·(S₁+S₂)/2`
3. Planck factor: `ℓ_P(Φ) = ℓ_P0·√(1-Φ²)` (canonical env: `ℓ_P0 = 1`)
4. Curvature scale: `𝒢 ~ 1/ℓ_P⁴` at the pair overlap
5. **Coupling ansatz (the one unproven step):** `η = ℓ_P(Φ)²`

## What is proved in Lean

- `etaFromPhi_eq`: η(Φ) = ℓ_P0²·(1-Φ²)
- `etaFromPhi_nonneg`, `etaFromPhi_le`: 0 ≤ η ≤ ℓ_P0²
- `etaFromPhi_zero` / `etaFromPhi_one`: η(0) = ℓ_P0², η(1) = 0
- `etaFromPhi_anti`: η strictly decreasing in Φ on [0,1]
- `etaPair_eq`, `etaPair_pos` (for Φ<1), `etaPair_self`: η(R,R) = 0
- `thresholdOverlap`: Φ* = √(1 - λ₁M²/ℓ_P0²) and
  `etaOverM2_at_threshold`: η(Φ*)/M² = λ₁ exactly

## Honest interpretation

- This **derives η from Φ**, closing the "η is not derived from the Q-region
  benchmarks" gap flagged in GAUSS_BONNET_SCALARIZATION.md §2 — but only by
  *identifying* η with ℓ_P(Φ)². That identification is a modeling choice,
  now explicit and machine-checked, not hidden.
- It **predicts the overlap** required for scalarization at mass M:
  Φ* = √(1 - λ₁M²/ℓ_P0²), with λ₁ ≈ 0.7256092. For astrophysical M ≫ ℓ_P0
  this is unphysical (imaginary), so the SGB branch is only relevant for
  Planck-scale black holes — a real constraint the derivation makes visible.
- It does NOT derive M, does not fix ℓ_P0, and does not prove stability.
