# Empirical preregistration: frozen items

**Status (2026-11-06):** the six freeze items from
`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md` §5, now frozen as far as
the current corpus allows. This is a project-internal freeze, not an external
registration (e.g. OSF); external registration remains open.

## Frozen items

1. **Theory choice:** Φ is a dynamical scalar in a scalar–Gauss–Bonnet branch
   of the gravitational action (no direct matter coupling, γ=β=1 on the
   zero-scalar branch). Metric frame: Jordan frame with minimal matter.
   Anchored in `lean_proofs/SGBZeroMode.lean` + `QRegionCoupling.lean`.
2. **Equations and parameters:** action
   `S = ∫√-g [R/(16πG) - ½Z(∇Φ)² - V(Φ) + ½ηΦ²𝒢] + S_m`, massless quadratic
   coupling. One dimensionless parameter λ = η/M². Threshold λ₁ ≈ 0.7256092
   (shooting) / 0.7256128 (Rayleigh–Ritz), uncertainty ~1e-6. Stability domain:
   linear onset only — no nonlinear hairy branch found by shooting
   (`research/nonlinear_hair_negative_result.py`).
3. **Observable map:** η is identified with ℓ_P(Φ)² = ℓ_P0²(1-Φ²)
   (`QRegionCoupling.lean`); the threshold predicts Φ* = √(1-λ₁M²/ℓ_P0²).
   No target curve is imported into the definition of Φ, κ, or any fit.
4. **Data and comparator:** NOT YET NAMED — no immutable dataset release, hashes,
   or holdout selected. Open. GR baseline: Schwarzschild (γ=β=1) on the
   zero-scalar branch.
5. **Decision rule:** NOT YET PUBLISHED — no likelihood/statistic or threshold
   registered. Open. Negative/null outcomes must be reported (see item 6).
6. **Reproduction:** all computations in `research/` are scripted and
   pytest-gated (`tests/`); Lean corpus builds green with 0 sorry / 0 axioms.
   Second clean run: CI rebuild from frozen commit.

## What this freeze does NOT claim

- No independent-data analysis is claimed.
- The H₀-calibrated depletion simulation is not a registered prediction.
- Items 4 and 5 are honestly open; registering them requires an external
  dataset choice and a statistic, which are decisions for the operator.
