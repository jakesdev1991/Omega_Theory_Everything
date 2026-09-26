# AI governor — prototype run results (2026-09-26)

**Verdict up front:** the first C++ governor run is a **negative result**,
and a useful one: it reproduces, in C++23, the exact failure mode the
Python benchmark already documented (`rcod/RESULTS.md` F5), and it
isolates the cause. No claim of benefit is made. This file exists so the
negative is on the record per the repo's own standard ("no formula
becomes valid merely because it is written mathematically").

## Setup

- Corpus: the Lean 4 volume sources, concatenated (333,167 bytes).
- Model: byte-level next-token MLP (context 4 bytes → one-hot ×256 →
  tanh hidden 128 → softmax 256), 164,224 parameters.
- Optimizer: plain SGD, lr 3e-3, batch 64. 4000 steps, fixed seed.
- Governor: RCOD multi-window (5/10/20/40), update-space states, spec
  thresholds (FLOW 0.15 / VISC 0.35 / SHOCK 0.70), with **both** fixes from
  `rcod/RESULTS.md` next-experiments enabled: (a) loss-gated clean-checkpoint
  refresh, (b) EMA-smoothed μ̄ (β=0.9) before thresholding.
- Baseline arm: same seed, governor thresholds raised out of reach
  (never engages).

## Outcome

| Arm | Final eval loss | Regimes (of 4000) | Mean α |
|---|---|---|---|
| baseline | **3.9401** | FLOW 4000 | 0 |
| governed | 5.5298 | WARMUP 39 / SHOCK 3961 | **1.0000** |

Delta (baseline − governed) = **−1.59** → the governor hurt at this scale.

## Diagnosis (the useful part)

The EMA smoothing (fix b) denoised μ̄ exactly as intended — but μ̄ now
lives at ~0.92–0.97 *continuously*, because batch-64 update directions
are weakly correlated step-to-step even in clean training (same floor as
the Python F2 finding). With the spec thresholds unchanged:

1. μ̄ ≥ 0.70 on essentially every step → SHOCK ~99% of steps.
2. α = clamp((μ̄ − 0.35)/(0.70 − 0.35)) saturates to 1.0 whenever μ̄ ≥ 0.70
   — which is always. Every step therefore rolls the weights back to the
   clean checkpoint; the model barely moves (eval 5.53 ≈ its step-400
   level).
3. The loss-gated checkpoint refresh (fix a) *widens* the damage here:
   whenever loss is within 10% of the EMA (rarely, given (2)), the
   polluted checkpoint gets refreshed, so even the rollback target is bad.

This is the F5 "hyperactive damped optimizer" mechanism, but worse: at
α=1.0 it is not damping, it is freezing. The two fixes must land
**together with threshold recalibration**, not with the spec constants:

## Next experiments (priority order)

1. **Calibrate thresholds from the clean-phase μ̄ distribution** (median /
   p97.5 of smoothed μ̄ during a governor-inert warmup), exactly as
   `rcod-updates-calibrated` did. Expected: SHOCK becomes rare, α < 1.
2. **Cap α** (e.g. α ≤ 0.5) so a reversal is always a partial blend, never
   a full rollback — the Reverse-With-Matter paper intent.
3. Larger batch (256+) to lower the gradient-noise floor in μ̄ (Python
   next-experiment #2) — cheap to test here since batch is a flag.
4. Only after 1–3: re-run the shock benchmark (inject label noise in a
   window) and measure detection enrichment *and* recovery, vs baseline.

## Reproduce

```bash
cd ai-governor
g++ -std=c++23 -O2 -march=native -Iinclude governor_train.cpp -lcrypto -o governor_train
cd .. && ./ai-governor/governor_train "lean_proofs/*.lean" 4000
```

Single seed, prototype scale, no significance tests. The verdict line in
the binary prints the same summary. Receipts (sha256 of weights+step+loss)
were minted every 1000 steps in both arms — the kind-31331 shape works;
the economics of the receipt loop is not in question, only the governor's
intervention quality.
