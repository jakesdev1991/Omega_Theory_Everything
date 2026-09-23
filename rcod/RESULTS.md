# RCOD benchmark results — label-noise recovery

**Date:** 2026-09-23 · **Code:** [`rcod_optimizer.py`](rcod_optimizer.py) +
[`benchmark_noise_recovery.py`](benchmark_noise_recovery.py)
**Verdict up front:** the designed claim — RCOD governance improves loss
recovery after a distribution shock — is **not supported** on this workload.
Two falsifiable sub-findings survive, and one unintended effect is
interesting enough to warrant separate study. Details below.

## Setup

Teacher-student synthetic task. A fixed teacher MLP (32→64→4, tanh) labels
Gaussian inputs; a student MLP (32→128→128→4, ~21k params, ReLU) trains
with AdamW (lr 1e-3, batch 256) for 3000 steps. In steps 1200–1700, 50% of
each batch's labels are flipped at random (the shock); afterwards labels are
clean again. 3 seeds; batch order identical across methods per seed. Test
loss/accuracy measured on a clean held-out set every 20 steps.

Methods:

| Method | State mode | Thresholds |
|---|---|---|
| `baseline` | — (plain AdamW) | — |
| `rcod-default` | weights (`s=w/‖w‖`), spec-as-written | 0.15 / 0.70 / 0.35 |
| `rcod-updates-default` | updates (`s=Δw/‖Δw‖`) | 0.15 / 0.70 / 0.35 |
| `rcod-updates-calibrated` | updates | clean-phase quantiles (median / p97.5 / p97.5) |

## Outcomes (mean over 3 seeds; full per-seed data in `results_noise_recovery.csv`)

| Method | Pre-noise best | Peak in noise | Damage | Recovery (steps) | Final loss | Final acc |
|---|---|---|---|---|---|---|
| baseline | 0.211 | 1.239 | +1.028 | 140 | 0.318 | 91.1% |
| rcod-default | 0.211 | 1.239 | +1.028 | 140 | 0.318 | 91.1% |
| rcod-updates-default | 0.244 | 1.079 | +0.834 | 487 | **0.232** | 91.2% |
| rcod-updates-calibrated | 0.211 | 1.227 | +1.016 | 193 | 0.311 | 91.3% |

## Governor telemetry (3 seeds pooled)

| Method | Regimes (steps) | Reversals | Mean α | μ̄ clean | μ̄ noise |
|---|---|---|---|---|---|
| rcod-default | FLOW 8602 / VISC 98 / SHOCK 0 | 0 | — | 0.028 | 0.042 |
| rcod-updates-default | VISC 371 / SHOCK 8329 | 8329 | 0.64 | 0.860 | 0.826 |
| rcod-updates-calibrated | FLOW 2737 / VISC 4773 / SHOCK 290 | 290 | 0.85 | 0.988 | 0.940 |

Shock timing, `rcod-updates-calibrated`: **clean phase 29 (~1.2% of governed
steps) · noise window 131 (~8.7%) · post-noise 130 (~3.3%)** — a ~7×
enrichment of SHOCK triggers inside the noise window.

## Findings

**F1 — Spec-as-written is inert at this scale.** With weight-vector states,
the shock moves μ̄ from 0.028 to 0.042 — far below the document's FLOW
boundary of 0.15, let alone SHOCK at 0.70. The governor never engages:
regime counts show zero SHOCKs and the trajectory is byte-identical to
baseline. The accumulated weight norm dominates per-step rotations, so the
normalized full-weight vector is a very stiff compass.

**F2 — Update-space μ̄ is dominated by batch gradient noise.** In update
space the *clean-phase* μ̄ already runs 0.86–0.99 (batch-256 gradient
directions are only weakly correlated step to step), so the document's fixed
SHOCK threshold (0.70) fires on ~96% of steps — including 3258 shocks during
clean training. Fixed universal thresholds are unusable in update space;
calibration per workload is mandatory.

**F3 — After calibration there is a real but weak detection signal.** With
thresholds set from clean-phase quantiles, SHOCK triggers rise ~7× inside
the noise window (8.7% of steps vs 1.2% clean). The metrics do carry
shock information — it is the intervention, not only the detector, that
fails next.

**F4 — The reversal intervention does not improve recovery.** Despite 131
noise-window reversals at mean α≈0.85, damage (+1.016 vs +1.028), recovery
(193 vs 140 steps) and final loss (0.311 vs 0.318) are within seed noise of
baseline. Plausible cause: **clean-checkpoint pollution** — any VISCOSITY
step refreshes `w_clean`, including steps inside the slowly developing noise
regime, so reversals increasingly target already-damaged weights. The
blueprint's own telemetry-section warning ("treat telemetry as noisy
evidence, not truth") applies to the governor's own checkpoint policy.

**F5 — The hyperactive variant behaves as a damped optimizer, not a
detector.** Perpetual α≈0.64 rollbacks (rcod-updates-default) act like a
heavy implicit regularizer: less peak damage (+0.83), much slower recovery
(487 steps), slower clean learning (pre-best 0.244), but the best final
loss (0.232 vs 0.318) at identical accuracy (~91%). This resembles
EMA/Polyak weight averaging and the Lookahead optimizer more than shock
detection, and is worth a separate controlled study — but it is not the
mechanism the RCOD document describes.

## Threats to validity

One task family (teacher-student classification), one model scale (~21k
params), one shock type (50% label flips), 3 seeds, no significance tests,
recovery tolerance ±10% of pre-noise best, thresholds and windows chosen
per the design document plus one calibration heuristic. None of this
establishes that RCOD governance cannot help on other workloads (LLM
fine-tuning, RL, larger batches, slower-onset distribution shift); it
establishes that it does not help here, and that the default thresholds
cannot work here.

## Next experiments (in priority order)

1. **Loss-aware / hysteresis clean-checkpoint policy** — stop refreshing
   `w_clean` when recent SHOCK rate is elevated; measure whether F4 flips.
2. **Larger batches** to lower the gradient-noise floor and widen the
   clean/noise μ̄ separation before testing interventions again.
3. **Per-layer state metrics** instead of the concatenated flat vector.
4. **EMA-smoothed update states** to denoise μ̄ before thresholding.
5. **F5 follow-up:** isolate the damped-optimizer effect (constant-α
   rollback control vs SHOCK-gated rollback) against Lookahead/EMA baselines.
6. More seeds + paired significance tests once a configuration shows promise.

## Reproduce

```bash
python benchmark_noise_recovery.py --csv results_noise_recovery.csv
```

Requires `torch` (CPU sufficient); see [`README.md`](README.md). Full
per-eval-step trajectories: `results_noise_recovery.csv`.
