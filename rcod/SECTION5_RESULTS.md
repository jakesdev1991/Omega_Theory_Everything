# RCOD benchmark results — §5 metrics-as-features shock prediction

**Date:** 2026-10-19/20 · **Design:** [`section5/DESIGN.md`](section5/DESIGN.md)
(pre-registered before Run 1; Amendments 1–4 recorded in the same file).
**Code:** [`section5/swarm.py`](section5/swarm.py),
[`section5/experiment.py`](section5/experiment.py). **Raw data:**
`section5/results*.json` (per-seed).
**Verdict up front:** the §5 claim of `RCOD_Operational_Logic.pdf` — that the
RCOD swarm metrics ([Median μ, Macro μ, Swarm Dissonance]) work as ML features
for predicting shock onset — is **not supported**. Four variants (single-level,
stacked, nested-temporal, dyadic ladder) each failed pre-registered criteria
against a plain rolling-statistics baseline. Per the design's pre-registered
one-shot rule (Amendment 4), the thread is **closed**.
**Provenance:** run externally at the owner's request (the test
[`RESULTS.md`](RESULTS.md) never ran — it tested the governor intervention,
not the §5 feature use) and filed into the repo on 2026-10-20. Environment:
Python 3, numpy 2.5.4, scikit-learn 1.9.1.

## Common setup (Runs 2–5)

Synthetic series, T = 24,000 steps, AR(1) φ = 0.9 base; 8 shocks per series.
Each shock is preceded by a 150-step precursor ramp (volatility 1×→3×, or mean
drift 0→+2σ), then the full break (volatility ×6 or mean +4σ, 600 steps).
Onset = break point; positive label = onset within the next 30 steps (~1%
positive rate). Time-ordered 60/20/20 split; L2 logistic regression; 10 seeds;
primary metric PR-AUC; paired Wilcoxon across seeds, α = 0.05; decision
threshold tuned on validation for F1.

## Run 1 — discarded (design flaw in the test, documented in Amendment 1)

Instantaneous shocks with no precursor phase: the primary target was
unlearnable by construction (all arms PR-AUC ≈ 0.00–0.03). Numbers discarded
as uninformative; the precursor design of Runs 2–5 replaces it.

## Run 2 — single-level swarm features

Arms: (A) RCOD trio + lags {0, 5, 10, 20}; (B) dissonance-only + lags;
(C) raw rolling statistics (mean/std/lag-1 autocorrelation/max |Δ|, 60-step
window) + lags.

| arm | mean PR-AUC | sd |
|---|---|---|
| A: RCOD trio | 0.036 | 0.058 |
| B: dissonance-only | 0.022 | 0.041 |
| C: raw baseline | 0.015 | 0.018 |

- Wilcoxon A>C: p = 0.37 → **not significant.** A>B: p = 0.29 → **not
  significant.**
- A small MLP on arm A scored PR-AUC 0.020 — no better than logistic.
- Operationally: no arm achieves usable detection (false alarms 0.4–13.8 per
  1,000 steps at the F1-tuned threshold; most shocks missed).

**Verdict: negative.** Even with precursor phases designed to give the method
its best shot, RCOD features do not significantly beat plain rolling
statistics at predicting shock onsets 30 steps ahead, and the trio does not
beat dissonance alone on this target.

### Exploratory (not pre-registered): the PDF's literal target

Predicting *dissonance crossing its train p99* within 30 steps: trio PR-AUC
0.391 vs dissonance-only 0.283 vs base rate 0.086 — the trio wins on **all 10
seeds** (post-hoc Wilcoxon p = 0.001). Median/Macro μ do carry independent
signal about *future dissonance* beyond dissonance's own history. That is
predicting the metric family from itself, not predicting physical shocks —
an interesting lead, not a vindication of §5.

## Run 3 — stacked (hierarchical) swarms (Amendment 2)

The §3 swarm transform applied recursively: L1: 3 metric series → L2: 9 →
L3: 27 series, 108 features with lags. Identical data, seeds, task, splits;
all arms evaluated on identical index grids.

| arm | mean PR-AUC | sd |
|---|---|---|
| D: hierarchical (3-level) | 0.010 | 0.018 |
| A: single-level trio | 0.038 | 0.062 |
| C: raw baseline | 0.016 | 0.018 |

- Wilcoxon D>C: p = 0.875 → **not significant.** 43.9 false alarms / 1,000
  steps — unusable as a detector.

**Verdict: negative** — numerically worse than single-level.

*Post-hoc observation (ROC-AUC instability):* per-seed ROC-AUCs ranged
0.006–0.986 across arms; two seeds showed single-level ROC-AUC of
0.986 / 0.977 while PR-AUC sat near 0.1 — textbook rare-event flattery. On
small positive samples ROC-AUC is unstable enough to produce eye-catching
values from noise, which is why PR-AUC was pre-registered as primary.

## Run 4 — nested temporal partitions (Amendment 3)

The owner's original "3 levels deep" structure, as clarified by the owner:
a base span partitioned at three nested resolutions (implemented as L1: 3×84,
L2: 12×21, L3: 42×6 steps, proportional to the owner's illustrative
minute → 3×20 s → 20×3 s → 150×0.4 s). Per level, μ between consecutive
z-scored intervals; 14 features (4 most recent μ per level + cross-level
median/std of latest μ).

| arm | mean PR-AUC | sd |
|---|---|---|
| E: temporal hierarchy | 0.026 | 0.036 |
| A: single-level trio | 0.037 | 0.059 |
| C: raw baseline | 0.016 | 0.018 |

- Wilcoxon E>C: p = 0.367 → **not significant.** 30.0 false alarms / 1,000
  steps.

**Verdict: negative.**

## Run 5 — dyadic ladder: the precise structure (Amendment 4, one-shot)

The owner noted the Run 4 structure was improvised and required a precise
one. Run 5 uses a **dyadic ladder** — block sizes 128/32/8 steps
(2^7/2^5/2^3, every other octave), the standard parameter-free
multiresolution decomposition; there are no hand-picked ratios to tune.
Everything else identical to Run 4. Amendment 4's one-shot rule: this was
the final redesign; failure closes §5.

| arm | mean PR-AUC | sd |
|---|---|---|
| E′: dyadic hierarchy | 0.012 | 0.011 |
| A: single-level trio | 0.037 | 0.059 |
| C: raw baseline | 0.015 | 0.018 |

- Wilcoxon E′>C: p = 0.633 → **not significant.** ROC-AUC 0.502 — a coin
  flip. 33.7 false alarms / 1,000 steps; 8 of 10 test shocks missed.

**Verdict: negative** — the worst of the four variants. **§5 is closed.**

## Threats / limits

Synthetic single-family data; z-scored windows are blind to pure level shifts
(documented in the design); horizon n = 30 and the lag set are choices;
10 seeds. A different horizon or real data could change the picture, but the
burden is now on a positive demonstration. Minor: Run 2's raw-baseline arm
built lagged features with `np.roll`, contaminating the first 20 of ~14k
training rows with wraparound values (<0.2% of train rows; negligible, and
Runs 3–5 recompute all arms without it).

## Bottom line

The §5 feature-extraction claim, tested under pre-registration for the first
time, fails in all four variants. The governor intervention was already
negative ([`RESULTS.md`](RESULTS.md)). RCOD's honest status: two
independently killed applications, one exploratory lead (Median/Macro μ
predict future dissonance, Run 2 secondary), and the untouched F5
damped-optimizer thread — a different mechanism (optimizer dynamics, not
time-series features), unaffected by these results.
