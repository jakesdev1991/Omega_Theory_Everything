# Pre-registration — RCOD §5 experiment: metrics-as-features shock prediction

**Date:** 2026-10-19 · **Status:** written BEFORE any code ran.
**Source hypothesis:** `RCOD_Operational_Logic.pdf` §5 — "DO NOT train the model
directly on raw time-series data. DO feed the model the extracted RCOD metrics:
[Median_Mu, Macro_Mu, Swarm_Dissonance]. Target: predict the onset of Swarm
Dissonance n-steps in the future."

**What the repo already tested (and killed):** `rcod/RESULTS.md` (2026-09-23)
tested the *governor intervention* (reversal on SHOCK) — negative result.
This experiment tests the *other* proposal in the PDF: RCOD metrics as
*predictive features*. It has never been run.

## Swarm implementation (per PDF, with one documented choice)

- Macro anchor window: 252 steps. Stochastic leaves: 12 windows, sizes drawn
  uniformly from {10..60} with fixed RNG seed (reproducible).
- State vector v_{t,w}: the w most recent values ending at t.
- Overlap Φ_w = cosine_similarity(v_{t,w}, v_{t-1,w}); novelty μ_w = sqrt(max(0, 1−Φ_w²)).
- Per-timestep aggregates: **Median μ** (median over windows), **Macro μ**
  (w=252), **Swarm Dissonance** = std(μ_w) across windows.
- **Documented choice:** each window is z-scored before cosine similarity.
  Rationale: on raw windows a nonzero series mean pins Φ≈1 and μ≈0
  (degenerate — same failure mode as RESULTS.md F1). Cost, stated up front:
  z-scoring blinds the metric to *pure level shifts*; mean-shift shocks are
  included in the data so this blind spot is measurable, not hidden.

## Data

- Synthetic series, T=24,000 steps: AR(1) φ=0.9 + Gaussian noise (seeded).
- 8 shock events per series, onsets spaced ≥1,500 apart, ≥1,500 from edges:
  alternating volatility shocks (σ×6 for 600 steps) and mean shifts (+4σ for
  600 steps). Ground-truth onset times known.
- 10 seeds (shock placement + noise redrawn).

## Task, features, models

- **Primary target:** y_t = 1 if a true shock onset falls in (t, t+30].
- **Secondary target (doc-literal):** y_t = 1 if Swarm Dissonance crosses its
  train-set p99 within (t, t+30].
- **Arms:** (A) RCOD trio [median, macro, dissonance] + lags {0,5,10,20};
  (B) dissonance-only + same lags (tests whether the trio beats the PDF's
  "primary indicator" alone); (C) raw baseline: rolling mean/std/lag-1
  autocorrelation/max|Δ| over last 60 steps + same lags (fair-dimensionality
  fight — the PDF says raw should lose).
- **Models:** primary = L2 logistic regression (linear probe = cleanest test
  of feature quality); secondary = small MLP on arm A only.
- **Split:** time-ordered 60/20/20; alarm threshold tuned on validation for
  F1; all metrics reported on test.

## Metrics

- Primary: **PR-AUC** on test (1% positive rate — ROC-AUC would flatter).
- Operational: median detection lead time (first alarm in [onset−200, onset];
  missed if none), false alarms per 1,000 steps (alarms outside any
  [onset−200, onset+50] window).
- Comparison: paired Wilcoxon signed-rank on PR-AUC across the 10 seeds.

## Kill criteria (decided now, applied after running)

1. If arm A PR-AUC is **not significantly greater** than arm C (p ≥ 0.05):
   **negative result** — RCOD features add nothing over raw rolling stats.
   The §5 lead closes as a published negative.
2. If arm B ≥ arm A (no significant difference): **partial negative** — the
   swarm aggregation adds nothing over dissonance alone.
3. If arm A wins significantly AND median lead time > 0 with false alarms
   ≤ 2 per 1,000 steps: **positive result**, worth a real-data follow-up.

## Amendment 1 (2026-10-19, after Run 1)

**Run 1 used instantaneous shocks with no precursor phase.** Onsets were
uniformly random times at which the series jumped discontinuously. A shock
with no precursor cannot be predicted from any feature set, by construction:
the pre-onset data is statistically identical to normal operation. All arms
(RCOD trio, dissonance-only, raw baseline, MLP) scored PR-AUC ≈ 0.00–0.03 on
the primary target. **This run tests nothing about the PDF's claim and its
primary-target numbers are discarded as uninformative** — the flaw is in the
experimental design, not (only) in the hypothesis.

The PDF's early-warning claim *presupposes* a pre-shock phase: "Scales are
decoupling. Micro-fractures are forming **before** macroscopic breakdown."
A fair test must include that phenomenon.

**Fix (Run 2):** each shock is preceded by a 150-step precursor ramp —
volatility ramps 1x→3x (vol shocks) or mean drifts 0→+2σ (mean shocks) —
before the full break (6x / +4σ for 600 steps). The predicted onset is the
break point; the 30-step positive label window then sits inside the
precursor ramp — exactly "catch the pre-shock phase." All arms, metrics,
splits, seeds, and kill criteria are otherwise unchanged.

**Addition:** the secondary (doc-literal) dissonance-crossing target gains a
dissonance-only control arm, since Run 1's trio-only score (PR-AUC 0.371)
is confounded by dissonance autocorrelation.

## Amendment 2 (2026-10-19, Run 3 — hierarchical swarms)

Jacob's recollection of the historical tokamak work: the evaluation metric
was AUC, the approach was machine-agnostic, and — critically — he used
**"rcod swarms to go 3 levels deep."** Runs 1–2 tested a *single-level*
swarm (PDF §3 as written). A 3-level stacked swarm is a different method
and is not covered by the Run 2 negative.

**Operationalization (documented interpretation):** "3 levels deep" =
recursive application of the §3 swarm transform:
- Level 1: swarm over the raw series → 3 metric series (median μ, macro μ,
  dissonance).
- Level 2: swarm over *each* level-1 series → 9 series.
- Level 3: swarm over each level-2 series → 27 series ("novelty of novelty
  of novelty" — acceleration of structural change).
Same window set (macro 252 + 12 leaves 10–60, fixed seed) at every level;
warmup ≈ 3×252 steps. Alternative reading (3 nested window scales in one
swarm) is noted but not implemented — the PDF already is multi-scale, so
stacking is the reading that adds something new.

**New arm D:** level-3 aggregates (median/macro/dissonance of each level-2
series → 27 series) + lags {0,5,10,20} = 108 features, logistic regression.
Arms A–C, task, splits, seeds unchanged.

**Metrics addition:** report ROC-AUC alongside PR-AUC on every arm.
Rationale: Jacob's historical metric was AUC (presumably ROC-AUC). On rare
events ROC-AUC flatters; if arm D shows high ROC-AUC with low PR-AUC, that
reconciles a "high AUC" memory with poor operational performance — and is
itself a finding about how the old result should be read.

**Kill criteria for arm D (primary metric remains PR-AUC):**
- If D PR-AUC significantly > C (raw) via paired Wilcoxon: POSITIVE —
  hierarchical swarms add predictive power; earns a real-data follow-up.
- Else: NEGATIVE for the hierarchical variant as well.

## Amendment 3 (2026-10-19, Run 4 — nested temporal partitions; CORRECTED interpretation)

**Correction:** Jacob clarified "3 levels deep" does NOT mean stacked/
recursive swarms (what Run 3 tested). It means **nested time intervals**:
one span partitioned at three resolutions — e.g. a minute as 3×20s
intervals, then 20×3s intervals, then 150×0.4s intervals (his numbers are
illustrative). Run 3's negative therefore stands only as a result about
stacked swarms, an exploratory variant Jacob did not propose; it does not
bear on his actual method. That misinterpretation was mine, and the record
is corrected here.

**Operationalization (documented):** base span = 252 steps (the PDF's macro
anchor; Jacob's "minute"). Three nested partition levels:
- L1: 3 blocks × 84 steps (Jacob's 3×20s),
- L2: 12 blocks × 21 steps (≈ his 20×3s),
- L3: 42 blocks × 6 steps (≈ his 150×0.4s).
Per level, for each block b≥1: Φ_b = cosine_similarity(z(block_b),
z(block_{b−1})); μ_b = sqrt(max(0,1−Φ_b²)) — the §3 primitive applied to
consecutive *intervals* rather than sliding windows. (The PDF's stochastic
leaves were multiple window sizes; here the multiple *levels* play that
role — documented mapping.)

**Features at time t (14):** the 4 most recently completed μ values per
level (12), plus cross-level median and std of the latest μ per level (2;
the PDF's "swarm dissonance" across levels rather than windows).

**New arm E** (temporal hierarchy) vs arms A (single-level trio) and C
(raw), same precursor data/seeds/task, all on identical index grids
(warmup = 5×84 = 420 steps so 4 coarse blocks + μ defined).

**Kill criteria for arm E (PR-AUC primary):** E significantly > C via
paired Wilcoxon → POSITIVE, earns real-data follow-up; else NEGATIVE for
the temporal-hierarchy variant.

## Amendment 4 (2026-10-20, Run 5 — dyadic ladder; LAST redesign)

**Context:** Jacob stated the Run 4 structure (and, by implication, the
historical one) was "made up... off the top of my head," and required that
"the structure needs to be something precise and not off the top of my
head." An arbitrary structure is a free parameter set; combined with
small-sample ROC-AUC it fully explains a spurious historical "high AUC."

**Precise structure (no hand-picked ratios):** a **dyadic ladder** —
block sizes are powers of two, the standard parameter-free multiresolution
decomposition (wavelet octave bands). Levels: **128, 32, 8 steps**
(2^7, 2^5, 2^3; every other octave). Everything else identical to Run 4:
μ between consecutive z-scored intervals per level; features = 4 most
recent μ per level + cross-level median/std (14 features); arms E′
(dyadic) vs A vs C; same data/seeds/task; warmup = 5×128 = 640.

**One-shot rule (binding):** this is the final structural redesign of the
§5 test. If E′ fails the kill criterion, the §5 thread is closed — no
further variants on this benchmark. Iterating designs until one passes
manufactures false positives; the discipline that makes a positive mean
something is stopping after a pre-registered negative.

**Kill criterion for E′ (PR-AUC primary):** E′ significantly > C via
paired Wilcoxon → POSITIVE; else NEGATIVE and §5 closes.

## Threats to validity (stated now)

Synthetic data only; one series family; z-scoring blinds level shifts;
horizon n=30 and lag set are choices; 10 seeds, no multiple-comparison
correction beyond the two planned contrasts. A positive here earns a
real-data replication, not a monetization claim.
