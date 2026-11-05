<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE and ../LICENSES/MIT.txt. -->

# What in this repository is portable, and what is not

**Date:** 2026-11-05 · **Assessed at:** `ee35581` (`main`).
**Scope:** the theory documents, `rcod/` (both the governor and the §5
experiment), and the Lean 4 audit stack in `lean_proofs/`.
**Question:** which parts of this work are domain-agnostic — usable on a
tokamak, a price series, a turbine, a patient monitor — and what has to change
for that to be true?

Every number below was produced by a command run against this checkout on this
date. Where a result is new, it says so; where it is a restatement of a
committed result, it links to the file that carries it.

---

## 1. Summary

| Layer | Portable today? | What was blocking | State after this pass |
|---|---|---|---|
| RCOD metrics (`Φ`, `μ`, median/macro/dissonance) | **Yes**, mechanically | Batch-only, O(T·Σw) memory, single-channel, Python 2-era layout | Streaming tracker, prefix-sum backend, multichannel, explicit level-shift and flat-window options; four implementations that agree to 1e-12 |
| §5 evaluation protocol (arms, split, PR-AUC, paired test, kill criteria) | **Yes**, and it is the most valuable export | Shipped as `main_*` functions inside one experiment | Published as the **`onset-eval`** distribution (`packages/onset_eval/`); installable, tested against its own wheel |
| The §5 *result* | **No** — a negative, correctly | — | Reproduces byte-identically; a second, unrelated family also negative (§5) |
| Optimizer governor (`rcod_optimizer.py`) | **No** — negative result | — | Unchanged; `rcod/RESULTS.md` still the record |
| Lean kernel/axiom audits | **Yes** | — | Unchanged; `audit_axioms`, `audit_kernel` take `--root` already |
| Lean statement audit | **Almost** — the mechanism yes, the rule no | File glob and lexicon hard-coded to physics | `--volume-glob` and `--lexicon` added; defaults unchanged |
| Lean corpus itself (Vol01–Vol54) | **No** — mostly a degenerate model | — | Unchanged; `PROOF_MAP.md` already grades each claim |
| Theory documents | **Partly** — see §2 | — | Unchanged |

The pattern is consistent: the *mechanisms* in this repo are portable and the
*claims* are not. That is the right way round for a research program, and it is
worth making explicit because the two are easy to confuse when everything sits
in one tree.

---

## 2. The theory portion: what is general mathematics, what is a model

The corpus is unusually honest about this already — `PROOF_MAP.md` grades every
prose claim as *Formalised*, *Conditional*, *Structural-check-only*,
*Python-only*, *Open* or *Ansatz*, and `statement_audit.json` classifies 163
declarations by disposition (73 `witness`, 43 `conditional`, 14 `renamed`, 33
`carried`). Reading those two files is the fastest way to separate the portable
mathematics from the model-specific bookkeeping.

**Domain-general (would survive a change of subject matter):**

| Result | File | Why it travels |
|---|---|---|
| Negative-log distance is a metric under a multiplicative kernel; equality case and counterexample for the opposite bound | `LogCorrelationMetric.lean` | Pure real analysis. No physics in the statement; the hypothesis is a kernel bound, which is a modelling assumption a different domain supplies. |
| Landauer-cost bookkeeping: positivity, saturation iff, strictness iff, batch decomposition and affordability | `InformationPhysics.lean` | Arithmetic over an explicit cost function. Useful to any resource-accounting domain (energy, tokens, budget), not just erasure. |
| Budget safety: finite-run success iff total cost fits, exact remaining-balance conservation | `CBwK_Budget_Pacer.lean` | A ledger invariant. This is the single most transferable object in the corpus — it is a pacer, not a physical law. |
| Context branching: input-independence and non-representability of identity | `APPA_Context_Branching.lean` | Computability-flavoured, model-independent. |
| Degenerate-model diagnostics | `audit_vacuity.py` | Not a theorem: a *method* (§4). |

**Model-specific (true of the encoded model, not of the world):** everything
whose row in `statement_audit.json` is `carried`, and the `zeroCurvature_*`
family — `maxwell_equations_of_sourced_field`, `zeroCurvature_bianchi_identity`,
`zeroCurvature_cosmological_constant`, `placeholder_optical_expansion`,
`SM_Gauge_Group_eq_product`, `legacy_density_constants_sum_eq_one`. These are
the declarations the Phase 6 rename pass renamed *because* their names
overstated their content. They are not defects to fix by proving harder; they
are the honest residue of a degenerate carrier model, and the 33-row
`carried_cap` ratchet makes them visible rather than rhetorically available.

**The practical consequence for reuse:** a different domain does not want
Vol01–Vol54. It wants `LogCorrelationMetric`'s lemma shape, the budget-ledger
invariants, and the audit method that keeps a formalization honest. Those are
three separable artifacts; right now they are one directory.

---

## 3. RCOD implementation: what was wrong with it as a library

`rcod_optimizer.py` is a negative result and stays as one. The reusable part is
the metric family in `rcod/section5/swarm.py`, and as a library it had four
problems that only show up when the consumer is not the original experiment.

**3.1 It could not run on a long series.** The reference path materialises a
`(T-w+1, w)` z-score block per window, so peak memory is O(T · Σw). Measured on
this machine:

| T | direct (reference) | prefix (new) | streaming (new) |
|---|---|---|---|
| 24,000 | 0.33 s, 124 MB | 0.03 s | 2.3 s |
| 200,000 | 2.91 s, 819 MB | 0.37 s | 19.4 s, **32 MB flat** |
| 1,000,000 | **killed by the OOM killer** | 2.5 s | ~97 s (measured rate) |

The fix is arithmetic, not engineering: the cosine between consecutive
z-scored windows needs `Σa`, `Σa²`, `Σb`, `Σb²` and `Σab`, and `Σab` is the sum
of the lag-1 product series `p_j = x_j x_{j+1}` over the window — so all five
come from prefix sums, and each (window, t) is O(1). `SwarmTracker` carries the
same sums incrementally for a genuine online consumer: ~97 µs/sample,
constant memory, which is 10k samples/s in pure Python.

**3.2 Two of its behaviours were accidents of the guard rails.** A flat window
(a stuck sensor, a frozen feed) is mapped to `Φ = 0`, i.e. **maximum** novelty,
because the guard exists only to avoid dividing by zero. And z-scoring is, by
construction, blind to amplitude precursors — which is exactly the signal that
a volatility-ramp precursor carries. Both are now explicit
(`degenerate="max"|"zero"`, `normalize="zscore"|"none"`) instead of implicit,
with the published behaviour as the default.

**3.3 It was single-channel and not importable.** Real deployments have more
than one sensor: `swarm_metrics` now takes `(T, C)` and returns `(T, C)`
metrics with `pool_channels(how="median"|"max"|...)`, and the module is a real
package (`rcod/`, `rcod/section5/`) importable from the repo root instead of
only from inside `section5/`.

**3.4 The experiment around it was not reusable.** `experiment.py` contains
three near-identical `main_*` functions (hier/temporal/dyadic), each re-deriving
the same split, threshold tuning, lead-time and false-alarm accounting, and each
printing its own verdict. That protocol is the genuinely portable artifact and
it was the most duplicated code in the directory. It is now `rcod/harness.py`
(§4), and the archival experiment is untouched — a pre-registration is a
historical record, and rewriting it to match new code would destroy exactly the
property that makes it evidence.

**3.5 What was verified, not assumed.** The three implementations (reference,
prefix, streaming) are checked against each other on five series families
(white noise, AR(1), mean shift, constant, single spike) × two normalisation
modes: maximum absolute difference **4e-12** (prefix) and **3e-13** (streaming),
with identical NaN warm-up patterns. And the frozen Run 5 was re-run after the
change: `results_dyadic.json` is **byte-identical**, with the same PR-AUCs
(0.0371 / 0.0153 / 0.0115) and the same Wilcoxon p = 0.6328 that closed §5.

---

## 4. The portable piece: a pre-registered claim harness

The harness is now a distribution rather than a file in a research directory:
**`packages/onset_eval/`**, importable as `onset_eval`, installable with
`pip install ./packages/onset_eval` (`[full]` extra for scikit-learn/SciPy).
`rcod/harness.py` remains as a shim so that the in-repository import path still
works; the canonical name outside this tree is `onset_eval`.

Its interface is deliberately domain-shaped, not RCOD-shaped:

```python
from onset_eval import Arm, Protocol, Series, run_benchmark, render_report

series = Series.from_csv("telemetry.csv", value="sensor", onset="fault")
arms = [Arm("rcod", my_extractor), Arm("rolling", rolling_stats)]
summary = run_benchmark(
    lambda seed: series, arms, seeds=range(10), protocol=Protocol(horizon=30)
)
print(render_report(summary, arms, baseline="rolling", treatment="rcod"))
```

What it enforces, because §5 learned each rule the hard way: the label window is
the horizon; thresholds are tuned on validation only; PR-AUC is primary at a ~1%
base rate (Run 3 showed ROC-AUC of 0.986 on PR-AUC ≈ 0.1 — rare-event flattery);
operational numbers (lead time, false alarms per 1,000 steps) are reported
alongside ranking quality; comparisons are paired by seed against the simplest
credible baseline; and arms with different warm-ups are scored on the
intersection of their index grids so an arm cannot win on an easier slice.

Running it from outside the repository is what caught two honesty bugs that the
in-tree tests had not: a single-seed run printed *"Wilcoxon p = 1.0000 — not
significant"* when no test was possible, and the verdict said KILL for an
untested claim. Both are now explicit — `paired_test` returns `p_value: None`
with a `reason`, the report says "no test possible (n = 1 seed)", and `decide`
has three outcomes (`PASS`, `KILL`, `UNDECIDED`) instead of two. The same run
also surfaced that a seed whose test slice has one class has no ROC-AUC at all;
the table now reports the count of valid seeds rather than silently averaging
NaN, which is the convention the RCOD record already used ("n = 9/10 valid").

`rcod/section5/domain_example.py` is the demonstration and the entry point:

```bash
python -m rcod.section5.domain_example                       # synthetic, not AR(1)
python -m rcod.section5.domain_example --csv yours.csv --value sensor --onset fault
```

---

## 5. A new result, on a family that is not the §5 one

The example runs the protocol on a **telemetry family** built to be unlike the
§5 generator: a random-walk baseline with a diurnal component, multiplicative
noise, and eight step faults, each preceded by a 120-step variance ramp (same
precursor *phenomenology*, different mechanics). Three seeds, default settings:

| arm | mean PR-AUC | ROC-AUC | false alarms / 1k | missed |
|---|---|---|---|---|
| RCOD trio (median/macro μ, dissonance + lags) | 0.016 | 0.524 | 6.75 | 0.7 |
| rolling stats (mean/std/AC1/max|Δ|, 60 samples + lags) | **0.600** | **0.994** | 15.58 | 0.0 |

**Negative for the RCOD arms, decisively** — and this time with a mechanism
rather than a shrug: on this family the precursor is *amplitude*, and z-scoring
each window before the cosine removes exactly that. The rolling baseline sees it
because `std` is not normalised away. This is the same blind spot `DESIGN.md`
threatens-to-validity §3 named in the abstract ("z-scoring blinds the metric to
pure level shifts"); the harness turns it into a measurable, domain-relevant
failure mode, and it predicts where the metric *would* work: precursors that are
*shape* changes (reordering, decorrelation, phase slips) rather than level or
scale changes.

`--normalize none` (the level-sensitive variant) does **not** rescue it here —
PR-AUC 0.010 vs 0.020 — because raw cosine is pinned near Φ = 1 by a nonzero
mean, exactly as `DESIGN.md` predicted. Both variants are negatives, stated as
such.

This is reported as an exploratory follow-up to §5, not as a pre-registered
test: it was run after seeing the §5 results, on a family chosen by the author
of this pass. It is a lead about *where* to look, and its own harness output says
KILL.

---

## 6. The Lean audits: what they establish, and what blocked reuse

Eight gates run in `lean-ci.yml`; none of them is a claim that the physics is
right, and the code says so repeatedly. Read as a *harness*, the inventory is:

| Gate | Establishes | Cannot establish |
|---|---|---|
| `grep` for `sorry`/`admit`/`native_decide`/`opaque`/`unsafe`/`partial` | No proof holes or escape hatches in sources | That a proof is about the intended object |
| `audit_axioms.py` | Zero declaration-level `axiom`s | That the model is non-degenerate |
| `audit_kernel.py` (260 targets) | Transitive axioms ⊆ {propext, Classical.choice, Quot.sound} | Coverage outside the target list |
| `audit_vacuity.py` (8 ratchets at 0) | No trivially-stated theorems outside the `bridge_*` convention, no decorative/trivial hypotheses, no identity proofs, alias debt not growing | Semantic non-vacuity (it is lexical and says so) |
| `check_declaration_parity.py` | Comment-only edits leave declarations byte-identical | Anything about meaning |
| `sync_txt_mirrors.py` | 68 plain-text mirrors track the sources | — |
| `check_mathlib_names.py` | Every cited name exists in the pinned mathlib/Lean sources | That the citation is apt |
| `check_statement_audit.py` (163 rows, cap 33) | Every lexicon-matching declaration has a recorded judgement, and each row's witness/condition/rename is real | That a judgement is *correct* — that is `lake build`'s half |

That is a genuinely reusable methodology: "prove something, then prove you
proved the thing you said". Its portability was blocked by three couplings, two
of which are now removed:

1. **The file glob** (`Vol*.lean`) and **the lexicon** (92 physics tokens) are
   what decide which declarations need a judgement. Hard-coded, they make the
   gate unusable outside this subject matter. Now `--volume-glob` and
   `--lexicon FILE` (one token per line, `#` comments, empty/`junk` rejected),
   with defaults unchanged and four tests pinning that a finance-style corpus is
   selected by its own lexicon and ignored by the physics one.
2. **The `bridge_*` naming convention** (in `audit_vacuity.py`) is still
   hard-coded. It is cheap to parameterise and will be the next step for anyone
   reusing the ratchets; it is left alone here because changing the convention
   that guards 33 carried rows is a judgement call, not a refactor.
3. **The corpus root** — the audits already take `--root`, but `lean-ci.yml`
   invokes them with `working-directory: lean_proofs` and `python3 -m unittest`,
   which is why the whole stack is `unittest` and not `pytest`. Reuse means
   copying the files, not importing them as a package.

Two gates remain **CI-only by necessity**: `lake build ToE` (needs the pinned
toolchain and mathlib oleans) and `check_mathlib_names.py` (needs the pinned
mathlib sources). Everything else runs offline in seconds, which is why the
offline consistency gate can run 16 checks before a Lean installation exists.

---

## 7. What to do next, in order of leverage

1. ~~**Split the three portable artifacts out of the one directory.**~~ **Done**
   for the harness: `packages/onset_eval/` is a hatchling-built distribution with
   its own tests, its own LICENSE and `py.typed`, and CI builds the wheel,
   installs it into a virtual environment that cannot see this checkout, and runs
   its tests from `/tmp`. Nothing in it imports this repository. Still open: the
   metrics (`rcod.section5.swarm`) could follow, but they are one import of
   NumPy and two functions — the argument for publishing them is weaker than the
   argument for the harness, which is the thing whose *rules* need to be shared.
2. **Parameterise the `bridge_*` convention** in `audit_vacuity.py` (the last
   hard-coded rule in the Lean stack), then document the eight gates as an
   adoptable recipe rather than eight scripts in this repo.
3. **Rewrite the streaming tracker in the Rust `telemetry` crate** if it is ever
   deployed in a training or monitoring path — 97 µs/sample in Python is fine
   for a monitor and hopeless inside a training loop. `rcod/README.md` already
   names that crate; the O(1)-per-sample recurrence above is the spec.
4. **Use the harness before building the next method, not after.** The expensive
   lesson of §5 and Run 1 is that a design flaw (instantaneous shocks with no
   precursor) is indistinguishable from a refuted hypothesis until you write the
   protocol down first. That is now code, so it can be reused and, better,
   reviewed.

---

## 8. Caveats

- The new telemetry result is **not pre-registered**. It post-dates the §5
  closure, was designed by the author of this pass, and uses three seeds. It is
  evidence about a mechanism, not a new claim about the world.
- "Portable" here means *mechanically applicable*, not *validated elsewhere*. No
  real-domain dataset (tokamak, financial, clinical) was tested in this pass;
  `Series.from_csv` exists so that it can be, and any such run needs its own
  pre-registration.
- The performance numbers are one machine, one NumPy build, and the streaming
  figure is a pure-Python loop; they are orders of magnitude, not benchmarks.
- The Lean side was reviewed by reading sources and running the offline audits
  here. `lake build` and the kernel audit were not re-run in this environment
  (toolchain and mathlib oleans are fetched in CI), and this document does not
  claim otherwise.
