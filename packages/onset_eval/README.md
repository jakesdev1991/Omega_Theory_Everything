<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE. -->

# onset-eval

Pre-registered evaluation of onset-prediction claims: *"these features predict
a regime change before it happens."*

That claim is easy to fool yourself about. Tune the threshold on the test set,
score at a 1% positive rate with ROC-AUC, skip the baseline, try four feature
designs until one passes — each is individually reasonable and together they
manufacture positives. `onset-eval` fixes the protocol instead: the label
window, the split, the metric, the threshold discipline, the baseline
comparison and the kill criterion, all decided before the numbers exist.

It is domain-agnostic on purpose. A tokamak discharge, a price series with known
breaks, a turbine with maintenance records, a patient monitor with annotated
events: if you have a series and the indices where something changed, it applies.

## Install

```bash
pip install onset-eval            # NumPy only
pip install "onset-eval[full]"    # + scipy, scikit-learn for the modelling steps
```

## Use

```python
from onset_eval import Arm, Protocol, Series, run_benchmark, render_report

series = Series.from_csv("telemetry.csv", value="sensor", onset="fault")


def my_features(series, warmup):
    idx = np.arange(warmup, len(series) - 30)  # rows from warmup to the horizon
    ...  # your extractor
    return ("my_feature",), matrix  # (names, n_rows × n_features)


def rolling(series, warmup):
    ...  # the simplest credible baseline
    return ("mean", "std"), matrix


arms = [Arm("mine", my_features, warmup=200), Arm("rolling", rolling, warmup=200)]
summary = run_benchmark(
    lambda seed: series, arms, seeds=range(10), protocol=Protocol(horizon=30)
)
print(render_report(summary, arms, baseline="rolling", treatment="mine"))
```

`Series.from_csv` accepts either a 0/1 onset flag column or a column holding an
explicit onset index; `time=` orders newest-first exports. If you would rather
not go through a file, construct `Series(values=..., onsets=...)` directly.

## What the protocol fixes

| Rule | Why it is not optional |
|---|---|
| Positives are onsets inside `(t, t + horizon]` | Otherwise the label leaks the event itself and every arm "works" |
| Model fit on the first split, threshold tuned on the second, everything reported on the third | Tuning on test is how a detector scores 0.99 and misses half the events |
| PR-AUC is primary at a rare base rate | ROC-AUC at a 1% positive rate produces eye-catching numbers from noise; the RCOD §5 record has a case of ROC-AUC 0.986 sitting on PR-AUC ≈ 0.1 |
| Operational numbers alongside ranking quality: median lead time, false alarms per 1,000 steps | A detector that alarms constantly has excellent recall and no use |
| Paired one-sided Wilcoxon across seeds against the simplest baseline | Per-seed noise dominates; an unpaired comparison of two means is not evidence |
| Arms with different warm-ups are scored on the intersection of their index grids | Otherwise an arm wins by being judged on an easier slice of the series |

The harness cannot make a weak feature good. It can stop a weak feature from
looking good.

## Requirements

NumPy for everything except the modelling and significance steps;
`evaluate_arm`, `run_benchmark`, `paired_test` and `decide` import
scikit-learn and SciPy lazily, so `Series`, `Arm`, `label_onsets` and the CSV
loader work in a minimal environment.

## Provenance

Extracted from the RCOD §5 experiment in the [Omega Theory
repository](https://github.com/jakesdev1991/Omega_Theory_Everything), where the
protocol was first written down before any code ran and then applied to four
pre-registered variants — all negative. The number of designs that failed
against a plain rolling-statistics baseline is the reason this package exists:
the protocol is the part that survived. What is portable there, and what is
not, is recorded in
[`docs/DOMAIN_AGNOSTIC_CORE.md`](https://github.com/jakesdev1991/Omega_Theory_Everything/blob/main/docs/DOMAIN_AGNOSTIC_CORE.md).

## License

MIT — the science side of the parent repository. See `LICENSE`.
