# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Domain-agnostic evaluation harness for onset-prediction claims.

The §5 experiment in this repository (`rcod/section5/experiment.py`) is one
instance of a protocol that is not about cosmology at all:

    a series, ground-truth event onsets, several feature extractors ("arms"),
    a time-ordered split, a rare-event metric, a paired test across seeds, and a
    decision rule written down *before* the numbers exist.

That protocol is the reusable part, and this module is the shell that makes it
work for any series with labelled onsets -- a tokamak discharge,
a price series with known breaks, a turbine with maintenance records, a
patient monitor with annotated events. Nothing here imports torch, sklearn or
scipy at module import time; the modelling and test steps take callables, and
the defaults import lazily.

Minimal use on your own data (published as the `onset-eval` distribution):

    from onset_eval import Arm, Protocol, Series, run_benchmark, render_report

    series = Series.from_csv("telemetry.csv", value="sensor", onset="fault")
    arms = [Arm("swarm", my_extractor), Arm("rolling", rolling_stats)]
    results = run_benchmark(lambda seed: series, arms, seeds=range(10),
                            protocol=Protocol(horizon=30))
    print(render_report(results, arms, baseline="rolling_stats"))

The rules that make a result mean something are the ones the §5 work learned
the hard way, and they are encoded here rather than left to the caller:

  * the label window is the horizon, and onsets inside it are the only
    positives (no leakage across the split);
  * the threshold is tuned on validation and applied unchanged to test;
  * the primary metric is PR-AUC, because at a 1% positive rate ROC-AUC
    flatters noise (the RCOD §5 record, Run 3: ROC-AUC 0.986 on PR-AUC ~0.1);
  * alarms are judged operationally too: lead time and false alarms per 1,000
    steps at that threshold, not just ranking quality;
  * a claim is compared against the *simplest* credible baseline, paired by
    seed, with a one-sided test and a pre-declared kill criterion.

A harness cannot make a weak feature good. It can stop a weak feature from
looking good.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "Arm",
    "Protocol",
    "Series",
    "decide",
    "evaluate_arm",
    "label_onsets",
    "paired_test",
    "render_report",
    "run_benchmark",
]


@dataclasses.dataclass(frozen=True)
class Series:
    """One time series with known event onsets.

    `onsets` are indices into `values` where a regime change begins; they are
    the only ground truth the harness needs, which is what makes it portable
    across domains.
    """

    values: np.ndarray
    onsets: np.ndarray
    name: str = "series"
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        onsets = np.asarray(self.onsets, dtype=int).ravel()
        if values.ndim != 1:
            raise ValueError(f"{self.name}: values must be 1-D, got {values.shape}")
        if values.size == 0:
            raise ValueError(f"{self.name}: empty series")
        if onsets.size and (onsets.min() < 0 or onsets.max() >= values.size):
            raise ValueError(f"{self.name}: onsets outside the series")
        if onsets.size > 1 and not np.all(np.diff(np.sort(onsets)) > 0):
            raise ValueError(f"{self.name}: onsets must be distinct")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "onsets", np.sort(onsets))

    def __len__(self) -> int:
        return int(self.values.size)

    @classmethod
    def from_csv(
        cls,
        path: str | pathlib.Path,
        value: str = "value",
        onset: str | None = "onset",
        time: str | None = None,
        name: str | None = None,
    ) -> Series:
        """Load a series from CSV.

        `onset` is a 0/1 column marking onset rows, or a column holding the
        onset index in each event row with 0/blank elsewhere. `time` is
        accepted and ignored beyond ordering the rows, because some exports
        are newest-first.
        """
        rows: list[dict[str, str]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{path}: no rows")
        if value not in rows[0]:
            raise ValueError(f"{path}: no column {value!r}; has {sorted(rows[0])}")
        if time is not None:
            rows.sort(key=lambda r: r[time])
        values = np.array([float(r[value]) for r in rows])
        onsets: list[int] = []
        if onset is not None and onset in rows[0]:
            for i, row in enumerate(rows):
                cell = (row[onset] or "").strip()
                if not cell:
                    continue
                flag = float(cell)
                if flag == 0:
                    continue
                # a 0/1 flag marks the row; a larger value is an explicit index
                onsets.append(i if flag == 1 else int(flag))
        return cls(
            values=values,
            onsets=np.array(sorted(set(onsets)), dtype=int),
            name=name or pathlib.Path(path).stem,
            metadata={"source": str(path)},
        )


@dataclasses.dataclass(frozen=True)
class Arm:
    """A feature extractor: name plus a function of (series, warmup).

    `extract` returns (feature_names, matrix) with one row per time index from
    `warmup` to `len(series) - protocol.horizon`. Keeping the warmup an
    argument rather than a constant is what lets arms with different warmups be
    compared on the same index grid.
    """

    name: str
    extract: Callable[[Series, int], tuple[Sequence[str], np.ndarray]]
    description: str = ""
    warmup: int = 0

    def build(
        self, series: Series, protocol: Protocol
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(self.warmup, len(series) - protocol.horizon)
        if idx.size <= 0:
            raise ValueError(
                f"arm {self.name!r}: series too short for warmup "
                f"{self.warmup} + horizon {protocol.horizon}"
            )
        _, features = self.extract(series, self.warmup)
        features = np.asarray(features, dtype=float)
        if features.shape[0] != idx.size:
            raise ValueError(
                f"arm {self.name!r}: {features.shape[0]} rows of features for "
                f"{idx.size} index positions"
            )
        if not np.isfinite(features).all():
            raise ValueError(f"arm {self.name!r}: non-finite feature values")
        return idx, features


def _default_model(seed: int):
    """L2 logistic regression: the linear probe is the honest default arm.

    Imported lazily so this module -- and the arm/result plumbing -- works in
    an environment without scikit-learn (the repo's CI installs only the
    simulation requirements).
    """
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=5000, random_state=seed)


@dataclasses.dataclass(frozen=True)
class Protocol:
    """Everything that must be fixed before the numbers are seen."""

    horizon: int = 30
    split: tuple[float, float, float] = (0.6, 0.2, 0.2)
    lead_window: int = 200
    failure_window: int = 50
    threshold_grid: tuple[float, ...] = tuple(np.linspace(0.05, 0.95, 19))
    alpha: float = 0.05
    max_false_alarms_per_1000: float = 2.0
    model_factory: Callable[[int], Any] = _default_model
    primary_metric: str = "pr_auc"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if len(self.split) != 3 or abs(sum(self.split) - 1.0) > 1e-9:
            raise ValueError(
                f"split must be three fractions summing to 1: {self.split}"
            )
        if self.lead_window < 0 or self.failure_window < 0:
            raise ValueError("lead/failure windows must be non-negative")


def label_onsets(idx: np.ndarray, onsets: np.ndarray, horizon: int) -> np.ndarray:
    """1 when a true onset falls in (t, t + horizon] -- the pre-registered target."""
    onsets = np.asarray(onsets)
    return np.array(
        [np.any((onsets > t) & (onsets <= t + horizon)) for t in idx]
    ).astype(int)


def evaluate_arm(
    features: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    onsets: np.ndarray,
    protocol: Protocol,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit on the first split, tune on the second, report on the third."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    n = len(idx)
    i1 = int(protocol.split[0] * n)
    i2 = int((protocol.split[0] + protocol.split[1]) * n)
    if i1 < 2 or i2 <= i1 or i2 >= n:
        raise ValueError(f"split too small for {n} rows: {protocol.split}")

    scaler = StandardScaler().fit(features[:i1])
    scaled = scaler.transform(features)
    model = protocol.model_factory(seed)
    model.fit(scaled[:i1], y[:i1])
    p_val = model.predict_proba(scaled[i1:i2])[:, 1]
    p_test = model.predict_proba(scaled[i2:])[:, 1]

    best_t, best_f1 = 0.5, -1.0
    for t_ in protocol.threshold_grid:
        f1 = f1_score(y[i1:i2], (p_val >= t_).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t_)

    alarms = idx[i2:][p_test >= best_t]
    test_onsets = onsets[(onsets >= idx[i2]) & (onsets < idx[-1])]
    leads: list[float] = []
    missed = 0
    for onset in test_onsets:
        candidates = alarms[
            (alarms >= onset - protocol.lead_window) & (alarms <= onset)
        ]
        if candidates.size:
            leads.append(float(onset - candidates.min()))
        else:
            missed += 1
    if test_onsets.size:
        near = np.abs(alarms[:, None] - test_onsets[None, :])
        usable = np.any(
            (near <= protocol.lead_window)
            & (alarms[:, None] <= test_onsets[None, :] + protocol.failure_window),
            axis=1,
        )
        false_alarms = float(np.sum(~usable))
    else:
        false_alarms = float(alarms.size)

    return {
        "pr_auc": float(average_precision_score(y[i2:], p_test)),
        "roc_auc": float(roc_auc_score(y[i2:], p_test)),
        "median_lead": float(np.median(leads)) if leads else None,
        "missed": int(missed),
        "n_test_onsets": int(test_onsets.size),
        "fa_per_1000": false_alarms / max(1, n - i2) * 1000.0,
        "threshold": best_t,
        "val_f1": float(best_f1),
        "n_test_pos": int(y[i2:].sum()),
        "n_test_rows": int(n - i2),
    }


def paired_test(
    a: Sequence[float], b: Sequence[float], alternative: str = "greater"
) -> dict[str, Any]:
    """One-sided paired Wilcoxon across seeds -- the pre-registered comparison.

    Returns `p_value: None` and a `reason` when the comparison cannot be made
    (`too_few_seeds`, `identical`). It is deliberately *not* reported as p = 1.0:
    a single-seed run is the most common case for a personal dataset, and
    "p = 1.0, not significant" reads like a tested-and-failed result when in fact
    nothing was tested.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"paired test needs equal-length arms, got {x.shape} and {y.shape}"
        )
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if x.size < 2:
        return {
            "n": float(x.size),
            "statistic": None,
            "p_value": None,
            "reason": "too_few_seeds",
        }
    if np.allclose(x - y, 0.0):
        return {
            "n": float(x.size),
            "statistic": None,
            "p_value": None,
            "reason": "identical",
        }
    from scipy.stats import wilcoxon

    stat, p = wilcoxon(x, y, alternative=alternative)
    return {"n": float(x.size), "statistic": float(stat), "p_value": float(p)}


def decide(
    treatment: dict[str, Any],
    baseline: dict[str, Any],
    detail: dict[str, Any],
    protocol: Protocol,
    kill_criterion: str,
) -> dict[str, Any]:
    """Apply a pre-declared rule, and say so when the rule cannot be applied.

    Three outcomes, not two: PASS needs a significant paired win *and* usable
    operation (a lead time, and no more false alarms than the protocol allows);
    KILL is a comparison that was made and lost; UNDECIDED is a comparison that
    could not be made at all -- one seed, or identical arms. Collapsing the last
    into KILL is how "we did not test it" becomes "it does not work" in a
    summary a year later.
    """
    p_value = treatment.get("p_value")
    reason = treatment.get("reason")
    if p_value is None:
        return {
            "verdict": "UNDECIDED",
            "reason": reason or "no_p_value",
            "significant": None,
            "operational": None,
            "criterion": kill_criterion,
            "p_value": None,
            "baseline": baseline,
        }
    significant = p_value < protocol.alpha
    lead = detail.get("median_lead")
    false_alarms = detail.get("fa_per_1000")
    operational = (
        lead is not None
        and false_alarms is not None
        and false_alarms <= protocol.max_false_alarms_per_1000
    )
    return {
        "verdict": "PASS" if (significant and operational) else "KILL",
        "reason": (
            "significant_and_operational"
            if significant and operational
            else "significant_but_unusable"
            if significant
            else "not_significant"
        ),
        "significant": bool(significant),
        "operational": bool(operational),
        "criterion": kill_criterion,
        "p_value": p_value,
        "baseline": baseline,
    }


def _median_lead(leads: Sequence[float | None]) -> float | None:
    """Median lead time across seeds, or None when no seed produced a lead."""
    usable = [v for v in leads if v is not None]
    return float(np.median(usable)) if usable else None


def run_benchmark(
    series_factory: Callable[[int], Series],
    arms: Iterable[Arm],
    seeds: Iterable[int],
    protocol: Protocol,
) -> dict[str, Any]:
    """Run every arm on every seed on one shared index grid.

    Arms with different warmups are compared on the intersection of their index
    grids, so an arm cannot win by evaluating on an easier slice of the series.
    """
    arms = list(arms)
    if not arms:
        raise ValueError("no arms")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("no seeds")

    per_seed: list[dict[str, Any]] = []
    grids: set[tuple[int, ...]] = set()
    for seed in seeds:
        series = series_factory(seed)
        built: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in arms:
            built[arm.name] = arm.build(series, protocol)
            grids.add(tuple(built[arm.name][0].tolist()))
        if len(grids) > 1:
            # Keep the rows every arm can score: drop the union's edges.
            common = np.array(sorted(set.intersection(*[set(g) for g in grids])))
            for name, (idx, feats) in built.items():
                keep = np.isin(idx, common)
                built[name] = (idx[keep], feats[keep])
        record: dict[str, Any] = {"seed": seed, "series": series.name}
        for arm in arms:
            idx, features = built[arm.name]
            y = label_onsets(idx, series.onsets, protocol.horizon)
            record[arm.name] = evaluate_arm(
                features, y, idx, series.onsets, protocol, seed=seed
            )
            record[arm.name]["n_test_pos"] = int(y[int(0.8 * len(idx)) :].sum())
        per_seed.append(record)

    summary: dict[str, Any] = {"seeds": seeds, "arms": [a.name for a in arms]}
    for arm in arms:
        pr = np.array([r[arm.name]["pr_auc"] for r in per_seed])
        roc = np.array([r[arm.name]["roc_auc"] for r in per_seed])
        # A seed whose test slice contains only one class has no ROC-AUC at all
        # (sklearn returns NaN). Averaging over the valid seeds and reporting how
        # many there were is what the RCOD record does for the same reason.
        roc_valid = roc[~np.isnan(roc)]
        summary[arm.name] = {
            "pr_auc_mean": float(np.nanmean(pr)),
            "pr_auc_sd": float(np.nanstd(pr)),
            "pr_auc_per_seed": pr.tolist(),
            "roc_auc_mean": float(roc_valid.mean()) if roc_valid.size else float("nan"),
            "roc_auc_n_valid": int(roc_valid.size),
            "fa_per_1000_mean": float(
                np.mean([r[arm.name]["fa_per_1000"] for r in per_seed])
            ),
            "missed_mean": float(np.mean([r[arm.name]["missed"] for r in per_seed])),
            "n_test_rows": int(np.mean([r[arm.name]["n_test_rows"] for r in per_seed])),
            "median_lead": _median_lead([r[arm.name]["median_lead"] for r in per_seed]),
        }
    summary["per_seed"] = per_seed
    return summary


def _describe_test(test: dict[str, Any], protocol: Protocol) -> str:
    """One line per comparison, including the ones that could not be made."""
    if test["p_value"] is None:
        if test["reason"] == "too_few_seeds":
            return (
                f"no test possible (n = {int(test['n'])} seed"
                f"{'' if int(test['n']) == 1 else 's'}); a paired test needs at least 2"
            )
        return f"no test possible (identical across all {int(test['n'])} seeds)"
    verdict = "significant" if test["p_value"] < protocol.alpha else "not significant"
    return (
        f"Wilcoxon p = {test['p_value']:.4f} (n = {int(test['n'])} seeds) "
        f"at alpha = {protocol.alpha} — {verdict}"
    )


def render_report(
    summary: dict[str, Any],
    arms: Iterable[Arm],
    baseline: str,
    treatment: str | None = None,
    kill_criterion: str = "",
    protocol: Protocol | None = None,
) -> str:
    """Markdown report: the table, the paired tests, and the verdict."""
    arms = list(arms)
    protocol = protocol or Protocol()
    lines = [
        "| arm | mean PR-AUC | sd | ROC-AUC | false alarms / 1k | missed |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        s = summary[arm.name]
        seeds = len(summary.get("seeds", []))
        roc = (
            f"{s['roc_auc_mean']:.3f}"
            if s["roc_auc_n_valid"] == seeds or not seeds
            else f"{s['roc_auc_mean']:.3f} ({s['roc_auc_n_valid']}/{seeds} seeds)"
        )
        lines.append(
            f"| {arm.name} | {s['pr_auc_mean']:.3f} | {s['pr_auc_sd']:.3f} | "
            f"{roc} | {s['fa_per_1000_mean']:.2f} | {s['missed_mean']:.1f} |"
        )
    lines.append("")
    if baseline not in summary:
        raise ValueError(f"baseline arm {baseline!r} not in results")
    for arm in arms:
        if arm.name == baseline:
            continue
        test = paired_test(
            summary[arm.name]["pr_auc_per_seed"], summary[baseline]["pr_auc_per_seed"]
        )
        lines.append(
            f"- `{arm.name}` vs `{baseline}`: {_describe_test(test, protocol)}"
        )
    if treatment and treatment in summary:
        test = paired_test(
            summary[treatment]["pr_auc_per_seed"], summary[baseline]["pr_auc_per_seed"]
        )
        detail = summary[treatment]
        decision = decide(
            test,
            {"pr_auc_mean": summary[baseline]["pr_auc_mean"]},
            {
                "median_lead": detail["median_lead"],
                "fa_per_1000": detail["fa_per_1000_mean"],
            },
            protocol,
            kill_criterion,
        )
        lines.append("")
        lines.append(f"**Verdict for `{treatment}`: {decision['verdict']}**")
        if decision["reason"] == "too_few_seeds":
            lines.append(
                "One seed cannot be tested. The table above is a description of a "
                "single run, not evidence; re-run with at least 5 seeds to get a "
                "verdict."
            )
        elif decision["reason"] == "significant_but_unusable":
            lines.append(
                "Significantly better ranking than the baseline, but not usable as "
                f"a detector at the protocol's limit "
                f"({protocol.max_false_alarms_per_1000:.1f} false alarms per 1,000 "
                "steps with a positive lead time)."
            )
        if decision["criterion"]:
            lines.append(f"Criterion (pre-registered): {decision['criterion']}")
    return "\n".join(lines)
