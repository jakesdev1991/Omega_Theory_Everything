# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Tests for the published `onset-eval` harness.

These run in two places, and both matter:

  * in this repository, collected by the root `pytest` run (conftest.py beside
    this file puts `src/` on the path, so no install is needed);
  * in CI's packaging step, against a wheel installed into a clean virtual
    environment with the repository absent -- which is the only way to know the
    distribution is actually complete.

Nothing here needs the RCOD metrics or anything else in this repository: the
harness is supposed to stand on its own, and if a test needs something from the
tree it is testing the wrong thing.
"""

from __future__ import annotations

import numpy as np
import onset_eval
import pytest


def test_label_onsets_marks_only_the_horizon_after_an_onset():
    idx = np.arange(0, 100)
    y = onset_eval.label_onsets(idx, np.array([10, 50]), horizon=5)
    # onset at 10 is positive for t in (10-horizon, 10) = 5..9, and never at 10
    assert set(idx[y == 1]) == set(range(5, 10)) | set(range(45, 50))
    assert y[idx[np.isin(idx, [10, 50])]].tolist() == [
        0,
        0,
    ]  # onset itself not positive


def test_series_validates_onsets():
    with pytest.raises(ValueError, match="outside"):
        onset_eval.Series(values=np.zeros(10), onsets=np.array([10]))
    with pytest.raises(ValueError, match="distinct"):
        onset_eval.Series(values=np.zeros(10), onsets=np.array([3, 3]))
    with pytest.raises(ValueError, match="1-D"):
        onset_eval.Series(values=np.zeros((4, 2)), onsets=np.array([1]))


def test_series_from_csv_flags_and_indices(tmp_path):
    flags = tmp_path / "flags.csv"
    flags.write_text("value,onset\n1,0\n2,1\n3,0\n4,0\n", encoding="utf-8")
    series = onset_eval.Series.from_csv(flags)
    assert series.onsets.tolist() == [1]
    assert series.values.tolist() == [1.0, 2.0, 3.0, 4.0]

    indices = tmp_path / "indices.csv"
    indices.write_text("value,onset\n1,\n2,\n3,2\n", encoding="utf-8")
    series = onset_eval.Series.from_csv(indices)
    assert series.onsets.tolist() == [2]


def test_protocol_rejects_impossible_splits():
    with pytest.raises(ValueError, match="split"):
        onset_eval.Protocol(split=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="horizon"):
        onset_eval.Protocol(horizon=0)


def test_paired_test_that_cannot_run_says_so():
    """No p-value beats a fake p-value.

    A single-seed run and an all-identical comparison cannot be tested. Both
    used to come back as p = 1.0, which reads as "tested, not significant" --
    the opposite of "not tested" -- and that is how an untested claim becomes a
    refuted one in a summary written six months later.
    """
    same = onset_eval.paired_test([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert same["p_value"] is None and same["reason"] == "identical"
    one = onset_eval.paired_test([0.1], [0.2])
    assert one["p_value"] is None and one["reason"] == "too_few_seeds"
    assert one["n"] == 1.0
    with pytest.raises(ValueError, match="equal-length"):
        onset_eval.paired_test([0.1, 0.2], [0.1])


def test_decide_distinguishes_killed_from_untested():
    protocol = onset_eval.Protocol()
    untested = onset_eval.decide(
        {"p_value": None, "reason": "too_few_seeds", "n": 1.0},
        {"pr_auc_mean": 0.1},
        {"median_lead": None, "fa_per_1000": 0.0},
        protocol,
        "kill criterion text",
    )
    assert untested["verdict"] == "UNDECIDED" and untested["significant"] is None

    killed = onset_eval.decide(
        {"p_value": 0.4, "n": 10.0},
        {"pr_auc_mean": 0.1},
        {"median_lead": 50.0, "fa_per_1000": 0.5},
        protocol,
        "",
    )
    assert killed["verdict"] == "KILL" and killed["reason"] == "not_significant"

    # significant ranking, but a detector that alarms far too often is not a win
    unusable = onset_eval.decide(
        {"p_value": 0.01, "n": 10.0},
        {"pr_auc_mean": 0.1},
        {"median_lead": 50.0, "fa_per_1000": 40.0},
        protocol,
        "",
    )
    assert (
        unusable["verdict"] == "KILL"
        and unusable["reason"] == "significant_but_unusable"
    )

    passed = onset_eval.decide(
        {"p_value": 0.01, "n": 10.0},
        {"pr_auc_mean": 0.1},
        {"median_lead": 50.0, "fa_per_1000": 0.5},
        protocol,
        "",
    )
    assert passed["verdict"] == "PASS"


def _summary(per_arm: dict[str, list[float]], seeds: list[int]) -> dict:
    """A run_benchmark-shaped summary, built by hand.

    Built here rather than produced by `run_benchmark` so that the reporting and
    verdict logic can be tested in an environment without scikit-learn -- which
    is the environment this repository's own CI uses. The one-seed path is
    exactly what a user hits first with their own CSV, so it has to be covered
    without the modelling stack.
    """
    summary: dict = {"seeds": seeds, "arms": list(per_arm), "per_seed": []}
    for name, values in per_arm.items():
        n = len(values)
        summary[name] = {
            "pr_auc_mean": sum(values) / n,
            "pr_auc_sd": 0.0,
            "pr_auc_per_seed": list(values),
            "roc_auc_mean": 0.5,
            "roc_auc_n_valid": n,
            "fa_per_1000_mean": 0.0,
            "missed_mean": 0.0,
            "n_test_rows": 100,
            "median_lead": 10.0,
        }
    return summary


def _arms_placeholder() -> list:
    def extract(series, warmup):  # never called by render_report
        raise AssertionError("render_report must not build features")

    return [onset_eval.Arm("a", extract), onset_eval.Arm("b", extract)]


def test_single_seed_report_is_descriptive_not_a_verdict():
    """One seed is the common case, and it must not read as evidence."""
    arms = _arms_placeholder()
    summary = _summary({"a": [0.30], "b": [0.10]}, seeds=[0])
    report = onset_eval.render_report(
        summary, arms, baseline="b", treatment="a", protocol=onset_eval.Protocol()
    )
    assert "no test possible (n = 1 seed)" in report
    assert "Verdict for `a`: UNDECIDED" in report
    assert "description of a single run, not evidence" in report
    assert "p = 1.0000" not in report


def test_multi_seed_report_reports_the_test_it_ran():
    arms = _arms_placeholder()
    # a wins on every seed by a clear margin, and is operational (0 false alarms)
    summary = _summary(
        {"a": [0.5, 0.6, 0.55, 0.52, 0.58], "b": [0.1] * 5}, seeds=list(range(5))
    )
    summary["a"]["fa_per_1000_mean"] = 0.5
    report = onset_eval.render_report(
        summary, arms, baseline="b", treatment="a", protocol=onset_eval.Protocol()
    )
    assert "Wilcoxon p = 0.0312" in report and "significant" in report
    assert "Verdict for `a`: PASS" in report


def test_seeds_with_one_class_have_no_roc_auc_and_the_table_says_how_many():
    arms = _arms_placeholder()
    summary = _summary({"a": [0.4, 0.4, 0.4], "b": [0.1, 0.1, 0.1]}, seeds=[0, 1, 2])
    summary["a"]["roc_auc_n_valid"] = 2
    report = onset_eval.render_report(summary, arms, baseline="b")
    assert "(2/3 seeds)" in report


def test_benchmark_and_report_end_to_end():
    """A whole claim, decided: two arms, three seeds, a verdict line."""
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")

    def family(seed):
        rng = np.random.default_rng(seed)
        n = 1200
        x = np.cumsum(rng.standard_normal(n)) * 0.1
        onsets = np.array([300, 700, 1000])
        for onset in onsets:  # visible level breaks at each onset
            x[onset:] += 3.0
        return onset_eval.Series(values=x, onsets=onsets, name=f"toy-{seed}")

    def informative(series, warmup):
        idx = np.arange(warmup, len(series) - 30)
        x = series.values
        level = np.array([x[max(0, t - 20) : t + 1].mean() for t in idx])
        slope = np.array([x[t] - x[max(0, t - 20)] if t >= 20 else 0.0 for t in idx])
        return ("level", "slope"), np.column_stack([level, slope])

    def flat(series, warmup):
        idx = np.arange(warmup, len(series) - 30)
        return ("constant",), np.ones((idx.size, 1))

    arms = [
        onset_eval.Arm("informative", informative, warmup=40),
        onset_eval.Arm("constant", flat, warmup=40),
    ]
    summary = onset_eval.run_benchmark(
        family, arms, [0, 1, 2], onset_eval.Protocol(horizon=30)
    )
    assert summary["informative"]["pr_auc_mean"] > summary["constant"]["pr_auc_mean"]
    assert len(summary["per_seed"]) == 3
    report = onset_eval.render_report(
        summary, arms, baseline="constant", treatment="informative"
    )
    assert (
        "| informative |" in report and "Wilcoxon p" in report and "Verdict" in report
    )


def test_arms_on_different_warmups_are_compared_on_one_grid():
    pytest.importorskip("sklearn")

    def family(_seed):
        n = 900
        x = (
            np.sin(np.arange(n) / 25.0)
            + np.random.default_rng(0).standard_normal(n) * 0.1
        )
        return onset_eval.Series(values=x, onsets=np.array([300, 600]), name="grid")

    def extract(series, warmup):
        idx = np.arange(warmup, len(series) - 20)  # must match Protocol.horizon
        return ("v",), series.values[idx][:, None]

    arms = [
        onset_eval.Arm("short", extract, warmup=50),
        onset_eval.Arm("long", extract, warmup=120),
    ]
    summary = onset_eval.run_benchmark(
        family, arms, [0], onset_eval.Protocol(horizon=20)
    )
    # The arms' warmups differ, so their natural grids differ; the harness
    # scores both on the intersection, which shows up as an equal test-slice
    # size. Without that, the longer-warmup arm would be judged on a different
    # slice of the series than the arm it is compared against.
    rows = {summary[arm]["n_test_rows"] for arm in ("short", "long")}
    assert len(rows) == 1, rows
    assert summary["short"]["pr_auc_mean"] == pytest.approx(
        summary["long"]["pr_auc_mean"], abs=1e-9
    )
