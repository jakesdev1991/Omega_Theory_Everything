# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Tests for the portable RCOD core: metrics backends, streaming, harness.

Three implementations of the same arithmetic exist on purpose:

    swarm._mu_direct   reference path, O(T*w) memory per window -- the one the
                       committed §5 results were produced with
    swarm._mu_prefix   same arithmetic from prefix sums, O(T) memory
    SwarmTracker       online, O(sum of windows) memory

The first is what the published numbers mean; the other two are what a
deployment can afford. This file is the proof that they agree, which is the
only thing that makes the faster paths safe to use.

Nothing here needs scikit-learn or scipy: the modelling-dependent checks skip
themselves when those packages are absent (the repo's CI requirements do not
include them), so the equivalence guarantees still run in CI.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rcod.harness as harness  # noqa: E402
from rcod.section5 import swarm  # noqa: E402


def _ar1(n=3000, phi=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


SERIES = {
    "white noise": np.random.default_rng(1).standard_normal(3000),
    "ar1": _ar1(),
    "mean shift": _ar1() + np.r_[np.zeros(1500), np.full(1500, 4.0)],
    "flat": np.zeros(3000),
    "single spike": np.r_[np.zeros(1500), 1.0, np.zeros(1499)],
}
KEYS = ("median_mu", "macro_mu", "dissonance")


# --------------------------------------------------------------------------
# equivalence: the fast paths are the reference path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(SERIES))
@pytest.mark.parametrize("normalize", ["zscore", "none"])
def test_prefix_backend_matches_reference(name, normalize):
    x = SERIES[name]
    ref = swarm.swarm_metrics(x, backend="direct", normalize=normalize)
    fast = swarm.swarm_metrics(x, backend="prefix", normalize=normalize)
    for key in KEYS:
        a, b = ref[key], fast[key]
        assert a.shape == b.shape
        both = ~np.isnan(a) & ~np.isnan(b)
        assert np.array_equal(np.isnan(a), np.isnan(b)), (
            f"{name}/{normalize}: NaN pattern"
        )
        if both.any():
            assert np.abs(a[both] - b[both]).max() < 1e-9


@pytest.mark.parametrize("name", sorted(SERIES))
def test_streaming_tracker_matches_prefix_backend(name):
    x = SERIES[name]
    batch = swarm.swarm_metrics(x, backend="prefix")
    tracker = swarm.SwarmTracker()
    for i, value in enumerate(x):
        state = tracker.update(value)
        if state is None:
            continue
        for key in KEYS:
            expected = batch[key][i]
            if np.isnan(expected):
                assert np.isnan(state[key])
            else:
                assert state[key] == pytest.approx(expected, abs=1e-9)


def test_tracker_metrics_matches_batch():
    x = _ar1(2000)
    tracker = swarm.SwarmTracker(macro_w=64, leaf_ws=[8, 16, 32])
    streamed = tracker.metrics(x)
    batch = swarm.swarm_metrics(x, macro_w=64, leaf_ws=[8, 16, 32], backend="prefix")
    for key in KEYS:
        both = ~np.isnan(batch[key])
        assert np.abs(streamed[key][both] - batch[key][both]).max() < 1e-9


def test_degenerate_windows_are_explicit():
    """A flat window is "maximum novelty" by default and zero on request.

    The default is the published behaviour (the guard in the reference path maps
    a zero-variance window to cos = 0, i.e. mu = 1). `degenerate="zero"` is the
    domain-agnostic reading -- a stuck sensor is not chaos -- and both are
    available rather than one being silently chosen.
    """
    flat = np.zeros(600)
    flat[300:] = 1.0
    maxed = swarm.swarm_metrics(flat, macro_w=64, leaf_ws=[16], backend="direct")
    zeroed = swarm.swarm_metrics(
        flat, macro_w=64, leaf_ws=[16], backend="direct", degenerate="zero"
    )
    # t = 100 sits deep inside a constant stretch: both windows are flat.
    assert maxed["macro_mu"][100] == pytest.approx(1.0)
    assert zeroed["macro_mu"][100] == pytest.approx(0.0)
    # A genuine step is novelty either way -- the knob is about flat windows,
    # not about suppressing real change.
    assert np.nanmax(zeroed["macro_mu"]) > 0.5


# --------------------------------------------------------------------------
# multichannel
# --------------------------------------------------------------------------
def test_multichannel_shapes_and_pooling():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((1200, 3))
    per_channel = swarm.swarm_metrics(X, macro_w=64, leaf_ws=[16, 32], backend="prefix")
    assert per_channel["channels"] == 3
    for key in KEYS:
        assert per_channel[key].shape == (1200, 3)
    pooled = swarm.pool_channels(per_channel)
    assert pooled["median_mu"].shape == (1200,)
    assert pooled["pooled"] == "median"
    # a single-channel batch pools to that channel exactly
    one = swarm.swarm_metrics(X[:, :1], macro_w=64, leaf_ws=[16, 32], backend="prefix")
    single_pooled = swarm.pool_channels(one)
    single = swarm.swarm_metrics(
        X[:, 0], macro_w=64, leaf_ws=[16, 32], backend="prefix"
    )
    assert np.allclose(single_pooled["macro_mu"], single["macro_mu"], equal_nan=True)


def test_multichannel_pool_max_is_the_most_sensitive_channel():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((800, 2))
    X[600:, 1] += 5.0  # one channel breaks
    per_channel = swarm.swarm_metrics(X, macro_w=64, leaf_ws=[16], backend="prefix")
    pooled = swarm.pool_channels(per_channel, how="max")
    assert np.nanmax(pooled["macro_mu"]) >= np.nanmax(per_channel["macro_mu"][:, 0])


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------
def test_level_shift_sensitivity_flag_changes_the_answer():
    x = _ar1(2000) + 2.0
    zscored = swarm.swarm_metrics(x, macro_w=64, leaf_ws=[16], backend="prefix")
    raw = swarm.swarm_metrics(
        x, macro_w=64, leaf_ws=[16], backend="prefix", normalize="none"
    )
    # The z-scored form is blind to a constant offset by construction; the
    # raw form is pinned near Phi = 1 (mu ~ 0) by it. Both behaviours are the
    # documented ones, and they differ.
    assert not np.allclose(
        np.nanmean(zscored["macro_mu"]), np.nanmean(raw["macro_mu"]), atol=1e-3
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"backend": "direct", "normalize": "robust"}, "normalize"),
        ({"backend": "turbo"}, "backend"),
        ({"backend": "direct", "macro_w": 1}, "window sizes"),
    ],
)
def test_invalid_options_are_rejected(kwargs, match):
    x = _ar1(500)
    with pytest.raises(ValueError, match=match):
        swarm.swarm_metrics(x, leaf_ws=[8], **kwargs)


def test_short_series_and_nan_are_rejected():
    with pytest.raises(ValueError, match="too short"):
        swarm.swarm_metrics(np.arange(50.0), macro_w=128, leaf_ws=[16])
    with pytest.raises(ValueError, match="NaN"):
        swarm.swarm_metrics(np.r_[np.zeros(300), np.nan, np.zeros(300)], macro_w=64)
    with pytest.raises(ValueError, match="empty"):
        swarm.swarm_metrics(np.array([]))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        swarm.swarm_metrics(np.zeros((10, 10, 10)))


def test_tracker_rejects_non_finite():
    tracker = swarm.SwarmTracker(macro_w=8, leaf_ws=[4])
    with pytest.raises(ValueError, match="NaN"):
        tracker.update(float("nan"))


# --------------------------------------------------------------------------
# harness: the portable protocol
# --------------------------------------------------------------------------
def test_label_onsets_marks_only_the_horizon_after_an_onset():
    idx = np.arange(0, 100)
    y = harness.label_onsets(idx, np.array([10, 50]), horizon=5)
    # onset at 10 is positive for t in (10-horizon, 10) = 5..9, and never at 10
    assert set(idx[y == 1]) == set(range(5, 10)) | set(range(45, 50))
    assert y[idx[np.isin(idx, [10, 50])]].tolist() == [
        0,
        0,
    ]  # onset itself not positive


def test_series_validates_onsets():
    with pytest.raises(ValueError, match="outside"):
        harness.Series(values=np.zeros(10), onsets=np.array([10]))
    with pytest.raises(ValueError, match="distinct"):
        harness.Series(values=np.zeros(10), onsets=np.array([3, 3]))
    with pytest.raises(ValueError, match="1-D"):
        harness.Series(values=np.zeros((4, 2)), onsets=np.array([1]))


def test_series_from_csv_flags_and_indices(tmp_path):
    flags = tmp_path / "flags.csv"
    flags.write_text("value,onset\n1,0\n2,1\n3,0\n4,0\n", encoding="utf-8")
    series = harness.Series.from_csv(flags)
    assert series.onsets.tolist() == [1]
    assert series.values.tolist() == [1.0, 2.0, 3.0, 4.0]

    indices = tmp_path / "indices.csv"
    indices.write_text("value,onset\n1,\n2,\n3,2\n", encoding="utf-8")
    series = harness.Series.from_csv(indices)
    assert series.onsets.tolist() == [2]


def test_protocol_rejects_impossible_splits():
    with pytest.raises(ValueError, match="split"):
        harness.Protocol(split=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="horizon"):
        harness.Protocol(horizon=0)


def test_paired_test_degenerate_inputs_report_no_evidence():
    same = harness.paired_test([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert same["p_value"] == 1.0
    one = harness.paired_test([0.1], [0.2])
    assert one["p_value"] == 1.0 and one["n"] == 1.0


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
        return harness.Series(values=x, onsets=onsets, name=f"toy-{seed}")

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
        harness.Arm("informative", informative, warmup=40),
        harness.Arm("constant", flat, warmup=40),
    ]
    summary = harness.run_benchmark(
        family, arms, [0, 1, 2], harness.Protocol(horizon=30)
    )
    assert summary["informative"]["pr_auc_mean"] > summary["constant"]["pr_auc_mean"]
    assert len(summary["per_seed"]) == 3
    report = harness.render_report(
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
        return harness.Series(values=x, onsets=np.array([300, 600]), name="grid")

    def extract(series, warmup):
        idx = np.arange(warmup, len(series) - 20)  # must match Protocol.horizon
        return ("v",), series.values[idx][:, None]

    arms = [
        harness.Arm("short", extract, warmup=50),
        harness.Arm("long", extract, warmup=120),
    ]
    summary = harness.run_benchmark(family, arms, [0], harness.Protocol(horizon=20))
    # The arms' warmups differ, so their natural grids differ; the harness
    # scores both on the intersection, which shows up as an equal test-slice
    # size. Without that, the longer-warmup arm would be judged on a different
    # slice of the series than the arm it is compared against.
    rows = {summary[arm]["n_test_rows"] for arm in ("short", "long")}
    assert len(rows) == 1, rows
    assert summary["short"]["pr_auc_mean"] == pytest.approx(
        summary["long"]["pr_auc_mean"], abs=1e-9
    )
