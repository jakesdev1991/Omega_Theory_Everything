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

The harness that used to be tested here now lives in `packages/onset_eval/` and
is tested there, including against an installed wheel; what remains here is the
metric core plus one check that the `rcod.harness` shim still resolves to the
published package.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rcod.harness as rcod_harness  # noqa: E402
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
# the published package: rcod.harness is a shim over onset_eval
# --------------------------------------------------------------------------
def test_rcod_harness_shim_resolves_to_the_published_package():
    """`rcod.harness` must keep working now that the code moved to a package."""
    import onset_eval
    import onset_eval.harness as published

    assert rcod_harness.__all__ == published.__all__
    for name in published.__all__:
        assert getattr(rcod_harness, name) is getattr(published, name), name
    assert onset_eval.__version__


def test_package_public_surface_matches_its_module():
    import onset_eval

    for name in (
        "Arm",
        "Protocol",
        "Series",
        "evaluate_arm",
        "label_onsets",
        "paired_test",
        "render_report",
        "run_benchmark",
        "decide",
    ):
        assert hasattr(onset_eval, name), name
