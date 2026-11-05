"""Stochastic Swarm metrics per RCOD_Operational_Logic.pdf sections 3-4.

For each window w at time t:
    v_{t,w}   = x[t-w+1 : t+1]          (recent history)
    Phi_w     = cosine_similarity(v_{t,w}, v_{t-1,w})
    mu_w      = sqrt(max(0, 1 - Phi_w^2))
Aggregates per t:
    median_mu  = median(mu_w over windows)
    macro_mu   = mu_w for the macro anchor window
    dissonance = std(mu_w over windows)

Documented choice (see DESIGN.md): each window is z-scored before the cosine
similarity, otherwise a nonzero series mean pins Phi ~= 1 and mu ~= 0.

Nothing in this module is specific to cosmology, to the Omega corpus, or to a
particular data source: it is a multiscale novelty measure on a real-valued
series. That is deliberate -- the metric family is the part of the RCOD work
that transfers to other domains, so the implementation is written to be used
that way:

    x           1-D (length T) or 2-D (T, channels)
    normalize   "zscore" (default, as specified) or "none" (raw overlap, which
                is level-shift sensitive -- the documented blind spot of the
                z-scored form)
    backend     "auto" picks the incremental prefix-sum path for long series;
                "direct" is the reference implementation used by the committed
                benchmark results; "prefix" forces the fast path. Both paths
                are the same arithmetic and agree to floating-point tolerance
                (tests/test_rcod.py checks that against random data).
    SwarmTracker  the same metrics computed online, one sample at a time, with
                memory independent of T -- what a monitoring system embeds.

Cost, which is why there are two backends: the direct path materialises a
(T, w) z-score block per window, so peak memory is O(T * sum(w)) -- about
0.9 GB at T = 200k with the default window set, and it is killed outright near
T = 1M. The prefix path keeps O(T * number_of_windows) and allocates no
per-window block; the streaming tracker keeps O(sum of windows) regardless of
how long the stream runs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Below this a window is numerically flat. The reference implementation maps a
# flat window to cos = 0, i.e. mu = 1 ("maximal novelty"), because the guard
# exists only to avoid a division by zero. For a domain-agnostic consumer that
# is a false-alarm generator (a stuck sensor reads as chaos), so `degenerate`
# makes the choice explicit; "max" preserves the published behaviour.
_FLAT = 1e-12


def _quiet():
    """Silence the all-NaN warnings the warm-up region legitimately produces."""
    import contextlib
    import warnings

    stack = contextlib.ExitStack()
    stack.callback(lambda: None)
    stack.enter_context(np.errstate(invalid="ignore"))
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter("ignore", RuntimeWarning)
    return stack


def _zscore_rows(M):
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd < _FLAT, 1.0, sd)
    return (M - mu) / sd


def _default_leaf_windows(seed=0):
    rng = np.random.default_rng(seed)
    return sorted(rng.integers(10, 61, size=12).tolist())


def _mu_from_cos(cos, flat, degenerate="max"):
    cos = np.clip(cos, -1.0, 1.0)
    mu = np.sqrt(np.maximum(0.0, 1.0 - cos**2))
    if degenerate == "zero":
        mu = np.where(flat, 0.0, mu)
    elif degenerate != "max":
        raise ValueError(f"degenerate must be 'max' or 'zero', got {degenerate!r}")
    return mu


def _check_normalize(normalize):
    if normalize not in ("zscore", "none"):
        raise ValueError(f"normalize must be 'zscore' or 'none', got {normalize!r}")


def _mu_direct(x, w, normalize="zscore", degenerate="max"):
    """Reference path: materialise the (T-w+1, w) window block for one w."""
    T = len(x)
    mu = np.full(T, np.nan)
    if w + 1 > T:
        return mu
    W = sliding_window_view(x, w)  # shape (T-w+1, w); W[i] ends at t=i+w-1
    if normalize == "zscore":
        Z = _zscore_rows(W)
        norms = np.linalg.norm(Z, axis=1)
        flat = norms < _FLAT
        norms = np.where(flat, 1.0, norms)
        Zn = Z / norms[:, None]
        cos = np.einsum("ij,ij->i", Zn[1:], Zn[:-1])
    else:
        norms = np.linalg.norm(W, axis=1)
        flat = norms < _FLAT
        norms = np.where(flat, 1.0, norms)
        Wn = W / norms[:, None]
        cos = np.einsum("ij,ij->i", Wn[1:], Wn[:-1])
    flat_pair = flat[1:] | flat[:-1]
    # mu[k] compares windows ending at t=k+w-1 and t=k+w-2 -> assign to t=k+w-1
    mu[w : w + len(cos)] = _mu_from_cos(cos, flat_pair, degenerate=degenerate)
    return mu


def _mu_prefix(x, w, normalize="zscore", degenerate="max"):
    """Incremental path: O(T) per window from prefix sums, no (T, w) block.

    With a_t = x[t-w+1 .. t] and b_t = x[t-w .. t-1], the cosine of the two
    z-scored windows needs only four prefix-summed quantities over the series:
    sum a_i, sum a_i^2, sum b_i, sum b_i^2 and sum a_i b_i -- and the last is
    the sum of the lag-1 product series p_j = x_j x_{j+1} over j in [t-w, t-1],
    which is another prefix sum. Everything is O(1) per (window, t).
    """
    T = len(x)
    mu = np.full(T, np.nan)
    if w + 1 > T:
        return mu

    # Z-scoring is invariant to a global shift, so centre once: this keeps the
    # cancellation in (sum x^2 - (sum x)^2/w) away from large offsets, which is
    # exactly the mean-shift regime that makes the metric ill-conditioned.
    # The raw ("none") overlap is *not* shift-invariant, so it is left alone.
    y = x - float(np.median(x)) if normalize == "zscore" else x

    c1 = np.concatenate(([0.0], np.cumsum(y)))  # length T+1
    c2 = np.concatenate(([0.0], np.cumsum(y * y)))
    p = y[:-1] * y[1:]  # length T-1, p[j] = y_j y_{j+1}
    cp = np.concatenate(([0.0], np.cumsum(p)))  # length T

    t = np.arange(w, T)
    a1 = c1[t + 1] - c1[t - w + 1]
    a2 = c2[t + 1] - c2[t - w + 1]
    b1 = c1[t] - c1[t - w]
    b2 = c2[t] - c2[t - w]
    ab = cp[t] - cp[t - w]

    if normalize == "zscore":
        # (w*sigma_a*sigma_b) and (w*covar) share the factor w, so it drops out
        var_a = np.clip(a2 - a1 * a1 / w, 0.0, None)
        var_b = np.clip(b2 - b1 * b1 / w, 0.0, None)
        flat = (var_a < _FLAT) | (var_b < _FLAT)
        denom = np.sqrt(var_a * var_b)
        cos = np.where(flat, 0.0, (ab - a1 * b1 / w) / np.where(flat, 1.0, denom))
    else:
        na = np.sqrt(np.clip(a2, 0.0, None))
        nb = np.sqrt(np.clip(b2, 0.0, None))
        denom = na * nb
        flat = denom < _FLAT
        cos = np.where(flat, 0.0, ab / np.where(flat, 1.0, denom))

    mu[w:] = _mu_from_cos(cos, flat, degenerate=degenerate)
    return mu


def _metrics_from_matrix(mu_mat):
    with _quiet():
        median_mu = np.nanmedian(mu_mat, axis=1)
        macro_mu = mu_mat[:, 0].copy()
        dissonance = np.nanstd(mu_mat, axis=1)
    return {
        "median_mu": median_mu,
        "macro_mu": macro_mu,
        "dissonance": dissonance,
    }


def _auto_backend(T, sum_windows):
    """Prefix sums win once the direct path's per-window block stops fitting.

    The direct path allocates roughly T * sum(window) floats. Below ~40 MB of
    that it is the faster of the two in practice and is the path the committed
    benchmark results were produced with, so it stays the default for
    benchmark-sized inputs.
    """
    return "direct" if T * sum_windows <= 5e6 else "prefix"


def swarm_metrics(
    x,
    macro_w=252,
    leaf_ws=None,
    seed=0,
    normalize="zscore",
    backend="auto",
    degenerate="max",
):
    """Multiscale overlap metrics for one series or a batch of channels.

    `x` is (T,) for a single series, or (T, C) for C channels sampled together;
    for 2-D input every metric comes back with shape (T, C) and
    `pool_channels` reduces across channels.

    Returns a dict with `median_mu`, `macro_mu`, `dissonance` (NaN until the
    corresponding windows are populated), plus `leaf_ws`, `windows`,
    `normalize` and the resolved `backend`.
    """
    _check_normalize(normalize)
    arr = np.asarray(x, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D (T, channels), got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("x is empty")
    if not np.isfinite(arr).all():
        raise ValueError(
            "x contains NaN/inf; interpolate or mask gaps first -- the metric is "
            "undefined across missing samples (see docs/DOMAIN_AGNOSTIC_CORE.md)"
        )

    if arr.ndim == 2:
        return _swarm_metrics_channels(
            arr,
            macro_w=macro_w,
            leaf_ws=leaf_ws,
            seed=seed,
            normalize=normalize,
            backend=backend,
            degenerate=degenerate,
        )

    if leaf_ws is None:
        leaf_ws = _default_leaf_windows(seed)
    windows = [int(macro_w)] + [int(w) for w in leaf_ws]
    if any(w < 2 for w in windows):
        raise ValueError(f"window sizes must be >= 2, got {windows}")
    if max(windows) + 1 > len(arr):
        raise ValueError(
            f"series of length {len(arr)} is too short for window {max(windows)}"
        )

    if backend == "auto":
        backend = _auto_backend(len(arr), sum(windows))
    if backend not in ("direct", "prefix"):
        raise ValueError(
            f"backend must be 'auto', 'direct' or 'prefix', got {backend!r}"
        )

    fn = _mu_prefix if backend == "prefix" else _mu_direct
    mu_mat = np.column_stack(
        [fn(arr, w, normalize=normalize, degenerate=degenerate) for w in windows]
    )
    out = _metrics_from_matrix(mu_mat)
    out.update(leaf_ws=leaf_ws, windows=windows, normalize=normalize, backend=backend)
    return out


def _swarm_metrics_channels(
    X,
    macro_w=252,
    leaf_ws=None,
    seed=0,
    normalize="zscore",
    backend="auto",
    degenerate="max",
):
    per_channel = [
        swarm_metrics(
            X[:, c],
            macro_w=macro_w,
            leaf_ws=leaf_ws,
            seed=seed,
            normalize=normalize,
            backend=backend,
            degenerate=degenerate,
        )
        for c in range(X.shape[1])
    ]
    out = {
        key: np.column_stack([m[key] for m in per_channel])
        for key in ("median_mu", "macro_mu", "dissonance")
    }
    out.update(
        leaf_ws=per_channel[0]["leaf_ws"],
        windows=per_channel[0]["windows"],
        normalize=normalize,
        backend=per_channel[0]["backend"],
        channels=X.shape[1],
    )
    return out


def pool_channels(metrics, how="median"):
    """Reduce a (T, C) metric dict to the single-series trio.

    A multi-channel deployment needs one alarm signal; `median` is the default
    because one bad channel should not raise it (use "max" when the most
    sensitive channel should decide).
    """
    if "channels" not in metrics:
        return metrics
    reductions: dict[str, Callable[..., np.ndarray]] = {
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "mean": np.nanmean,
    }
    if how not in reductions:
        raise ValueError(f"how must be one of {sorted(reductions)}, got {how!r}")
    fn = reductions[how]
    with _quiet():
        out = {
            key: fn(metrics[key], axis=1)
            for key in ("median_mu", "macro_mu", "dissonance")
        }
    out.update(
        leaf_ws=metrics["leaf_ws"],
        windows=metrics["windows"],
        normalize=metrics["normalize"],
        pooled=how,
    )
    return out


class SwarmTracker:
    """Online `swarm_metrics`: one sample in, one metric triple out.

    Memory is O(sum of window sizes) and each `update` costs O(number of
    windows) regardless of how long the stream has run, because every window's
    sums are maintained incrementally. Values equal
    `swarm_metrics(series, ..., backend="prefix")` from the point where the
    macro window is populated; `median_mu`/`dissonance` also populate earlier,
    as they do in the batch path.

    >>> tracker = SwarmTracker(macro_w=64, leaf_ws=[8, 16])
    >>> state = tracker.update(0.13)   # None until a window pair exists
    """

    def __init__(
        self,
        macro_w=252,
        leaf_ws=None,
        seed=0,
        normalize="zscore",
        degenerate="max",
    ):
        _check_normalize(normalize)
        if leaf_ws is None:
            leaf_ws = _default_leaf_windows(seed)
        self.windows = [int(macro_w)] + [int(w) for w in leaf_ws]
        if any(w < 2 for w in self.windows):
            raise ValueError(f"window sizes must be >= 2, got {self.windows}")
        self.normalize = normalize
        self.degenerate = degenerate
        self._max_w = max(self.windows)
        self._buf = np.full(self._max_w + 2, np.nan)
        self._n = 0
        self._offset = None
        self._a1 = {w: 0.0 for w in self.windows}
        self._a2 = {w: 0.0 for w in self.windows}
        self._ab = {w: 0.0 for w in self.windows}

    def _at(self, t):
        """Value at absolute index t, or NaN when it has left the ring."""
        if t < 0 or t < self._n - len(self._buf):
            return np.nan
        return float(self._buf[t % len(self._buf)])

    def update(self, value):
        """Append one sample; return the metric triple, or None before warmup."""
        v = float(value)
        if not np.isfinite(v):
            raise ValueError("SwarmTracker does not accept NaN/inf samples")
        if self._offset is None:
            self._offset = v
        y = v - self._offset
        t = self._n
        self._buf[t % len(self._buf)] = y
        self._n += 1

        entering = self._at(t - 1) * y if t >= 1 else 0.0
        mus = np.full(len(self.windows), np.nan)
        with np.errstate(invalid="ignore"):
            for j, w in enumerate(self.windows):
                if t + 1 <= w:  # still filling the first window
                    self._a1[w] += y
                    self._a2[w] += y * y
                    self._ab[w] += entering
                    continue

                leaving = self._at(t - w)
                a1_old, a2_old = self._a1[w], self._a2[w]
                self._a1[w] = a1_old + y - leaving
                self._a2[w] = a2_old + y * y - leaving * leaving
                # P over j in [t-w, t-1]: add p_{t-1}, drop p_{t-w-1}
                dropping = 0.0
                if t - w - 1 >= 0:
                    prev = self._at(t - w - 1)
                    if np.isfinite(prev):
                        dropping = prev * self._at(t - w)
                self._ab[w] += entering - dropping

                a1, a2 = self._a1[w], self._a2[w]
                b1, b2, ab = a1_old, a2_old, self._ab[w]
                if self.normalize == "zscore":
                    var_a = max(a2 - a1 * a1 / w, 0.0)
                    var_b = max(b2 - b1 * b1 / w, 0.0)
                    flat = var_a < _FLAT or var_b < _FLAT
                    denom = (var_a * var_b) ** 0.5
                    cos = 0.0 if flat else (ab - a1 * b1 / w) / denom
                else:
                    na, nb = max(a2, 0.0) ** 0.5, max(b2, 0.0) ** 0.5
                    denom = na * nb
                    flat = denom < _FLAT
                    cos = 0.0 if flat else ab / denom
                cos = min(1.0, max(-1.0, cos))
                mu = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
                if self.degenerate == "zero" and flat:
                    mu = 0.0
                mus[j] = mu

        available = mus[~np.isnan(mus)]
        if not available.size:
            return None
        return {
            "median_mu": float(np.median(available)),
            "macro_mu": float(mus[0]),
            "dissonance": float(np.std(available)),
        }

    def metrics(self, series):
        """Metrics for a whole series, using the vectorised path.

        This deliberately does not replay `update` per sample: the Python loop
        costs about 100 us/sample here against about 2 us/sample for the
        prefix-sum backend, and the two agree to floating-point tolerance. Use
        `update` only when the data genuinely arrives one sample at a time.
        """
        return swarm_metrics(
            np.asarray(series, dtype=float),
            macro_w=self.windows[0],
            leaf_ws=self.windows[1:],
            normalize=self.normalize,
            backend="prefix",
            degenerate=self.degenerate,
        )


def hierarchical_swarm(x, levels=3, macro_w=252, seed=0):
    """Stacked swarms: the section-3 transform applied recursively.

    Level 1: swarm over the raw series -> 3 metric series.
    Level 2: swarm over each level-1 series -> 9 series.
    Level 3: swarm over each level-2 series -> 27 series.
    Returns {level: (matrix T x 3^level, names)}.
    """
    prev = [np.asarray(x, dtype=float)]
    prev_names = ["raw"]
    out = {}
    for lv in range(1, levels + 1):
        cur, names = [], []
        for s, nm in zip(prev, prev_names):
            m = swarm_metrics(s, macro_w=macro_w, seed=seed)
            for k in ("median_mu", "macro_mu", "dissonance"):
                cur.append(m[k])
                names.append(f"L{lv}({nm}).{k}")
        out[lv] = (np.column_stack(cur), names)
        prev, prev_names = cur, names
    return out


def temporal_hierarchy(x, block_sizes=(84, 21, 6), normalize="zscore"):
    """Jacob's '3 levels deep': nested temporal partitions.

    One span partitioned at three resolutions (coarse -> fine), e.g. his
    minute as 3x20s / 20x3s / 150x0.4s. For each level and each block b>=1:
        Phi_b = cosine_similarity(zscore(block_b), zscore(block_{b-1}))
        mu_b  = sqrt(max(0, 1 - Phi_b^2)); mu_0 = NaN.
    This is the section-3 primitive (overlap of consecutive windows -> mu)
    applied to consecutive *intervals*; the multiple levels play the role
    the stochastic leaves played in the single-level swarm.
    Returns {level: {'mu': array length n_blocks, 'W': block size}}.
    """
    _check_normalize(normalize)
    x = np.asarray(x, dtype=float)
    out = {}
    for lv, W in enumerate(block_sizes, start=1):
        n = len(x) // W
        B = x[: n * W].reshape(n, W)
        if normalize == "none":
            norms = np.linalg.norm(B, axis=1)
            flat = norms < _FLAT
            norms = np.where(flat, 1.0, norms)
            Bn = B / norms[:, None]
            cos = np.einsum("ij,ij->i", Bn[1:], Bn[:-1])
        else:
            Z = _zscore_rows(B)
            norms = np.linalg.norm(Z, axis=1)
            flat = norms < _FLAT
            norms = np.where(flat, 1.0, norms)
            Bn = Z / norms[:, None]
            cos = np.einsum("ij,ij->i", Bn[1:], Bn[:-1])
        mu = np.full(n, np.nan)
        mu[1:] = _mu_from_cos(cos, flat[1:] | flat[:-1])
        out[lv] = {"mu": mu, "W": W, "n": n}
    return out
