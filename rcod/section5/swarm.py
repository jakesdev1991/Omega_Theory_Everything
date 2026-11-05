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
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _zscore_rows(M):
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd


def swarm_metrics(x, macro_w=252, leaf_ws=None, seed=0):
    x = np.asarray(x, dtype=float)
    T = len(x)
    rng = np.random.default_rng(seed)
    if leaf_ws is None:
        leaf_ws = sorted(rng.integers(10, 61, size=12).tolist())
    windows = [macro_w] + list(leaf_ws)

    mu_mat = np.full((T, len(windows)), np.nan)
    for j, w in enumerate(windows):
        if w + 1 > T:
            continue
        W = sliding_window_view(x, w)          # shape (T-w+1, w); W[i] ends at t=i+w-1
        Z = _zscore_rows(W)
        norms = np.linalg.norm(Z, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        Zn = Z / norms[:, None]
        cos = np.einsum("ij,ij->i", Zn[1:], Zn[:-1])   # cos between window t and t-1
        cos = np.clip(cos, -1.0, 1.0)
        mu = np.sqrt(np.maximum(0.0, 1.0 - cos ** 2))
        # mu[k] compares windows ending at t=k+w-1 and t=k+w-2 -> assign to t=k+w-1
        mu_mat[w : w + len(mu), j] = mu

    median_mu = np.nanmedian(mu_mat, axis=1)
    macro_mu = mu_mat[:, 0]
    dissonance = np.nanstd(mu_mat, axis=1)
    return {
        "median_mu": median_mu,
        "macro_mu": macro_mu,
        "dissonance": dissonance,
        "leaf_ws": leaf_ws,
    }


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


def temporal_hierarchy(x, block_sizes=(84, 21, 6)):
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
    x = np.asarray(x, dtype=float)
    out = {}
    for lv, W in enumerate(block_sizes, start=1):
        n = len(x) // W
        B = x[: n * W].reshape(n, W)
        Z = _zscore_rows(B)
        norms = np.linalg.norm(Z, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        Zn = Z / norms[:, None]
        cos = np.einsum("ij,ij->i", Zn[1:], Zn[:-1])
        cos = np.clip(cos, -1.0, 1.0)
        mu = np.full(n, np.nan)
        mu[1:] = np.sqrt(np.maximum(0.0, 1.0 - cos ** 2))
        out[lv] = {"mu": mu, "W": W, "n": n}
    return out
