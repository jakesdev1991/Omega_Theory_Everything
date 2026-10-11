# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT

# Simulation 1: Exponential Kernel and 1D Embedding (toy)

> **Status: manuscript, not an executable simulation.** This file spent its
> life as `Sim1_Emergent_Geometry.py` at the repository root, where the
> Quick Start instructions told readers to run it. It never parsed. The v4.0
> paste of this document passed the text through a Markdown round-trip, which
> consumed underscores and double asterisks, so the appendix listing below is
> damaged beyond mechanical repair: `build_mutual_info` appears as
> `buildmutualinfo`, `D**2` as `D  2`, and the final expression is truncated
> mid-line. The prose is intact and is the actual content of this document.
>
> To restore a runnable Sim1, drop the original script in as
> `Sim1_Emergent_Geometry.py`; the damaged listing here should not be pasted
> back into a `.py` file.

Abstract
This historical manuscript describes a one-dimensional numerical toy in which a prescribed exponential kernel is transformed with a logarithm and embedded by classical multidimensional scaling (MDS). The construction can reproduce a line because the kernel is chosen to decay exponentially with lattice separation; it does not calculate mutual information from a quantum state, derive a physical spatial metric, or establish an emergent-geometry theory. The appendix is damaged and the reported numerical results are not currently reproducible from this file.

---

1. Introduction
The exercise tests an elementary construction: start with an exponentially decaying positive kernel on a line, take its negative logarithm, and apply MDS. Since the decay is specified as a function of the line separation and its scale is calibrated to a target end-to-end length, recovering a line is expected by construction. The result is a code/model sanity check only; it does not show that a physical correlation pattern determines space or support higher-dimensional or dynamical extensions.

---

2. Methods

2.1 Prescribed Exponential Kernel
We prescribe a dimensionless toy kernel:
\[
C_{ij} = \exp\!\left(-\frac{|i-j|}{\xi}\right),
\]
where \(\xi\) is a chosen decay length in lattice steps. This is not a mutual-information matrix calculated from a quantum state. It is a mathematical input selected to have exponential decay.

2.2 Kernel Normalization and Distance Mapping
- The prescribed kernel is symmetrized and normalized so that \(K_{ii}=1\).
- A toy distance matrix is defined by
\[
r_{ij}=-\lambda_P\ln K_{ij},
\]
where \(\lambda_P\) is calibrated so that the end-to-end distance \(r_{0,N-1}\) matches a chosen target. This is a definition plus calibration, not an independently predicted length.

2.3 Embedding and Alignment
- Classical MDS is applied to the distance matrix, yielding a one‑dimensional embedding.  
- Procrustes alignment maps the emergent coordinates onto the expected linear chain, allowing direct comparison.  
- Stress‑1 and triangle inequality checks quantify metric consistency.

2.4 Kernel Perturbation (not a quantum noise channel)
A numerical perturbation is applied directly to the kernel:
\[
K \;\to\; (1-p)K + p\mathbb{1}.
\]
This operation is not derived from a quantum channel or a decoherence model. Its effect on the transformed distances is a property of this chosen perturbation.

---

3. Historical Results (not independently reproducible here)

The earlier text reported that the calibrated MDS embedding aligned closely with a target linear chain, that numerical metric checks passed, and that the kernel perturbation stretched the embedding by about 10%. The appendix below is corrupted and the original executable is not present, so those values cannot be reproduced from this manuscript and should not be treated as verified results. Even if reproduced, the alignment is expected from the exponential kernel, logarithm, and target-length calibration; it is not evidence about physical space or decoherence.

---

4. Discussion
The exercise illustrates that a selected positive kernel can be transformed into a distance-like matrix and embedded. Because the input kernel already encodes a line through its exponential dependence on index separation, this is not a reconstruction from independent physical data. The kernel perturbation is likewise not a decoherence experiment.

This manuscript motivates possible mathematical experiments with other kernels, but it supplies no physical result and does not support a claim that geometry, dynamics, or physical law have been derived from information-theoretic primitives.

---

5. Conclusion
This historical toy construction does not establish that geometry is emergent from information. Because the code listing is damaged, it should first be recovered and independently tested before its numerical values are reused. A future model would need to calculate correlations from a specified state, avoid calibrating to the target geometry, and test whether the resulting metric recovers known limits.

---

Appendix A: Cleaned Code (Python)

```python

# sim1.py — recovered listing (DAMAGED, see the note at the top of this file)

Simulation 1: Prescribed Exponential Kernel and 1D Embedding

- PSD-normalized kernel (no projection needed for exp-decay)

- Exact MDS centering (double-centering)

- λP calibrated to target r_{0,N-1}

- Optional geodesic metricization (k-NN shortest paths)

- Proper stress-1 and triangle checks

- Kernel-level depolarizing noise (dilates distances)



Dependencies: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt

def buildmutualinfo(N, xi):
    idx = np.arange(N)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.exp(-d / xi)

def buildkernel(I, alpha, jitter=1e-12, psdproject=False):
    K0 = np.power(I, alpha)
    K0 = 0.5 * (K0 + K0.T)
    if psd_project:
        w, v = np.linalg.eigh(K0)
        w = np.clip(w, 0, None)
        K_psd = (v * w) @ v.T
    else:
        K_psd = K0
    return Kpsd + jitter * np.eye(Kpsd.shape[0])

def normalize_kernel(K, eps=1e-12):
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    return K / (d[:, None] * d[None, :])

def kerneltodistances(K_norm, lP, eps=1e-12):
    Kc = np.clip(K_norm, eps, 1.0)
    R = -lP * np.log(Kc)
    np.fill_diagonal(R, 0.0)
    return 0.5 * (R + R.T)

def classicalmds1d(D):
    N = D.shape[0]
    D2 = D  2
    J = np.ones((N, N)) / N
    H = np.eye(N) - J
    B = -0.5 * H @ D2 @ H
    w, v = np.linalg.eigh(B)
    idx = np.argmax(w)
    lam = max(w[idx], 0.0)
    x = v[:, idx] * np.sqrt(lam)
    if x[0] > x[-1]:
        x = -x
    return x

def procrustes1d(xsrc, x_tgt):
    stdsrc, stdtgt = np.std(xsrc), np.std(xtgt)
    if stdsrc < 1e-15 or stdtgt < 1e-15:
        return x_tgt.copy(), 1.0, 0.0, 1.0
    corr = np.corrcoef(xsrc, xtgt)[0, 1]
    scale = corr * (stdtgt / stdsrc)
    shift = np.mean(xtgt) - scale * np.mean(xsrc)
    xaligned = scale * xsrc + shift
    return x_aligned, scale, shift, corr

def stress1(dembed, dtrue):
    N = d_true.shape[0]
    iu = np.triu_indices(N, 1)
    num = np.sum((dembed[iu] - dtrue[iu])  2)
    den = np.sum(d_true[iu]  2)
    return np.sqrt(num / den)

def pairwise_dist(x):
    return np.abs(x[:, None] - x[None, :])

def mintriangleviolation(D):
    n = D.shape[0]
    vmin = np.inf
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                v = D[i, j] + D[j, k] - D[i, k]
                if v < vmin:
                    vmin = v
    return vmin

def main():
    N, alpha, xi = 10, 0.5, 2.02
    targetr0N = 1.797
    p_noise = 0.10
    I = buildmutualinfo(N, xi)
    K0 = build_kernel(I, alpha)
    K = normalize_kernel(K0)
    step_coeff = alpha / xi
    lP = targetr0N / ((N - 1) * step_coeff)
    print(f"Calibrated lP = {lP:.6f} to match r0,{N-1} = {targetr_0N}")
    R = kerneltodistances(K, lP)
    xembed = classicalmds_1d(R)
    xtrue = np.linspace(-targetr0N / 2, targetr_0N / 2, N)
    xaligned, scale, shift, corr = procrustes1d(xembed, xtrue)
    relrms = np.sqrt(np.mean((xaligned - xtrue)  2)) / np.std(xtrue)
    Dembed = pairwisedist(x_aligned
```
