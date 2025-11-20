Simulation 1: Emergent Distances and 1D Embedding

Abstract
We present a toy simulation demonstrating how information‑theoretic correlations can generate emergent spatial structure. Starting from a mutual‑information kernel on a one‑dimensional chain, we construct effective distances via logarithmic mapping, embed the system using classical multidimensional scaling (MDS), and quantify the fidelity of the emergent geometry. This simulation establishes the baseline case for a hierarchy of models supporting a unification framework in which geometry arises from correlation structure.

---

1. Introduction
A central theme of the proposed framework is that geometry is not fundamental but emergent. Correlation patterns between subsystems encode effective distances, and from these distances one can reconstruct a spatial embedding. Simulation 1 provides the simplest demonstration: a one‑dimensional chain with exponentially decaying correlations. By calibrating the decay length to a target end‑to‑end separation, we recover a linear embedding consistent with classical metric space structure. This serves as the seed case for higher‑dimensional and dynamical extensions (Simulations 2 and 3).

---

2. Methods

2.1 Mutual Information Kernel
We define a toy mutual information matrix:
\[
I_{ij} = \exp\!\left(-\frac{|i-j|}{\xi}\right),
\]
where \(\xi\) is the correlation length in units of lattice steps. This captures the intuition that nearby sites share stronger correlations.

2.2 Kernel Normalization and Distance Mapping
- The kernel is symmetrized and normalized so that \(K_{ii} = 1\).  
- Effective distances are defined by
\[
r{ij} = -\lambdaP \ln K_{ij},
\]
where \(\lambdaP\) is calibrated so that the end‑to‑end distance \(r{0,N-1}\) matches a chosen target.

2.3 Embedding and Alignment
- Classical MDS is applied to the distance matrix, yielding a one‑dimensional embedding.  
- Procrustes alignment maps the emergent coordinates onto the expected linear chain, allowing direct comparison.  
- Stress‑1 and triangle inequality checks quantify metric consistency.

2.4 Noise Experiment
To model decoherence or depolarization, we introduce kernel‑level noise:
\[
K \;\to\; (1-p)K + pI,
\]
which dilates distances and stretches the embedding.

---

3. Results

- Calibration: The kernel decay length was tuned so that the reconstructed chain spanned the target end‑to‑end distance.  
- Embedding: The MDS embedding aligned closely with the true linear chain, with high correlation and low relative RMS error.  
- Metricity: Stress‑1 values were small, and triangle inequality violations were within numerical tolerance, confirming that the emergent distances form a valid metric.  
- Noise: Adding depolarizing noise stretched the embedding by ~10%, consistent with the interpretation that information loss increases effective separation.

---

4. Discussion
This simulation demonstrates that spatial structure can be reconstructed from correlation data alone. The one‑dimensional chain provides a controlled baseline: correlations decay exponentially, distances emerge logarithmically, and the embedding recovers the expected geometry. The noise experiment illustrates how decoherence manifests geometrically as dilation.  

In the broader framework, this result is the first step toward:
- Simulation 2: Higher‑dimensional embeddings from more complex correlation topologies.  
- Simulation 3: Dynamical evolution of emergent geometry under time‑dependent correlations.  

Together, these simulations support the thesis that geometry, dynamics, and ultimately physical law can be derived from information‑theoretic primitives.

---

5. Conclusion
Simulation 1 establishes the principle that geometry is emergent from information. By grounding this in a simple, reproducible toy model, we prepare the foundation for scaling to richer structures. This baseline case will serve as the reference point for the subsequent simulations in the series.

---

Appendix A: Cleaned Code (Python)

`python

sim1.py

Simulation 1: Emergent Distances and 1D Embedding

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
