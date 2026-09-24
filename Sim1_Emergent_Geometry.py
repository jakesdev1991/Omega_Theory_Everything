# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 1: Emergent Geometry from Quantum Information (Omega Theory v4.0)

Implements:
- Logarithmic Distance Mapping: d_ij = -l_P(Phi) * ln(I_ij / I_max)
- Dynamic Planck Scale: l_P(Phi) = l_P0 * exp((1 - Phi) / phi_c)
- Multi-dimensional Classical Multidimensional Scaling (MDS)
- Stress-1 Evaluation and Triangle Inequality Metricity Verification
- Depolarizing Noise Dilation Modeling
"""

import numpy as np


def build_mutual_info(N: int, xi: float) -> np.ndarray:
    idx = np.arange(N)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.exp(-d / xi)


def normalize_kernel(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    return K / (d[:, None] * d[None, :])


def lP_func(phi: float, lP0: float = 1.0, phi_c: float = 0.1) -> float:
    return lP0 * float(np.exp((1.0 - phi) / phi_c))


def kernel_to_distances(
    K_norm: np.ndarray, lP: float, eps: float = 1e-12
) -> np.ndarray:
    Kc = np.clip(K_norm, eps, 1.0)
    R = -lP * np.log(Kc)
    np.fill_diagonal(R, 0.0)
    return 0.5 * (R + R.T)


def classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    N = D.shape[0]
    D2 = D**2
    J = np.ones((N, N)) / N
    H = np.eye(N) - J
    B = -0.5 * H @ D2 @ H
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    lam = np.maximum(w[idx], 0.0)
    X = v[:, idx] * np.sqrt(lam)
    return X


def stress1(D_embed: np.ndarray, D_true: np.ndarray) -> float:
    N = D_true.shape[0]
    iu = np.triu_indices(N, 1)
    num = float(np.sum((D_embed[iu] - D_true[iu]) ** 2))
    den = float(np.sum(D_true[iu] ** 2))
    return float(np.sqrt(num / max(den, 1e-12)))


def min_triangle_violation(D: np.ndarray) -> float:
    n = D.shape[0]
    vmin = float("inf")
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                v = float(D[i, j] + D[j, k] - D[i, k])
                if v < vmin:
                    vmin = v
    return vmin


def apply_depolarizing_noise(
    K: np.ndarray, p: float = 0.1, eps: float = 1e-12
) -> np.ndarray:
    K_noisy = (1.0 - p) * K + p * np.eye(K.shape[0])
    return normalize_kernel(K_noisy, eps)


def main() -> None:
    N, xi = 15, 2.5
    phi_val = 0.95
    lP = lP_func(phi_val)

    I = build_mutual_info(N, xi)
    K = normalize_kernel(I)
    D = kernel_to_distances(K, lP)

    # Multi-dimensional MDS embeddings
    X_1d = classical_mds(D, dim=1)
    X_2d = classical_mds(D, dim=2)

    D_embed_1d = np.abs(X_1d[:, 0][:, None] - X_1d[:, 0][None, :])
    D_embed_2d = np.linalg.norm(X_2d[:, None, :] - X_2d[None, :, :], axis=-1)

    s1_1d = stress1(D_embed_1d, D)
    s1_2d = stress1(D_embed_2d, D)
    tri_check = min_triangle_violation(D)

    # Depolarizing noise experiment
    K_noisy = apply_depolarizing_noise(K, p=0.1)
    D_noisy = kernel_to_distances(K_noisy, lP)
    mean_dil = float(np.mean(D_noisy - D))

    print("Simulation 1 (Omega Theory v4.0 Geometry) completed successfully.")
    print(f"Dynamic Planck Length l_P(Phi={phi_val}): {lP:.4f}")
    print(f"1D Embedding Stress-1: {s1_1d:.6f} | 2D Embedding Stress-1: {s1_2d:.6f}")
    print(f"Triangle Inequality Minimum Margin: {tri_check:.6f}")
    print(f"Mean Distance Dilation under 10% Depolarizing Noise: {mean_dil:.4f}")


if __name__ == "__main__":
    main()
