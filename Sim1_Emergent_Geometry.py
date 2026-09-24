# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 1: Emergent Geometry from Quantum Information (Omega Theory v4.0)

Implements logarithmic distance mapping d_ij = -l_P(Phi) * ln(I_ij / I_max)
where l_P(Phi) = l_P0 * exp((1 - Phi) / phi_c) from Omega Theory v4.0 Technical Specification.
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


def classical_mds_1d(D: np.ndarray) -> np.ndarray:
    N = D.shape[0]
    D2 = D**2
    J = np.ones((N, N)) / N
    H = np.eye(N) - J
    B = -0.5 * H @ D2 @ H
    w, v = np.linalg.eigh(B)
    idx = np.argmax(w)
    lam = max(float(w[idx]), 0.0)
    x = v[:, idx] * np.sqrt(lam)
    if x[0] > x[-1]:
        x = -x
    return x


def main() -> None:
    N, xi = 10, 2.02
    phi_val = 0.95
    lP = lP_func(phi_val)

    I = build_mutual_info(N, xi)
    K = normalize_kernel(I)
    R = kernel_to_distances(K, lP)
    xembed = classical_mds_1d(R)

    print(f"Simulation 1 completed successfully.")
    print(f"Emergent 1D embedding range: [{xembed[0]:.4f}, {xembed[-1]:.4f}]")
    print(f"Calibrated l_P(Phi={phi_val}): {lP:.4f}")


if __name__ == "__main__":
    main()
