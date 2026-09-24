# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 3: Dynamic Planck Length and Disformal Causality Band (Omega Theory v4.0)

Implements dynamic stiffness l_P(Phi) = l_P0 * exp((1 - Phi) / phi_c) and checks
causality bounds for Phi field updates.
"""

import numpy as np


def lP_func(phi: float, lP0: float = 1.0, phi_c: float = 0.5) -> float:
    return lP0 * float(np.exp((1.0 - phi) / phi_c))


def causality_bound(phi: float, beta: float = 1.0) -> float:
    return float(np.exp(-(phi + 1.0) / 2.0) / np.sqrt(beta))


def main() -> None:
    phi_vals = np.linspace(0, 5, 11)
    beta_sweep = [1.0, 0.5, 0.05]

    lP_vals = [lP_func(p) for p in phi_vals]
    bounds = {b: [causality_bound(p, beta=b) for p in phi_vals] for b in beta_sweep}

    print("Simulation 3 completed successfully.")
    print(f"Planck length l_P at Phi=0: {lP_vals[0]:.4f}")
    print(f"Planck length l_P at Phi=5: {lP_vals[-1]:.4f}")
    print(f"Causality bound (beta=1.0) at Phi=0: {bounds[1.0][0]:.4f}")


if __name__ == "__main__":
    main()
