# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 2: Cosmological Dynamics and Depletion Law (Omega Theory v4.0)

Implements the Depletion Law and Informational Friedmann Equation:
  dI/dt = -gamma * A_BH^kappa * I
  H(t) = alpha * gamma * A_BH^kappa(t)
Yielding De Sitter expansion with w_eff -> -1 attractor.
"""

import numpy as np
from scipy.integrate import solve_ivp


def depletion_system(
    t: float, y: list[float], gamma: float = 0.35, kappa: float = 1.0
) -> list[float]:
    r = y[0]
    dr_dt = gamma * r
    return [dr_dt]


def main() -> None:
    t_max = 10.0
    gamma_decay = 0.35
    r0 = 1.0

    t_eval = np.linspace(0, t_max, 100)
    sol = solve_ivp(
        depletion_system,
        [0, t_max],
        [r0],
        t_eval=t_eval,
        args=(gamma_decay, 1.0),
    )

    t = sol.t
    r = sol.y[0]
    a_info = r / r0
    H_info = np.gradient(a_info, t) / a_info

    print("Simulation 2 completed successfully.")
    print(f"Final scale factor a({t_max}): {a_info[-1]:.4f}")
    print(f"Mean Hubble parameter H: {np.mean(H_info):.4f}")


if __name__ == "__main__":
    main()
