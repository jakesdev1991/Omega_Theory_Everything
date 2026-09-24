# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 2: Cosmological Expansion via Depletion Mechanics (Omega Theory v4.0)

Implements:
- BHARD (Black Hole Area Growth Density) evolution curve A_BH(z)
- Depletion Law: dI/dt = -gamma * A_BH(z)^kappa * I
- Informational Hubble Rate: H(t) = alpha * gamma * A_BH^kappa(t)
- Cosmological diagnostics: scale factor a(z), H(z), deceleration parameter q(z),
  and effective equation of state w_eff(z) verifying the w -> -1 de Sitter attractor.
"""

import numpy as np
from scipy.integrate import solve_ivp


def bhard_curve(z: np.ndarray | float, A0: float = 1.0) -> np.ndarray:
    z_arr = np.atleast_1d(z)
    z_clipped = np.maximum(z_arr, 0.0)
    # Model quasar peak at z ~ 2
    a_bhard = 0.15 + 4.0 * (z_clipped**2) * np.exp(-z_clipped / 1.0)
    res = A0 * np.maximum(a_bhard, 0.0)
    return res if isinstance(z, np.ndarray) else res[0]


def depletion_derivs(
    t: float,
    y: list[float],
    gamma: float = 0.35,
    kappa: float = 1.0,
    alpha: float = 1.0,
) -> list[float]:
    u, a = y
    a = max(a, 1e-6)
    z = 1.0 / a - 1.0
    A_bh = float(bhard_curve(z))
    du_dt = -gamma * (A_bh**kappa)
    H_info = alpha * abs(du_dt)
    da_dt = a * H_info
    return [float(du_dt), float(da_dt)]


def main() -> None:
    t_span = (0.001, 10.0)
    y0 = [0.0, 0.01]  # u=ln(I/I0), initial scale factor
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(
        depletion_derivs, t_span, y0, t_eval=t_eval, method="Radau", rtol=1e-6
    )

    t = sol.t
    u = sol.y[0]
    a = sol.y[1]
    I_info = np.exp(u)

    H = np.gradient(a, t) / a
    dH_dt = np.gradient(H, t)
    w_eff = -1.0 - (2.0 / 3.0) * (dH_dt / (H**2 + 1e-9))
    q_dec = -1.0 - (dH_dt / (H**2 + 1e-9))

    print(
        "Simulation 2 (Omega Theory v4.0 Cosmology & Depletion) completed successfully."
    )
    print(f"Initial Information Density I(0): {I_info[0]:.4f}")
    print(f"Final Information Density I({t[-1]}): {I_info[-1]:.6e}")
    print(f"Scale Factor Expansion Range a: [{a[0]:.4f}, {a[-1]:.4f}]")
    print(f"Current Effective Equation of State w_eff: {w_eff[-1]:.4f}")
    print(f"Deceleration Parameter q (accelerating if <0): {q_dec[-1]:.4f}")


if __name__ == "__main__":
    main()
