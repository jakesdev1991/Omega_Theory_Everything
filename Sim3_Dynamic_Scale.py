# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0

"""
Simulation 3: Dynamic Planck Scale & Disformal Causality Band (Omega Theory v4.0)

Implements:
- Dynamic Planck length l_P(Phi) = l_P0 * exp((1 - Phi) / phi_c)
- Conformal factor C(Phi) = exp(-2 * Phi)
- Disformal causality bound |dPhi/dt| < exp(-(Phi + 1)/2) / sqrt(beta)
- Time-dependent scalar field EOM: d2Phi/dt2 + 3H(Phi)dPhi/dt + m^2 * Phi = 0
- Ringdown frequency shift predictions under informational gradients.
"""

import numpy as np
from scipy.integrate import solve_ivp


def lP_func(phi: float, lP0: float = 1.0, phi_c: float = 0.5) -> float:
    return lP0 * float(np.exp((1.0 - phi) / phi_c))


def conformal_factor(phi: float) -> float:
    return float(np.exp(-2.0 * phi))


def causality_bound(phi: float, beta: float = 1.0) -> float:
    return float(np.exp(-(phi + 1.0) / 2.0) / np.sqrt(beta))


def phi_eom(
    t: float, y: list[float], H0: float = 0.5, m: float = 0.1
) -> list[float]:
    phi, phidot = y
    H_phi = H0 * np.exp(phi)
    phiddot = -3.0 * H_phi * phidot - (m**2) * phi
    return [float(phidot), float(phiddot)]


def main() -> None:
    phi_vals = np.linspace(0, 5, 11)
    beta_sweep = [1.0, 0.5, 0.05]

    bounds = {b: [causality_bound(p, beta=b) for p in phi_vals] for b in beta_sweep}

    # Dynamic scalar field simulation
    phi0 = 0.0
    phidot0 = 0.5 * causality_bound(phi0, beta=1.0)
    t_span = (0.0, 10.0)
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(
        phi_eom, t_span, [phi0, phidot0], t_eval=t_eval, method="RK45", rtol=1e-6
    )

    t = sol.t
    phi_num = sol.y[0]
    phidot_num = sol.y[1]

    bound_vals = np.array([causality_bound(p, beta=1.0) for p in phi_num])
    causality_ratio = np.abs(phidot_num) / bound_vals
    causality_satisfied = bool(np.all(causality_ratio < 1.0))

    # Ringdown frequency shift proxy
    ringdown_shift_pct = (
        100.0 * np.abs(phidot_num) / (np.exp(phi_num) * lP_func(0.0))
    )

    print(
        "Simulation 3 (Omega Theory v4.0 Dynamic Scale & Causality) completed successfully."
    )
    print(
        f"Dynamic Planck Scale l_P Range: [{lP_func(phi_vals[0]):.4f}, {lP_func(phi_vals[-1]):.6f}]"
    )
    print(f"Causality Band Bound (beta=1.0) at Phi=0: {bounds[1.0][0]:.4f}")
    print(f"Scalar Field Causality Satisfied Throughout EOM: {causality_satisfied}")
    print(f"Peak Estimated Ringdown Frequency Shift: {np.max(ringdown_shift_pct):.2f}%")


if __name__ == "__main__":
    main()
