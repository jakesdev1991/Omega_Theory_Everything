# Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT
"""Linear scalarization threshold on a Schwarzschild background.

This tests a candidate gravitational-scalar branch, not a completed Omega
model. See docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class ZeroModeThreshold:
    """First sampled-bracket static s-wave zero mode on Schwarzschild."""

    eta_over_mass_squared: float
    dimensionless_outer_radius: float
    asymptotic_constant_mode: float
    scan_points: int


def asymptotic_constant_mode(
    eta_over_mass_squared: float,
    *,
    outer_radius: float = 250.0,
    horizon_offset: float = 1e-6,
    max_step: float = 0.2,
) -> float:
    """Shoot the regular static scalar mode and estimate its constant tail.

    With x=r/(2M), let Z0>0 be the constant kinetic coefficient and define
    η=f''(0)/Z0 (a length-squared effective coupling). The linearized,
    spherically symmetric, static equation on Schwarzschild is

        d/dx [x(x-1) dφ/dx] + 3(η/M²) φ/x⁴ = 0.

    A decaying zero mode has φ=A+B/x at large x with A=0. This function
    returns the asymptotic estimate A≈φ+x φ' at outer_radius.
    """
    if not math.isfinite(eta_over_mass_squared) or eta_over_mass_squared < 0.0:
        raise ValueError("eta_over_mass_squared must be finite and nonnegative")
    if not math.isfinite(outer_radius) or outer_radius <= 2.0:
        raise ValueError("outer_radius must be finite and greater than 2")
    if not math.isfinite(horizon_offset) or not 0.0 < horizon_offset < 0.01:
        raise ValueError("horizon_offset must lie between 0 and 0.01")
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be finite and positive")

    coupling = eta_over_mass_squared
    if coupling == 0.0:
        return 1.0

    # Regular Frobenius data at x=1+epsilon. The horizon condition is
    # phi'(1)=-3*coupling*phi(1); the second coefficient reduces start error.
    first = -3.0 * coupling
    second = (18.0 * coupling + 9.0 * coupling**2) / 4.0
    offset = horizon_offset
    initial_phi = 1.0 + first * offset + second * offset**2
    initial_derivative = first + 2.0 * second * offset

    def radial_system(x: float, state: np.ndarray) -> tuple[float, float]:
        phi, derivative = state
        p = x * (x - 1.0)
        p_prime = 2.0 * x - 1.0
        potential = 3.0 * coupling / x**4
        second_derivative = -(p_prime * derivative + potential * phi) / p
        return derivative, second_derivative

    solution = solve_ivp(
        radial_system,
        (1.0 + offset, outer_radius),
        (initial_phi, initial_derivative),
        method="DOP853",
        rtol=2e-10,
        atol=2e-12,
        max_step=max_step,
    )
    if not solution.success:
        raise ArithmeticError(f"zero-mode integration failed: {solution.message}")

    phi_outer, derivative_outer = solution.y[:, -1]
    return float(phi_outer + outer_radius * derivative_outer)


def first_zero_mode_threshold(
    *,
    maximum_coupling: float = 0.8,
    scan_points: int = 17,
    outer_radius: float = 250.0,
) -> ZeroModeThreshold:
    """Find the first sign-changing zero-mode bracket in a fixed scan interval."""
    if not math.isfinite(maximum_coupling) or maximum_coupling <= 0.0:
        raise ValueError("maximum_coupling must be finite and positive")
    if (
        not isinstance(scan_points, int)
        or isinstance(scan_points, bool)
        or scan_points < 3
    ):
        raise ValueError("scan_points must be an integer of at least three")

    couplings = np.linspace(0.0, maximum_coupling, scan_points)
    amplitudes = [
        asymptotic_constant_mode(coupling, outer_radius=outer_radius)
        for coupling in couplings
    ]
    bracket: tuple[float, float] | None = None
    for left, right, left_value, right_value in zip(
        couplings[:-1], couplings[1:], amplitudes[:-1], amplitudes[1:]
    ):
        if left_value == 0.0:
            bracket = (float(left), float(left))
            break
        if left_value * right_value < 0.0:
            bracket = (float(left), float(right))
            break
    if bracket is None:
        raise ValueError("scan interval did not bracket a static zero mode")

    if bracket[0] == bracket[1]:
        threshold = bracket[0]
    else:
        threshold = float(
            brentq(
                lambda coupling: asymptotic_constant_mode(
                    coupling, outer_radius=outer_radius
                ),
                *bracket,
                xtol=2e-10,
                rtol=2e-10,
            )
        )
    residual = asymptotic_constant_mode(threshold, outer_radius=outer_radius)
    return ZeroModeThreshold(
        eta_over_mass_squared=threshold,
        dimensionless_outer_radius=outer_radius,
        asymptotic_constant_mode=residual,
        scan_points=scan_points,
    )


def format_report() -> str:
    """Return a reproducible report for the lowest bracketed zero mode."""
    result = first_zero_mode_threshold()
    return "\n".join(
        (
            "Linear scalar-Gauss-Bonnet zero-mode check on Schwarzschild",
            "eta/M^2=f''(0)/(Z0*M^2); regular horizon; s-wave",
            "A decaying zero mode requires the asymptotic constant coefficient A=0.",
            f"eta/M^2 = {result.eta_over_mass_squared:.9f}",
            f"outer radius x=r/(2M) = {result.dimensionless_outer_radius:.1f}",
            f"residual A = {result.asymptotic_constant_mode:.3e}",
            "This is a linear onset diagnostic, not a nonlinear hairy black-hole solution.",
            "No coupling is inferred from data and no Q-region map is supplied.",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(format_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
