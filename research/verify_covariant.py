# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Machine checks for the covariant-action note (COVARIANT_COMPLETION_AND_TEST_GATE.md).

Phase 3 items 1 and 2 of docs/research/CLAIM_TIGHTENING_ROADMAP.md:

* scalar field equation from the minisuperspace action, and its covariant form
  Z box(Phi) + (1/2) Z' (grad Phi)^2 - V' = 0 on flat FLRW;
* energy-momentum components of the scalar and the two FLRW equations, derived
  from the Einstein tensor computed by SymPy;
* linearized mass for the minimal branch and for the curvature-coupled branch;
* the integration-by-parts identity behind the static no-hair argument, on a
  Schwarzschild exterior with a stated boundary condition.

Passing checks establish only the stated algebra for the stated model. They do not
establish that the model describes nature, nor that a nonlinear hairy solution exists.

Run: python research/verify_covariant.py
Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp
from scipy.integrate import quad


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


# Explicit polynomial model used throughout: Z(Phi) = z0 + z1 Phi + z2 Phi^2,
# V(Phi) = v2 Phi^2 + v3 Phi^3 (so V(0) = V'(0) = 0), f(Phi) = f2 Phi^2 / 2.
_P = sp.Symbol("P")
_z0, _z1, _z2, _v2, _v3, _f2 = sp.symbols("z0 z1 z2 v2 v3 f2", real=True)
Z_OF_P = _z0 + _z1 * _P + _z2 * _P**2
V_OF_P = _v2 * _P**2 + _v3 * _P**3
F_OF_P = _f2 * _P**2 / 2


def _at_phi(expr: sp.Expr, phi: sp.Expr) -> sp.Expr:
    return expr.subs(_P, phi)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _einstein_tensor(coords: tuple[sp.Symbol, ...], g: sp.Matrix) -> sp.Matrix:
    n = len(coords)
    ginv = g.inv()
    gamma = [
        [
            [
                sp.simplify(
                    sum(
                        ginv[i, lam]
                        * (
                            sp.diff(g[lam, k], coords[j])
                            + sp.diff(g[lam, j], coords[k])
                            - sp.diff(g[j, k], coords[lam])
                        )
                        for lam in range(n)
                    )
                    / 2
                )
                for k in range(n)
            ]
            for j in range(n)
        ]
        for i in range(n)
    ]

    def ricci(mu: int, nu: int) -> sp.Expr:
        expr = sp.Integer(0)
        for lam in range(n):
            expr += sp.diff(gamma[lam][mu][nu], coords[lam])
            expr -= sp.diff(gamma[lam][mu][lam], coords[nu])
            for s in range(n):
                expr += gamma[lam][lam][s] * gamma[s][mu][nu]
                expr -= gamma[lam][nu][s] * gamma[s][mu][lam]
        return sp.simplify(expr)

    ric = sp.Matrix(n, n, lambda m, v: ricci(m, v))
    scalar = sp.simplify(
        sum(ginv[m, v] * ric[m, v] for m in range(n) for v in range(n))
    )
    return sp.simplify(ric - scalar * g / 2)


def _flrw() -> tuple[tuple[sp.Symbol, ...], sp.Function, sp.Matrix]:
    t, x, y, z = sp.symbols("t x y z", real=True)
    a = sp.Function("a")(t)
    return (t, x, y, z), a, sp.diag(-1, a**2, a**2, a**2)


# ---------------------------------------------------------------------------
# Check 1: minisuperspace scalar equation equals the covariant form
# ---------------------------------------------------------------------------


def check_scalar_eom_flrw() -> CheckResult:
    coords, a, g = _flrw()
    t = coords[0]
    phi = sp.Function("Phi")(t)
    # Minisuperspace Lagrangian density sqrt(-g) [ -(1/2) Z g^{tt} Phidot^2 - V ], N = 1.
    lag = a**3 * (
        sp.Rational(1, 2) * _at_phi(Z_OF_P, phi) * sp.diff(phi, t) ** 2
        - _at_phi(V_OF_P, phi)
    )
    el = sp.euler_equations(lag, phi, t)[0].lhs
    hub = sp.diff(a, t) / a
    phi_dot = sp.diff(phi, t)
    phi_ddot = sp.diff(phi, t, 2)
    zp = _at_phi(sp.diff(Z_OF_P, _P), phi)
    vp = _at_phi(sp.diff(V_OF_P, _P), phi)
    flrw_form = (
        _at_phi(Z_OF_P, phi) * (phi_ddot + 3 * hub * phi_dot)
        + sp.Rational(1, 2) * zp * phi_dot**2
        + vp
    )
    # sympy's euler_equations returns dL/dPhi - d/dt(dL/dPhidot) = 0, i.e. -a^3 * FLRW form.
    if sp.simplify(el + a**3 * flrw_form) != 0:
        return CheckResult(
            "COV: scalar EOM from minisuperspace", False, "EL != -a^3 * FLRW form"
        )

    # Covariant form: Z box(Phi) + (1/2) Z' (grad Phi)^2 - V' = 0 with sqrt(-g) = a^3.
    sqrt_g = a**3
    ginv_tt = -1
    box = sp.diff(sqrt_g * ginv_tt * phi_dot, t) / sqrt_g
    grad2 = ginv_tt * phi_dot**2
    covariant = _at_phi(Z_OF_P, phi) * box + sp.Rational(1, 2) * zp * grad2 - vp
    if sp.simplify(covariant + flrw_form) != 0:
        return CheckResult(
            "COV: covariant scalar eq", False, "covariant form != -(FLRW form)"
        )
    return CheckResult(
        "COV: scalar EOM (minisuperspace and covariant)",
        True,
        "EL = a^3 [Z(Phi''+3H Phi')+Z' Phi'^2/2+V'] and covariant form = -(same)",
    )


# ---------------------------------------------------------------------------
# Check 2: FLRW equations from the Einstein tensor and the scalar stress tensor
# ---------------------------------------------------------------------------


def check_flrw_equations() -> CheckResult:
    coords, a, g = _flrw()
    t = coords[0]
    ein = _einstein_tensor(coords, g)
    hub = sp.diff(a, t) / a
    adot = sp.diff(a, t)
    addot = sp.diff(a, t, 2)
    if sp.simplify(ein[0, 0] - 3 * hub**2) != 0:
        return CheckResult("COV: G_tt = 3H^2", False, f"G_tt = {ein[0, 0]}")
    if sp.simplify(ein[1, 1] + (2 * a * addot + adot**2)) != 0:
        return CheckResult("COV: G_xx", False, f"G_xx = {ein[1, 1]}")

    # Scalar stress tensor T_{mu nu} = Z d_mu Phi d_nu Phi - g_{mu nu}[ Z (grad Phi)^2 / 2 + V ].
    phi = sp.Function("Phi")(t)
    dphi = [sp.diff(phi, t), 0, 0, 0]
    ginv = g.inv()
    grad2 = sum(ginv[m, n] * dphi[m] * dphi[n] for m in range(4) for n in range(4))
    zz = _at_phi(Z_OF_P, phi)
    vv = _at_phi(V_OF_P, phi)
    tens = sp.Matrix(
        4, 4, lambda m, n: zz * dphi[m] * dphi[n] - g[m, n] * (zz * grad2 / 2 + vv)
    )
    rho_phi = sp.simplify(tens[0, 0])
    p_phi = sp.simplify(tens[1, 1] / a**2)
    if sp.simplify(rho_phi - (zz * sp.diff(phi, t) ** 2 / 2 + vv)) != 0:
        return CheckResult("COV: rho_Phi", False, f"rho = {rho_phi}")
    if sp.simplify(p_phi - (zz * sp.diff(phi, t) ** 2 / 2 - vv)) != 0:
        return CheckResult("COV: p_Phi", False, f"p = {p_phi}")
    if sp.simplify(rho_phi + p_phi - zz * sp.diff(phi, t) ** 2) != 0:
        return CheckResult("COV: rho+p", False, "rho+p != Z Phidot^2")

    # Solve the two metric equations for the matter sources, G = K T with K = 8 pi G:
    #   Friedmann:  3 H^2 = K (rho_m + rho_Phi)          -> rho_m
    #   xx-equation: G_xx / a^2 = K (p_m + p_Phi)          -> p_m
    # Then -2 Hdot must equal K (rho_m + p_m + Z Phidot^2), identically in the scalar history.
    kappa = sp.Symbol("K", positive=True)  # kappa = 8 pi G
    rho_m = 3 * hub**2 / kappa - rho_phi
    p_m = ein[1, 1] / (a**2 * kappa) - p_phi
    accel_rhs = kappa * (rho_m + p_m + zz * sp.diff(phi, t) ** 2)
    accel_lhs = -2 * sp.diff(hub, t)
    if sp.simplify(accel_lhs - accel_rhs) != 0:
        return CheckResult(
            "COV: acceleration equation",
            False,
            f"residual = {sp.simplify(accel_lhs - accel_rhs)}",
        )
    return CheckResult(
        "COV: FLRW Friedmann and acceleration",
        True,
        "G_tt=3H^2, G_xx=-(2a a''+a'^2); rho_Phi=Z Phi'^2/2+V, p_Phi=Z Phi'^2/2-V; "
        "-2Hdot = K(rho+p+Z Phi'^2)",
    )


# ---------------------------------------------------------------------------
# Check 3: linearized mass, minimal and curvature-coupled branches
# ---------------------------------------------------------------------------


def check_linear_mass() -> CheckResult:
    t, x = sp.symbols("t x", real=True)
    eps = sp.Symbol("epsilon")
    psi = sp.Function("psi")(t, x)
    big_g = sp.Symbol(
        "Gb", real=True
    )  # Gauss-Bonnet value on the background (a constant here)
    phi = eps * psi
    # Flat space, signature (-+++): box = -d_t^2 + d_x^2 (one spatial direction suffices).
    box = -sp.diff(phi, t, 2) + sp.diff(phi, x, 2)
    grad2 = -(sp.diff(phi, t) ** 2) + sp.diff(phi, x) ** 2
    zp = _at_phi(sp.diff(Z_OF_P, _P), phi)
    vp = _at_phi(sp.diff(V_OF_P, _P), phi)
    fp = _at_phi(sp.diff(F_OF_P, _P), phi)
    # Minimal branch: Z box + (1/2) Z' grad^2 - V' = 0.
    minimal = sp.expand(_at_phi(Z_OF_P, phi) * box + zp * grad2 / 2 - vp)
    # Curvature-coupled branch: add f'(Phi) * G_background (Gauss-Bonnet term in the scalar equation).
    coupled = sp.expand(minimal + fp * big_g)
    if (
        sp.simplify(minimal.coeff(eps, 0)) != 0
        or sp.simplify(coupled.coeff(eps, 0)) != 0
    ):
        return CheckResult(
            "COV: flat background solves scalar eq", False, "eps^0 term nonzero"
        )
    lin_min = minimal.coeff(eps, 1)
    lin_cpl = coupled.coeff(eps, 1)
    box_psi = -sp.diff(psi, t, 2) + sp.diff(psi, x, 2)
    # Expected: z0 box(psi) - V''(0) psi  and  z0 box(psi) - (V''(0) - f''(0) G) psi.
    v_dd = 2 * _v2
    f_dd = _f2
    ok_min = sp.simplify(lin_min - (_z0 * box_psi - v_dd * psi)) == 0
    ok_cpl = sp.simplify(lin_cpl - (_z0 * box_psi - (v_dd - f_dd * big_g) * psi)) == 0
    if not ok_min:
        return CheckResult(
            "COV: linear mass (minimal)", False, f"linear term = {lin_min}"
        )
    if not ok_cpl:
        return CheckResult(
            "COV: linear mass (coupled)", False, f"linear term = {lin_cpl}"
        )
    return CheckResult(
        "COV: linear mass m_eff^2",
        True,
        "minimal m^2 = V''(0)/Z(0) = 2 v2/z0; coupled m^2 = (V''(0) - f''(0) G)/Z(0)",
    )


# ---------------------------------------------------------------------------
# Check 4: integration-by-parts identity behind the static no-hair argument
# ---------------------------------------------------------------------------


def check_no_hair_identity() -> CheckResult:
    r = sp.Symbol("r", positive=True)
    rs = sp.Integer(1)  # horizon radius in units of r_s; the identity is scale-free
    f = 1 - rs / r
    # Weighted operator: for Schwarzschild, N sqrt(h) h^{rr} = r^2 f (per unit solid angle).
    phi = sp.Rational(3, 10) * sp.exp(-(r - rs)) * (1 + (r - rs) ** 2)
    v_prime = phi + phi**3  # V = phi^2/2 + phi^4/4
    residual = sp.diff(r**2 * f * sp.diff(phi, r), r) - r**2 * v_prime
    # Exact integration-by-parts identity for any smooth phi:
    #   d/dr [ r^2 f phi phi' ] = r^2 f phi'^2 + phi * d/dr(r^2 f phi') .
    pointwise = sp.simplify(
        sp.diff(r**2 * f * phi * sp.diff(phi, r), r)
        - (
            r**2 * f * sp.diff(phi, r) ** 2
            + phi * sp.diff(r**2 * f * sp.diff(phi, r), r)
        )
    )
    if pointwise != 0:
        return CheckResult(
            "NOHAIR: pointwise identity", False, f"residual = {pointwise}"
        )

    flux = sp.lambdify(r, r**2 * f * phi * sp.diff(phi, r), "numpy")
    integrand = sp.lambdify(
        r, r**2 * f * sp.diff(phi, r) ** 2 + r**2 * phi * v_prime, "numpy"
    )
    res_fn = sp.lambdify(r, phi * residual, "numpy")

    # The identity holds for any upper limit R. Use R = 3 r_s, where the asymptotic boundary
    # term is O(1), so that dropping it would break the identity; the test would otherwise be vacuous.
    big = 3.0
    lhs_int, _ = quad(lambda s: float(integrand(s)), float(rs), big, limit=400)
    boundary = float(flux(big)) - float(flux(float(rs)))
    rhs_int, _ = quad(lambda s: float(res_fn(s)), float(rs), big, limit=400)
    # Identity: int (r^2 f phi'^2 + r^2 phi V') dr - [flux] = - int phi * Residual dr.
    gap = (lhs_int - boundary) + rhs_int
    horizon_term = float(flux(float(rs)))
    ok = abs(gap) < 1e-7 and abs(horizon_term) < 1e-12 and abs(boundary) > 1e-3
    return CheckResult(
        "NOHAIR: integral identity with boundary terms",
        ok,
        f"R = 3 r_s: gap = {gap:.2e}; boundary term = {boundary:.3f} (nonzero); "
        f"horizon term = {horizon_term:.2e} (zero, since N = 0 there)",
    )


CHECKS: list[Callable[[], CheckResult]] = [
    check_scalar_eom_flrw,
    check_flrw_equations,
    check_linear_mass,
    check_no_hair_identity,
]


def run_all() -> list[CheckResult]:
    return [check() for check in CHECKS]


def main() -> int:
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name.ljust(width)}  ({r.detail})")
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} covariant checks passed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
