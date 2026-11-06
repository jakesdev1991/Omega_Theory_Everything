"""Fast nonlinear SGB amplitude scan at a single λ (test-friendly).

Coarse φ_h scan only — no brentq refinement. Used by the unit test so the
test stays under a few seconds.
"""
import numpy as np
from scipy.integrate import solve_ivp

X_MAX = 100.0


def nl(x, y, lam):
    m, nu, phi, psi = y
    mu = 1.0 - m / x
    mu = mu if mu > 1e-9 else 1e-9
    dm = 2 * x**2 * (0.5 * mu * psi**2 + lam * phi**2 / x**6)
    denom = x * (x - m)
    denom = denom if abs(denom) > 1e-9 else (1e-9 if denom >= 0 else -1e-9)
    dnu = (m + 2 * x**3 * (0.5 * mu * psi**2 - lam * phi**2 / x**6)) / denom
    dpsi = (-psi * (2 * x - m) / denom - 3 * lam * phi / x**4) / mu
    return [dm, dnu, psi, dpsi]


def A(lam, ph, xmax=X_MAX):
    x0 = 1.0 + 1e-6
    sol = solve_ivp(nl, (x0, xmax), [1.0, 0.0, ph, -3 * lam * ph], args=(lam,),
                    method="LSODA", rtol=1e-7, atol=1e-9)
    if not sol.success:
        return np.nan
    phi, psi = sol.y[2, -1], sol.y[3, -1]
    return phi + xmax * psi


def coarse_scan(lam, phs=None):
    if phs is None:
        phs = np.geomspace(1e-3, 2.0, 10)
    return [(p, A(lam, p)) for p in phs]


if __name__ == '__main__':
    for lam in [0.75, 1.0, 2.0]:
        print(f"λ={lam}")
        for p, a in coarse_scan(lam):
            print(f"  φ_h={p:.4f}  A={a:+.4e}")
