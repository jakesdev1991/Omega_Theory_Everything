"""Nonlinear SGB hairy branch — log-x shooting with Radau (stiff-stable).

Each integration is fast in log-x. We scan φ_h at each λ, find sign changes
of A(λ,φ_h)=φ(X_MAX)+X_MAX·φ'(X_MAX), refine with brentq.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

X_MAX = 100.0
LAM1 = 0.7256091671055829


def nl_log(t, y, lam):
    # t = log(x), x = e^t
    x = np.exp(t)
    m, nu, phi, psi = y
    mu = 1.0 - m / x
    mu = mu if mu > 1e-9 else 1e-9
    dxdt = x
    dm = 2 * x**2 * (0.5 * mu * psi**2 + lam * phi**2 / x**6) / dxdt
    denom = x * (x - m)
    denom = denom if abs(denom) > 1e-9 else (1e-9 if denom >= 0 else -1e-9)
    dnu = (m + 2 * x**3 * (0.5 * mu * psi**2 - lam * phi**2 / x**6)) / denom / dxdt
    dphi = psi / dxdt
    dpsi = (-psi * (2 * x - m) / denom - 3 * lam * phi / x**4) / mu / dxdt
    return [dm, dnu, dphi, dpsi]


def shoot(lam, ph, xmax=X_MAX):
    t0 = np.log(1 + 1e-6)
    t1 = np.log(xmax)
    sol = solve_ivp(nl_log, (t0, t1), [1.0, 0.0, ph, -3 * lam * ph], args=(lam,),
                    method="Radau", rtol=1e-7, atol=1e-9)
    return sol


def A(lam, ph):
    sol = shoot(lam, ph)
    if not sol.success:
        return np.nan
    x = X_MAX
    return sol.y[2, -1] + x * sol.y[3, -1]


def mass(lam, ph):
    sol = shoot(lam, ph)
    return sol.y[0, -1] if sol.success else np.nan


def branch(lams, n_ph=12):
    phs = np.geomspace(1e-3, 3.0, n_ph)
    out = []
    for lam in lams:
        vals = [A(lam, p) for p in phs]
        root = None
        for i in range(len(phs) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                root = brentq(lambda p: A(lam, p), phs[i], phs[i + 1], xtol=1e-9)
                break
        if root is not None:
            out.append((lam, root, mass(lam, root)))
            print(f"λ={lam:.5f}  φ_h={root:+.6f}  m(∞)={mass(lam, root):.6f}  NONTRIVIAL", flush=True)
        else:
            print(f"λ={lam:.5f}  no root", flush=True)
    return out


if __name__ == '__main__':
    branch([LAM1 + 0.005, 0.75, 0.8, 1.0, 1.5, 2.0, 3.0])
