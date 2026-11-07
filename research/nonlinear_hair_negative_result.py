"""Negative-result record: no regular nonlinear SGB hairy branch by shooting.

Static system in x=r/(2M), horizon amplitude φ_h, regularity ψ(1)=-3λφ_h.
Decay test A(λ,φ_h)=φ(X)+X·φ'(X) at X=80.

Findings (LSODA, rtol=1e-6):
- For φ_h ≲ 1 (λ=1.0): A>0, monotonically increasing in φ_h; metric stays
  near-regular (min μ ≈ 0 at horizon only). No decaying root.
- For φ_h ≳ 1.17: solutions cross an inner horizon (μ<0 deep in), A jumps
  chaotically between ±1e3..1e4 — these are not regular asymptotically-flat
  configurations; the integrator passes through μ=0.
- At fixed small φ_h, A(λ) stays positive across the linear threshold
  λ₁=0.7256092 (scan λ∈[0.5,2]) — no sign change.
Conclusion: within this ansatz and coupling sign, amplitude shooting detects
no regular nonlinear hairy black hole. The linear zero mode is an onset
diagnostic only — consistent with docs/research/GAUSS_BONNET_SCALARIZATION.md §4.
"""
import numpy as np
from scipy.integrate import solve_ivp

X_MAX = 80.0
LAM1 = 0.7256091671


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


def shoot(lam, ph, xmax=X_MAX):
    return solve_ivp(nl, (1 + 1e-6, xmax), [1.0, 0.0, ph, -3 * lam * ph], args=(lam,),
                     method="LSODA", rtol=1e-6, atol=1e-8)


def A(lam, ph):
    sol = shoot(lam, ph)
    if not sol.success:
        return np.nan
    return sol.y[2, -1] + X_MAX * sol.y[3, -1]


def min_mu(lam, ph):
    sol = shoot(lam, ph)
    if not sol.success:
        return np.nan
    return float((1 - sol.y[0] / sol.t).min())


if __name__ == '__main__':
    lam = 1.0
    print('φ_h scan at λ=1.0 (regular branch requires A=0 with min μ ≥ 0):', flush=True)
    for p in np.geomspace(0.1, 1.9, 14):
        a = A(lam, p)
        mu = min_mu(lam, p)
        tag = 'ROOT' if abs(a) < 1e-6 else ('irregular' if mu < -1e-3 else 'regular, A>0')
        print(f'  φ_h={p:.3f}  A={a:+.4e}  min μ={mu:+.4f}  {tag}', flush=True)
    print('λ scan at fixed small φ_h (looking for sign change across λ₁):', flush=True)
    for ph in [1e-3, 1e-2]:
        vals = [(lam, A(lam, ph)) for lam in np.linspace(0.5, 2.0, 16)]
        signs = [np.sign(v) for _, v in vals if np.isfinite(v)]
        print(f'  φ_h={ph:.0e}: A>0 for all λ scanned: {all(s > 0 for s in signs)}', flush=True)
    print('NEGATIVE-RESULT SCAN DONE', flush=True)
