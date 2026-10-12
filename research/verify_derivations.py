# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Machine checks for the hand derivations in the exploratory radial-metric,
technical, Markov-benchmark and Gauss-Bonnet notes.

Each check re-derives one displayed identity or numerical claim from its stated
inputs, using SymPy (exact), NumPy/SciPy (floating point), or an independent
numerical method. Passing checks establish only the stated algebra or numerics
for the stated model. They say nothing about whether the model describes nature.

Run: python research/verify_derivations.py
Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.linalg import eigh
from scipy.optimize import brentq


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Radial note, section 4: flat background, Phi = A/rho, h = Omega^2 delta
# ---------------------------------------------------------------------------


def check_flat_obstruction() -> CheckResult:
    rho, A = sp.symbols("rho A", positive=True)
    omega = sp.sqrt(1 - A**2 / rho**2)
    r_areal = sp.simplify(omega * rho)
    if sp.simplify(r_areal - sp.sqrt(rho**2 - A**2)) != 0:
        return CheckResult("flat: areal radius", False, f"R = {r_areal}")
    d_r = sp.diff(r_areal, rho)
    g_rr = sp.simplify(omega**2 / d_r**2)
    if sp.simplify(g_rr - (1 - A**2 / rho**2) ** 2) != 0:
        return CheckResult("flat: g_RR = (1-Phi^2)^2", False, f"g_RR = {g_rr}")
    m_ms = sp.simplify(r_areal / 2 * (1 - 1 / g_rr))
    expected_m = A**2 * (A**2 - 2 * rho**2) / (2 * (rho**2 - A**2) ** sp.Rational(3, 2))
    if sp.simplify(m_ms - expected_m) != 0:
        return CheckResult("flat: Misner-Sharp mass", False, f"m = {m_ms}")
    # Negative for every rho > A: A^2 - 2 rho^2 < 0 because rho^2 > A^2 > A^2/2.
    neg_everywhere = all(
        float(expected_m.subs({A: 1, rho: 1 + k / 10})) < 0 for k in range(1, 200)
    )
    # Energy density from the Hamiltonian constraint, psi^4 = Omega^2 (time-symmetric, conformally flat).
    psi = omega ** sp.Rational(1, 2)
    lap = sp.diff(rho**2 * sp.diff(psi, rho), rho) / rho**2
    eps = sp.simplify(-8 * psi ** (-5) * lap / (16 * sp.pi))
    claimed = A**2 * (A**2 + 2 * rho**2) / (8 * sp.pi * (rho**2 - A**2) ** 3)
    if sp.simplify(eps - claimed) != 0:
        return CheckResult("flat: energy density", False, f"eps = {eps}")
    # ADM mass: psi has no 1/rho term.
    x = sp.symbols("x", positive=True)
    series = sp.series(psi.subs(rho, 1 / x), x, 0, 2).removeO()
    no_monopole = sp.simplify(series.coeff(x, 1)) == 0
    ok = neg_everywhere and no_monopole
    return CheckResult(
        "flat: g_RR, R, Misner-Sharp<0, energy density, zero ADM mass",
        ok,
        f"neg_everywhere={neg_everywhere}, no_1/rho_term={no_monopole}",
    )


# ---------------------------------------------------------------------------
# Radial note, section 5: isotropic Schwarzschild written in Phi
# ---------------------------------------------------------------------------


def check_isotropic_schwarzschild() -> CheckResult:
    t, rho, th, ph, M = sp.symbols("t rho theta phi M", positive=True)
    coords = [t, rho, th, ph]
    phi_field = M / (2 * rho)
    psi = 1 + phi_field
    lapse = (1 - phi_field) / (1 + phi_field)
    metric = sp.diag(
        -(lapse**2), psi**4, psi**4 * rho**2, psi**4 * rho**2 * sp.sin(th) ** 2
    )
    inv = metric.inv()
    n = 4
    gam = [
        [
            [
                sp.simplify(
                    sum(
                        inv[a, d]
                        * (
                            sp.diff(metric[d, b], coords[c])
                            + sp.diff(metric[d, c], coords[b])
                            - sp.diff(metric[b, c], coords[d])
                        )
                        for d in range(n)
                    )
                    / 2
                )
                for c in range(n)
            ]
            for b in range(n)
        ]
        for a in range(n)
    ]
    ricci_zero = True
    for b in range(n):
        for c in range(n):
            expr = 0
            for a in range(n):
                expr += sp.diff(gam[a][b][c], coords[a]) - sp.diff(
                    gam[a][b][a], coords[c]
                )
                for d in range(n):
                    expr += gam[a][a][d] * gam[d][b][c] - gam[a][c][d] * gam[d][b][a]
            if sp.simplify(expr) != 0:
                ricci_zero = False
    # Static lapse equation D_i(psi^2 D^i N) = 0 reduces to div(psi^2 grad N) = 0 on flat space.
    lapse_eq = sp.simplify(sp.diff(rho**2 * psi**2 * sp.diff(lapse, rho), rho) / rho**2)
    # Areal gauge: R = rho psi^2, g_RR = psi^4 / (dR/drho)^2 = ((1+Phi)/(1-Phi))^2.
    r_areal = rho * psi**2
    g_rr = sp.simplify(psi**4 / sp.diff(r_areal, rho) ** 2)
    p = sp.symbols("Phi", positive=True)
    target = (((1 + p) / (1 - p)) ** 2).subs(p, phi_field)
    areal_ok = sp.simplify(g_rr - target) == 0
    # Weak field: -g_tt = N^2 = 1 - 4 Phi + O(Phi^2).
    weak_ok = (
        sp.simplify(sp.series(lapse**2, M, 0, 2).removeO().coeff(M, 1) + 4 / (2 * rho))
        == 0
    )
    ok = ricci_zero and lapse_eq == 0 and areal_ok and weak_ok
    return CheckResult(
        "Schwarzschild in Phi: Ricci-flat, lapse equation, areal g_RR, weak field",
        ok,
        f"ricci_zero={ricci_zero}, lapse_eq={lapse_eq}, areal={areal_ok}, weak={weak_ok}",
    )


def check_lichnerowicz_and_required_kappa() -> CheckResult:
    p = sp.symbols("Phi", positive=True)
    # Lichnerowicz form: 16 pi eps = -8 psi^-5 Lap psi  <=>  Lap psi = -2 pi psi^5 eps.
    eps = sp.symbols("epsilon")
    lap_psi = sp.symbols("lap_psi")
    psi_sym = 1 + p
    lichn = sp.simplify(-8 * psi_sym ** (-5) * lap_psi / (16 * sp.pi) - eps)
    solved = sp.solve(sp.Eq(lichn, 0), lap_psi)[0]
    lichn_ok = sp.simplify(solved + 2 * sp.pi * psi_sym**5 * eps) == 0
    # Required kappa: (l_P kappa)^2 = h_rr = (1+Phi)^4 with l_P = sqrt(1-Phi^2).
    l_p = sp.sqrt(1 - p**2)
    kappa_req = (1 + p) ** 2 / l_p
    kappa_ok = sp.simplify((l_p * kappa_req) ** 2 - (1 + p) ** 4) == 0
    # Dilation criterion: d ln l_P / dPhi = -Phi/(1-Phi^2); margin 2/(1+Phi) for kappa_req.
    dlnl = sp.simplify(sp.diff(sp.log(l_p), p))
    dilation_ok = sp.simplify(dlnl + p / (1 - p**2)) == 0
    dlnk = sp.simplify(sp.diff(sp.log(kappa_req), p))
    margin_ok = sp.simplify(dlnk - (2 / (1 + p) + p / (1 - p**2))) == 0
    ok = lichn_ok and kappa_ok and dilation_ok and margin_ok
    return CheckResult(
        "Lichnerowicz source, required kappa, dilation criterion and margin",
        ok,
        f"lichnerowicz={lichn_ok}, kappa={kappa_ok}, dilation={dilation_ok}, margin={margin_ok}",
    )


def check_conformal_radial_identity() -> CheckResult:
    rho = sp.symbols("rho", positive=True)
    omega = sp.Function("Omega")(rho)
    r_areal = omega * rho
    g_rr = sp.simplify(omega**2 / sp.diff(r_areal, rho) ** 2)
    s = sp.symbols("s")
    claimed = 1 / (1 + s) ** 2
    s_expr = sp.diff(sp.log(omega), rho) * rho
    ok = sp.simplify(g_rr - claimed.subs(s, s_expr)) == 0
    return CheckResult(
        "conformal identity g_RR = (1 + dlnOmega/dlnrho)^-2", ok, f"g_RR={g_rr}"
    )


# ---------------------------------------------------------------------------
# Radial note, section 8: golden bottleneck
# ---------------------------------------------------------------------------


def check_golden_bottleneck() -> CheckResult:
    p = sp.symbols("Phi", real=True)
    phi_star = (sp.sqrt(5) - 1) / 2
    f = sp.exp(2 * p) * (1 - p**2)
    crit_zero = sp.simplify(sp.diff(f, p).subs(p, phi_star)) == 0
    value_ok = sp.simplify(f.subs(p, phi_star) - phi_star * sp.exp(2 * phi_star)) == 0
    # Global maximum on (-1,1): check on a dense grid, excluding endpoints.
    grid = np.linspace(-0.999999, 0.999999, 200001)
    vals = np.exp(2 * grid) * (1 - grid**2)
    bound = float(phi_star * np.exp(2 * float(phi_star)))
    global_ok = (
        bool(np.max(vals) <= bound * (1 + 1e-12))
        and abs(float(grid[np.argmax(vals)]) - float(phi_star)) < 1e-4
    )
    # Robustness: maximiser of exp(2 a Phi)(1-Phi^2)^p solves a(1-Phi^2)=p Phi.
    a, q = sp.symbols("a p", positive=True)
    log_f = 2 * a * p + q * sp.log(1 - p**2)
    crit = sp.simplify(sp.diff(log_f, p))
    root = (sp.sqrt(q**2 + 4 * a**2) - q) / (2 * a)
    root_ok = sp.simplify(crit.subs(p, root)) == 0
    # Golden iff a = p: since 1 - phi*^2 = phi*, the criterion at phi* is a*phi* - p*phi*.
    residual_at_star = sp.simplify(a * (1 - phi_star**2) - q * phi_star)
    golden_iff = sp.simplify(residual_at_star - (a - q) * phi_star) == 0
    # Away from a = p the root moves: a = 2, p = 1 gives 0.7808, not 0.6180.
    not_golden = abs(float(root.subs({a: 2, q: 1})) - float(phi_star)) > 0.1
    ok = crit_zero and value_ok and global_ok and root_ok and golden_iff and not_golden
    return CheckResult(
        "golden bottleneck: critical point, global maximum, robustness root, golden iff a=p",
        ok,
        f"crit={crit_zero}, value={value_ok}, global={global_ok}, root={root_ok}, iff={golden_iff}",
    )


def check_horizon_locations() -> CheckResult:
    # Schwarzschild with r_s = 2M: Phi = M/(2 rho) = r_s/(4 rho); horizon at Phi = 1, i.e. rho = r_s/4.
    # Areal radius R = rho (1 + Phi)^2 = r_s (1 + Phi)^2 / (4 Phi). Areal dictionary: Phi = r_s / R.
    p = sp.symbols("Phi", positive=True)
    r_s = sp.symbols("r_s", positive=True)
    phi_star = (sp.sqrt(5) - 1) / 2
    r_iso = r_s * (1 + p) ** 2 / (4 * p)
    iso_ok = sp.simplify(r_iso.subs(p, phi_star) / r_s - (2 + sp.sqrt(5)) / 4) == 0
    areal_ok = sp.simplify(r_s / phi_star - (1 + sp.sqrt(5)) / 2 * r_s) == 0
    # Horizon: Phi = 1 gives R = r_s (1+1)^2/4 = r_s, i.e. the areal radius 2M = r_s.
    horizon_ok = sp.simplify(r_iso.subs(p, 1) - r_s) == 0
    ok = iso_ok and areal_ok and horizon_ok
    return CheckResult(
        "horizon locations: isotropic R*=(2+sqrt5)/4 r_s, areal R*=phi r_s, Phi=1 is R=r_s",
        ok,
        f"iso={iso_ok}, areal={areal_ok}, horizon={horizon_ok}",
    )


def check_disformal_bound_depends_on_phidot() -> CheckResult:
    """Radial note section 8 caveat: the causality bound constrains Phi_dot only.

    Model, exactly as Sim3 section 2.1 states it: the disformed metric
    ``g~ = C(Phi) g + D(Phi) dPhi (x) dPhi`` with ``C = exp(-2 Phi)`` and
    ``D = beta lP(Phi)^2 = beta (1 - Phi^2)``, on flat spacetime in spherical
    coordinates, signature ``(-,+,+,+)``, with ``Phi = Phi(t, r)``. Domain:
    ``|Phi| < 1``, ``beta > 0``.

    Claims checked:

    1. the disformed ``(t, r)`` block has determinant
       ``C (D PhiDot^2 - C - D PhiPrime^2)``, so it stays Lorentzian iff
       ``D PhiDot^2 < C + D PhiPrime^2`` — a condition on the time derivative,
       never an upper bound on the gradient;
    2. at ``PhiPrime = 0`` that is exactly ``|PhiDot| < sqrt(C/D)``, the closed
       form ``exp(-Phi)/(sqrt(beta) sqrt(1 - Phi^2))`` printed by Sim3. With a
       nonzero gradient the exact condition is ``|PhiDot| < sqrt(C/D + PhiPrime^2)``,
       so the published bound is the gradient-free case: sufficient for every
       profile, and *conservative* rather than tight when ``PhiPrime != 0``;
    3. ``g~_rr = C + D PhiPrime^2 > 0`` and ``g~_tt = -C < 0`` for every
       gradient, so a static profile cannot threaten the signature — the note's
       caveat, now machine-checked rather than hand-checked;
    4. the bound's minimum on ``(-1, 1)`` is at ``Phi*`` with value
       ``exp(-Phi*)/sqrt(beta Phi*)``, matching the note's ``0.6856/sqrt(beta)``.
    """
    phi, beta, phi_dot, phi_prime = sp.symbols("Phi beta PhiDot PhiPrime", real=True)
    c_sym = sp.exp(-2 * phi)
    d_sym = beta * (1 - phi**2)
    block = sp.Matrix(
        [
            [-c_sym + d_sym * phi_dot**2, d_sym * phi_dot * phi_prime],
            [d_sym * phi_dot * phi_prime, c_sym + d_sym * phi_prime**2],
        ]
    )
    det_ok = (
        sp.simplify(
            block.det() - c_sym * (d_sym * phi_dot**2 - c_sym - d_sym * phi_prime**2)
        )
        == 0
    )

    # (1) Lorentzian (one negative eigenvalue) iff D PhiDot^2 < C + D PhiPrime^2.
    rng = np.random.default_rng(20261012)
    samples = 200_000
    phi_s = rng.uniform(-0.99, 0.99, samples)
    beta_s = rng.choice([0.05, 0.5, 1.0], samples)
    dot_s = rng.uniform(-3.0, 3.0, samples)
    prime_s = rng.uniform(-3.0, 3.0, samples)
    c_s = np.exp(-2 * phi_s)
    d_s = beta_s * (1 - phi_s**2)
    gtt = -c_s + d_s * dot_s**2
    grr = c_s + d_s * prime_s**2
    gtr = d_s * dot_s * prime_s
    lorentzian = (gtt * grr - gtr**2) < 0
    equivalence = bool(np.all(lorentzian == (d_s * dot_s**2 < c_s + d_s * prime_s**2)))

    # (2) the printed closed form is sqrt(C/D): square it and compare exactly.
    printed = sp.exp(-phi) / (sp.sqrt(beta) * sp.sqrt(1 - phi**2))
    closed_form_ok = sp.simplify(printed**2 - c_sym / d_sym) == 0

    # (2b) sufficiency and non-necessity of the published bound sqrt(C/D).
    published = np.sqrt(c_s / d_s)
    exact = np.sqrt(c_s / d_s + prime_s**2)
    beyond_published = np.abs(dot_s) >= published
    still_lorentzian = bool(np.any(beyond_published & lorentzian))
    beyond_exact = dot_s**2 >= c_s / d_s + prime_s**2
    exact_blocks_everything = bool(np.all(lorentzian[beyond_exact] == False))  # noqa: E712
    example = int(np.argmax(beyond_published & lorentzian)) if still_lorentzian else 0

    # (3) static profiles at any gradient magnitude, on the model's domain.
    gradients = np.concatenate([np.linspace(-1e6, 1e6, 10_001), [0.0]])
    static_ok = True
    for b in (0.05, 0.5, 1.0):
        grid = np.linspace(-0.9999, 0.9999, 2_001)
        cc = np.exp(-2 * grid)
        dd = b * (1 - grid**2)
        timelike = bool(np.all(-cc < 0))
        spacelike = bool(np.all(cc + dd * gradients[:, None] ** 2 > 0))
        static_ok = static_ok and timelike and spacelike

    # (4) minimum of the bound on (-1, 1), cross-checked against the note.
    phi_star = (sp.sqrt(5) - 1) / 2
    bound = sp.exp(-phi) / (sp.sqrt(beta) * sp.sqrt(1 - phi**2))
    min_exact = float(sp.N(bound.subs({phi: phi_star, beta: 1}), 12))
    grid = np.linspace(-0.999999, 0.999999, 200_001)
    numeric_min = float(np.min(np.exp(-grid) / np.sqrt(1 - grid**2)))
    argmin_phi = float(grid[int(np.argmin(np.exp(-grid) / np.sqrt(1 - grid**2)))])
    min_ok = (
        abs(min_exact - numeric_min) < 1e-6
        and abs(argmin_phi - float(phi_star)) < 1e-4
        and abs(min_exact - 0.6856) < 5e-4
    )

    ok = (
        det_ok
        and closed_form_ok
        and equivalence
        and still_lorentzian
        and exact_blocks_everything
        and static_ok
        and min_ok
    )
    return CheckResult(
        "disformal causality bound: constrains Phi_dot only, static gradients are safe",
        ok,
        f"det={det_ok}, closed_form={closed_form_ok}, equivalence={equivalence} "
        f"({samples} samples), published bound "
        f"sufficient but not necessary={still_lorentzian} (Phi={phi_s[example]:.3f}, "
        f"beta={beta_s[example]:.2f}, |Phidot|={abs(dot_s[example]):.3f} vs published "
        f"{published[example]:.3f} and exact {exact[example]:.3f}), exact bound "
        f"necessary={exact_blocks_everything}, static={static_ok}, "
        f"min={min_exact:.6f}/sqrt(beta) at Phi*={min_ok}",
    )


# ---------------------------------------------------------------------------
# Markov benchmark (docs/research/QREGION_MARKOV_MODEL.md)
# ---------------------------------------------------------------------------


def _markov_rho(n_sites: int, lam: float) -> np.ndarray:
    dim = 2**n_sites
    diag = np.zeros(dim)
    for idx, bits in enumerate(itertools.product([1, -1], repeat=n_sites)):
        s = bits
        prob = 0.5
        for k in range(n_sites - 1):
            prob *= (1 + lam * s[k] * s[k + 1]) / 2
        diag[idx] = prob
    return np.diag(diag)


def _reduced(rho: np.ndarray, n_sites: int, keep: list[int]) -> np.ndarray:
    tensor = rho.reshape([2] * n_sites * 2)
    current = tensor
    remaining = n_sites
    for site in sorted(set(range(n_sites)) - set(keep), reverse=True):
        current = np.trace(current, axis1=site, axis2=site + remaining)
        remaining -= 1
    d = 2 ** len(keep)
    return current.reshape(d, d)


def _entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-300]
    return float(-(w * np.log(w)).sum())


def _pair_mi(rho: np.ndarray, n_sites: int, i: int, j: int) -> float:
    return (
        _entropy(_reduced(rho, n_sites, [i]))
        + _entropy(_reduced(rho, n_sites, [j]))
        - _entropy(_reduced(rho, n_sites, [i, j]))
    )


def _i_closed(lam_n: float) -> float:
    def h2(p: float) -> float:
        return -p * np.log(p) - (1 - p) * np.log(1 - p)

    return float(np.log(2) - h2((1 + lam_n) / 2))


def check_markov_mutual_information() -> CheckResult:
    lam = 0.5
    rho = _markov_rho(4, lam)
    errs = [abs(_pair_mi(rho, 4, 0, n) - _i_closed(lam**n)) for n in (1, 2, 3)]
    max_err = max(errs)
    i1 = _i_closed(lam)
    i2 = _i_closed(lam**2)
    not_composing = abs(i2 / np.log(2) - (i1 / np.log(2)) ** 2) > 1e-3
    # derivative dI1/dlambda = atanh(lambda)
    h = 1e-6
    deriv = (_i_closed(0.5 + h) - _i_closed(0.5 - h)) / (2 * h)
    deriv_ok = abs(deriv - np.arctanh(0.5)) < 1e-6
    ok = max_err < 1e-12 and not_composing and deriv_ok
    return CheckResult(
        "Markov: explicit density-matrix MI equals closed form; MI does not compose; dI/dlambda=atanh",
        ok,
        f"max_err={max_err:.2e}, I2/ln2={i2 / np.log(2):.6f}, (I1/ln2)^2={(i1 / np.log(2)) ** 2:.6f}, deriv_ok={deriv_ok}",
    )


def check_markov_kappa_table() -> CheckResult:
    iv = _i_closed(0.5)
    d_gap = np.log(2) - iv

    def lam_of(phi: float) -> float:
        return float(
            brentq(
                lambda ll: _i_closed(ll) - (iv + phi * d_gap),
                1e-12,
                1 - 1e-12,
                xtol=1e-14,
            )
        )

    # Table rows (Phi label, expected lambda, expected kappa/kappa0, expected toy h_rr ratio).
    phi_star = (np.sqrt(5) - 1) / 2
    rows = [
        (0.1, 0.5917005, 0.7570609, 0.5674098),
        (0.3, 0.7323062, 0.4494810, 0.1838502),
        (0.5, 0.8380308, 0.2549249, 0.0487400),
        (phi_star, 0.8887238, 0.1701929, 0.0179017),
        (0.9, 0.9798990, 0.0292951, 0.0001631),
        (0.99, 0.9986444, 0.0019571, 0.0000001),
    ]
    k0 = -np.log(0.5)
    worst = 0.0
    for phi, lam_exp, kap_exp, toy_exp in rows:
        lam_v = lam_of(phi)
        kap = -np.log(lam_v) / k0
        toy = (1 - phi**2) * kap**2
        worst = max(worst, abs(lam_v - lam_exp), abs(kap - kap_exp), abs(toy - toy_exp))
    # Table row labelled 0.618 is evaluated at phi_star (0.6180340), not at the rounded label.
    rounded_label_lam = lam_of(0.618)
    label_defect = abs(rounded_label_lam - 0.8887238) > 1e-5
    ok = worst < 2e-6 and label_defect
    return CheckResult(
        "Markov table: lambda, kappa/kappa0, toy h_rr at each row (row 0.618 is Phi*)",
        ok,
        f"worst_abs_err={worst:.2e}, rounded_0.618_lambda={rounded_label_lam:.7f}",
    )


# ---------------------------------------------------------------------------
# Gauss-Bonnet note (docs/research/GAUSS_BONNET_SCALARIZATION.md)
# ---------------------------------------------------------------------------


def check_gauss_bonnet_reduction() -> CheckResult:
    r, big_m, eta, x = sp.symbols("r M eta x", positive=True)
    g = sp.Function("g")
    f = 1 - 2 * big_m / r
    phi_r = g(r / (2 * big_m))
    box = sp.diff(r**2 * f * sp.diff(phi_r, r), r) / r**2
    gb = 48 * big_m**2 / r**6
    lhs = (box + eta * gb * phi_r).subs(r, 2 * big_m * x).doit()
    claimed = (
        sp.diff(x * (x - 1) * sp.diff(g(x), x), x) + 3 * (eta / big_m**2) * g(x) / x**4
    )
    ratio = sp.simplify(sp.simplify(lhs) / claimed)
    ok = sp.simplify(ratio - 1 / (4 * big_m**2 * x**2)) == 0
    return CheckResult(
        "Gauss-Bonnet: linearised static equation = (1/(4M^2x^2)) * stated x-ODE",
        ok,
        f"ratio={ratio}",
    )


def check_kretschmann() -> CheckResult:
    r, big_m, th, t, ph = sp.symbols("r M theta t phi", positive=True)
    f = 1 - 2 * big_m / r
    coords = [t, r, th, ph]
    metric = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)
    inv = metric.inv()
    n = 4
    gam = [
        [
            [
                sp.simplify(
                    sum(
                        inv[a, d]
                        * (
                            sp.diff(metric[d, b], coords[c])
                            + sp.diff(metric[d, c], coords[b])
                            - sp.diff(metric[b, c], coords[d])
                        )
                        for d in range(n)
                    )
                    / 2
                )
                for c in range(n)
            ]
            for b in range(n)
        ]
        for a in range(n)
    ]

    def riem_up(a: int, b: int, c: int, d: int) -> sp.Expr:
        e = sp.diff(gam[a][b][d], coords[c]) - sp.diff(gam[a][b][c], coords[d])
        for k in range(n):
            e += gam[a][c][k] * gam[k][b][d] - gam[a][d][k] * gam[k][b][c]
        return sp.simplify(e)

    kretsch = 0
    for a, b, c, d in itertools.product(range(n), repeat=4):
        comp = sp.simplify(sum(metric[a, e] * riem_up(e, b, c, d) for e in range(n)))
        if comp != 0:
            kretsch += comp**2 * inv[a, a] * inv[b, b] * inv[c, c] * inv[d, d]
    ok = sp.simplify(kretsch - 48 * big_m**2 / r**6) == 0
    return CheckResult(
        "Kretschmann invariant on Schwarzschild equals 48 M^2 / r^6",
        ok,
        f"K={sp.simplify(kretsch)}",
    )


def _galerkin_lambda1(n_basis: int) -> float:
    y = sp.symbols("y")
    basis = [y**k for k in range(1, n_basis + 1)]
    stiff = np.zeros((n_basis, n_basis))
    mass = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            stiff[i, j] = float(
                sp.integrate(
                    (1 - y) * sp.diff(basis[i], y) * sp.diff(basis[j], y), (y, 0, 1)
                )
            )
            mass[i, j] = float(sp.integrate(y**2 * basis[i] * basis[j], (y, 0, 1)))
    mu1 = float(eigh(stiff, mass, eigvals_only=True)[0])
    return mu1 / 3.0


def check_gauss_bonnet_threshold() -> CheckResult:
    # y = 1/x maps the horizon to y = 1 and infinity to y = 0. The problem becomes
    #   -((1-y) phi')' = mu y^2 phi,   phi(0) = 0,   mu = 3 lambda,
    # with the horizon regular automatically. Rayleigh-Ritz gives an upper bound that
    # converges as the basis grows; this is independent of the shooting code.
    vals = {nb: _galerkin_lambda1(nb) for nb in (6, 8)}
    converged = abs(vals[8] - vals[6]) < 1e-7
    value = vals[8]
    ok = converged and abs(value - 0.7256128) < 2e-6
    return CheckResult(
        "Gauss-Bonnet threshold: Rayleigh-Ritz (independent of shooting) converges to 0.7256128",
        ok,
        f"basis6={vals[6]:.9f}, basis8={vals[8]:.9f}",
    )


CHECKS: list[Callable[[], CheckResult]] = [
    check_flat_obstruction,
    check_isotropic_schwarzschild,
    check_lichnerowicz_and_required_kappa,
    check_conformal_radial_identity,
    check_golden_bottleneck,
    check_horizon_locations,
    check_disformal_bound_depends_on_phidot,
    check_markov_mutual_information,
    check_markov_kappa_table,
    check_gauss_bonnet_reduction,
    check_kretschmann,
    check_gauss_bonnet_threshold,
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
    print(f"\n{len(results) - len(failed)}/{len(results)} derivation checks passed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
