# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0
"""
Simulation 7: Toward the Macroscopic Radial Metric g_rr(Phi)
============================================================

Companion to ``Omega_Theory_v4.0_Radial_Metric.md``. Every formula quoted in
that note is checked here: numerically (NumPy/SciPy) and, when SymPy is
installed, symbolically. ``self_check()`` runs on every invocation.

Convention (the one used by ``lean_proofs/DynamicCODScale.lean``, section 2.2
of the technical note and ``Sim3_Dynamic_Scale.py``): Phi = 0 is the vacuum,
Phi -> 1 is the bound-state / horizon end, and

    l_P(Phi) = l_P0 sqrt(1 - Phi^2)          (imported from Sim3, not copied).

Units: G = c = 1 and l_P0 = kappa_0 = 1.

What is checked
---------------
1. Two-factor law (Derived from Markov factorisation + additivity + locality):
       ds = l_P(Phi) dsigma,   dsigma = -d ln K   =>   g_rr = (l_P(Phi) kappa)^2.
   The finite section 2.1 formula -l_P ln K is exact only at constant Phi.
2. Obstruction (Derived): with uniform correlation decay and a harmonic profile
   Phi = A / rho, l_P alone gives g_RR = (1 - Phi^2)^2 <= 1, zero ADM mass and
   a negative Misner-Sharp mass.
3. Schwarzschild in Phi form (Identified, not derived): psi = 1 + Phi with
   Phi = M / (2 rho) gives the exact vacuum metric
       ds^2 = -((1-Phi)/(1+Phi))^2 dt^2 + (1+Phi)^4 (drho^2 + rho^2 dOmega^2),
   g_RR = ((1+Phi)/(1-Phi))^2, horizon at Phi = 1, and the correlation-decay
   rate it requires, kappa / kappa_0 = (1+Phi)^2 / sqrt(1-Phi^2).
4. Convention checks: g_rr ~ 1/Phi is not asymptotically flat in this
   convention; Sim3's potential V = m^2 (1-Phi)^2 / 2 has V'(0) = -m^2 != 0.
5. Golden-ratio bottleneck: an elementary global-minimum inequality and the
   generalisation Phi* = (sqrt(p^2 + 4 a^2) - p) / (2 a) for C = exp(-2 a Phi),
   D ~ (1 - Phi^2)^p (golden only when a = p).

Usage
-----
    python Sim7_Radial_Metric.py [--no-plots] [--no-symbolic] [--outdir .]
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad

from Sim3_Dynamic_Scale import GOLDEN_CONJUGATE, causality_bound, lP

PHI_STAR = GOLDEN_CONJUGATE  # (sqrt(5) - 1) / 2


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def lp_ratio(phi: np.ndarray | float) -> np.ndarray:
    """l_P(Phi) / l_P0 from the canonical Sim3 / Lean profile."""
    return np.asarray(lP(np.asarray(phi, dtype=float)), dtype=float)


def chain_distance(phi_links: np.ndarray, k_links: np.ndarray) -> float:
    """Additive proper length of a chain: sum_i l_P(Phi_i) * (-ln K_i)."""
    k = np.asarray(k_links, dtype=float)
    if np.any((k <= 0.0) | (k > 1.0)):
        raise ValueError("link correlations must lie in (0, 1]")
    return float(np.sum(lp_ratio(phi_links) * (-np.log(k))))


def g_rr_lp_only(phi: np.ndarray | float) -> np.ndarray:
    """Areal-gauge g_RR when l_P(Phi) is the only Phi-dependence (flat gamma)."""
    return np.asarray((1.0 - np.square(phi)) ** 2, dtype=float)


def g_rr_isotropic_dictionary(phi: np.ndarray | float) -> np.ndarray:
    """Areal-gauge g_RR of Schwarzschild written in Phi = psi - 1."""
    p = np.asarray(phi, dtype=float)
    return np.asarray(((1.0 + p) / (1.0 - p)) ** 2, dtype=float)


def kappa_required(phi: np.ndarray | float) -> np.ndarray:
    """Correlation-decay rate (per isotropic coordinate length, units of
    kappa_0) that makes l_P(Phi) * kappa reproduce Schwarzschild."""
    p = np.asarray(phi, dtype=float)
    return np.asarray((1.0 + p) ** 2 / lp_ratio(p), dtype=float)


def misner_sharp(areal: np.ndarray, g_rr: np.ndarray) -> np.ndarray:
    """Quasi-local mass of time-symmetric spherical data: (R/2)(1 - 1/g_RR)."""
    return np.asarray(0.5 * areal * (1.0 - 1.0 / g_rr), dtype=float)


def golden_bottleneck(a: float, p: float) -> float:
    """Tightest point of |Phidot| < exp(-a Phi) / (1 - Phi^2)^(p/2)."""
    return (math.sqrt(p * p + 4.0 * a * a) - p) / (2.0 * a)


def _areal_g_rr(rho: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Areal radius R = omega rho and g_RR = (omega / R'(rho))^2, numerically."""
    areal = omega * rho
    return areal, (omega / np.gradient(areal, rho, edge_order=2)) ** 2


def _flat_laplacian(f: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Radial flat-space Laplacian (1/rho^2) d/drho (rho^2 df/drho)."""
    df = np.gradient(f, rho, edge_order=2)
    return np.gradient(rho**2 * df, rho, edge_order=2) / rho**2


# ---------------------------------------------------------------------------
# Checks (each returns a dict of headline numbers for the report)
# ---------------------------------------------------------------------------
def check_two_factor_law() -> dict[str, float]:
    """Chain sums converge to int l_P(Phi) kappa dr; the finite form needs
    constant Phi."""

    def phi_of(r: float) -> float:
        return 0.9 / r

    def kappa_of(r: float) -> float:
        return 1.0 + 0.5 / r

    r0, r1 = 1.0, 5.0
    exact = quad(lambda r: float(lp_ratio(phi_of(r))) * kappa_of(r), r0, r1)[0]
    errors = []
    for n in (100, 1000):
        edges = np.linspace(r0, r1, n + 1)
        mids = 0.5 * (edges[1:] + edges[:-1])
        # Markov-consistent links: -ln K_i = int_link kappa dr (closed form).
        sigma = np.diff(edges) + 0.5 * np.log(edges[1:] / edges[:-1])
        d = chain_distance(np.array([phi_of(m) for m in mids]), np.exp(-sigma))
        errors.append(abs(d - exact))
    assert errors[1] < 1e-5 and errors[0] / errors[1] > 50.0  # O(1/n^2)

    sigma_tot = (r1 - r0) + 0.5 * math.log(r1 / r0)  # = -ln K_total
    naive = {
        r: float(lp_ratio(phi_of(r))) * sigma_tot for r in (r0, 0.5 * (r0 + r1), r1)
    }
    rel = [abs(v - exact) / exact for v in naive.values()]
    assert min(rel) > 0.05  # single-l_P finite formula fails for varying Phi

    # Constant Phi: the chain sum IS the section 2.1 formula -l_P ln K_total.
    k_links = np.exp(-np.full(50, sigma_tot / 50))
    const = chain_distance(np.full(50, 0.3), k_links)
    assert math.isclose(
        const, -float(lp_ratio(0.3)) * math.log(float(np.prod(k_links))), rel_tol=1e-12
    )
    return {
        "chain_err_n1000": errors[1],
        "naive_rel_err_min": min(rel),
        "naive_rel_err_max": max(rel),
    }


def check_lp_only_obstruction(amp: float = 1.0) -> dict[str, float]:
    """h = l_P(Phi)^2 delta with Phi = A/rho: contraction, M_ADM = 0, m_MS < 0."""
    rho = np.linspace(1.05 * amp, 40.0 * amp, 400001)
    phi = amp / rho
    omega = lp_ratio(phi)
    areal, g_num = _areal_g_rr(rho, omega)
    inner = slice(10, -10)
    assert np.allclose(areal, np.sqrt(rho**2 - amp**2), rtol=1e-12)
    assert np.allclose(g_num[inner], g_rr_lp_only(phi)[inner], rtol=1e-6)
    assert np.all(g_rr_lp_only(phi) < 1.0)  # contraction, never dilation

    m_ms = misner_sharp(areal, g_rr_lp_only(phi))
    assert np.all(m_ms < 0.0)  # negative quasi-local mass everywhere

    # ADM mass = lim rho (omega - 1) for h = omega^2 delta.
    far = np.array([1e3, 1e4, 1e5]) * amp
    adm = far * (lp_ratio(amp / far) - 1.0)
    assert np.all(np.abs(adm) < amp / far)  # ~ -A^2/(2 rho) -> 0

    # Hamiltonian-constraint energy density, eps = -psi^-5 lap(psi) / (2 pi).
    psi = np.sqrt(omega)
    eps_num = -(psi**-5) * _flat_laplacian(psi, rho) / (2.0 * math.pi)
    eps_exact = amp**2 * (amp**2 + 2 * rho**2) / (8 * math.pi * (rho**2 - amp**2) ** 3)
    sel = slice(100, 40000)
    assert np.allclose(eps_num[sel], eps_exact[sel], rtol=1e-4)
    assert np.all(eps_exact > 0.0)
    return {
        "g_RR_at_Phi_0.9": float(g_rr_lp_only(0.9)),
        "ADM_mass_estimate": float(adm[-1]),
        "misner_sharp_at_rho_2A": float(np.interp(2.0 * amp, rho, m_ms)),
    }


def check_schwarzschild_phi_form(mass: float = 1.0) -> dict[str, float]:
    """Isotropic dictionary Phi = M/(2 rho): exact Schwarzschild, horizon at 1.

    Finite differences are ill-conditioned at the throat (dR/drho -> 0), so the
    derivative checks use Phi <= 0.99; the identities hold exactly on the whole
    range [0, 1) (see ``symbolic_checks``), and the chain-sum check below runs
    on a log grid up to Phi = 0.9999.
    """
    rho = np.linspace(mass / (2 * 0.99), 60.0 * mass, 400001)
    phi = mass / (2.0 * rho)
    psi = 1.0 + phi
    lapse = (1.0 - phi) / (1.0 + phi)
    areal, g_num = _areal_g_rr(rho, psi**2)
    inner = slice(10, -10)
    assert np.allclose(1.0 - 2.0 * mass / areal, lapse**2, rtol=1e-12)
    assert np.allclose(g_num[inner], g_rr_isotropic_dictionary(phi)[inner], rtol=1e-5)
    assert np.allclose(misner_sharp(areal, g_rr_isotropic_dictionary(phi)), mass)
    far = 1e6 * mass
    assert math.isclose(far * ((1 + mass / (2 * far)) ** 2 - 1), mass, rel_tol=1e-6)

    # Horizon, throat isometry rho -> M^2/(4 rho) (Phi -> 1/Phi), area 16 pi M^2.
    r_h = 0.5 * mass * (1 + 1) ** 2
    assert math.isclose(r_h, 2 * mass) and math.isclose(
        4 * math.pi * r_h**2, 16 * math.pi * mass**2
    )
    mirror = mass**2 / (4.0 * rho)
    assert np.allclose((mass / (2 * mirror)) * phi, 1.0)
    assert np.allclose(mirror * (1 + mass / (2 * mirror)) ** 2, areal)

    # Phi harmonic (static massless s3.1 equation) and the static lapse equation
    # div(psi^2 grad N) = 0, checked in flux form (conserved radial flux).
    flux_phi = rho**2 * np.gradient(phi, rho, edge_order=2)
    flux_lapse = rho**2 * psi**2 * np.gradient(lapse, rho, edge_order=2)
    assert np.allclose(flux_phi, -mass / 2.0, rtol=1e-6)
    assert np.allclose(flux_lapse, mass, rtol=1e-6)

    # Remark identity: lapse * psi^2 = 1 - Phi^2 = (l_P / l_P0)^2 (coordinate-dependent).
    assert np.allclose(lapse * psi**2, lp_ratio(phi) ** 2, rtol=1e-12)

    # Required decay rate reproduces the proper radial distance via the chain
    # sum, right down to Phi = 0.9999 where kappa_required diverges.
    rho_h = mass / 2 * (1.0 + np.logspace(-4, np.log10(120.0), 200001))
    phi_h = mass / (2.0 * rho_h)
    phim = 0.5 * (phi_h[1:] + phi_h[:-1])
    k_links = np.exp(-kappa_required(phim) * np.diff(rho_h))
    d_chain = chain_distance(phim, k_links)
    d_exact = quad(lambda r: (1 + mass / (2 * r)) ** 2, rho_h[0], rho_h[-1])[0]
    assert math.isclose(d_chain, d_exact, rel_tol=1e-8)

    # Dilation criterion d ln kappa / dPhi > Phi / (1 - Phi^2) (margin 2/(1+Phi)).
    grid, step = np.linspace(0.0, 0.99, 2001), 1e-6
    dlnk = (
        np.log(kappa_required(grid + step)) - np.log(kappa_required(grid - step))
    ) / (2 * step)
    assert np.allclose(dlnk - grid / (1 - grid**2), 2.0 / (1.0 + grid), rtol=1e-6)
    return {"chain_vs_exact_rel": abs(d_chain - d_exact) / d_exact}


def check_conventions(m: float = 0.1) -> list[tuple[str, bool, bool]]:
    """Candidate g_RR(Phi) forms: asymptotically flat at Phi=0? horizon at 1?"""
    candidates: dict[str, Callable[[float], float]] = {
        "1/Phi            (s1.2 convention)": lambda p: 1.0 / p,
        "1/(1-Phi)        (areal dictionary)": lambda p: 1.0 / (1.0 - p),
        "((1+Phi)/(1-Phi))^2 (isotropic)": lambda p: float(
            g_rr_isotropic_dictionary(p)
        ),
        "(1-Phi^2)^2      (l_P alone)": lambda p: float(g_rr_lp_only(p)),
    }
    rows = []
    for name, g in candidates.items():
        flat = abs(g(1e-9) - 1.0) < 1e-6
        horizon = g(1.0 - 1e-9) > 1e6
        rows.append((name, flat, horizon))
    verdict = {name: (flat, hor) for name, flat, hor in rows}
    assert verdict["1/Phi            (s1.2 convention)"] == (False, False)
    assert verdict["((1+Phi)/(1-Phi))^2 (isotropic)"] == (True, True)
    assert verdict["(1-Phi^2)^2      (l_P alone)"] == (True, False)

    def v_sim3(p: float) -> float:
        return 0.5 * m**2 * (1.0 - p) ** 2

    h = 1e-6
    dv0 = (v_sim3(h) - v_sim3(-h)) / (2 * h)
    assert math.isclose(dv0, -(m**2), rel_tol=1e-6)  # Phi=0 not a static solution
    return rows


def check_golden_bottleneck(beta: float = 1.0) -> dict[str, float]:
    """Elementary proof ingredients and robustness of Phi* = (sqrt5 - 1)/2."""
    ps = PHI_STAR
    rng = np.random.default_rng(0)
    t = rng.uniform(-3.0, 3.0, 1000)
    assert np.allclose(1 - (ps + t) ** 2, ps * (1 - 2 * t) - t**2, atol=1e-12)
    assert np.all(1 - 2 * t <= np.exp(-2 * t))  # e^x >= 1 + x
    x = np.linspace(-3.0, 3.0, 600001)
    gap = np.exp(2 * x) * (1 - x**2) - ps * math.exp(2 * ps)
    assert np.max(gap) <= 1e-12 and abs(x[np.argmax(gap)] - ps) < 1e-4

    phi = np.linspace(0.0, 0.999, 200001)
    bound = np.asarray(causality_bound(phi, beta), dtype=float)
    assert abs(phi[np.argmin(bound)] - ps) < 1e-4
    b_min = math.exp(-ps) / math.sqrt(beta * ps)
    assert math.isclose(float(bound.min()), b_min, rel_tol=1e-8)

    for a, p in ((1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (2.0, 2.0)):
        f = -a * phi - 0.5 * p * np.log(1 - phi**2)
        assert abs(phi[np.argmin(f)] - golden_bottleneck(a, p)) < 1e-4
    return {
        "min_bound_sqrt_beta": b_min * math.sqrt(beta),
        "R_star_isotropic_over_rs": (1 + ps) ** 2 / (4 * ps),
        "R_star_areal_over_rs": 1 / ps,
    }


def symbolic_checks() -> bool:
    """Exact SymPy checks; returns False (and skips) if SymPy is unavailable."""
    try:
        import sympy as sp
    except ImportError:  # pragma: no cover - optional dependency
        print("  SymPy not available; symbolic checks skipped.")
        return False

    t, rho, th, ph, x = sp.symbols("t rho theta phi x", real=True)
    mass = sp.symbols("M", positive=True)
    phi = mass / (2 * rho)
    lapse, psi4 = (1 - phi) / (1 + phi), (1 + phi) ** 4
    coords = [t, rho, th, ph]
    g = sp.diag(-(lapse**2), psi4, psi4 * rho**2, psi4 * rho**2 * sp.sin(th) ** 2)
    ginv = g.inv()
    n = 4
    gam = [
        [
            [
                sp.simplify(
                    sum(
                        ginv[a, d]
                        * (
                            sp.diff(g[d, b], coords[c])
                            + sp.diff(g[d, c], coords[b])
                            - sp.diff(g[b, c], coords[d])
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
    for b in range(n):
        for c in range(n):
            ric = sum(
                sp.diff(gam[a][b][c], coords[a]) - sp.diff(gam[a][b][a], coords[c])
                for a in range(n)
            ) + sum(
                gam[a][a][d] * gam[d][b][c] - gam[a][c][d] * gam[d][b][a]
                for a in range(n)
                for d in range(n)
            )
            assert sp.simplify(ric) == 0, "Phi-form metric is not Ricci-flat"

    areal = rho * (1 + phi) ** 2
    assert sp.simplify(1 - 2 * mass / areal - lapse**2) == 0
    g_rr = sp.simplify(((1 + phi) ** 2 / sp.diff(areal, rho)) ** 2)
    assert sp.simplify(g_rr - ((1 + phi) / (1 - phi)) ** 2) == 0
    assert sp.simplify(lapse * (1 + phi) ** 2 - (1 - phi**2)) == 0
    amp = sp.symbols("A", positive=True)
    omega = sp.sqrt(1 - amp**2 / rho**2)
    g_lp = sp.simplify((omega / sp.diff(omega * rho, rho)) ** 2)
    assert sp.simplify(g_lp - (1 - amp**2 / rho**2) ** 2) == 0
    s5 = (sp.sqrt(5) - 1) / 2
    assert sp.simplify(s5**2 + s5 - 1) == 0
    assert sp.expand(1 - (s5 + x) ** 2 - (s5 * (1 - 2 * x) - x**2)) == 0
    return True


def self_check(symbolic: bool = True) -> dict[str, float]:
    """Run every check; raise AssertionError on the first failure."""
    out: dict[str, float] = {}
    out.update(check_two_factor_law())
    out.update(check_lp_only_obstruction())
    out.update(check_schwarzschild_phi_form())
    check_conventions()
    out.update(check_golden_bottleneck())
    if symbolic:
        out["symbolic"] = float(symbolic_checks())
    return out


# ---------------------------------------------------------------------------
# Report and plots
# ---------------------------------------------------------------------------
def report(results: dict[str, float]) -> None:
    print("Simulation 7: macroscopic radial metric (convention: Phi=0 vacuum)")
    print("  1. two-factor law  g_rr = (l_P(Phi) kappa)^2")
    print(f"     chain -> integral error (n=1000): {results['chain_err_n1000']:.1e}")
    print(
        "     single-l_P finite formula with varying Phi: "
        f"{results['naive_rel_err_min']:.0%} - {results['naive_rel_err_max']:.0%} error"
    )
    print("  2. l_P alone (flat gamma, Phi = A/rho)")
    print(f"     g_RR(Phi=0.9) = {results['g_RR_at_Phi_0.9']:.4f}  (< 1: contraction)")
    print(f"     ADM mass estimate at rho=1e5 A: {results['ADM_mass_estimate']:.1e}")
    print(f"     Misner-Sharp mass at rho=2A: {results['misner_sharp_at_rho_2A']:.4f}")
    print("  3. Schwarzschild in Phi form (Phi = M/(2 rho))")
    print(
        f"     chain sum vs exact proper distance: {results['chain_vs_exact_rel']:.1e}"
    )
    header = "     Phi     g_RR[l_P only]  g_RR[Schw.]  kappa_req/kappa0  l_P/l_P0"
    print(header)
    for p in (0.0, 0.1, 0.3, PHI_STAR, 0.9, 0.99):
        print(
            f"     {p:<7.4f} {float(g_rr_lp_only(p)):>13.4f} "
            f"{float(g_rr_isotropic_dictionary(p)):>12.4f} "
            f"{float(kappa_required(p)):>17.4f} {float(lp_ratio(p)):>9.4f}"
        )
    print("  4. candidate g_RR(Phi): asymptotically flat at 0? horizon at 1?")
    for name, flat, hor in check_conventions():
        print(f"     {name:<40} flat={flat!s:<5} horizon={hor}")
    print("     Sim3 potential m^2 (1-Phi)^2/2: V'(0) = -m^2 -> Phi=0 is not static")
    print("  5. golden bottleneck")
    print(f"     min bound * sqrt(beta) = {results['min_bound_sqrt_beta']:.4f}")
    print(
        f"     location around a hole: R*/r_s = {results['R_star_isotropic_over_rs']:.4f}"
        f" (isotropic), {results['R_star_areal_over_rs']:.4f} (areal dictionary)"
    )
    for a, p in ((0.5, 1.0), (2.0, 1.0), (2.0, 2.0)):
        print(f"     a={a}, p={p}: Phi* = {golden_bottleneck(a, p):.4f}")
    if "symbolic" in results:
        ok = "passed" if results["symbolic"] else "skipped"
        print(f"  symbolic checks (SymPy: Ricci-flatness, identities): {ok}")


def make_plots(outdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi = np.linspace(0.0, 0.98, 400)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].semilogy(
        phi, g_rr_isotropic_dictionary(phi), label=r"Schwarzschild, isotropic $\Phi$"
    )
    axs[0].semilogy(phi, 1.0 / (1.0 - phi), "--", label=r"Schwarzschild, areal $\Phi$")
    axs[0].semilogy(phi, g_rr_lp_only(phi) + 1e-12, label=r"$\ell_P(\Phi)$ alone")
    axs[0].axhline(1.0, color="k", lw=0.6)
    axs[0].axvline(PHI_STAR, color="k", ls=":", lw=0.8)
    axs[0].set_xlabel(r"$\Phi$")
    axs[0].set_ylabel(r"$g_{RR}$ (areal gauge)")
    axs[0].set_title("Radial metric: dilation vs contraction")
    axs[0].legend(fontsize=8)
    axs[1].semilogy(phi, kappa_required(phi), label=r"required $\kappa/\kappa_0$")
    axs[1].semilogy(phi, 1.0 / lp_ratio(phi), "--", label=r"$\ell_{P0}/\ell_P(\Phi)$")
    axs[1].set_xlabel(r"$\Phi$")
    axs[1].set_ylabel("per isotropic coordinate length")
    axs[1].set_title("Correlation-decay rate needed for Schwarzschild")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig7_radial_metric.pdf")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulation 7: checks for the macroscopic radial metric g_rr(Phi)."
    )
    parser.add_argument("--no-plots", action="store_true", help="do not write the PDF")
    parser.add_argument(
        "--no-symbolic", action="store_true", help="skip the SymPy checks"
    )
    parser.add_argument("--outdir", default=".", help="directory for figures")
    args = parser.parse_args(argv)

    results = self_check(symbolic=not args.no_symbolic)
    report(results)
    if not args.no_plots:
        make_plots(args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
