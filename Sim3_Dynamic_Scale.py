# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: Apache-2.0
"""
Simulation 3: Dynamic Planck Length and Disformal Causality Band
================================================================

Abstract
--------
We investigate the interplay between a dynamically varying Planck length,
conformal rescaling, and disformal causality bounds. The Planck-length profile
is the Chain-Overlap-Density (COD) form formally verified in
``lean_proofs/DynamicCODScale.lean``:

    l_P(Phi) = l_P0 * sqrt(1 - Phi^2),      0 <= Phi <= 1.

Static relations are derived across the physical COD range, including a
beta-sweep of the disformal causality band. An optional dynamical evolution is
implemented via a scalar-field equation of motion with disformal-inspired
Hubble scaling, allowing causality checks and a preliminary estimate of
ringdown frequency shifts.

History: an earlier version of this simulation used the heuristic profile
``l_P(phi) = exp((1 - phi)/2)`` over ``phi in [0, 5]``. It was replaced by the
COD profile so that the simulation matches the Lean-proven module. The
differences are stated explicitly in Section 3.

Convention: ``Phi = 0`` is the vacuum and ``Phi -> 1`` the bound-state/horizon
end, matching ``lean_proofs/DynamicCODScale.lean`` and Section 2.2 of the v4.0
technical note. The consensus potential of Section 3.1 is oriented the same way
(minimum at ``Phi = 0``); an earlier revision used ``V = m^2 (1 - Phi)^2 / 2``,
whose minimum at ``Phi = 1`` followed the opposite reading of Section 1.2 and
therefore made the horizon end the attractor.

1. Introduction
---------------
Simulations 1 and 2 established that geometry can emerge from correlations and
that expansion can be modelled as a chain-break process. Simulation 3 probes
the ultraviolet and causal domain: how does a dynamic Planck length interact
with conformal and disformal factors, and what bounds ensure Lorentzian
causality?

2. Methods
----------
2.1 Static relations (l_P0 = 1 throughout)

- Dynamic Planck length (Lean: ``DynamicCODScale.dynamicPlanckCOD``):
      l_P(Phi) = sqrt(1 - Phi^2).
  Vacuum baseline l_P(0) = 1 (``dynamicPlanckCOD_zero``); maximal overlap
  l_P(1) = 0 (``dynamicPlanckCOD_one``); strictly decreasing on [0, 1]
  (``dynamicPlanckCOD_strictAntiOn``).
- Conformal factor:  C(Phi) = exp(-2 Phi).
- Disformal coupling: D(Phi) = beta * l_P(Phi)^2 = beta * (1 - Phi^2).
- Disformal causality bound. For the disformal metric
  g_hat = C g + D dPhi dPhi, the Lorentzian signature is preserved iff
  D * Phidot^2 < C, i.e.
      |Phidot| < sqrt(C / D) = exp(-Phi) / (sqrt(beta) * sqrt(1 - Phi^2)).
  (With the old exponential profile the same derivation gives the previous
  bound exp(-(phi+1)/2)/sqrt(beta); the derivation is unchanged, only l_P is.)

2.2 beta-sweep
  The bound is evaluated for beta = 1.0, 0.5 and 0.05.

2.3 Optional dynamics
- Scalar-field equation of motion
      Phiddot + 3 H(Phi) Phidot + V'(Phi) = 0,   H(Phi) = H0 exp(Phi),
  with consensus potential V(Phi) = m^2 Phi^2 / 2, whose minimum at Phi = 0
  is the vacuum of the COD convention used throughout (Section 2.2 and
  DynamicCODScale.lean: l_P(0) = l_P0). This is the orientation stated in
  Section 3.1 of the v4.0 technical note. An earlier revision used
  V = m^2 (1 - Phi)^2 / 2 (minimum at 1); under the present convention that
  would put the attractor at the horizon end, where l_P vanishes, and would
  leave V'(0) = -m^2 != 0 so that the vacuum is not a static solution.
- Initial conditions lie inside the causality band; integration stops if Phi
  reaches 1, where l_P vanishes.
- Diagnostics: causality ratio |Phidot| / bound, emergent scale factor
  a(t) = exp(int H dt), ringdown-shift proxy 100 * |Phidot| / (a l_P), and
  whether Phi stays inside the physical COD range [0, 1].

3. Results
----------
All numbers below are printed by ``python Sim3_Dynamic_Scale.py`` and can be
regenerated; the values quoted are for the default parameters.

3.1 Static outputs
- l_P(Phi) decreases monotonically from 1 at Phi = 0 to 0 at Phi = 1.
- The causality bound is NOT monotone (unlike the old exponential profile,
  where it tightened monotonically). d ln(bound)/dPhi = -1 + Phi/(1 - Phi^2)
  vanishes at Phi* = (sqrt(5) - 1)/2 ~ 0.618, so the band tightens on
  [0, Phi*] to a minimum of ~0.686/sqrt(beta) and then widens, diverging as
  Phi -> 1 because the disformal coupling D switches off with l_P.
- Decreasing beta widens the whole band by 1/sqrt(beta) (beta = 0.05 widens
  it by ~4.5x at every Phi).

3.2 Ringdown shift (static estimate)
  With a toy gradient |grad Phi| ~ 0.1 at Phi = 0 the proxy gives ~10 %
  (l_P(0) = 1). The same gradient at higher Phi gives a larger shift because
  l_P shrinks; the proxy diverges as Phi -> 1.

3.3 Dynamic evolution (optional)
  The initial kick (|Phidot| = 0.5 * bound(0) = 0.5) lifts Phi off the vacuum
  to a peak of 0.153 within t ~ 1; the restoring force V' = m^2 Phi then
  relaxes it back toward the consensus minimum at 0 with rate m^2 / (3 H)
  ~ 0.0033, so Phi is still 0.138 at t = 40. Phi stays inside [0, 1] and the
  horizon is not reached, which is the expected behaviour now that the
  attractor is the vacuum rather than the horizon end. The causality ratio
  starts at 0.5 by construction and only decreases, so |Phidot| < bound holds
  throughout. The ringdown proxy peaks at 50 % at t = 0 (it is
  100 * 0.5 * bound(0) / (a l_P) with a = l_P = 1 there) and decays with
  Phidot. ln a(40) ~ 46.

4. Discussion
-------------
The disformal causality bound is inherited directly from l_P(Phi). Replacing
the exponential heuristic by the Lean-proven COD profile changes the
qualitative picture in one place: the band no longer closes monotonically but
has a single tightest point at Phi* ~ 0.618 and opens up toward maximal
overlap. This is a consequence of the model choice D = beta l_P^2 and should
be read as such; neither the profile nor the coupling is derived from the
Omega axioms (see the "Model assumptions" section of DynamicCODScale.lean).

The potential is a second, independent model choice. V = m^2 Phi^2 / 2 has its
minimum at the vacuum Phi = 0, so the field relaxes toward the unshared state
and the horizon end Phi -> 1 is never an attractor; a quadratic potential is
also Yukawa-screened, Phi ~ e^(-m rho) / rho, which is what an asymptotically
flat vacuum needs. Both properties are consequences of the choice, not
derivations of it. A quartic or cosine potential with the same minimum would
change the late-time approach but not the vacuum value.

5. Conclusion
-------------
- Sim 1: emergent distances from correlations.
- Sim 2: cosmological expansion from chain-break processes.
- Sim 3: ultraviolet cutoff and causal band from the COD-driven Planck length.

Usage
-----
    python Sim3_Dynamic_Scale.py [--beta 1.0 0.5 0.05] [--no-dynamic]
                                 [--no-plots] [--outdir .]
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

import numpy as np

GOLDEN_CONJUGATE = (math.sqrt(5.0) - 1.0) / 2.0  # tightest point of the band


# ---------------------------------------------------------------------------
# Static relations
# ---------------------------------------------------------------------------
def lP(phi: np.ndarray | float, lP0: float = 1.0) -> np.ndarray | float:
    """COD-driven Planck length l_P(Phi) = l_P0 sqrt(1 - Phi^2).

    Mirrors ``DynamicCODScale.dynamicPlanckCOD``. Like Mathlib's ``Real.sqrt``
    the function is total: it returns 0 outside the physical range |Phi| >= 1.
    """
    return lP0 * np.sqrt(np.clip(1.0 - np.square(phi), 0.0, None))


def C_conformal(phi: np.ndarray | float) -> np.ndarray | float:
    """Conformal factor C(Phi) = exp(-2 Phi)."""
    return np.exp(-2.0 * phi)


def D_disformal(phi: np.ndarray | float, beta: float = 1.0) -> np.ndarray | float:
    """Disformal coupling D(Phi) = beta l_P(Phi)^2."""
    return beta * np.square(lP(phi))


def causality_bound(phi: np.ndarray | float, beta: float = 1.0) -> np.ndarray | float:
    """Signature-preservation bound |Phidot| < sqrt(C / D).

    Closed form: exp(-Phi) / (sqrt(beta) sqrt(1 - Phi^2)). Diverges as Phi -> 1.
    """
    return np.exp(-phi) / (math.sqrt(beta) * lP(phi))


def ringdown_shift_percent(
    phidot: np.ndarray | float, a: np.ndarray | float, phi: np.ndarray | float
) -> np.ndarray | float:
    """Ringdown-shift proxy 100 |Phidot| / (a l_P(Phi))."""
    return 100.0 * np.abs(phidot) / (a * lP(phi))


def self_check() -> None:
    """Numerical sanity checks mirroring the Lean theorems and the derivation."""
    phi = np.linspace(0.0, 0.999, 1000)
    assert math.isclose(float(lP(0.0)), 1.0)  # dynamicPlanckCOD_zero
    assert float(lP(1.0)) == 0.0  # dynamicPlanckCOD_one
    assert np.all(np.diff(lP(phi)) < 0)  # dynamicPlanckCOD_strictAntiOn
    assert np.all(lP(phi[1:]) < 1.0)  # cod_scale_contraction
    for beta in (1.0, 0.5, 0.05):
        derived = np.sqrt(C_conformal(phi) / D_disformal(phi, beta))
        assert np.allclose(derived, causality_bound(phi, beta))
    # Consensus potential V = m^2 Phi^2 / 2: minimum at the vacuum Phi = 0.
    m = 0.1
    v = 0.5 * m**2 * np.square(phi)
    assert math.isclose(float(v[0]), 0.0)  # V(0) = 0
    assert np.all(np.diff(v) >= 0.0)  # V increasing away from the vacuum
    dv = np.gradient(v, phi)[1:-1]
    assert np.allclose(dv, m**2 * phi[1:-1], rtol=1e-6, atol=1e-12)  # V' = m^2 Phi


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def run_static(betas: Sequence[float], make_plots: bool, outdir: str) -> None:
    phi = np.linspace(0.0, 0.95, 20)
    lP_vals = lP(phi)
    bounds = {b: causality_bound(phi, beta=b) for b in betas}

    print("Static relations (l_P0 = 1)")
    print(
        f"  l_P(0) = {float(lP(0.0)):.3f}, l_P(0.5) = {float(lP(0.5)):.3f}, l_P(1) = {float(lP(1.0)):.3f}"
    )
    print(f"  band tightest at Phi* = {GOLDEN_CONJUGATE:.3f}")
    for b in betas:
        bmin = float(causality_bound(GOLDEN_CONJUGATE, b))
        b0 = float(causality_bound(0.0, b))
        print(
            f"  beta={b:<5}: bound(0) = {b0:.3f}, min bound = {bmin:.3f}, bound(0.95) = {float(causality_bound(0.95, b)):.3f}"
        )
    grad = 0.1
    print(
        f"  static ringdown proxy for |grad Phi| = {grad}: {float(ringdown_shift_percent(grad, 1.0, 0.0)):.1f} % at Phi=0, {float(ringdown_shift_percent(grad, 1.0, 0.9)):.1f} % at Phi=0.9"
    )

    if not make_plots:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(phi, lP_vals, "g-o", label=r"$\ell_P(\Phi)=\sqrt{1-\Phi^2}$")
    plt.xlabel(r"$\Phi$")
    plt.ylabel(r"$\ell_P$")
    plt.title("Dynamic Planck Length (COD profile)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{outdir}/fig4_1a.pdf")

    plt.figure()
    for b in betas:
        plt.semilogy(phi, bounds[b], "o-", label=f"beta={b}")
    plt.axvline(GOLDEN_CONJUGATE, color="k", ls=":", label=r"$\Phi^*=(\sqrt{5}-1)/2$")
    plt.xlabel(r"$\Phi$")
    plt.ylabel("Bound on |dPhi/dt|")
    plt.title("Disformal Causality Band")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{outdir}/fig4_1b.pdf")


def run_dynamic(
    beta: float,
    make_plots: bool,
    outdir: str,
    H0: float = 1.0,
    m: float = 0.1,
    t_end: float = 40.0,
) -> bool:
    try:
        from scipy.integrate import cumulative_trapezoid, solve_ivp
    except ImportError:  # pragma: no cover - SciPy is a listed requirement
        print("SciPy not available; skipping dynamic run.")
        return True

    def phi_eom(_t: float, y: Sequence[float]) -> list[float]:
        phival, phidot = y
        H = H0 * math.exp(phival)
        # V = m^2 Phi^2 / 2, so V' = m^2 Phi: the minimum sits at the vacuum
        # Phi = 0 of the COD convention (l_P(0) = l_P0), not at Phi = 1.
        phiddot = -3.0 * H * phidot - m**2 * phival
        return [phidot, phiddot]

    def reach_horizon(_t: float, y: Sequence[float]) -> float:
        return y[0] - 1.0

    reach_horizon.terminal = True  # type: ignore[attr-defined]

    phi0 = 0.0
    phidot0 = 0.5 * float(causality_bound(phi0, beta=beta))
    sol = solve_ivp(
        phi_eom,
        (0.0, t_end),
        [phi0, phidot0],
        t_eval=np.linspace(0.0, t_end, 800),
        events=reach_horizon,
        rtol=1e-8,
        atol=1e-10,
    )
    phi_num, phidot_num = sol.y
    H_num = H0 * np.exp(phi_num)
    a_num = np.exp(cumulative_trapezoid(H_num, sol.t, initial=0.0))
    bound_num = causality_bound(phi_num, beta=beta)
    ratio = np.abs(phidot_num) / bound_num
    shift = np.asarray(ringdown_shift_percent(phidot_num, a_num, phi_num))
    in_range = bool(np.all((phi_num >= 0.0) & (phi_num <= 1.0)))
    horizon = bool(sol.status == 1)
    ok = bool(np.all(ratio < 1.0)) and in_range

    print(f"Dynamic run (beta={beta}, H0={H0}, m={m}, t_end={t_end})")
    print(
        f"  Phi: {phi_num[0]:.3f} -> {phi_num[-1]:.3f}; "
        f"range [{phi_num.min():.4f}, {phi_num.max():.4f}]; "
        f"inside [0, 1]: {in_range}"
    )
    print(
        f"  causality ratio: initial {ratio[0]:.3f}, max {ratio.max():.3f}  -> satisfied: {ok}"
    )
    print(f"  ringdown proxy: peak {shift.max():.2f} %")
    print(f"  ln a(t_end) = {math.log(a_num[-1]):.2f}")
    if horizon:
        print("  stopped: horizon (Phi = 1) reached")
    elif not in_range:
        print("  warning: Phi left the physical COD range [0, 1]")

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].plot(sol.t, phi_num)
        axs[0, 0].set_title(r"$\Phi(t)$")
        axs[0, 1].plot(sol.t, np.abs(phidot_num), label=r"$|\dot\Phi|$")
        axs[0, 1].plot(sol.t, bound_num, "g--", label="bound")
        axs[0, 1].set_yscale("log")
        axs[0, 1].legend()
        axs[0, 1].set_title("Causality band")
        axs[1, 0].semilogy(sol.t, a_num)
        axs[1, 0].set_title("a(t)")
        axs[1, 1].plot(sol.t, shift)
        axs[1, 1].set_title("Ringdown Shift (%)")
        fig.tight_layout()
        fig.savefig(f"{outdir}/fig41_eom.pdf")
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulation 3: COD-driven dynamic Planck length and disformal causality band."
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.05],
        help="disformal parameters for the sweep (first is used for dynamics)",
    )
    parser.add_argument(
        "--no-dynamic", action="store_true", help="skip the scalar-field evolution"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="do not write PDF figures"
    )
    parser.add_argument("--outdir", default=".", help="directory for figures")
    args = parser.parse_args(argv)

    self_check()
    run_static(args.beta, not args.no_plots, args.outdir)
    if not args.no_dynamic:
        ok = run_dynamic(args.beta[0], not args.no_plots, args.outdir)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
