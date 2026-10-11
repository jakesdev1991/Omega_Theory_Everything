# Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT
"""Finite transverse-field Ising-chain benchmark with entangled Q-regions.

This exact-diagonalization study is a finite-lattice diagnostic, not a
continuum gravitational model. See docs/research/ENTANGLED_TFIM_BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

LN2 = math.log(2.0)
PAULI_Z = np.diag([1.0, -1.0])
ZZ = np.kron(PAULI_Z, PAULI_Z)
MAX_TWO_QUBIT_MUTUAL_INFORMATION = 2.0 * LN2


def schwarzschild_required_kappa_ratio(phi: float) -> float:
    """Identified isotropic-Schwarzschild comparison, not a model input."""
    if not 0.0 <= phi <= 1.0:
        raise ValueError("phi must lie in [0, 1]")
    if phi == 1.0:
        return math.inf
    return (1.0 + phi) ** 2 / math.sqrt(1.0 - phi**2)


@dataclass(frozen=True)
class IsingBenchmark:
    """Observables from one finite-chain ground state."""

    sites: int
    h_over_j: float
    energy_per_site: float
    excitation_gap: float
    phi: float
    nearest_mutual_information: float
    pair_negativity: float
    separations: tuple[int, ...]
    mutual_information_by_separation: tuple[float, ...]
    zz_correlation_by_separation: tuple[float, ...]
    effective_correlation_decay_rate: float
    correlation_fit_r_squared: float


def transverse_field_ising_hamiltonian(sites: int, h_over_j: float) -> coo_matrix:
    """Build H/J=-sum_i Z_i Z_(i+1) - (h/J) sum_i X_i, with open boundaries."""
    if not isinstance(sites, int) or isinstance(sites, bool) or sites < 2:
        raise ValueError("sites must be an integer of at least two")
    if not math.isfinite(h_over_j) or h_over_j <= 1.0:
        raise ValueError("h_over_j must be finite and greater than 1")

    dimension = 1 << sites
    basis = np.arange(dimension, dtype=np.uint64)
    z_values = np.empty((sites, dimension), dtype=np.int8)
    for site in range(sites):
        bit_position = sites - site - 1
        z_values[site] = 1 - 2 * ((basis >> bit_position) & 1)

    diagonal = np.zeros(dimension, dtype=float)
    for site in range(sites - 1):
        diagonal -= z_values[site] * z_values[site + 1]

    rows = np.repeat(np.arange(dimension, dtype=np.int64), sites)
    site_bits = 1 << np.arange(sites - 1, -1, -1, dtype=np.int64)
    bit_masks = np.tile(site_bits, dimension)
    columns = np.bitwise_xor(rows, bit_masks)
    data = np.full(rows.size, -h_over_j, dtype=float)
    flip_terms = coo_matrix((data, (rows, columns)), shape=(dimension, dimension))
    return (diags(diagonal, format="coo") + flip_terms).tocoo()


def lowest_two_states(sites: int, h_over_j: float) -> tuple[float, float, np.ndarray]:
    """Return the ground energy, first gap, and normalized ground-state vector."""
    hamiltonian = transverse_field_ising_hamiltonian(sites, h_over_j).tocsr()
    dimension = 1 << sites
    initial = np.full(dimension, 1.0 / math.sqrt(dimension), dtype=float)
    energies, states = eigsh(
        hamiltonian,
        k=2,
        which="SA",
        tol=1e-11,
        v0=initial,
        maxiter=max(1000, 20 * dimension),
    )
    order = np.argsort(energies)
    ground_energy = float(energies[order[0]])
    gap = float(energies[order[1]] - energies[order[0]])
    ground_state = np.asarray(states[:, order[0]], dtype=float)
    ground_state /= np.linalg.norm(ground_state)
    if gap <= 0.0:
        raise ArithmeticError("finite-chain ground state is not isolated")
    return ground_energy, gap, ground_state


def reduced_density_matrix(
    state: np.ndarray, sites: int, keep_sites: tuple[int, ...]
) -> np.ndarray:
    """Partial trace, with site 0 as the leftmost tensor/Hamiltonian factor."""
    if state.ndim != 1 or state.size != (1 << sites):
        raise ValueError("state dimension must be exactly 2**sites")
    if not keep_sites or len(set(keep_sites)) != len(keep_sites):
        raise ValueError("keep_sites must be a nonempty tuple of distinct sites")
    if any(site < 0 or site >= sites for site in keep_sites):
        raise ValueError("keep_sites contains an out-of-range site")

    tensor = state.reshape((2,) * sites)
    trace_sites = tuple(site for site in range(sites) if site not in keep_sites)
    matrix = tensor.transpose(keep_sites + trace_sites).reshape(
        1 << len(keep_sites), -1
    )
    return matrix @ matrix.conj().T


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """Von Neumann entropy in nats, with small roundoff eigenvalues clipped."""
    hermitian = 0.5 * (density_matrix + density_matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if np.min(eigenvalues) < -1e-9:
        raise ArithmeticError("reduced density matrix is not positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    nonzero = eigenvalues[eigenvalues > 1e-14]
    return float(-np.sum(nonzero * np.log(nonzero)))


def pair_mutual_information(
    state: np.ndarray, sites: int, left: int, right: int
) -> float:
    """Two-site quantum mutual information S(left)+S(right)-S(left,right)."""
    rho_left = reduced_density_matrix(state, sites, (left,))
    rho_right = reduced_density_matrix(state, sites, (right,))
    rho_pair = reduced_density_matrix(state, sites, (left, right))
    return (
        von_neumann_entropy(rho_left)
        + von_neumann_entropy(rho_right)
        - von_neumann_entropy(rho_pair)
    )


def pair_zz_correlation(state: np.ndarray, sites: int, left: int, right: int) -> float:
    """Absolute-value-ready Z_i Z_j order-parameter correlator."""
    rho_pair = reduced_density_matrix(state, sites, (left, right))
    return float(np.real(np.trace(rho_pair @ ZZ)))


def pair_negativity(state: np.ndarray, sites: int, left: int, right: int) -> float:
    """Two-qubit negativity; positive value certifies pairwise entanglement."""
    rho = reduced_density_matrix(state, sites, (left, right)).reshape(2, 2, 2, 2)
    partial_transpose = rho.transpose(0, 3, 2, 1).reshape(4, 4)
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (partial_transpose + partial_transpose.conj().T)
    )
    return float(np.sum(np.abs(eigenvalues[eigenvalues < 0.0])))


def analyze_chain(
    sites: int = 12,
    h_over_j: float = 1.4,
    max_separation: int | None = None,
) -> IsingBenchmark:
    """Compute local QMI, pair negativity, and a stated finite-range decay fit.

    Phi is the nearest-neighbour QMI divided by the two-qubit upper bound
    2*ln(2). The effective decay rate is the negative least-squares slope of
    log(abs(<Z_i Z_(i+r)>)) versus r over the selected finite separation range.
    It is not asserted to be an exact continuum kappa or a Markov rate.
    """
    if not isinstance(sites, int) or isinstance(sites, bool) or sites < 6:
        raise ValueError("sites must be an integer of at least six")
    ground_energy, gap, state = lowest_two_states(sites, h_over_j)
    left = sites // 2 - 1
    furthest_available = sites - left - 1
    if max_separation is None:
        max_separation = min(5, furthest_available)
    if (
        not isinstance(max_separation, int)
        or isinstance(max_separation, bool)
        or not 3 <= max_separation <= furthest_available
    ):
        raise ValueError(
            "max_separation must be an integer from 3 to the available chain extent"
        )

    separations = tuple(range(1, max_separation + 1))
    informations = tuple(
        pair_mutual_information(state, sites, left, left + separation)
        for separation in separations
    )
    correlations = tuple(
        pair_zz_correlation(state, sites, left, left + separation)
        for separation in separations
    )
    fit_y = np.log(np.maximum(np.abs(correlations), 1e-30))
    slope, intercept = np.polyfit(np.asarray(separations, dtype=float), fit_y, 1)
    predicted = intercept + slope * np.asarray(separations, dtype=float)
    residual_sum = float(np.sum((fit_y - predicted) ** 2))
    total_sum = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
    fit_r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    nearest_information = informations[0]
    phi = nearest_information / MAX_TWO_QUBIT_MUTUAL_INFORMATION
    return IsingBenchmark(
        sites=sites,
        h_over_j=h_over_j,
        energy_per_site=ground_energy / sites,
        excitation_gap=gap,
        phi=phi,
        nearest_mutual_information=nearest_information,
        pair_negativity=pair_negativity(state, sites, left, left + 1),
        separations=separations,
        mutual_information_by_separation=informations,
        zz_correlation_by_separation=correlations,
        effective_correlation_decay_rate=float(-slope),
        correlation_fit_r_squared=fit_r_squared,
    )


def format_report(sites: int = 12) -> str:
    """Run a fixed, unfitted field sweep and return its reproducible report."""
    field_ratios = (4.0, 2.5, 1.8, 1.4, 1.2, 1.1, 1.05)
    lines = [
        f"Open transverse-field Ising chain; N={sites}; J=1; h/J>1",
        "Q-regions are single spins; Phi=nearest-neighbour QMI/(2 ln 2).",
        "kappa_eff is the finite-range log(|ZZ|) slope, not an exact GR metric.",
        "",
        " h/J       Phi       I1[nats]    negativity    kappa_eff    R^2   kappa_GR/kappa0",
    ]
    for field in field_ratios:
        result = analyze_chain(sites=sites, h_over_j=field)
        required = schwarzschild_required_kappa_ratio(result.phi)
        lines.append(
            f"{field:5.2f}   {result.phi:8.6f}   "
            f"{result.nearest_mutual_information:10.7f}   "
            f"{result.pair_negativity:10.7f}   "
            f"{result.effective_correlation_decay_rate:10.7f}   "
            f"{result.correlation_fit_r_squared:5.3f}   {required:14.7f}"
        )
    lines.extend(
        (
            "",
            "The ground-state gap is positive for every finite sweep member.",
            "Nearest-neighbour negativity > 0 certifies pairwise entanglement.",
            "The QMI and ZZ-correlation sequences are not fitted to Schwarzschild.",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites", type=int, default=12, help="finite-chain length (default: 12)"
    )
    args = parser.parse_args()
    print(format_report(sites=args.sites))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
