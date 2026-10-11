# Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT
"""A minimal qubit-chain benchmark for correlation-derived distance.

This is a deliberately small diagnostic model, not a model of quantum gravity.
The density operator is diagonal in the Z basis, so it is a classical Markov
chain encoded as a valid, full-rank multi-qubit state for 0 <= lambda < 1.
See docs/research/QREGION_MARKOV_MODEL.md for assumptions and interpretation.
"""

from __future__ import annotations

import itertools
import math

LN2 = math.log(2.0)
DEFAULT_VACUUM_CORRELATION = 0.5


def _check_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _check_baseline(correlation: float) -> None:
    if not 0.0 < correlation < 1.0:
        raise ValueError("vacuum_correlation must lie strictly between 0 and 1")


def binary_entropy(probability: float) -> float:
    """Binary Shannon entropy in nats, with the continuous endpoint values."""
    _check_unit_interval(probability, "probability")
    entropy = 0.0
    if probability > 0.0:
        entropy -= probability * math.log(probability)
    if probability < 1.0:
        complement = 1.0 - probability
        entropy -= complement * math.log(complement)
    return entropy


def markov_chain_probabilities(
    region_count: int, step_correlation: float
) -> tuple[float, ...]:
    """Diagonal entries of rho_N, returned in lexicographic spin order."""
    _check_unit_interval(step_correlation, "step_correlation")
    if (
        not isinstance(region_count, int)
        or isinstance(region_count, bool)
        or region_count < 1
    ):
        raise ValueError("region_count must be a positive integer")
    probabilities = []
    for spins in itertools.product((-1, 1), repeat=region_count):
        probability = 0.5
        for left, right in zip(spins, spins[1:]):
            probability *= 0.5 * (1.0 + step_correlation * left * right)
        probabilities.append(probability)
    return tuple(probabilities)


def mutual_information_from_correlation(correlation: float) -> float:
    """Two-qubit mutual information for rho=(I+c Z⊗Z)/4, in nats."""
    _check_unit_interval(correlation, "correlation")
    probability = 0.5 * (1.0 + correlation)
    return LN2 - binary_entropy(probability)


def correlation_from_mutual_information(mutual_information: float) -> float:
    """Invert the two-qubit mutual information map by monotone bisection."""
    if not 0.0 <= mutual_information <= LN2:
        raise ValueError("mutual_information must lie in [0, ln(2)]")
    if mutual_information == 0.0:
        return 0.0
    if mutual_information == LN2:
        return 1.0

    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if mutual_information_from_correlation(midpoint) < mutual_information:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def mutual_information_at_separation(step_correlation: float, separation: int) -> float:
    """Pairwise mutual information at `separation` edges in the chain."""
    _check_unit_interval(step_correlation, "step_correlation")
    if (
        not isinstance(separation, int)
        or isinstance(separation, bool)
        or separation < 1
    ):
        raise ValueError("separation must be a positive integer")
    return mutual_information_from_correlation(step_correlation**separation)


def raw_phi_from_mutual_information(mutual_information: float) -> float:
    """Literal qubit normalization Phi=I/ln(2), including the product-state end."""
    if not 0.0 <= mutual_information <= LN2:
        raise ValueError("mutual_information must lie in [0, ln(2)]")
    return mutual_information / LN2


def phi_from_mutual_information(
    mutual_information: float,
    vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION,
) -> float:
    """Normalize excess adjacent-pair MI above a nonzero vacuum baseline.

    This regularized convention has Phi=0 at I(vacuum)>0 and Phi=1 at the
    perfectly correlated limit I=ln(2). It is an explicit toy-model choice,
    not the original literal Phi=I/I_max definition.
    """
    _check_baseline(vacuum_correlation)
    vacuum_information = mutual_information_from_correlation(vacuum_correlation)
    if not vacuum_information <= mutual_information <= LN2:
        raise ValueError(
            "mutual_information must be between its vacuum value and ln(2)"
        )
    return (mutual_information - vacuum_information) / (LN2 - vacuum_information)


def mutual_information_from_phi(
    phi: float,
    vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION,
) -> float:
    """Adjacent-pair mutual information implied by the regularized Phi."""
    _check_unit_interval(phi, "phi")
    _check_baseline(vacuum_correlation)
    vacuum_information = mutual_information_from_correlation(vacuum_correlation)
    return vacuum_information + phi * (LN2 - vacuum_information)


def correlation_from_phi(
    phi: float,
    vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION,
) -> float:
    """Invert the regularized Phi map to the one-step Markov correlation."""
    _check_unit_interval(phi, "phi")
    _check_baseline(vacuum_correlation)
    if phi == 0.0:
        return vacuum_correlation
    if phi == 1.0:
        return 1.0
    return correlation_from_mutual_information(
        mutual_information_from_phi(phi, vacuum_correlation)
    )


def kappa_from_phi(
    phi: float,
    spacing: float = 1.0,
    vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION,
) -> float:
    """Log-correlation decay rate -ln(lambda)/spacing, in inverse coordinate units."""
    _check_unit_interval(phi, "phi")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    correlation = correlation_from_phi(phi, vacuum_correlation)
    if correlation == 1.0:
        return 0.0
    return -math.log(correlation) / spacing


def kappa_derivative_from_phi(
    phi: float,
    spacing: float = 1.0,
    vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION,
) -> float:
    """Analytic derivative d(kappa)/d(Phi) for the regularized model."""
    _check_unit_interval(phi, "phi")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    correlation = correlation_from_phi(phi, vacuum_correlation)
    if correlation == 1.0:
        return 0.0
    information_headroom = LN2 - mutual_information_from_correlation(vacuum_correlation)
    return -information_headroom / (spacing * correlation * math.atanh(correlation))


def kappa_ratio_to_vacuum(
    phi: float,
    vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION,
) -> float:
    """Dimensionless kappa(Phi)/kappa(0) for the regularized model."""
    baseline = -math.log(vacuum_correlation)
    return kappa_from_phi(phi, vacuum_correlation=vacuum_correlation) / baseline


def toy_radial_metric_ratio(
    phi: float,
    vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION,
) -> float:
    """h_rr(Phi)/h_rr(0) under the chosen ell_P and fixed-coordinate chain ansatz."""
    _check_unit_interval(phi, "phi")
    if phi == 1.0:
        return 0.0
    return (1.0 - phi**2) * kappa_ratio_to_vacuum(phi, vacuum_correlation) ** 2


def schwarzschild_required_kappa_ratio(phi: float) -> float:
    """Required kappa ratio for isotropic Schwarzschild under the chosen ell_P."""
    _check_unit_interval(phi, "phi")
    if phi == 1.0:
        return math.inf
    return (1.0 + phi) ** 2 / math.sqrt(1.0 - phi**2)


def schwarzschild_isotropic_h_rr(phi: float) -> float:
    """Spatial radial coefficient (1+Phi)^4 in the isotropic Schwarzschild chart."""
    _check_unit_interval(phi, "phi")
    return (1.0 + phi) ** 4


def raw_kappa_from_phi(phi: float, spacing: float = 1.0) -> float:
    """Kappa for literal Phi=I/ln(2); it diverges at the product-state vacuum."""
    _check_unit_interval(phi, "phi")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    correlation = correlation_from_mutual_information(phi * LN2)
    if correlation == 0.0:
        return math.inf
    if correlation == 1.0:
        return 0.0
    return -math.log(correlation) / spacing


def format_report(vacuum_correlation: float = DEFAULT_VACUUM_CORRELATION) -> str:
    """Return the reproducible comparison table used in the research note."""
    _check_baseline(vacuum_correlation)
    vacuum_information = mutual_information_from_correlation(vacuum_correlation)
    phi_values = (0.0, 0.1, 0.3, 0.5, (math.sqrt(5.0) - 1.0) / 2.0, 0.9, 0.99)
    lines = [
        "Minimal diagonal-qubit Markov-chain benchmark (all information in nats)",
        f"vacuum correlation lambda_v = {vacuum_correlation:.6g}",
        f"vacuum adjacent-pair mutual information = {vacuum_information:.12g}",
        "",
        " Phi       lambda       kappa/kappa0   h_rr(toy)   h_rr(Schwarzschild)",
    ]
    for phi in phi_values:
        correlation = correlation_from_phi(phi, vacuum_correlation)
        lines.append(
            f"{phi:7.4f}   {correlation:10.7f}   "
            f"{kappa_ratio_to_vacuum(phi, vacuum_correlation):12.7f}   "
            f"{toy_radial_metric_ratio(phi, vacuum_correlation):10.7f}   "
            f"{schwarzschild_isotropic_h_rr(phi):19.7f}"
        )

    lambda_half = 0.5
    mi_kernel_1 = mutual_information_at_separation(lambda_half, 1) / LN2
    mi_kernel_2 = mutual_information_at_separation(lambda_half, 2) / LN2
    lines.extend(
        (
            "",
            "Literal normalized-MI kernel check at lambda=0.5:",
            f"  K_MI(1)={mi_kernel_1:.12g}; K_MI(2)={mi_kernel_2:.12g}; "
            f"K_MI(1)^2={mi_kernel_1**2:.12g}",
            f"Literal Phi=I/ln(2): kappa(Phi=0)={raw_kappa_from_phi(0.0)}",
        )
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report())
