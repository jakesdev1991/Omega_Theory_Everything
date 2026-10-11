# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE.
"""Checks for the explicit diagonal-qubit Markov-chain benchmark."""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.qregion_markov import (  # noqa: E402
    DEFAULT_VACUUM_CORRELATION,
    LN2,
    correlation_from_mutual_information,
    correlation_from_phi,
    kappa_derivative_from_phi,
    kappa_from_phi,
    kappa_ratio_to_vacuum,
    markov_chain_probabilities,
    mutual_information_at_separation,
    mutual_information_from_correlation,
    mutual_information_from_phi,
    phi_from_mutual_information,
    raw_kappa_from_phi,
    schwarzschild_isotropic_h_rr,
    schwarzschild_required_kappa_ratio,
    toy_radial_metric_ratio,
)


def test_qubit_markov_state_is_normalized_and_full_rank_below_one() -> None:
    for step_correlation in (0.0, 0.2, 0.5, 0.99):
        probabilities = markov_chain_probabilities(5, step_correlation)
        assert sum(probabilities) == pytest.approx(1.0)
        assert all(probability > 0.0 for probability in probabilities)


def test_qubit_mutual_information_has_expected_endpoints_and_inverse() -> None:
    assert mutual_information_from_correlation(0.0) == pytest.approx(0.0)
    assert mutual_information_from_correlation(1.0) == pytest.approx(LN2)
    for correlation in (0.01, 0.2, 0.5, 0.9, 0.999):
        information = mutual_information_from_correlation(correlation)
        assert correlation_from_mutual_information(information) == pytest.approx(
            correlation, abs=1e-12
        )


def test_phi_information_map_is_invertible_on_the_declared_family() -> None:
    baseline = DEFAULT_VACUUM_CORRELATION
    for phi in (0.0, 0.01, 0.1, 0.3, 0.5, 0.9, 0.99, 1.0):
        information = mutual_information_from_phi(phi, baseline)
        assert phi_from_mutual_information(information, baseline) == pytest.approx(
            phi, abs=1e-12
        )
        assert correlation_from_phi(phi, baseline) >= baseline


def test_markov_correlator_composes_but_mutual_information_does_not() -> None:
    step_correlation = 0.5
    assert step_correlation**3 == pytest.approx(step_correlation * step_correlation**2)
    mi_one = mutual_information_at_separation(step_correlation, 1) / LN2
    mi_two = mutual_information_at_separation(step_correlation, 2) / LN2
    assert not math.isclose(mi_two, mi_one**2, rel_tol=1e-6, abs_tol=1e-9)


def test_regularized_kappa_decreases_with_overlap() -> None:
    values = [kappa_from_phi(phi) for phi in (0.0, 0.1, 0.3, 0.5, 0.9, 0.99)]
    assert values == sorted(values, reverse=True)
    assert kappa_from_phi(1.0) == 0.0
    assert all(kappa_derivative_from_phi(phi) < 0.0 for phi in (0.0, 0.2, 0.8))


def test_analytic_kappa_derivative_matches_finite_difference() -> None:
    phi, step = 0.4, 1e-6
    numerical = (kappa_from_phi(phi + step) - kappa_from_phi(phi - step)) / (2 * step)
    assert kappa_derivative_from_phi(phi) == pytest.approx(numerical, rel=1e-7)


def test_literal_zero_mutual_information_vacuum_has_infinite_log_decay_rate() -> None:
    assert math.isinf(raw_kappa_from_phi(0.0))
    assert raw_kappa_from_phi(1.0) == 0.0


def test_regularized_toy_radial_factor_moves_opposite_to_schwarzschild() -> None:
    assert toy_radial_metric_ratio(0.0) == pytest.approx(1.0)
    for phi in (0.1, 0.3, 0.5, 0.9, 0.99):
        assert toy_radial_metric_ratio(phi) < 1.0
        assert schwarzschild_isotropic_h_rr(phi) > 1.0
        assert kappa_ratio_to_vacuum(phi) < 1.0
        assert schwarzschild_required_kappa_ratio(phi) > 1.0
    assert toy_radial_metric_ratio(1.0) == 0.0
    assert math.isinf(schwarzschild_required_kappa_ratio(1.0))


def test_invalid_model_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        mutual_information_from_correlation(-0.1)
    with pytest.raises(ValueError):
        phi_from_mutual_information(0.0)
    with pytest.raises(ValueError):
        kappa_from_phi(0.5, spacing=0.0)
    with pytest.raises(ValueError):
        correlation_from_phi(0.5, vacuum_correlation=0.0)
    with pytest.raises(ValueError):
        markov_chain_probabilities(1.5, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mutual_information_at_separation(0.5, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        kappa_from_phi(0.5, spacing=math.nan)
