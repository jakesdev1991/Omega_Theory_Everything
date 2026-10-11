# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE.
"""Tests for the finite transverse-field Ising Q-region benchmark."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.quantum_ising_chain import (  # noqa: E402
    MAX_TWO_QUBIT_MUTUAL_INFORMATION,
    analyze_chain,
    lowest_two_states,
    reduced_density_matrix,
    transverse_field_ising_hamiltonian,
)


def test_ising_hamiltonian_is_real_symmetric_and_ground_state_is_non_degenerate() -> (
    None
):
    small_hamiltonian = transverse_field_ising_hamiltonian(2, 1.4).toarray()
    expected = np.array(
        [
            [-1.0, -1.4, -1.4, 0.0],
            [-1.4, 1.0, 0.0, -1.4],
            [-1.4, 0.0, 1.0, -1.4],
            [0.0, -1.4, -1.4, -1.0],
        ]
    )
    np.testing.assert_allclose(small_hamiltonian, expected)

    hamiltonian = transverse_field_ising_hamiltonian(6, 1.4).tocsr()
    assert hamiltonian.shape == (64, 64)
    assert (hamiltonian - hamiltonian.T).nnz == 0

    _, gap, state = lowest_two_states(8, 1.4)
    assert gap > 0.0
    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_pair_reduced_state_is_valid_and_entangled() -> None:
    result = analyze_chain(sites=8, h_over_j=1.4, max_separation=3)
    _, _, state = lowest_two_states(8, 1.4)
    rho_pair = reduced_density_matrix(state, 8, (3, 4))
    np.testing.assert_allclose(rho_pair, rho_pair.conj().T, atol=1e-12)
    assert np.trace(rho_pair) == pytest.approx(1.0)
    assert np.linalg.eigvalsh(rho_pair).min() >= -1e-10

    partial_transpose = rho_pair.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    partial_transpose_eigenvalues = np.linalg.eigvalsh(partial_transpose)
    assert partial_transpose_eigenvalues.min() < -1e-8
    assert result.pair_negativity == pytest.approx(
        np.abs(partial_transpose_eigenvalues[partial_transpose_eigenvalues < 0]).sum()
    )
    assert result.nearest_mutual_information > 0.0
    assert result.pair_negativity > 0.0
    assert 0.0 < result.phi < 1.0
    assert result.phi == pytest.approx(
        result.nearest_mutual_information / MAX_TWO_QUBIT_MUTUAL_INFORMATION
    )


def test_entangled_chain_decay_rate_decreases_as_nearest_pair_phi_rises() -> None:
    near_critical = analyze_chain(sites=8, h_over_j=1.2, max_separation=3)
    intermediate = analyze_chain(sites=8, h_over_j=2.0, max_separation=3)
    more_disordered = analyze_chain(sites=8, h_over_j=4.0, max_separation=3)

    assert near_critical.phi > intermediate.phi > more_disordered.phi
    assert (
        near_critical.effective_correlation_decay_rate
        < intermediate.effective_correlation_decay_rate
        < more_disordered.effective_correlation_decay_rate
    )
    assert (
        min(
            near_critical.pair_negativity,
            intermediate.pair_negativity,
            more_disordered.pair_negativity,
        )
        > 0.0
    )


@pytest.mark.parametrize("sites", (8, 10, 12, 14))
def test_opposite_decay_rate_trend_persists_across_chain_sizes(sites: int) -> None:
    weakly_correlated = analyze_chain(sites=sites, h_over_j=4.0, max_separation=3)
    more_correlated = analyze_chain(sites=sites, h_over_j=1.05, max_separation=3)

    assert more_correlated.phi > weakly_correlated.phi
    assert (
        more_correlated.effective_correlation_decay_rate
        < weakly_correlated.effective_correlation_decay_rate
    )
    assert weakly_correlated.pair_negativity > 0.0
    assert more_correlated.pair_negativity > 0.0


def test_invalid_ising_benchmark_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        transverse_field_ising_hamiltonian(8, 1.0)
    with pytest.raises(ValueError):
        analyze_chain(sites=5, h_over_j=1.5)
    with pytest.raises(ValueError):
        analyze_chain(sites=8, h_over_j=1.5, max_separation=5)
