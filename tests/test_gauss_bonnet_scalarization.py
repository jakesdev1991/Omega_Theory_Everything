# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE.
"""Regression tests for the candidate curvature-coupled scalar zero mode."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gauss_bonnet_scalarization import (  # noqa: E402
    asymptotic_constant_mode,
    first_zero_mode_threshold,
)


def test_zero_coupling_is_the_regular_constant_mode() -> None:
    assert asymptotic_constant_mode(0.0) == pytest.approx(1.0)


def test_coupling_bracket_changes_the_asymptotic_constant_mode_sign() -> None:
    assert asymptotic_constant_mode(0.7) > 0.0
    assert asymptotic_constant_mode(0.8) < 0.0


def test_first_static_zero_mode_has_small_boundary_residual() -> None:
    result = first_zero_mode_threshold()
    assert result.eta_over_mass_squared == pytest.approx(0.7256, abs=2e-4)
    assert abs(result.asymptotic_constant_mode) < 1e-8
    assert result.dimensionless_outer_radius == 250.0


def test_invalid_zero_mode_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        asymptotic_constant_mode(-0.1)
    with pytest.raises(ValueError):
        asymptotic_constant_mode(0.5, outer_radius=1.0)
    with pytest.raises(ValueError):
        first_zero_mode_threshold(scan_points=2)
