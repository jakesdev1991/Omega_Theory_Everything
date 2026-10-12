# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""CI gate for the derivation checks in research/verify_derivations.py."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.verify_derivations import CHECKS, CheckResult, run_all  # noqa: E402


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_each_derivation_check_passes(check: Callable[[], CheckResult]) -> None:
    result = check()
    assert result.passed, f"{result.name}: {result.detail}"


def test_all_checks_pass_in_one_run() -> None:
    results = run_all()
    assert len(results) == len(CHECKS) == 12
    assert all(r.passed for r in results), [r.name for r in results if not r.passed]
