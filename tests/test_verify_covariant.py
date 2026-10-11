# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Runs the covariant-note machine checks (research/verify_covariant.py) under pytest."""

from research.verify_covariant import run_all


def test_all_covariant_checks_pass() -> None:
    results = run_all()
    failed = [r for r in results if not r.passed]
    assert not failed, "\n".join(f"{r.name}: {r.detail}" for r in failed)
    assert len(results) == 4
