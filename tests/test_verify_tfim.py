# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Runs the TFIM benchmark cross-checks (research/verify_tfim.py) under pytest."""

from research.verify_tfim import run_all


def test_all_tfim_checks_pass() -> None:
    results = run_all()
    failed = [r for r in results if not r.passed]
    assert not failed, "\n".join(f"{r.name}: {r.detail}" for r in failed)
