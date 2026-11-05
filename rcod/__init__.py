# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Portable core of the RCOD work: metrics, streaming tracker, claim harness.

Two things in this package are useful outside this repository:

    rcod.section5.swarm   multiscale overlap metrics (Φ, μ) on any real series
    rcod.harness          pre-registered evaluation of an onset-prediction claim

    rcod.section5         the frozen §5 experiment and its pre-registration

Everything else in `rcod/` (the optimizer governor and its benchmark) is the
negative result it says it is. See docs/DOMAIN_AGNOSTIC_CORE.md for what is
portable and what is not, with the measurements.
"""

from __future__ import annotations

from .harness import (
    Arm,
    Protocol,
    Series,
    evaluate_arm,
    label_onsets,
    paired_test,
    render_report,
    run_benchmark,
)
from .section5.swarm import SwarmTracker, pool_channels, swarm_metrics

__all__ = [
    "Arm",
    "Protocol",
    "Series",
    "SwarmTracker",
    "evaluate_arm",
    "label_onsets",
    "paired_test",
    "pool_channels",
    "render_report",
    "run_benchmark",
    "swarm_metrics",
]
