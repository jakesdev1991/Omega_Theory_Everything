# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""onset-eval: pre-registered evaluation of onset-prediction claims.

A harness for the specific kind of scientific claim that is easy to fool
yourself about: "these features predict a regime change before it happens".
It fixes the protocol the claim is judged by -- label window, split, metric,
threshold-tuning discipline, paired baseline comparison, kill criterion -- so
the numbers can be compared across methods and across domains.

    from onset_eval import Arm, Protocol, Series, run_benchmark, render_report

Only NumPy is required. The modelling and test steps import scikit-learn and
SciPy when they run, so `pip install "onset-eval[full]"` (or your own copies of
those two) is needed for `evaluate_arm`, `run_benchmark`, `paired_test` and
`decide`; building arms, labelling onsets and loading CSVs work without them.
"""

from __future__ import annotations

from .harness import (
    Arm,
    Protocol,
    Series,
    decide,
    evaluate_arm,
    label_onsets,
    paired_test,
    render_report,
    run_benchmark,
)

__version__ = "0.1.0"

__all__ = [
    "Arm",
    "Protocol",
    "Series",
    "__version__",
    "decide",
    "evaluate_arm",
    "label_onsets",
    "paired_test",
    "render_report",
    "run_benchmark",
]
