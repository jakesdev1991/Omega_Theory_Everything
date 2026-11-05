# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""In-repository import path for the published `onset_eval` harness.

The harness now lives in its own distribution (`packages/onset_eval/`) so that
another project can install it rather than copy it. This module keeps the
`rcod.harness` import path working for the code in this repository -- the
frozen experiment's README, `rcod/section5/domain_example.py`,
`tests/test_rcod.py` and `docs/DOMAIN_AGNOSTIC_CORE.md` all referred to it, and
a moved module should not break its readers.

Outside this repository, use the distribution name:

    from onset_eval import Arm, Protocol, Series, run_benchmark, render_report
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "packages" / "onset_eval" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    # Not installed (the normal case when working in this checkout): fall back
    # to the package sources rather than requiring `pip install -e`.
    sys.path.insert(0, str(_SRC))

from onset_eval.harness import *  # noqa: E402,F403
from onset_eval.harness import __all__  # noqa: E402,F401
