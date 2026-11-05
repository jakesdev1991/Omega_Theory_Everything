# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Make `import onset_eval` work without installing the distribution.

In CI the packaging step installs a wheel and the package is already importable;
in the repository's own pytest run it is not, and asking every contributor to
run `pip install -e packages/onset_eval` before the suite works would be a tax on
the thing this change is supposed to remove.
"""

from __future__ import annotations

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
