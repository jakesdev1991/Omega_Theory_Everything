# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Independent checks for docs/research/ENTANGLED_TFIM_BENCHMARK.md (Phase 3, item 3).

1. Schwarzschild-required column: every row of the first table equals
   (1 + Phi)^2 / sqrt(1 - Phi^2) evaluated at that row's Phi.
2. Exact-diagonalization cross-check of one row: a dense ground state built here from
   Kronecker products (not from research/quantum_ising_chain.py) reproduces the note's
   Phi at h/J = 4 and matches the module's analyze_chain output.

Both checks support only the finite-chain diagnostic stated in the note. They say
nothing about a continuum gravitational model.

Run: python research/verify_tfim.py
Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.quantum_ising_chain import (  # noqa: E402
    analyze_chain,
    schwarzschild_required_kappa_ratio,
)

NOTE = _REPO / "docs" / "research" / "ENTANGLED_TFIM_BENCHMARK.md"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _single_site(op: np.ndarray, site: int, sites: int) -> np.ndarray:
    out = np.array([[1.0]])
    for k in range(sites):
        out = np.kron(out, op if k == site else np.eye(2))
    return out


def dense_ground_state(sites: int, h_over_j: float) -> np.ndarray:
    """Ground state of H = -sum Z_i Z_{i+1} - h sum X_i (open chain), built independently."""
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    dim = 2**sites
    ham = np.zeros((dim, dim))
    for i in range(sites - 1):
        ham -= _single_site(z, i, sites) @ _single_site(z, i + 1, sites)
    for i in range(sites):
        ham -= h_over_j * _single_site(x, i, sites)
    _, vecs = np.linalg.eigh(ham)
    return vecs[:, 0]


def _entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log(w)))


def dense_nearest_phi(sites: int, h_over_j: float) -> float:
    """Phi = I(left:right) / (2 ln 2) for the central nearest-neighbour pair."""
    psi = dense_ground_state(sites, h_over_j).reshape((2,) * sites)
    left = sites // 2 - 1
    right = left + 1

    def reduced(keep: list[int]) -> np.ndarray:
        drop = [k for k in range(sites) if k not in keep]
        moved = np.transpose(psi, keep + drop).reshape(2 ** len(keep), -1)
        return moved @ moved.conj().T

    s_l = _entropy(reduced([left]))
    s_r = _entropy(reduced([right]))
    s_lr = _entropy(reduced([left, right]))
    return (s_l + s_r - s_lr) / (2.0 * math.log(2.0))


def _table_rows() -> list[tuple[float, float, float]]:
    text = NOTE.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*\d+\.\d+\s*\|\s*\d+\.\d+\s*\|"
        r"\s*\d+\.\d+\s*\|\s*\d\.\d+\s*\|\s*(\d+\.\d+)\s*\|",
        re.MULTILINE,
    )
    return [(float(h), float(phi), float(req)) for h, phi, req in pattern.findall(text)]


def check_schwarzschild_column() -> CheckResult:
    rows = _table_rows()
    if len(rows) != 7:
        return CheckResult(
            "TFIM: Schwarzschild column parsed", False, f"found {len(rows)} rows"
        )
    worst = 0.0
    for _h, phi, required in rows:
        worst = max(worst, abs(schwarzschild_required_kappa_ratio(phi) - required))
    # The note prints four decimals, so the rounding error is at most 5e-5.
    ok = worst < 6e-5
    return CheckResult(
        "TFIM: Schwarzschild column = (1+Phi)^2/sqrt(1-Phi^2)",
        ok,
        f"7 rows; max |formula - table| = {worst:.1e}",
    )


def check_exact_diagonalization_row() -> CheckResult:
    sites, h = 8, 4.0
    dense_phi = dense_nearest_phi(sites, h)
    module_phi = analyze_chain(sites=sites, h_over_j=h).phi
    rows = {round(hj, 2): phi for hj, phi, _ in _table_rows()}
    table_phi = rows[h]
    ok = abs(dense_phi - module_phi) < 1e-9 and abs(dense_phi - table_phi) < 1e-6
    return CheckResult(
        "TFIM: dense ED row (N=8, h/J=4)",
        ok,
        f"dense Phi = {dense_phi:.9f}; module = {module_phi:.9f}; note = {table_phi:.6f}",
    )


CHECKS = [check_schwarzschild_column, check_exact_diagonalization_row]


def run_all() -> list[CheckResult]:
    return [check() for check in CHECKS]


def main() -> int:
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name.ljust(width)}  ({r.detail})")
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} TFIM checks passed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
