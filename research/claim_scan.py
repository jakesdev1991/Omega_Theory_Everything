# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Triage counter for overclaim language in the prose surfaces of the repository.

This is a grep-level metric used as a progress gate in
docs/research/CLAIM_TIGHTENING_ROADMAP.md. It is not a certificate of correctness:
a line that says "this does not prove X" is counted as hedged and is not a finding,
but any unhedged hit must still be read by a person before it is accepted or fixed.

Run:  python research/claim_scan.py            (summary per surface)
      python research/claim_scan.py --list     (every unhedged hit, file:line)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Claim-language families. Each is a case-insensitive pattern unless noted.
PATTERNS: dict[str, re.Pattern[str]] = {
    "prove": re.compile(r"\bprove[sd]?\b", re.IGNORECASE),
    "label": re.compile(r"\bGENUINE\b|\bREAL (?:mathematical|formalization)\b"),
    "hype": re.compile(
        r"fundamental truth|exactly like this|finally bridges|don'?t need to invent"
        r"|true infinities do not exist|smoking gun|operates exactly|giant computer",
        re.IGNORECASE,
    ),
}

# A line containing one of these is treated as hedged (a limitation statement, not a claim).
# Deliberately narrow: words such as 'if' or 'open' would mask real claims in long one-line paragraphs.
HEDGE = re.compile(
    r"\b(?:not|no|never|neither|nor|cannot|can't|without|unless|conditional)\b",
    re.IGNORECASE,
)

# Which families apply to which surface. In Lean and its text mirrors, "proved" is the
# ordinary term for a kernel-checked theorem, so only self-description labels and hype
# are counted there; in prose surfaces every family applies.
SURFACE_FAMILIES: dict[str, set[str]] = {
    "latex_docs": {"prove", "label", "hype"},
    "core_notes": {"prove", "label", "hype"},
    "txt_proofs": {"label", "hype"},
    "lean_proofs": {"label", "hype"},
}

SURFACES: dict[str, list[str]] = {
    "latex_docs": ["latex_docs/*.tex"],
    "txt_proofs": ["txt_proofs/*.txt"],
    "lean_proofs": ["lean_proofs/*.lean"],
    "core_notes": [
        "README.md",
        "Omega_Theory_Laymans_Guide.md",
        "Omega_Theory_v4.0_Technical.md",
        "Omega_Theory_v4.0_Radial_Metric.md",
        "docs/research/*.md",
    ],
}


# Files excluded from the scan because they quote the phrases they police, by design.
EXCLUDE: set[str] = {"docs/research/CLAIM_TIGHTENING_ROADMAP.md"}


@dataclass(frozen=True)
class Hit:
    surface: str
    path: str
    line_no: int
    family: str
    text: str
    hedged: bool


def scan_file(surface: str, path: pathlib.Path) -> list[Hit]:
    hits: list[Hit] = []
    rel = str(path.relative_to(ROOT))
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        for family, pattern in PATTERNS.items():
            if family not in SURFACE_FAMILIES[surface]:
                continue
            if pattern.search(line):
                hits.append(
                    Hit(
                        surface=surface,
                        path=rel,
                        line_no=line_no,
                        family=family,
                        text=line.strip()[:200],
                        hedged=bool(HEDGE.search(line)),
                    )
                )
    return hits


def collect() -> list[Hit]:
    hits: list[Hit] = []
    for surface, globs in SURFACES.items():
        seen: set[pathlib.Path] = set()
        for pattern in globs:
            for path in sorted(ROOT.glob(pattern)):
                rel = str(path.relative_to(ROOT))
                if path.is_file() and path not in seen and rel not in EXCLUDE:
                    seen.add(path)
                    hits.extend(scan_file(surface, path))
    return hits


def summarise(hits: list[Hit]) -> str:
    lines = [f"{'surface':<14}{'files':>6}{'unhedged':>10}{'hedged':>9}"]
    surfaces = list(SURFACES)
    for surface in surfaces:
        s_hits = [h for h in hits if h.surface == surface]
        n_files = len({h.path for h in s_hits})
        unhedged = sum(1 for h in s_hits if not h.hedged)
        hedged = sum(1 for h in s_hits if h.hedged)
        lines.append(f"{surface:<14}{n_files:>6}{unhedged:>10}{hedged:>9}")
    total_unhedged = sum(1 for h in hits if not h.hedged)
    lines.append(
        f"{'TOTAL':<14}{'':>6}{total_unhedged:>10}{sum(1 for h in hits if h.hedged):>9}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print every unhedged hit")
    args = parser.parse_args(argv)
    hits = collect()
    print(summarise(hits))
    if args.list:
        for h in hits:
            if not h.hedged:
                print(f"{h.path}:{h.line_no} [{h.family}] {h.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
