#!/usr/bin/env python3
"""Audit trusted Lean declarations without counting comments or field names.

The audit is intentionally lexical: it is useful in CI before a full Lean
installation is available.  It reports declaration-level ``axiom`` commands,
not occurrences of the word in documentation or structure field names.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

AXIOM = re.compile(r"^\s*axiom\s+([A-Za-z0-9_.'₁₂₃]+)")


def declarations(root: Path) -> list[tuple[Path, int, str]]:
    found: list[tuple[Path, int, str]] = []
    for path in sorted(root.glob("*.lean")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = AXIOM.match(line)
            if match:
                found.append((path, line_number, match.group(1)))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max", type=int, default=None, dest="maximum")
    args = parser.parse_args()

    found = declarations(args.root)
    print(f"axiom declarations: {len(found)}")
    by_file: dict[Path, int] = {}
    for path, line, name in found:
        by_file[path] = by_file.get(path, 0) + 1
        print(f"{path.name}:{line}: {name}")
    print("by file:")
    for path, count in sorted(by_file.items()):
        print(f"  {count:3d} {path.name}")

    if args.maximum is not None and len(found) > args.maximum:
        print(f"FAIL: {len(found)} declarations exceed --max {args.maximum}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
