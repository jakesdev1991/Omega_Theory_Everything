#!/usr/bin/env python3
"""Keep the plain-text mirrors in ``txt_proofs/`` byte-identical to ``lean_proofs/``.

``PROOF_AUDIT.md`` records that the mirrors are regenerated from the Lean
sources rather than edited by hand, and roadmap Phase 5 item 3 calls for that
regeneration. This module is both halves of it:

* ``--write`` copies every Lean module to its ``.txt`` companion, creating
  mirrors that do not exist yet and refreshing the ones that drifted.
* no flag checks that the working tree is already in sync and exits non-zero
  with the list of offenders otherwise, which is what CI runs.

The Lake build file is not a proof module and has no mirror. Orphans -- text
files with no Lean source -- are reported but never deleted: removing an
archived volume's companion is a decision for the owner, not for a sync.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lean_source import lean_sources

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
TXT_DIR = REPO_ROOT / "txt_proofs"
NO_MIRROR = {"lakefile.lean"}


def expected_mirrors(lean_dir: Path) -> dict[str, Path]:
    """Map ``txt_proofs`` relative path to its Lean source."""
    return {
        path.relative_to(lean_dir).with_suffix(".txt").as_posix(): path
        for path in lean_sources(lean_dir)
        if path.name not in NO_MIRROR
    }


def plan(lean_dir: Path, txt_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (missing, stale, orphan) mirror paths relative to ``txt_dir``."""
    wanted = expected_mirrors(lean_dir)
    missing: list[str] = []
    stale: list[str] = []
    for name, source in sorted(wanted.items()):
        mirror = txt_dir / name
        if not mirror.exists():
            missing.append(name)
        elif mirror.read_bytes() != source.read_bytes():
            stale.append(name)
    present = {path.name for path in txt_dir.glob("*.txt")}
    orphan = sorted(present - set(wanted))
    return missing, stale, orphan


def sync(lean_dir: Path, txt_dir: Path) -> list[str]:
    """Write every mirror that is missing or stale. Returns what changed."""
    wanted = expected_mirrors(lean_dir)
    changed: list[str] = []
    for name, source in sorted(wanted.items()):
        mirror = txt_dir / name
        if mirror.exists() and mirror.read_bytes() == source.read_bytes():
            continue
        txt_dir.mkdir(parents=True, exist_ok=True)
        mirror.write_bytes(source.read_bytes())
        changed.append(name)
    return changed


def check(lean_dir: Path, txt_dir: Path) -> list[str]:
    """Human-readable problems; empty means the mirrors are in sync."""
    missing, stale, orphan = plan(lean_dir, txt_dir)
    problems = [f"{name}: mirror missing" for name in missing]
    problems += [f"{name}: mirror out of date" for name in stale]
    problems += [f"{name}: orphan (no Lean source; not deleted)" for name in orphan]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lean-dir", type=Path, default=ROOT, help="Lean source directory"
    )
    parser.add_argument(
        "--txt-dir", type=Path, default=TXT_DIR, help="plain-text mirror directory"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="create or refresh mirrors instead of only checking them",
    )
    args = parser.parse_args(argv)

    if args.write:
        changed = sync(args.lean_dir, args.txt_dir)
        print(f"txt mirrors: {len(changed)} written")
        for name in changed:
            print(f"  {name}")
        problems = check(args.lean_dir, args.txt_dir)
        if problems:
            print("remaining problems (not auto-fixed):")
            for problem in problems:
                print(f"  {problem}")
        return 0

    problems = check(args.lean_dir, args.txt_dir)
    if problems:
        print(f"txt mirrors: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem}")
        print(
            "\nRegenerate them with:\n  python3 lean_proofs/sync_txt_mirrors.py --write"
        )
    else:
        mirrors = expected_mirrors(args.lean_dir)
        print(f"txt mirrors: {len(mirrors)} files in sync")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
