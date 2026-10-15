#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Fail closed when Dependabot misses a top-level dependency project.

Dependabot's YAML is deliberately small and uses only the `package-ecosystem`,
`directory`, `schedule.interval`, and `open-pull-requests-limit` fields.  This
checker reads that restricted shape with the standard library so that a new
npm lockfile, Python project, Cargo root, or workflow cannot quietly become a
maintenance blind spot.

It validates configuration coverage; it does not prove that GitHub has enabled
Dependabot alerts or that a human has reviewed its pull requests.  Those are
separate operational gates in launch/readiness-gates.json.

Run from the repository root:

    python tools/check_dependabot_coverage.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / ".github" / "dependabot.yml"
IGNORED_SOURCE_PARTS = frozenset({".git", "node_modules", "target", "vendor", ".venv"})

ENTRY_RE = re.compile(
    r'^\s*-\s*package-ecosystem:\s*["\'](?P<value>[^"\']+)["\']\s*(?:#.*)?$'
)
FIELD_RE = re.compile(
    r"^\s*(?P<key>directory|interval|open-pull-requests-limit):\s*"
    r"(?P<value>[^#]+?)\s*(?:#.*)?$"
)


@dataclass(frozen=True, order=True)
class UpdateEntry:
    """The subset of one Dependabot update stanza this repository relies on."""

    ecosystem: str
    directory: str
    interval: str
    limit: int


def _unquote(value: str) -> str:
    """Return a scalar in the deliberately restricted Dependabot configuration."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_entries(path: Path) -> list[UpdateEntry]:
    """Parse the small, reviewed configuration subset without a YAML dependency."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read Dependabot config {path}: {error}") from error

    raw_entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line_number, line in enumerate(lines, start=1):
        entry_match = ENTRY_RE.match(line)
        if entry_match:
            if current is not None:
                raw_entries.append(current)
            current = {"package-ecosystem": entry_match.group("value")}
            continue

        field_match = FIELD_RE.match(line)
        if field_match and current is not None:
            current[field_match.group("key")] = _unquote(field_match.group("value"))
            continue

        if line.lstrip().startswith("-") and current is not None:
            raise ValueError(
                f"unsupported list item in Dependabot stanza near line {line_number}; "
                "use the reviewed scalar-only shape"
            )

    if current is not None:
        raw_entries.append(current)
    if not raw_entries:
        raise ValueError("Dependabot config contains no package-ecosystem entries")

    entries: list[UpdateEntry] = []
    for index, raw in enumerate(raw_entries, start=1):
        missing = {
            "package-ecosystem",
            "directory",
            "interval",
            "open-pull-requests-limit",
        }.difference(raw)
        if missing:
            raise ValueError(
                f"Dependabot update entry {index} is missing fields: {sorted(missing)}"
            )
        try:
            limit = int(raw["open-pull-requests-limit"])
        except ValueError as error:
            raise ValueError(
                f"Dependabot update entry {index} has a non-integer "
                "open-pull-requests-limit"
            ) from error
        entries.append(
            UpdateEntry(
                ecosystem=raw["package-ecosystem"],
                directory=raw["directory"],
                interval=raw["interval"],
                limit=limit,
            )
        )
    return entries


def dependabot_directory(root: Path, directory: Path) -> str:
    """Render one repository directory in Dependabot's leading-slash syntax."""
    relative = directory.resolve().relative_to(root.resolve()).as_posix()
    return "/" if relative == "." else f"/{relative}"


def source_files(root: Path, filename: str) -> set[Path]:
    """Discover source project files without walking generated/vendor trees."""
    return {
        path
        for path in root.rglob(filename)
        if not IGNORED_SOURCE_PARTS.intersection(path.relative_to(root).parts)
    }


def top_level_cargo_project_dirs(root: Path) -> set[Path]:
    """Find Cargo project roots, not each crate nested below a workspace root."""
    manifests = {path.parent.resolve() for path in source_files(root, "Cargo.toml")}
    return {
        directory
        for directory in manifests
        if not any(parent in manifests for parent in directory.parents)
    }


def expected_coverage(root: Path) -> set[tuple[str, str]]:
    """Derive every source project Dependabot must be configured to maintain."""
    expected: set[tuple[str, str]] = set()

    if (root / ".github" / "workflows").is_dir():
        expected.add(("github-actions", "/"))
    if (root / "requirements.txt").is_file():
        expected.add(("pip", "/"))

    for pyproject in source_files(root, "pyproject.toml"):
        expected.add(("pip", dependabot_directory(root, pyproject.parent)))
    for lockfile in source_files(root, "package-lock.json"):
        expected.add(("npm", dependabot_directory(root, lockfile.parent)))
    for directory in top_level_cargo_project_dirs(root):
        expected.add(("cargo", dependabot_directory(root, directory)))

    return expected


def validate(entries: list[UpdateEntry], root: Path) -> list[str]:
    """Return every coverage/configuration error rather than stopping early."""
    errors: list[str] = []
    pairs = [(entry.ecosystem, entry.directory) for entry in entries]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    if duplicates:
        errors.append(f"duplicate Dependabot update entries: {duplicates}")

    for entry in entries:
        if entry.interval != "weekly":
            errors.append(
                f"{entry.ecosystem} {entry.directory} must use a weekly schedule, "
                f"not {entry.interval!r}"
            )
        if entry.limit < 1:
            errors.append(
                f"{entry.ecosystem} {entry.directory} must allow at least one open PR"
            )

    configured = set(pairs)
    expected = expected_coverage(root)
    missing = sorted(expected.difference(configured))
    if missing:
        errors.append(f"Dependabot coverage missing: {missing}")
    unexpected = sorted(configured.difference(expected))
    if unexpected:
        errors.append(
            "Dependabot configuration has no matching discovered source project: "
            f"{unexpected}"
        )

    supported_ecosystems = {"github-actions", "pip", "npm", "cargo"}
    unsupported = sorted(
        entry.ecosystem
        for entry in entries
        if entry.ecosystem not in supported_ecosystems
    )
    if unsupported:
        errors.append(f"unsupported Dependabot ecosystems in checker: {unsupported}")

    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Dependabot YAML path"
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root to inspect"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        entries = parse_entries(args.config)
        errors = validate(entries, args.root)
    except ValueError as exc:
        print(f"Dependabot coverage invalid: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("Dependabot coverage invalid:", file=sys.stderr)
        for message in errors:
            print(f"- {message}", file=sys.stderr)
        return 1

    print(f"Dependabot coverage valid: {len(entries)} update entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
