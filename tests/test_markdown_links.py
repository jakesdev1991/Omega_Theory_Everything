# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Offline gate for relative links in the tracked markdown.

The Documentation Check job runs ``markdown-link-check`` over every ``*.md``
file, but that is a network check: it can only report on hosts it can reach,
and it fails for reasons that have nothing to do with this repository (a 403
from a bot-protected host, a runner-side timeout). Half of its input, the
in-repo relative links, needs no network at all and is worth checking where the
verdict is deterministic.

This walks the same markdown set, resolves every relative link against the file
that contains it, and fails when the target does not exist. Anchors and query
strings are dropped before resolving, and links to directories count as valid.
"""

from __future__ import annotations

import pathlib
import re
from urllib.parse import unquote

ROOT = pathlib.Path(__file__).resolve().parents[1]

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"```.*?```", re.DOTALL)

# Directories that are generated, vendored, or build output. ``web/content`` in
# particular is produced by ``npm run sync:docs`` and is gitignored, so its
# relative links point at repository-root paths that do not exist under it.
SKIP_DIRS = {
    ".git",
    ".lake",
    ".next",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "out",
    "target",
}
SKIP_PATHS = {("web", "content")}
SKIP_SCHEMES = ("http://", "https://", "mailto:", "#", "tel:", "ftp:", "data:")


def _markdown_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in ROOT.rglob("*.md"):
        parts = path.relative_to(ROOT).parts
        if SKIP_DIRS.intersection(parts):
            continue
        if any(parts[:2] == skip for skip in SKIP_PATHS):
            continue
        files.append(path)
    return sorted(files)


def _relative_links(text: str) -> list[str]:
    return [target for target in LINK.findall(FENCE.sub("", text))]


def test_every_markdown_file_is_reachable() -> None:
    """Guard the walk itself: an empty set would make the check vacuous."""
    files = _markdown_files()
    assert len(files) > 20, f"expected the repository's markdown, found {len(files)}"
    assert ROOT / "README.md" in files


def test_relative_markdown_links_resolve() -> None:
    """A relative link must point at something that exists in the checkout."""
    missing: list[str] = []
    checked = 0
    for path in _markdown_files():
        for target in _relative_links(path.read_text(encoding="utf-8")):
            if target.startswith(SKIP_SCHEMES):
                continue
            relative = unquote(target.split("#")[0].split("?")[0])
            if not relative:
                continue
            checked += 1
            resolved = pathlib.Path((path.parent / relative).resolve())
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")

    assert checked > 100, f"expected the repository's cross-links, found {checked}"
    assert not missing, (
        "markdown links point at paths that do not exist:\n" + "\n".join(missing)
    )
