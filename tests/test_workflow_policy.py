# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Keep hosted CI images and core GitHub actions deliberately pinned."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

CORE_ACTIONS = re.compile(
    r"^\s*(?:-\s*)?uses:\s*"
    r"(actions/(?:checkout|setup-node|setup-python|upload-artifact|download-artifact))"
    r"@([^\s#]+)",
    re.MULTILINE,
)


def test_workflows_do_not_use_the_implicit_ubuntu_latest_image() -> None:
    """A GitHub image migration must be an intentional, reviewed change."""
    floating_runners: list[str] = []
    for workflow in WORKFLOWS:
        for line_number, line in enumerate(
            workflow.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if re.match(r"^\s*runs-on:\s*ubuntu-latest\s*(?:#.*)?$", line):
                floating_runners.append(f"{workflow.relative_to(ROOT)}:{line_number}")

    assert not floating_runners, (
        "Pin Linux runner images to a specific Ubuntu release; found ubuntu-latest at "
        f"{floating_runners}"
    )


def test_core_github_actions_are_pinned_to_full_commit_shas() -> None:
    """Prevent tag movement from changing checkout, setup, or artifact behavior."""
    pinned_actions: list[tuple[str, str]] = []
    for workflow in WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        pinned_actions.extend(CORE_ACTIONS.findall(source))

    assert pinned_actions, "expected core GitHub Actions in workflow files"
    invalid_refs = [
        f"{action}@{reference}"
        for action, reference in pinned_actions
        if re.fullmatch(r"[0-9a-f]{40}", reference) is None
    ]
    assert not invalid_refs, (
        f"Core GitHub Actions must use immutable full SHAs: {invalid_refs}"
    )
