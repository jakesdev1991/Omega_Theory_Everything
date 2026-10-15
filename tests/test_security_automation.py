# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Keep source-supply-chain controls broad when a new workspace is added."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "tools" / "audit_npm_production.mjs"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_npm_production_audit_covers_every_committed_root_workspace_lockfile() -> None:
    source = AUDIT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"const WORKSPACES = \[(.*?)\];", source, re.DOTALL)
    assert match, "expected an explicit WORKSPACES list in the audit script"
    configured = set(re.findall(r'"([a-z0-9-]+)"', match.group(1)))
    discovered = {path.parent.name for path in ROOT.glob("*/package-lock.json")}
    assert configured == discovered


def test_npm_production_audit_uses_lockfiles_and_excludes_dev_dependencies_explicitly() -> (
    None
):
    source = AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert '"--omit=dev"' in source
    assert '"--package-lock-only"' in source
    assert '"--audit-level=high"' in source


def test_ci_runs_the_cross_workspace_production_dependency_audit() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "node tools/audit_npm_production.mjs" in workflow
    assert "Audit production npm dependencies across all workspaces" in workflow
