# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Keep source-supply-chain controls broad when a new workspace is added."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "tools" / "audit_npm_production.mjs"
DEVELOPMENT_AUDIT_SCRIPT = ROOT / "tools" / "audit_npm_development.mjs"
DEVELOPMENT_EXCEPTIONS = (
    ROOT / "docs" / "security" / "npm-development-audit-exceptions.json"
)
REQUIREMENTS = ROOT / "requirements.txt"
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


def test_ci_runs_the_bounded_development_dependency_audit() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    source = DEVELOPMENT_AUDIT_SCRIPT.read_text(encoding="utf-8")
    exception_register = json.loads(DEVELOPMENT_EXCEPTIONS.read_text(encoding="utf-8"))

    assert "node tools/audit_npm_development.mjs" in workflow
    assert "Ratchet reviewed npm development dependency exceptions" in workflow
    assert '"--package-lock-only", "--json"' in source
    assert "review_by" in source
    assert exception_register["schema_version"] == 1
    assert len(exception_register["exceptions"]) == 1

    exception = exception_register["exceptions"][0]
    assert exception["advisory"] == "GHSA-vfj7-8cjw-p6xm"
    assert exception["advisory_package"] == "braces"
    assert exception["root_dev_dependency"] == "eslint-config-next"
    assert exception["review_by"] == "2026-11-15"
    assert set(exception["affected_packages"]) == {
        "@next/eslint-plugin-next",
        "braces",
        "eslint-config-next",
        "fast-glob",
        "micromatch",
    }


def test_python_advisory_tool_is_pinned_and_runs_in_each_ci_project() -> None:
    requirements = REQUIREMENTS.read_text(encoding="utf-8")
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "pip-audit==2.10.1" in requirements
    assert "pip-audit -r requirements.txt --strict" in workflow
    assert "Audit Python dependency resolution" in workflow
    assert "python -m pip install pip-audit==2.10.1" in workflow
    assert "pip-audit mcp --strict" in workflow
    assert "Audit MCP hub dependency resolution" in workflow


def test_development_dependency_exception_expires_before_running_npm(
    tmp_path: Path,
) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required to exercise the npm audit ratchet")
    exception_register = json.loads(DEVELOPMENT_EXCEPTIONS.read_text(encoding="utf-8"))
    exception_register["exceptions"][0]["review_by"] = "2000-01-01"
    expired = tmp_path / "expired-exception.json"
    expired.write_text(json.dumps(exception_register), encoding="utf-8")

    result = subprocess.run(
        ["node", str(DEVELOPMENT_AUDIT_SCRIPT), "--exceptions", str(expired)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "review expired on 2000-01-01" in result.stderr
