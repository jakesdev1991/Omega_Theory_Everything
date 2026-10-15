# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Keep the generated source-and-lockfile dependency inventory trustworthy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_dependency_inventory.py"
INVENTORY = ROOT / "docs" / "sbom" / "omega-theory-everything.cdx.json"


def run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_committed_dependency_inventory_is_current() -> None:
    result = run_generator("--check")
    assert result.returncode == 0, result.stderr
    assert "dependency inventory is current" in result.stdout


def test_inventory_declares_its_scope_and_has_expected_component_kinds() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert inventory["bomFormat"] == "CycloneDX"
    assert inventory["specVersion"] == "1.5"
    assert inventory["metadata"]["timestamp"] == "2026-10-15T00:00:00Z"

    components = inventory["components"]
    assert len(components) > 1_000
    names = {component["name"] for component in components}
    assert {"next", "mathlib4", "ethers", "actions/checkout"}.issubset(names)
    assert any(
        property_["name"] == "omega:vendored" and property_["value"] == "true"
        for component in components
        for property_ in component.get("properties", [])
    )


def test_generator_accepts_an_explicit_date_and_output_path(tmp_path: Path) -> None:
    output = tmp_path / "inventory.json"
    result = run_generator("--as-of", "2026-10-16", "--output", str(output))
    assert result.returncode == 0, result.stderr
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["metadata"]["timestamp"] == "2026-10-16T00:00:00Z"
    assert generated["serialNumber"].startswith("urn:uuid:")
