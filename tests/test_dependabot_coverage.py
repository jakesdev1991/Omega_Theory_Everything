# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Regression coverage for the source-project Dependabot maintenance boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_dependabot_coverage.py"
CONFIG = ROOT / ".github" / "dependabot.yml"


def run_checker(config: Path = CONFIG) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(ROOT), "--config", str(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shipped_dependabot_config_covers_every_supported_project() -> None:
    result = run_checker()
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Dependabot coverage valid: ")
    assert result.stdout.rstrip().endswith("update entries.")


def test_missing_independent_desktop_cargo_project_fails_closed(tmp_path: Path) -> None:
    config = CONFIG.read_text(encoding="utf-8")
    desktop_cargo_stanza = """  # The desktop wrapper also has an independent Tauri/Rust dependency graph.
  - package-ecosystem: "cargo"
    directory: "/desktop/src-tauri"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10

"""
    assert desktop_cargo_stanza in config
    incomplete = tmp_path / "dependabot.yml"
    incomplete.write_text(config.replace(desktop_cargo_stanza, ""), encoding="utf-8")

    result = run_checker(incomplete)
    assert result.returncode == 1
    assert "('cargo', '/desktop/src-tauri')" in result.stderr


def test_stale_dependabot_project_fails_closed(tmp_path: Path) -> None:
    stale = tmp_path / "dependabot.yml"
    stale.write_text(
        CONFIG.read_text(encoding="utf-8")
        + """
  - package-ecosystem: "npm"
    directory: "/retired-project"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
""",
        encoding="utf-8",
    )

    result = run_checker(stale)
    assert result.returncode == 1
    assert "('npm', '/retired-project')" in result.stderr
