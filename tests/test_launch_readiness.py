# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Regression tests for the public launch-readiness control plane."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "launch" / "check_readiness.py"
LEDGER = ROOT / "launch" / "readiness-gates.json"


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shipped_readiness_ledger_is_valid() -> None:
    result = run_checker("--validate")
    assert result.returncode == 0, result.stderr
    assert "Launch-readiness ledger valid: 13 gates." in result.stdout


def test_mainnet_is_explicitly_blocked_until_all_gates_are_complete() -> None:
    result = run_checker("--require", "mainnet")
    assert result.returncode == 1
    assert "REQUIREMENT NOT MET: mainnet remains blocked" in result.stderr
    assert "legal_and_compliance_review" in result.stderr
    assert "care_mainnet_safety" in result.stderr


def test_public_ledger_has_one_unique_gate_for_each_required_control() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    gates = ledger["gates"]
    gate_ids = [gate["id"] for gate in gates]
    assert len(gate_ids) == len(set(gate_ids)) == 13
    assert {"public-manuscript", "testnet-pilot", "mainnet"} == set(ledger["scopes"])
    assert all(gate["evidence"] for gate in gates)
    assert all(gate["owner_role"] for gate in gates)
    assert all(gate["last_reviewed"] == ledger["status_date"] for gate in gates)


def test_unsafe_or_missing_evidence_fails_closed(tmp_path: Path) -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    ledger["gates"][0]["evidence"] = ["../not-public-evidence.txt"]
    malformed = tmp_path / "malformed-ledger.json"
    malformed.write_text(json.dumps(ledger), encoding="utf-8")

    result = run_checker("--ledger", str(malformed), "--root", str(ROOT), "--validate")
    assert result.returncode == 2
    assert "evidence reference does not exist or is unsafe" in result.stderr
