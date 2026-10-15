#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Validate and report the evidence-backed launch-readiness ledger.

This script does not approve a deployment.  It makes the opposite failure mode
harder: an incomplete legal, security, operational, or deployment condition
cannot be presented as a green launch merely because a software test passed.

Examples:

    python launch/check_readiness.py --validate
    python launch/check_readiness.py --report --scope testnet-pilot
    python launch/check_readiness.py --require mainnet  # exits 1 while blocked

The ledger deliberately contains no credentials, wallet addresses, private
incident contacts, deployment manifests, or legal advice.  Evidence references
must be public repository paths or HTTPS URLs; sensitive evidence belongs in an
access-controlled register maintained by the responsible owner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "launch" / "readiness-gates.json"

ALLOWED_SCOPES = frozenset({"public-manuscript", "testnet-pilot", "mainnet"})
ALLOWED_STATUSES = frozenset(
    {"complete", "in_progress", "blocked", "decision_required"}
)
ALLOWED_RISKS = frozenset({"low", "medium", "high", "critical"})
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REQUIRED_GATE_IDS = frozenset(
    {
        "public_manuscript_access",
        "provenance_and_rights_review",
        "source_inventory_and_license_review",
        "security_visibility",
        "production_dependency_audit",
        "independent_security_review",
        "evm_sepolia_drill",
        "twc_devnet_drill",
        "amity_testnet_infrastructure",
        "operator_incident_ownership",
        "mainnet_venue_decision",
        "legal_and_compliance_review",
        "care_mainnet_safety",
    }
)


def load_ledger(path: Path) -> dict[str, Any]:
    """Load one JSON ledger, preserving a useful error for callers."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read ledger {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"ledger {path} must be a JSON object")
    return data


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _local_evidence_exists(root: Path, evidence: str) -> bool:
    """Return whether a repository-relative evidence reference is present."""
    if evidence.startswith("https://"):
        return True
    candidate = Path(evidence)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    return (root / candidate).exists()


def validate_ledger(ledger: dict[str, Any], root: Path) -> list[str]:
    """Return every schema/evidence problem instead of stopping at the first."""
    errors: list[str] = []

    if ledger.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not _is_nonempty_string(ledger.get("status_date")) or not DATE_RE.fullmatch(
        ledger.get("status_date", "")
    ):
        errors.append("status_date must use YYYY-MM-DD")
    if not _is_nonempty_string(ledger.get("scope_note")):
        errors.append("scope_note must be a non-empty string")

    scope_descriptions = ledger.get("scopes")
    if not isinstance(scope_descriptions, dict):
        errors.append("scopes must be an object")
    else:
        missing_scopes = ALLOWED_SCOPES.difference(scope_descriptions)
        extra_scopes = set(scope_descriptions).difference(ALLOWED_SCOPES)
        if missing_scopes:
            errors.append(f"scopes missing descriptions for: {sorted(missing_scopes)}")
        if extra_scopes:
            errors.append(f"scopes contains unknown entries: {sorted(extra_scopes)}")
        for scope, description in scope_descriptions.items():
            if not _is_nonempty_string(description):
                errors.append(f"scope {scope!r} must have a non-empty description")

    gates = ledger.get("gates")
    if not isinstance(gates, list) or not gates:
        return [*errors, "gates must be a non-empty list"]

    seen_ids: set[str] = set()
    represented_scopes: set[str] = set()
    for index, gate in enumerate(gates):
        label = f"gates[{index}]"
        if not isinstance(gate, dict):
            errors.append(f"{label} must be an object")
            continue

        gate_id = gate.get("id")
        if not isinstance(gate_id, str) or not gate_id.strip():
            errors.append(f"{label}.id must be a non-empty string")
        elif gate_id in seen_ids:
            errors.append(f"duplicate gate id: {gate_id}")
        else:
            seen_ids.add(gate_id)

        for field in ("title", "owner_role", "next_action", "last_reviewed"):
            if not _is_nonempty_string(gate.get(field)):
                errors.append(f"{label}.{field} must be a non-empty string")

        reviewed = gate.get("last_reviewed")
        if (
            isinstance(reviewed, str)
            and reviewed.strip()
            and not DATE_RE.fullmatch(reviewed)
        ):
            errors.append(f"{label}.last_reviewed must use YYYY-MM-DD")

        status = gate.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{label}.status must be one of {sorted(ALLOWED_STATUSES)}")

        risk = gate.get("risk")
        if risk not in ALLOWED_RISKS:
            errors.append(f"{label}.risk must be one of {sorted(ALLOWED_RISKS)}")

        scopes = gate.get("scopes")
        if (
            not isinstance(scopes, list)
            or not scopes
            or not all(isinstance(scope, str) for scope in scopes)
        ):
            errors.append(f"{label}.scopes must be a non-empty list of strings")
        else:
            unknown_scopes = set(scopes).difference(ALLOWED_SCOPES)
            if unknown_scopes:
                errors.append(
                    f"{label}.scopes contains unknown values: {sorted(unknown_scopes)}"
                )
            represented_scopes.update(scopes)

        evidence = gate.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(_is_nonempty_string(item) for item in evidence)
        ):
            errors.append(f"{label}.evidence must be a non-empty list of strings")
        else:
            for item in evidence:
                if not _local_evidence_exists(root, item):
                    errors.append(
                        f"{label}.evidence reference does not exist or is unsafe: {item}"
                    )

        blocker = gate.get("blocker")
        if status == "complete":
            if blocker not in (None, ""):
                errors.append(f"{label}.blocker must be empty when status is complete")
        elif not _is_nonempty_string(blocker):
            errors.append(f"{label}.blocker must explain a non-complete status")

    missing_gate_ids = REQUIRED_GATE_IDS.difference(seen_ids)
    if missing_gate_ids:
        errors.append(f"required gate ids missing: {sorted(missing_gate_ids)}")
    unexpected_gate_ids = seen_ids.difference(REQUIRED_GATE_IDS)
    if unexpected_gate_ids:
        errors.append(f"unrecognized gate ids: {sorted(unexpected_gate_ids)}")
    missing_coverage = ALLOWED_SCOPES.difference(represented_scopes)
    if missing_coverage:
        errors.append(f"no gates cover scopes: {sorted(missing_coverage)}")

    return errors


def gates_for_scope(
    gates: Iterable[dict[str, Any]], scope: str
) -> list[dict[str, Any]]:
    """Select gates affecting one declared readiness scope."""
    if scope == "all":
        return list(gates)
    return [gate for gate in gates if scope in gate["scopes"]]


def print_report(ledger: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    """Print a concise public report and return its selected gates."""
    selected = gates_for_scope(ledger["gates"], scope)
    scope_label = "all scopes" if scope == "all" else scope
    print(f"Launch-readiness ledger — {scope_label} — reviewed {ledger['status_date']}")
    print("This is a blocker report, not legal, security, or deployment approval.\n")
    for gate in selected:
        marker = "PASS" if gate["status"] == "complete" else "BLOCKED"
        print(f"[{marker}] {gate['id']}: {gate['title']}")
        print(
            f"  status={gate['status']} risk={gate['risk']} owner={gate['owner_role']}"
        )
        if gate["status"] != "complete":
            print(f"  blocker: {gate['blocker']}")
        print(f"  next: {gate['next_action']}")
    complete = sum(gate["status"] == "complete" for gate in selected)
    print(f"\n{complete}/{len(selected)} scoped gates complete.")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path, default=DEFAULT_LEDGER, help="ledger JSON path"
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root for evidence paths"
    )
    parser.add_argument(
        "--validate", action="store_true", help="validate the ledger and exit"
    )
    parser.add_argument(
        "--report", action="store_true", help="print gates for the selected scope"
    )
    parser.add_argument(
        "--require",
        metavar="SCOPE",
        choices=sorted(ALLOWED_SCOPES),
        help="exit 1 unless every gate in this scope is complete",
    )
    parser.add_argument(
        "--scope",
        choices=["all", *sorted(ALLOWED_SCOPES)],
        default="all",
        help="scope used with --report (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ledger = load_ledger(args.ledger)
    except ValueError as error:
        print(f"READINESS LEDGER ERROR: {error}", file=sys.stderr)
        return 2

    errors = validate_ledger(ledger, args.root)
    if errors:
        print("READINESS LEDGER INVALID:", file=sys.stderr)
        for message in errors:
            print(f"- {message}", file=sys.stderr)
        return 2

    if args.validate:
        print(f"Launch-readiness ledger valid: {len(ledger['gates'])} gates.")

    if args.report or args.require:
        scope = args.require or args.scope
        selected = print_report(ledger, scope)
        if args.require:
            blocked = [gate for gate in selected if gate["status"] != "complete"]
            if blocked:
                print(
                    f"\nREQUIREMENT NOT MET: {args.require} remains blocked by "
                    f"{', '.join(gate['id'] for gate in blocked)}.",
                    file=sys.stderr,
                )
                return 1
            print(f"\nREQUIREMENT MET: every {args.require} gate is complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
