# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Tests for research/toe_census.py, the open-distance ledger."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from research import toe_census
from research.toe_census import (
    carried_laws,
    check_evidence,
    main,
    tally,
    validate,
)

REPO_ITEM = {
    "id": "example-gap",
    "kind": "constant",
    "name": "an example free constant",
    "status": "undetermined",
    "evidence": "Omega_Theory_v4.0_Technical.md#7",
    "note": "A note long enough to count as a judgement rather than a placeholder.",
}


def _write(tmp_path: pathlib.Path, items: list[dict], ratchet: dict) -> pathlib.Path:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"items": items, "ratchet": ratchet}), encoding="utf-8"
    )
    return ledger


def _run(tmp_path: pathlib.Path, items: list[dict], ratchet: dict, *extra: str) -> int:
    ledger = _write(tmp_path, items, ratchet)
    return main(["--ledger", str(ledger), "--root", str(toe_census.ROOT), *extra])


def test_the_shipped_ledger_passes() -> None:
    """The committed ledger must be clean, or the gate is red for the wrong reason."""
    assert main([]) == 0


def test_section_anchors_resolve_by_number() -> None:
    assert check_evidence("Omega_Theory_v4.0_Technical.md#7", toe_census.ROOT) == []
    assert check_evidence("Omega_Theory_v4.0_Technical.md#5", toe_census.ROOT) == []


def test_a_missing_section_anchor_is_reported() -> None:
    problems = check_evidence("Omega_Theory_v4.0_Technical.md#99", toe_census.ROOT)
    assert problems and "no heading matches" in problems[0]


def test_a_missing_declaration_is_reported() -> None:
    problems = check_evidence(
        "lean_proofs/RadialMetric.lean:noSuchDefinition", toe_census.ROOT
    )
    assert problems and "does not occur" in problems[0]


def test_a_missing_file_is_reported() -> None:
    problems = check_evidence("docs/no_such_note.md#1", toe_census.ROOT)
    assert problems and "no such file" in problems[0]


def test_validate_rejects_an_unknown_status() -> None:
    item = dict(REPO_ITEM, status="probably-fine")
    problems = validate([item], toe_census.ROOT)
    assert any("unknown status" in problem for problem in problems)


def test_validate_rejects_a_duplicate_id() -> None:
    problems = validate([REPO_ITEM, dict(REPO_ITEM)], toe_census.ROOT)
    assert any("duplicate id" in problem for problem in problems)


def test_validate_rejects_a_placeholder_note() -> None:
    problems = validate([dict(REPO_ITEM, note="todo")], toe_census.ROOT)
    assert any("note too short" in problem for problem in problems)


def test_validate_rejects_a_broken_evidence_pointer() -> None:
    problems = validate(
        [dict(REPO_ITEM, evidence="lean_proofs/Nope.lean:thing")], toe_census.ROOT
    )
    assert any("no such file" in problem for problem in problems)


def test_open_counts_exclude_postulates_and_externals() -> None:
    items = [
        dict(REPO_ITEM),
        dict(REPO_ITEM, id="a-postulate", status="postulate"),
        dict(REPO_ITEM, id="a-measured", status="external"),
        dict(REPO_ITEM, id="a-derived", status="derived"),
        dict(REPO_ITEM, id="a-chosen", status="chosen"),
    ]
    counts = tally(items)
    assert counts["open"] == 2  # undetermined + chosen
    assert counts["postulate"] == 1


def test_carried_laws_are_discovered_from_the_sources() -> None:
    laws = carried_laws(toe_census.ROOT / "lean_proofs")
    assert "law_dual_maxwell" in laws  # the maxwell bridge is still carried
    assert "law_einstein_eqs" in laws or "law_einstein_equations" in laws
    assert all(name.startswith("law_") for name in laws)


def test_a_new_open_item_above_the_ratchet_fails(tmp_path: pathlib.Path) -> None:
    ratchet = {
        "max_open": 0,
        "max_carried_laws": len(carried_laws(toe_census.ROOT / "lean_proofs")),
    }
    assert _run(tmp_path, [dict(REPO_ITEM)], ratchet) == 1


def test_a_closed_gap_below_the_ratchet_fails(tmp_path: pathlib.Path) -> None:
    """Ratchet is two-sided: closing a gap must lower the number, not leave slack."""
    ratchet = {
        "max_open": 5,
        "max_carried_laws": len(carried_laws(toe_census.ROOT / "lean_proofs")),
    }
    assert _run(tmp_path, [dict(REPO_ITEM)], ratchet) == 1


def test_a_new_carried_law_above_the_ratchet_fails(tmp_path: pathlib.Path) -> None:
    ratchet = {"max_open": 1, "max_carried_laws": 0}
    assert _run(tmp_path, [dict(REPO_ITEM)], ratchet) == 1


def test_a_consistent_ledger_passes(tmp_path: pathlib.Path) -> None:
    ratchet = {
        "max_open": 1,
        "max_carried_laws": len(carried_laws(toe_census.ROOT / "lean_proofs")),
    }
    assert _run(tmp_path, [dict(REPO_ITEM)], ratchet) == 0


@pytest.mark.parametrize("payload", ["[]", "{}", '{"items": []}', '{"items": [{}]}'])
def test_a_malformed_ledger_is_unreadable_not_failing(
    tmp_path: pathlib.Path, payload: str
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(payload, encoding="utf-8")
    assert main(["--ledger", str(ledger)]) == 2


def test_json_output_is_machine_readable(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ratchet = {
        "max_open": 1,
        "max_carried_laws": len(carried_laws(toe_census.ROOT / "lean_proofs")),
    }
    assert _run(tmp_path, [dict(REPO_ITEM)], ratchet, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["open"] == 1 and payload["problems"] == []


def test_the_gate_runs_the_ledger() -> None:
    from research.consistency_gate import gate_commands

    names = {command.name for command in gate_commands(sys.executable)}
    assert "theory ledger" in names
