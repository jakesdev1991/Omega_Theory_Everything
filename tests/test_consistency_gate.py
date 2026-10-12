# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""CI gate for research/consistency_gate.py (roadmap Phase 7)."""

from __future__ import annotations

import io
import pathlib
import sys

from research.consistency_gate import (
    ROOT,
    GateCommand,
    gate_commands,
    main,
    run_gate,
    summarize,
)


def test_gate_commands_reference_files_that_exist() -> None:
    for command in gate_commands(sys.executable):
        if command.argv[0] == "bash" or command.requires:
            continue  # a shell one-liner, or an external tool rather than a repo script
        if command.argv[1] == "-m":
            assert command.argv[2] == "pytest"
            continue
        target = ROOT / command.cwd / command.argv[1]
        assert target.exists(), f"{command.name}: {target} does not exist"


def test_gate_covers_every_check_script() -> None:
    names = {command.name for command in gate_commands(sys.executable)}
    for expected in (
        "derivation checks",
        "covariant checks",
        "TFIM checks",
        "LaTeX citations",
        "claim scan",
        "axiom audit",
        "vacuity ratchets",
        "declaration parity",
        "text mirrors",
        "pytest suite",
    ):
        assert expected in names


def test_run_gate_records_pass_fail_and_skip() -> None:
    commands = [
        GateCommand("ok", (sys.executable, "-c", "print('fine')"), ".", "passes"),
        GateCommand("bad", (sys.executable, "-c", "raise SystemExit(3)"), ".", "fails"),
        GateCommand(
            "skipped",
            ("not-a-real-tool",),
            ".",
            "missing tool",
            requires="not-a-real-tool-xyz",
        ),
    ]
    results = run_gate(commands, ROOT)
    assert [r.status for r in results] == ["pass", "fail", "skipped"]
    assert results[0].output == "fine"
    assert results[1].failed


def test_run_gate_quick_skips_slow_commands() -> None:
    commands = [
        GateCommand(
            "slow",
            (sys.executable, "-c", "raise SystemExit(1)"),
            ".",
            "slow",
            slow=True,
        ),
        GateCommand("fast", (sys.executable, "-c", "pass"), ".", "fast"),
    ]
    results = run_gate(commands, ROOT, quick=True)
    assert [r.status for r in results] == ["skipped", "pass"]


def test_summarize_reports_failure_and_skips() -> None:
    commands = [
        GateCommand("ok", (sys.executable, "-c", "pass"), ".", "passes"),
        GateCommand("bad", (sys.executable, "-c", "raise SystemExit(1)"), ".", "fails"),
    ]
    results = run_gate(commands, ROOT)
    stream = io.StringIO()
    assert summarize(results, stream) == 1
    text = stream.getvalue()
    assert "[FAIL] bad" in text
    assert "1 passed, 1 failed, 0 skipped." in text
    assert "CI-only" in text

    stream = io.StringIO()
    assert summarize(run_gate(commands[:1], ROOT), stream) == 0
    assert "1 passed, 0 failed, 0 skipped." in stream.getvalue()


def test_skipped_commands_do_not_fail_the_gate() -> None:
    commands = [
        GateCommand(
            "tool", ("nope",), ".", "needs a tool", requires="not-a-real-tool-xyz"
        ),
    ]
    stream = io.StringIO()
    assert summarize(run_gate(commands, ROOT), stream) == 0
    assert "0 passed, 0 failed, 1 skipped." in stream.getvalue()
    assert "install the pinned requirements.txt" in stream.getvalue()


def test_list_prints_commands_without_running_them() -> None:
    stream = io.StringIO()
    saved = sys.stdout
    sys.stdout = stream
    try:
        assert main(["--list"]) == 0
    finally:
        sys.stdout = saved
    text = stream.getvalue()
    assert "research/verify_derivations.py" in text
    assert "audit_vacuity.py" in text


def test_gate_script_declares_the_ci_only_gap() -> None:
    source = (ROOT / "research" / "consistency_gate.py").read_text(encoding="utf-8")
    assert "lake build ToE" in source
    assert "CI-only" in source
    assert "Phase 7" in source


def test_scripts_run_from_the_right_directory() -> None:
    for command in gate_commands(sys.executable):
        expected = "lean_proofs" if command.cwd == "lean_proofs" else "."
        assert command.cwd == expected
        if command.cwd == "lean_proofs":
            assert pathlib.Path(ROOT, "lean_proofs", command.argv[1]).is_file()
