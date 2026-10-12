# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""One command that runs every local check the repository can run on its own.

Phase 7 of docs/research/CLAIM_TIGHTENING_ROADMAP.md asks for a single command
that runs every check, so a reviewer does not have to remember the list. This is
that command.

It runs, in order:

* the three derivation-check scripts (`verify_derivations.py`,
  `verify_covariant.py`, `verify_tfim.py`) and the two LaTeX gates
  (`check_latex_citations.py`, `check_latex_structure.py`);
* the claim scanner;
* the Lean-side static gates in `lean_proofs/`: the sorry/trivial grep, the
  axiom audit, the vacuity ratchets, the declaration-parity gate, the
  text-mirror gate and the statement audit of the physics volumes;
* the whole pytest suite;
* `ruff check`, `ruff format --check` and `mypy`, when those executables are
  installed (they are the pinned versions in `requirements.txt`; CI runs them
  from there, so a missing tool here is reported as skipped, not as a pass).

It cannot run three of the CI checks: `lake build ToE`, the transitive
kernel-axiom audit, and the `lean_proofs/check_mathlib_names.py` name check.
All three need the pinned Lean sources -- the toolchain and the mathlib
checkout it builds against -- whose download hosts are not reachable from every
environment. That part of the verification is CI-only and this script says so
in its summary rather than implying the tree is fully verified.

Nothing here is a substitute for reading a diff. The gate says a set of
commands exited 0.

Run: python research/consistency_gate.py [--quick] [--list]
Exit status is 0 only if every command that ran exited 0.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import IO

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateCommand:
    """One command in the gate, with the reason it is there."""

    name: str
    argv: tuple[str, ...]
    cwd: str  # relative to the repository root
    purpose: str
    requires: str | None = None  # an executable that must exist, else skipped
    slow: bool = False


def gate_commands(python: str) -> list[GateCommand]:
    """The commands the gate runs, in order."""
    return [
        GateCommand(
            "derivation checks",
            (python, "research/verify_derivations.py"),
            ".",
            "every machine-checked core-note identity",
        ),
        GateCommand(
            "covariant checks",
            (python, "research/verify_covariant.py"),
            ".",
            "scalar equation, FLRW equations, no-hair identity",
        ),
        GateCommand(
            "TFIM checks",
            (python, "research/verify_tfim.py"),
            ".",
            "Schwarzschild column and one dense exact-diagonalization row",
        ),
        GateCommand(
            "LaTeX citations",
            (python, "research/check_latex_citations.py"),
            ".",
            "every declaration cited by a volume exists",
        ),
        GateCommand(
            "LaTeX structure",
            (python, "research/check_latex_structure.py"),
            ".",
            "braces, environment nesting and math delimiters in the volumes",
        ),
        GateCommand(
            "claim scan",
            (python, "research/claim_scan.py"),
            ".",
            "unhedged claim language in the notes, volumes and proofs",
        ),
        GateCommand(
            "Lean sorry/trivial grep",
            (
                "bash",
                "-c",
                r"if grep -rnE '(^|[^[:alnum:]_])(sorry|admit|native_decide|sorryAx)([^[:alnum:]_]|$)' "
                r"--include='*.lean' lean_proofs | grep -vE '^\s*--'; then exit 1; fi",
            ),
            ".",
            "no sorry, admit, native_decide or sorryAx in the Lean sources",
        ),
        GateCommand(
            "axiom audit",
            (python, "audit_axioms.py", "--max", "0"),
            "lean_proofs",
            "no axiom, opaque or unsafe declarations",
        ),
        GateCommand(
            "vacuity ratchets",
            (
                python,
                "audit_vacuity.py",
                "--max-misleading",
                "0",
                "--max-axiom-names",
                "0",
                "--max-unit-stubs",
                "0",
                "--max-unit-types",
                "0",
                "--max-decorative-hypotheses",
                "0",
                "--max-trivial-hypotheses",
                "0",
                "--max-identity-proofs",
                "0",
                "--max-trivial-proofs",
                "0",
                "--alias-baseline",
                "legacy_statement_aliases.json",
            ),
            "lean_proofs",
            "all eight static ratchets plus the legacy-alias baseline at zero",
        ),
        GateCommand(
            "declaration parity",
            (python, "check_declaration_parity.py"),
            "lean_proofs",
            "no Lean code changed under a comment-only edit",
        ),
        GateCommand(
            "text mirrors",
            (python, "sync_txt_mirrors.py"),
            "lean_proofs",
            "txt_proofs/ is byte-identical to lean_proofs/",
        ),
        GateCommand(
            "statement audit",
            (python, "check_statement_audit.py", "--repo", "."),
            "lean_proofs",
            "every physics-sounding volume declaration has an audit row"
            " (witness/conditional/renamed/carried, carried count capped)",
        ),
        GateCommand(
            "cpp pacer suite",
            ("bash", "cpp/build.sh", "--quick"),
            ".",
            "C++23 CBwK shadow-price pacer build and test suite",
            requires="g++",
            slow=True,
        ),
        GateCommand(
            "omni-bridge cpp suite",
            ("bash", "omni-bridge/cpp/build.sh", "--quick"),
            ".",
            "C++23 Omni-Bridge control-boundary acceptance suite",
            requires="g++",
            slow=True,
        ),
        GateCommand(
            "ai-governor suite",
            ("bash", "ai-governor/build.sh"),
            ".",
            "C++23 PoUW governor build, FIPS 180-4 SHA-256 self-check and smoke run",
            requires="g++",
            slow=True,
        ),
        GateCommand(
            "pytest suite",
            (python, "-m", "pytest", "-q"),
            ".",
            "every script check plus the source and build audits",
            slow=True,
        ),
        GateCommand(
            "ruff check",
            ("ruff", "check", "."),
            ".",
            "lint",
            requires="ruff",
        ),
        GateCommand(
            "ruff format",
            ("ruff", "format", "--check", "."),
            ".",
            "formatting",
            requires="ruff",
        ),
        GateCommand(
            "mypy",
            ("mypy", ".", "--ignore-missing-imports"),
            ".",
            "types",
            requires="mypy",
        ),
    ]


@dataclass(frozen=True)
class GateResult:
    command: GateCommand
    status: str  # "pass" | "fail" | "skipped"
    output: str

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def run_gate(
    commands: list[GateCommand],
    root: pathlib.Path = ROOT,
    quick: bool = False,
) -> list[GateResult]:
    """Run each command; capture its output; skip missing tooling."""
    results: list[GateResult] = []
    for command in commands:
        if quick and command.slow:
            results.append(GateResult(command, "skipped", "skipped by --quick"))
            continue
        if command.requires and shutil.which(command.requires) is None:
            results.append(
                GateResult(command, "skipped", f"{command.requires} is not on PATH")
            )
            continue
        completed = subprocess.run(
            command.argv,
            cwd=root / command.cwd,
            capture_output=True,
            text=True,
        )
        output = (completed.stdout + completed.stderr).strip()
        results.append(
            GateResult(command, "pass" if completed.returncode == 0 else "fail", output)
        )
    return results


def summarize(results: list[GateResult], stream: IO[str] = sys.stdout) -> int:
    """Print one line per command and the tail of any failure."""
    width = max(len(r.command.name) for r in results)
    for result in results:
        label = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[result.status]
        print(
            f"[{label}] {result.command.name.ljust(width)}  {result.command.purpose}",
            file=stream,
        )
    for result in results:
        if result.failed:
            print(f"\n--- {result.command.name} failed ---", file=stream)
            print("\n".join(result.output.splitlines()[-25:]), file=stream)
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.failed]
    if skipped:
        print(
            f"\nskipped: {', '.join(r.command.name for r in skipped)}"
            " (install the pinned requirements.txt to run these here)",
            file=stream,
        )
    print(
        f"\n{len(results) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
        f"{len(skipped)} skipped.",
        file=stream,
    )
    print(
        "CI-only and therefore not run here: `lake build ToE`, the transitive kernel-axiom "
        "audit and the cited-name check, which need the pinned Lean sources.",
        file=stream,
    )
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the pytest suite")
    parser.add_argument(
        "--root", type=pathlib.Path, default=ROOT, help="repository root"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the commands, run nothing"
    )
    args = parser.parse_args(argv)

    commands = gate_commands(sys.executable)
    if args.list:
        for command in commands:
            print(f"{' '.join(command.argv)}\n    ({command.cwd})  {command.purpose}")
        return 0
    return summarize(run_gate(commands, args.root, args.quick))


if __name__ == "__main__":
    sys.exit(main())
