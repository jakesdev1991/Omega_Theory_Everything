#!/usr/bin/env python3
"""Build the project, then inspect transitive axioms of selected key declarations.

This is deliberately fail-closed: no Lake, a failed build, missing/duplicate
reports, or unexpected axioms are errors. It is not a semantic non-vacuity
checker and does not claim coverage of declarations outside the target list.
Only Lean's standard logical foundations are permitted, not custom postulates
or the proof-hole axiom. The source audits remain complementary.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
NAME = re.compile(NAME_PATTERN + r"\Z")
REPORT = re.compile(
    rf"'(?P<name>{NAME_PATTERN})'\s+"
    r"(?:depends on axioms:\s*\[(?P<axioms>[^\]]*)\]|does not depend on any axioms)"
)


def validate_targets(targets: object) -> list[str]:
    if not isinstance(targets, list) or not targets:
        raise ValueError("Target list must be a nonempty JSON array")
    if any(not isinstance(name, str) or not NAME.fullmatch(name) for name in targets):
        raise ValueError(
            "Targets must be fully specified ordinary Lean declaration names"
        )
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate audit targets")
    return targets


def parse_axiom_report(text: str, targets: list[str]) -> dict[str, set[str]]:
    """Parse Lean 4.32's #print axioms messages, rejecting incomplete reports."""
    validate_targets(targets)
    result: dict[str, set[str]] = {}
    for match in REPORT.finditer(text):
        name = match.group("name")
        if name not in targets:
            raise ValueError(f"Unexpected declaration in axiom report: {name}")
        if name in result:
            raise ValueError(f"Duplicate axiom report: {name}")
        raw = match.group("axioms")
        parts = [] if raw is None or not raw.strip() else raw.split(",")
        axioms = {part.strip() for part in parts}
        if any(not NAME.fullmatch(axiom) for axiom in axioms):
            raise ValueError(f"Malformed axiom list for {name}")
        unexpected = axioms - ALLOWED_AXIOMS
        if unexpected:
            raise ValueError(
                f"Unapproved transitive axioms for {name}: {sorted(unexpected)}"
            )
        result[name] = axioms
    missing = set(targets) - result.keys()
    if missing:
        raise ValueError(f"Missing axiom reports: {sorted(missing)}")
    return result


def run_checked(command: list[str], root: Path, timeout: int) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}: {' '.join(command)}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    lake = shutil.which("lake")
    if lake is None:
        print("FAIL: Lake is unavailable; no kernel audit was performed.")
        return 2
    root = args.root.resolve()
    target_path = args.targets or root / "kernel_audit_targets.json"
    try:
        targets = validate_targets(json.loads(target_path.read_text(encoding="utf-8")))
        # Never certify stale oleans after a source edit.
        run_checked([lake, "build", "ToE"], root, args.timeout)
        scratch = root / ".lake"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="kernel-audit-", dir=scratch) as temp:
            report = Path(temp) / "Report.lean"
            report.write_text(
                "import ToE\nset_option pp.fullNames true\n"
                + "".join(f"#print axioms {name}\n" for name in targets),
                encoding="utf-8",
            )
            output = run_checked([lake, "env", "lean", str(report)], root, args.timeout)
            results = parse_axiom_report(output, targets)
        print(f"Kernel axiom audit: {len(results)} selected declarations passed.")
        print("Permitted foundations: " + ", ".join(sorted(ALLOWED_AXIOMS)))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
