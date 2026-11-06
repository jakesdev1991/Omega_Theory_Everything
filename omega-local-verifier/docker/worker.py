#!/usr/bin/env python3
"""Container-side worker.

Reads one JSON request from stdin and writes exactly one JSON response to stdout.
All Lean and SafeVerify output is captured inside the response so stdout remains
machine-readable.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

VERIFIER_ROOT = Path(os.environ.get("LEAN_VERIFIER_ROOT", "/opt/SafeVerify"))
SAFE_VERIFY = VERIFIER_ROOT / ".lake" / "build" / "bin" / "safe_verify"
MAX_SOURCE_BYTES = 250_000
MAX_LOG_CHARS = 40_000

# The SafeVerify project's own library plus every Mathlib dependency it was
# built against. Project libraries are appended by lean_path_with_project().
# Paths MUST be absolute: `compile_source` passes `--root=<job dir>` to `lean`
# (Lean 4.32.0 requires the input file to be contained in the root), and Lean
# resolves relative LEAN_PATH entries against the root, not the cwd -- so
# relative entries would point into the temp job dir and Mathlib would not be
# found. Absolute paths are immune to the root change.
# NOTE: entries point at each package's `lib/lean` dir (where the .olean files
# live), NOT `lib` -- Lean resolves a module `Foo` by looking for `Foo.olean`
# directly in a search-path entry, and the oleans sit under `lib/lean/`.
# Also note the directory is `batteries` (lowercase, the package dir name),
# even though the Lean modules it provides are `Batteries.*`.
_BASE_LEAN_PATH_REL = [
    ".lake/build/lib/lean",
    ".lake/packages/mathlib/.lake/build/lib/lean",
    ".lake/packages/Cli/.lake/build/lib/lean",
    ".lake/packages/plausible/.lake/build/lib/lean",
    ".lake/packages/LeanSearchClient/.lake/build/lib/lean",
    ".lake/packages/importGraph/.lake/build/lib/lean",
    ".lake/packages/proofwidgets/.lake/build/lib/lean",
    ".lake/packages/Qq/.lake/build/lib/lean",
    ".lake/packages/aesop/.lake/build/lib/lean",
    ".lake/packages/batteries/.lake/build/lib/lean",
    ".lake/packages/mathlib/.lake/build/lib/lean",
]
BASE_LEAN_PATH = ":".join(str(VERIFIER_ROOT / p) for p in _BASE_LEAN_PATH_REL)


def lean_path_with_project(project_lean_path: list[str] | None) -> str:
    """Base LEAN_PATH plus any extra container-side project library dirs."""
    if not project_lean_path:
        return BASE_LEAN_PATH
    return BASE_LEAN_PATH + ":" + ":".join(project_lean_path)


# Source policy is defense in depth. The Docker sandbox is still required because
# Lean elaboration and tactics are executable programs. Comments and strings are
# removed before matching to avoid most accidental false positives.
BANNED_SOURCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bprelude\b", "prelude files are not accepted"),
    (r"\baxiom\b", "custom axiom declarations are not accepted"),
    (r"\bunsafe\b", "unsafe declarations are not accepted"),
    (r"\bpartial\b", "partial declarations are not accepted"),
    (r"\bunsafeCast\b", "unsafeCast is not accepted"),
    (r"\bextern\b", "FFI extern declarations are not accepted"),
    (r"\bimplemented_by\b", "implemented_by is not accepted"),
    (r"\brun_tac\b", "run_tac is not accepted"),
    (r"\belab(?:_rules)?\b", "custom elaborators are not accepted"),
    (r"\bmacro(?:_rules)?\b", "custom macros are not accepted"),
    (r"\bsyntax\b", "custom syntax is not accepted"),
    (r"\binitialize\b", "module initializers are not accepted"),
    (r"\bbuiltin_initialize\b", "module initializers are not accepted"),
    (r"\bopen\s+private\b", "open private is not accepted"),
    (r"\binclude_str\b", "compile-time file inclusion is not accepted"),
    (r"#eval!?\b", "#eval is not accepted"),
    (r"#compile\b", "#compile is not accepted"),
    (r"#run\b", "#run is not accepted"),
    (r"\bdebug\.skipKernelTC\b", "disabling kernel checking is not accepted"),
)


def clipped(value: str, limit: int = MAX_LOG_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[: limit // 2] + "\n... output clipped ...\n" + value[-limit // 2 :]


def scrub_comments_and_strings(source: str) -> str:
    """Replace Lean comments and string contents with spaces, preserving offsets."""
    chars = list(source)
    result = list(source)
    i = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False

    while i < len(chars):
        if in_line_comment:
            if chars[i] == "\n":
                in_line_comment = False
            else:
                result[i] = " "
            i += 1
            continue

        if block_depth:
            if i + 1 < len(chars) and chars[i] == "/" and chars[i + 1] == "-":
                result[i] = result[i + 1] = " "
                block_depth += 1
                i += 2
            elif i + 1 < len(chars) and chars[i] == "-" and chars[i + 1] == "/":
                result[i] = result[i + 1] = " "
                block_depth -= 1
                i += 2
            else:
                if chars[i] != "\n":
                    result[i] = " "
                i += 1
            continue

        if in_string:
            if chars[i] != "\n":
                result[i] = " "
            if escaped:
                escaped = False
            elif chars[i] == "\\":
                escaped = True
            elif chars[i] == '"':
                in_string = False
            i += 1
            continue

        if i + 1 < len(chars) and chars[i] == "-" and chars[i + 1] == "-":
            result[i] = result[i + 1] = " "
            in_line_comment = True
            i += 2
        elif i + 1 < len(chars) and chars[i] == "/" and chars[i + 1] == "-":
            result[i] = result[i + 1] = " "
            block_depth = 1
            i += 2
        elif chars[i] == '"':
            result[i] = " "
            in_string = True
            i += 1
        else:
            i += 1

    return "".join(result)


def policy_violations(label: str, source: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    size = len(source.encode("utf-8"))
    if size > MAX_SOURCE_BYTES:
        violations.append(
            {
                "source": label,
                "line": 1,
                "rule": "source-size",
                "message": f"source is {size} bytes; maximum is {MAX_SOURCE_BYTES}",
            }
        )
        return violations
    if "\x00" in source:
        violations.append(
            {
                "source": label,
                "line": 1,
                "rule": "nul-byte",
                "message": "source contains a NUL byte",
            }
        )

    scrubbed = scrub_comments_and_strings(source)
    for pattern, message in BANNED_SOURCE_PATTERNS:
        match = re.search(pattern, scrubbed)
        if match:
            violations.append(
                {
                    "source": label,
                    "line": scrubbed.count("\n", 0, match.start()) + 1,
                    "rule": pattern,
                    "message": message,
                }
            )
    return violations


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("verification deadline expired")
    return value


def run_command(
    command: list[str],
    cwd: Path,
    deadline: float,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()

    env = None
    if env_vars is not None:
        import os

        env = os.environ.copy()
        env.update(env_vars)

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=remaining(deadline),
            check=False,
            start_new_session=True,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": clipped(completed.stdout),
            "stderr": clipped(completed.stderr),
            "duration_ms": round((time.monotonic() - started) * 1000),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "exit_code": None,
            "stdout": clipped(stdout),
            "stderr": clipped(stderr),
            "duration_ms": round((time.monotonic() - started) * 1000),
            "timed_out": True,
        }


def parse_json_diagnostics(*streams: str) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for stream in streams:
        for line in stream.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                diagnostics.append(value)
    return diagnostics


def compile_source(
    source: str,
    directory: Path,
    prefix: str,
    deadline: float,
    project_lean_path: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    source_path = directory / f"{module_name}.lean"
    olean_path = directory / f"{module_name}.olean"
    source_path.write_text(source, encoding="utf-8")

    # Lean 4.32.0 requires the input file to be contained in the root directory
    # (default: the cwd). Our source lives in a temp dir under /tmp, outside the
    # verifier root, so pass --root=<job dir> to make the file contained in the root.
    # The root only determines the module name of the input file; imports are
    # resolved via LEAN_PATH, so this does not affect dependency resolution.
    result = run_command(
        [
            "lean",
            "--json",
            f"--root={directory}",
            "-o",
            str(olean_path),
            str(source_path),
        ],
        cwd=VERIFIER_ROOT,
        deadline=deadline,
        env_vars={"LEAN_PATH": lean_path_with_project(project_lean_path)},
    )
    result["diagnostics"] = parse_json_diagnostics(result["stdout"], result["stderr"])
    result["object_created"] = olean_path.is_file()
    return olean_path, result


def health(deadline: float) -> dict[str, Any]:
    lean = run_command(["lean", "--version"], VERIFIER_ROOT, deadline)
    lake = run_command(["lake", "--version"], VERIFIER_ROOT, deadline)
    return {
        "okay": lean["exit_code"] == 0
        and lake["exit_code"] == 0
        and SAFE_VERIFY.is_file(),
        "stage": "health",
        "lean": lean,
        "lake": lake,
        "safe_verify": {
            "path": str(SAFE_VERIFY),
            "exists": SAFE_VERIFY.is_file(),
            "executable": os.access(SAFE_VERIFY, os.X_OK),
        },
    }


ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def summarize_audit_report(report: Any) -> dict[str, Any]:
    """Summarize a safe_verify JSON report (array of [name, outcome] pairs)."""
    declarations: list[dict[str, Any]] = []
    sorry_declarations: list[str] = []
    nonstandard: list[dict[str, Any]] = []
    if isinstance(report, list):
        for entry in report:
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            name, outcome = entry
            if not isinstance(outcome, dict):
                continue
            target_info = outcome.get("targetInfo") or {}
            if not isinstance(target_info, dict):
                continue
            axioms = target_info.get("axioms") or []
            if not isinstance(axioms, list):
                axioms = []
            const_info = target_info.get("constInfo") or {}
            kind = const_info.get("kind") if isinstance(const_info, dict) else None
            declarations.append(
                {
                    "name": name,
                    "kind": kind,
                    "axioms": axioms,
                    "failure": outcome.get("failureMode"),
                }
            )
            if "sorryAx" in axioms:
                sorry_declarations.append(name)
            for axiom in axioms:
                if axiom not in ALLOWED_AXIOMS:
                    nonstandard.append({"declaration": name, "axiom": axiom})
    return {
        "declarations": len(declarations),
        "declaration_list": declarations,
        "sorry_declarations": sorry_declarations,
        "sorry_count": len(sorry_declarations),
        "nonstandard_axioms": nonstandard,
        "nonstandard_axiom_uses": len(nonstandard),
    }


def audit(request: dict[str, Any], deadline: float) -> dict[str, Any]:
    """Audit one Lean source file: compile it, then kernel-replay and
    axiom-audit every declaration it defines via safe_verify (self-compare).

    Unlike verify(), this is a reporting tool for trusted project code, so the
    untrusted-submission source policy (banned patterns) is NOT applied here:
    project files may legitimately use macros, syntax, etc. The size limit
    still applies. Compile success implies the kernel accepted the file;
    safe_verify additionally replays every declaration in the kernel and
    reports each one's axioms.
    """
    source = request.get("source")
    if not isinstance(source, str):
        return {
            "okay": False,
            "stage": "input",
            "error": "source must be a string",
        }
    file_name = request.get("file_name")
    if file_name is not None and not isinstance(file_name, str):
        return {
            "okay": False,
            "stage": "input",
            "error": "file_name must be a string",
        }
    project_lean_path = request.get("project_lean_path")
    if project_lean_path is not None:
        if not isinstance(project_lean_path, list) or not all(
            isinstance(entry, str) for entry in project_lean_path
        ):
            return {
                "okay": False,
                "stage": "input",
                "error": "project_lean_path must be a list of directory strings",
            }

    size = len(source.encode("utf-8"))
    if size > MAX_SOURCE_BYTES:
        return {
            "okay": False,
            "stage": "input",
            "error": f"source is {size} bytes; maximum is {MAX_SOURCE_BYTES}",
        }
    if "\x00" in source:
        return {
            "okay": False,
            "stage": "input",
            "error": "source contains a NUL byte",
        }

    # Cheap pre-signal only; the authoritative sorry check is the sorryAx
    # axiom in the safe_verify report below.
    scrubbed = scrub_comments_and_strings(source)
    sorry_mentions = len(re.findall(r"\bsorry\b", scrubbed))

    job_root = Path(tempfile.mkdtemp(prefix="lean-audit-", dir="/tmp"))
    try:
        olean_path, compile_result = compile_source(
            source, job_root, "AuditModule", deadline, project_lean_path
        )
        if compile_result["timed_out"]:
            return {
                "okay": False,
                "stage": "compile",
                "file_name": file_name,
                "timed_out": True,
                "sorry_mentions": sorry_mentions,
                "compile_result": compile_result,
            }
        if compile_result["exit_code"] != 0 or not olean_path.is_file():
            return {
                "okay": False,
                "stage": "compile",
                "file_name": file_name,
                "sorry_mentions": sorry_mentions,
                "compile_result": compile_result,
            }

        # Axiom audit: safe_verify the file against itself. It kernel-replays
        # every declaration the file defines and reports each one's axioms.
        report_path = job_root / "safe_verify_report.json"
        verification_result = run_command(
            [
                str(SAFE_VERIFY),
                "--disallow-partial",
                "--verbose",
                "--save",
                str(report_path),
                str(olean_path),
                str(olean_path),
            ],
            cwd=VERIFIER_ROOT,
            deadline=deadline,
            env_vars={"LEAN_PATH": lean_path_with_project(project_lean_path)},
        )

        report: Any = None
        report_error: str | None = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                report_error = str(exc)

        summary = summarize_audit_report(report)
        okay = (
            verification_result["exit_code"] == 0
            and not verification_result["timed_out"]
            and summary["nonstandard_axiom_uses"] == 0
        )
        response: dict[str, Any] = {
            "okay": okay,
            "stage": "audit",
            "file_name": file_name,
            "kernel_checked": True,
            "sorry_mentions": sorry_mentions,
            "compile_result": compile_result,
            "verification_result": verification_result,
            "verification_report": report,
            "summary": summary,
        }
        if report_error:
            response["verification_report_error"] = report_error
        return response
    finally:
        shutil.rmtree(job_root, ignore_errors=True)


def verify(request: dict[str, Any], deadline: float) -> dict[str, Any]:
    formal = request.get("formal_statement")
    candidate = request.get("candidate")
    if not isinstance(formal, str) or not isinstance(candidate, str):
        return {
            "okay": False,
            "stage": "input",
            "error": "formal_statement and candidate must both be strings",
        }

    project_lean_path = request.get("project_lean_path")
    if project_lean_path is not None:
        if not isinstance(project_lean_path, list) or not all(
            isinstance(entry, str) for entry in project_lean_path
        ):
            return {
                "okay": False,
                "stage": "input",
                "error": "project_lean_path must be a list of directory strings",
            }

    violations = policy_violations("formal_statement", formal)
    violations.extend(policy_violations("candidate", candidate))
    if violations:
        return {
            "okay": False,
            "stage": "policy",
            "policy_violations": violations,
        }

    job_root = Path(tempfile.mkdtemp(prefix="lean-verify-", dir="/tmp"))
    target_dir = job_root / "target"
    candidate_dir = job_root / "candidate"
    target_dir.mkdir()
    candidate_dir.mkdir()

    try:
        target_olean, target_result = compile_source(
            formal, target_dir, "TrustedSpec", deadline, project_lean_path
        )
        if target_result["timed_out"]:
            return {
                "okay": False,
                "stage": "compile_formal_statement",
                "timed_out": True,
                "formal_statement_result": target_result,
            }
        if target_result["exit_code"] != 0 or not target_olean.is_file():
            return {
                "okay": False,
                "stage": "compile_formal_statement",
                "formal_statement_result": target_result,
            }

        candidate_olean, candidate_result = compile_source(
            candidate, candidate_dir, "Candidate", deadline, project_lean_path
        )
        if candidate_result["timed_out"]:
            return {
                "okay": False,
                "stage": "compile_candidate",
                "timed_out": True,
                "formal_statement_result": target_result,
                "candidate_result": candidate_result,
            }
        if candidate_result["exit_code"] != 0 or not candidate_olean.is_file():
            return {
                "okay": False,
                "stage": "compile_candidate",
                "formal_statement_result": target_result,
                "candidate_result": candidate_result,
            }

        report_path = job_root / "safe_verify_report.json"
        verification_result = run_command(
            [
                str(SAFE_VERIFY),
                "--disallow-partial",
                "--verbose",
                "--save",
                str(report_path),
                str(target_olean),
                str(candidate_olean),
            ],
            cwd=VERIFIER_ROOT,
            deadline=deadline,
            # safe_verify loads the compiled modules, so it needs the same
            # LEAN_PATH (including any project library) as the compiler.
            env_vars={"LEAN_PATH": lean_path_with_project(project_lean_path)},
        )

        report: Any = None
        report_error: str | None = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                report_error = str(exc)

        okay = (
            verification_result["exit_code"] == 0
            and not verification_result["timed_out"]
        )
        response: dict[str, Any] = {
            "okay": okay,
            "stage": "verified" if okay else "safe_verify",
            "formal_statement_result": target_result,
            "candidate_result": candidate_result,
            "verification_result": verification_result,
            "verification_report": report,
            "guarantees": {
                "declaration_names_kinds_and_types_match": okay,
                "proof_terms_kernel_replayed": okay,
                "only_standard_axioms_allowed": okay,
                "unsafe_rejected": True,
                "partial_rejected": True,
                "runtime_network_disabled": True,
            },
        }
        if report_error:
            response["verification_report_error"] = report_error
        return response
    finally:
        shutil.rmtree(job_root, ignore_errors=True)


def main() -> int:
    try:
        Path(os.environ.get("HOME", "/tmp/home")).mkdir(parents=True, exist_ok=True)
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")

        timeout = request.get("timeout_seconds", 120)
        if not isinstance(timeout, int):
            raise ValueError("timeout_seconds must be an integer")
        timeout = max(5, min(timeout, 900))
        deadline = time.monotonic() + timeout

        mode = request.get("mode", "verify")
        if mode == "health":
            response = health(deadline)
        elif mode == "verify":
            response = verify(request, deadline)
        elif mode == "audit":
            response = audit(request, deadline)
        else:
            response = {
                "okay": False,
                "stage": "input",
                "error": f"unknown mode: {mode!r}",
            }
    except Exception as exc:
        response = {
            "okay": False,
            "stage": "worker",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
