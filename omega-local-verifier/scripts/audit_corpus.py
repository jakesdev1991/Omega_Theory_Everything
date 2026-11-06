#!/usr/bin/env python3
"""Audit every .lean file in a directory with the local Lean verifier.

Runs the verifier's `audit` mode (compile + kernel replay + per-declaration
axiom audit) over each file, using the project's built library so files may
import project modules. Writes a machine-checkable JSON report with per-file
results and a totals line.

Usage (from the omega-local-verifier directory):
    uv run python scripts/audit_corpus.py \
      --lean-proofs-dir /path/to/lean_proofs \
      --project-lib /path/to/lean_proofs/.lake/build/lib \
      --output audit-report-2026-11-05.json \
      --concurrency 3 --timeout 600

Exit code 0 only if every file passed the audit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lean_local_verifier.config import load_settings  # noqa: E402
from lean_local_verifier.runner import DockerVerifier  # noqa: E402


async def audit_one(
    verifier: DockerVerifier,
    source_path: Path,
    project_lib: Path,
    timeout: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        started = time.monotonic()
        try:
            result = await verifier.audit(
                source_path.read_text(encoding="utf-8"),
                file_name=source_path.name,
                timeout_seconds=timeout,
                project_lib=project_lib,
            )
        except Exception as exc:  # noqa: BLE001 - record infra failures per file
            result = {
                "okay": False,
                "stage": "infrastructure",
                "file_name": source_path.name,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        result["duration_seconds"] = round(time.monotonic() - started, 1)
        return result


def summarize(results: list[dict]) -> dict:
    passed = [r for r in results if r.get("okay")]
    failed = [r for r in results if not r.get("okay")]
    declarations = sum((r.get("summary") or {}).get("declarations", 0) for r in results)
    sorry = sum((r.get("summary") or {}).get("sorry_count", 0) for r in results)
    nonstandard = sum(
        (r.get("summary") or {}).get("nonstandard_axiom_uses", 0) for r in results
    )
    return {
        "files_total": len(results),
        "files_passed": len(passed),
        "files_failed": len(failed),
        "failed_files": [r.get("file_name") for r in failed],
        "declarations_checked": declarations,
        "sorry_count": sorry,
        "nonstandard_axiom_uses": nonstandard,
    }


async def main_async(args: argparse.Namespace) -> int:
    lean_proofs = Path(args.lean_proofs_dir).expanduser().resolve()
    project_lib = Path(args.project_lib).expanduser().resolve()
    if not lean_proofs.is_dir():
        raise FileNotFoundError(f"lean_proofs dir not found: {lean_proofs}")
    if not project_lib.is_dir():
        raise FileNotFoundError(f"project lib not found: {project_lib}")

    # The runner caps timeouts at max_timeout_seconds (default 300); allow
    # the per-file timeout the sweep requests.
    os.environ["LEAN_VERIFIER_MAX_TIMEOUT"] = str(max(args.timeout, 300))
    settings = load_settings()
    verifier = DockerVerifier(settings)

    files = sorted(lean_proofs.glob("*.lean"))
    print(
        f"Auditing {len(files)} files (concurrency={args.concurrency}, timeout={args.timeout}s) ..."
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    results = await asyncio.gather(
        *(audit_one(verifier, f, project_lib, args.timeout, semaphore) for f in files)
    )
    elapsed = round(time.monotonic() - started, 1)

    totals = summarize(results)
    report = {
        "generated_by": "omega-local-verifier audit_corpus.py",
        "lean_proofs_dir": str(lean_proofs),
        "project_lib": str(project_lib),
        "image": settings.image,
        "timeout_seconds": args.timeout,
        "elapsed_seconds": elapsed,
        "totals": totals,
        "files": results,
    }
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size} bytes)")
    print(json.dumps(totals, indent=2))
    print(f"Elapsed: {elapsed}s")
    return 0 if totals["files_failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-proofs-dir", required=True)
    parser.add_argument("--project-lib", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
