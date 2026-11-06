from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import load_settings
from .runner import DockerVerifier, list_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lean-verify",
        description="Run sandboxed local Lean 4 verification",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Check Docker and the verifier image")
    subparsers.add_parser("list-specs", help="List trusted local specifications")

    verify = subparsers.add_parser(
        "verify", help="Verify a candidate against a specification"
    )
    verify.add_argument(
        "--spec", required=True, help="Path to the trusted specification .lean file"
    )
    verify.add_argument(
        "--candidate", required=True, help="Path to the candidate .lean file"
    )
    verify.add_argument(
        "--timeout", type=int, default=None, help="Total timeout in seconds"
    )
    verify.add_argument(
        "--project-lib",
        default=None,
        help="Host path to a built Lake project's .lake/build/lib directory; "
        "mounted read-only so the candidate may import that project's modules",
    )

    audit_cmd = subparsers.add_parser(
        "audit",
        help="Audit a Lean source file (compile + kernel replay + axiom audit)",
    )
    audit_cmd.add_argument(
        "--file", required=True, help="Path to the Lean source file to audit"
    )
    audit_cmd.add_argument(
        "--timeout", type=int, default=None, help="Total timeout in seconds"
    )
    audit_cmd.add_argument(
        "--project-lib",
        default=None,
        help="Host path to a built Lake project's .lake/build/lib directory; "
        "mounted read-only so the file may import that project's modules",
    )

    raw = subparsers.add_parser(
        "verify-raw", help="Verify two explicitly supplied Lean files"
    )
    raw.add_argument(
        "--formal", required=True, help="Path to the formal statement .lean file"
    )
    raw.add_argument(
        "--candidate", required=True, help="Path to the candidate .lean file"
    )
    raw.add_argument(
        "--timeout", type=int, default=None, help="Total timeout in seconds"
    )
    raw.add_argument(
        "--project-lib",
        default=None,
        help="Host path to a built Lake project's .lake/build/lib directory; "
        "mounted read-only so the candidate may import that project's modules",
    )

    return parser


async def run(args: argparse.Namespace) -> dict:
    settings = load_settings()
    verifier = DockerVerifier(settings)

    if args.command == "health":
        return await verifier.health()

    if args.command == "list-specs":
        return {
            "okay": True,
            "specs_directory": str(settings.specs_dir),
            "specs": list_specs(settings),
        }

    if args.command == "verify":
        spec = Path(args.spec).expanduser().resolve()
        candidate = Path(args.candidate).expanduser().resolve()
        if not spec.is_file():
            raise FileNotFoundError(f"Specification does not exist: {spec}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Candidate does not exist: {candidate}")
        return await verifier.verify(
            spec.read_text(encoding="utf-8"),
            candidate.read_text(encoding="utf-8"),
            timeout_seconds=args.timeout,
            project_lib=Path(args.project_lib) if args.project_lib else None,
        )

    if args.command == "audit":
        source_path = Path(args.file).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        return await verifier.audit(
            source_path.read_text(encoding="utf-8"),
            file_name=source_path.name,
            timeout_seconds=args.timeout,
            project_lib=Path(args.project_lib) if args.project_lib else None,
        )

    if args.command == "verify-raw":
        formal = Path(args.formal).expanduser().resolve()
        candidate = Path(args.candidate).expanduser().resolve()
        if not formal.is_file():
            raise FileNotFoundError(f"Formal statement does not exist: {formal}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Candidate does not exist: {candidate}")
        return await verifier.verify(
            formal.read_text(encoding="utf-8"),
            candidate.read_text(encoding="utf-8"),
            timeout_seconds=args.timeout,
            project_lib=Path(args.project_lib) if args.project_lib else None,
        )

    raise AssertionError(f"Unhandled command: {args.command}")


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        result = {
            "okay": False,
            "stage": "client",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.get("okay") else 1)


if __name__ == "__main__":
    main()
