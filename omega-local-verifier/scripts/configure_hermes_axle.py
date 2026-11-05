#!/usr/bin/env python3
"""Safely add AXLE and the local Lean verifier to the active Hermes profile."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Configure AXLE MCP and local Lean verification for Hermes Agent"
    )
    p.add_argument("--profile", help="Hermes profile name; omit for the active profile")
    p.add_argument("--api-key", help="AXLE key (discouraged: visible in shell history)")
    p.add_argument(
        "--lean-project",
        type=Path,
        help="Also add lean-lsp-mcp for this Lean project's absolute path",
    )
    p.add_argument(
        "--skip-local-verifier",
        action="store_true",
        help="Configure AXLE only; do not register this project's local verifier",
    )
    p.add_argument("--no-test", action="store_true", help="Write config without running Hermes tests")
    return p


def command_path(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Required command not found on PATH: {name}")
    return str(Path(found).resolve())


def hermes_command(hermes: str, profile: str | None, *args: str) -> list[str]:
    result = [hermes]
    if profile:
        result.extend(["--profile", profile])
    result.extend(args)
    return result


def hermes_path(hermes: str, profile: str | None, kind: str) -> Path:
    completed = subprocess.run(
        hermes_command(hermes, profile, "config", kind),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Could not obtain Hermes config {kind!r}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    text = completed.stdout.strip().splitlines()
    if not text:
        raise RuntimeError(f"Hermes returned no path for config {kind!r}")
    return Path(text[-1]).expanduser().resolve()


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.before-axle-{stamp}.bak")
    shutil.copy2(path, destination)
    return destination


def update_env_file(path: Path, api_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    # JSON string quoting is also valid dotenv double-quote syntax and safely
    # handles spaces, #, quotes, and backslashes without exposing the value.
    replacement = f"AXLE_API_KEY={json.dumps(api_key)}"
    output: list[str] = []
    replaced = False
    for line in lines:
        if line.lstrip().startswith("AXLE_API_KEY="):
            if not replaced:
                output.append(replacement)
                replaced = True
        else:
            output.append(line)
    if not replaced:
        if output and output[-1] != "":
            output.append("")
        output.append(replacement)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def as_map(value: Any) -> CommentedMap:
    return value if isinstance(value, CommentedMap) else CommentedMap()


def configure_yaml(
    config_path: Path,
    uvx: str,
    uv: str,
    repository_root: Path,
    lean_project: Path | None,
    include_local_verifier: bool,
) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and config_path.stat().st_size:
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.load(stream)
    else:
        data = CommentedMap()
    if not isinstance(data, CommentedMap):
        raise RuntimeError(f"Hermes config root must be a YAML mapping: {config_path}")

    servers = as_map(data.get("mcp_servers"))
    data["mcp_servers"] = servers

    # No include allowlist here: AXLE generates tools dynamically from /v1/endpoints,
    # and omitting the filter ensures new tools (including extract_proof_states) appear.
    servers["axle"] = CommentedMap(
        {
            "command": uvx,
            "args": ["--from", "axiom-axle-mcp", "axle-mcp-server"],
            "env": {"AXLE_API_KEY": "${AXLE_API_KEY}"},
            "enabled": True,
            "timeout": 300,
            "connect_timeout": 90,
            "trust": "untrusted",
            "supports_parallel_tool_calls": False,
            "tools": {"resources": False, "prompts": False},
        }
    )

    if include_local_verifier:
        servers["lean_verify_local"] = CommentedMap(
            {
                "command": uv,
                "args": [
                    "--directory",
                    str(repository_root),
                    "run",
                    "lean-verifier-mcp",
                ],
                "env": {
                    "LEAN_VERIFIER_IMAGE": "local-lean-verifier:4.27.0",
                    "LEAN_VERIFIER_SPECS_DIR": str(repository_root / "specs"),
                },
                "enabled": True,
                "timeout": 320,
                "connect_timeout": 60,
                "trust": "full",
                "supports_parallel_tool_calls": False,
            }
        )

    if lean_project is not None:
        project = lean_project.expanduser().resolve()
        if not project.is_dir():
            raise RuntimeError(f"Lean project directory does not exist: {project}")
        if not ((project / "lakefile.toml").exists() or (project / "lakefile.lean").exists()):
            raise RuntimeError(f"No lakefile.toml or lakefile.lean found in: {project}")
        servers["lean_lsp"] = CommentedMap(
            {
                "command": uvx,
                "args": ["lean-lsp-mcp"],
                "env": {
                    "LEAN_PROJECT_PATH": str(project),
                    "LEAN_LOG_LEVEL": "NONE",
                },
                "enabled": True,
                "timeout": 180,
                "connect_timeout": 120,
                "trust": "full",
                "supports_parallel_tool_calls": False,
            }
        )

    mode = config_path.stat().st_mode if config_path.exists() else None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=config_path.parent,
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        yaml.dump(data, stream)
    os.replace(temporary, config_path)
    if mode is not None:
        try:
            os.chmod(config_path, mode)
        except OSError:
            pass


def run_test(hermes: str, profile: str | None, server: str) -> int:
    command = hermes_command(hermes, profile, "mcp", "test", server)
    print(f"\n$ {' '.join(command)}")
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]

    try:
        hermes = command_path("hermes")
        uvx = command_path("uvx")
        uv = command_path("uv")
        config_path = hermes_path(hermes, args.profile, "path")
        env_path = hermes_path(hermes, args.profile, "env-path")

        api_key = args.api_key or os.environ.get("AXLE_API_KEY")
        if not api_key:
            api_key = getpass.getpass("AXLE API key (input hidden): ").strip()
        if not api_key:
            raise RuntimeError("AXLE API key cannot be empty")
        if any(ch in api_key for ch in "\r\n"):
            raise RuntimeError("AXLE API key contains a newline")

        config_backup = backup(config_path)
        env_backup = backup(env_path)
        update_env_file(env_path, api_key)
        configure_yaml(
            config_path=config_path,
            uvx=uvx,
            uv=uv,
            repository_root=repository_root,
            lean_project=args.lean_project,
            include_local_verifier=not args.skip_local_verifier,
        )

        print(f"Updated Hermes config: {config_path}")
        print(f"Updated Hermes secrets: {env_path}")
        if config_backup:
            print(f"Config backup: {config_backup}")
        if env_backup:
            print(f"Secrets backup: {env_backup}")
        print("AXLE key was stored only in the profile .env file, not config.yaml.")

        if args.no_test:
            print("\nNext: hermes mcp test axle && hermes mcp list")
            return 0

        code = run_test(hermes, args.profile, "axle")
        if not args.skip_local_verifier:
            code = run_test(hermes, args.profile, "lean_verify_local") or code
        if args.lean_project:
            code = run_test(hermes, args.profile, "lean_lsp") or code

        list_command = hermes_command(hermes, args.profile, "mcp", "list")
        print(f"\n$ {' '.join(list_command)}")
        code = subprocess.run(list_command, check=False).returncode or code

        if code:
            print(
                "\nOne or more probes failed. Run `hermes logs errors -n 200` and "
                "check HERMES_AXLE_SETUP.md.",
                file=sys.stderr,
            )
        else:
            print(
                "\nAXLE is registered. Restart Hermes (or reload MCP in your active UI). "
                "Tools will be named mcp_axle_<tool>, e.g. mcp_axle_verify_proof."
            )
        return code
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
