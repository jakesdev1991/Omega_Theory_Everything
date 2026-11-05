from __future__ import annotations

from typing import Any

try:
    # MCP Python SDK 2.x
    from mcp.server import MCPServer
except ImportError:  # pragma: no cover - compatibility with MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from .config import load_settings
from .runner import (
    DockerVerifier,
    VerifierInfrastructureError,
    list_specs,
    resolve_spec,
)

mcp = MCPServer("local-lean4-verifier")


def failure(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "okay": False,
        "stage": stage,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


@mcp.tool()
async def verifier_health() -> dict[str, Any]:
    """Check Docker, the verifier image, Lean, and SafeVerify."""
    try:
        return await DockerVerifier().health()
    except Exception as exc:
        return failure("infrastructure", exc)


@mcp.tool()
async def list_verification_specs() -> dict[str, Any]:
    """List trusted .lean specifications available to verify against."""
    settings = load_settings()
    return {
        "okay": True,
        "specs_directory": str(settings.specs_dir),
        "specs": list_specs(settings),
    }


@mcp.tool()
async def verify_against_spec(
    spec_name: str,
    candidate: str,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Verify candidate Lean code against a trusted local specification.

    The specification must be a .lean file underneath LEAN_VERIFIER_SPECS_DIR.
    Its declarations may use `sorry`; the candidate must supply matching,
    kernel-checked declarations without disallowed axioms.
    """
    settings = load_settings()
    try:
        spec_path = resolve_spec(settings.specs_dir, spec_name)
        formal_statement = spec_path.read_text(encoding="utf-8")
        result = await DockerVerifier(settings).verify(
            formal_statement,
            candidate,
            timeout_seconds=timeout_seconds,
        )
        result["spec_name"] = spec_name
        return result
    except (ValueError, FileNotFoundError) as exc:
        return failure("input", exc)
    except VerifierInfrastructureError as exc:
        return failure("infrastructure", exc)
    except Exception as exc:
        return failure("unexpected", exc)


@mcp.tool()
async def verify_proof(
    formal_statement: str,
    candidate: str,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Verify candidate Lean code against the supplied formal statement.

    IMPORTANT: this proves only that candidate matches formal_statement. If an
    AI is allowed to change both strings, it can weaken the theorem. For an
    immutable challenge use verify_against_spec instead.
    """
    try:
        return await DockerVerifier().verify(
            formal_statement,
            candidate,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        return failure("input", exc)
    except VerifierInfrastructureError as exc:
        return failure("infrastructure", exc)
    except Exception as exc:
        return failure("unexpected", exc)


def main() -> None:
    # Do not print to stdout: stdio MCP uses stdout for JSON-RPC messages.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
