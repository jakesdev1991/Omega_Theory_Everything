"""Smoke test: import the server, list tools, and exercise a couple of tool paths."""

from __future__ import annotations

import asyncio
import json
import sys

# Point at the repo-local package so we don't depend on a global install.
sys.path.insert(0, "/tmp/omwga/mcp")

from omega_mcp import OmegaState, mcp


async def async_checks() -> list[str]:
    out: list[str] = []

    # Tool count
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    out.append(f"tool_count={len(tool_names)}")
    out.append(f"tools={','.join(tool_names)}")

    # State plane existence
    STATE = OmegaState()
    for plane in (
        "sov_balances",
        "use_receipts",
        "care_claims",
        "amity_balances",
        "omega_balances",
    ):
        present = hasattr(STATE, plane)
        out.append(f"plane.{plane}={present}")

    # Snapshot smoke
    s = STATE.ledger()
    out.append(f"ledger_ok=true planes={sorted(s.keys())}")

    return out


async def run_stdio_serve_then_stop() -> str:
    """Spin the stdio server for one tool-call round trip and report the result.

    We run a second process so the server's event loop is real, then speak MCP over
    stdin/stdout using the JSON-RPC payload the stdio transport expects.
    """
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-m", "omega_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/tmp/omwga/mcp",
        text=True,
    )
    try:
        # MCP stdio initialization + tool call in one batch.
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "omega-smoke", "version": "0.1.0"},
            },
        }
        initialized = {
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed",
        }
        call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ledger",
                "arguments": {},
            },
        }
        payload = (
            json.dumps(init)
            + "\n"
            + json.dumps(initialized)
            + "\n"
            + json.dumps(call)
            + "\n"
        )
        stdout, stderr = await asyncio.to_thread(
            lambda: proc.communicate(input=payload, timeout=15)
        )
        if proc.returncode != 0:
            return f"server_exit={proc.returncode} stderr={stderr!r}"
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        last = lines[-1] if lines else "(no output)"
        return f"roundtrip_ok line_count={len(lines)} last={last[:300]}"
    except subprocess.TimeoutExpired:
        proc.kill()
        return "timeout_killed"
    except Exception as exc:  # noqa: BLE001
        proc.kill()
        return f"roundtrip_error={exc!r}"


async def main() -> None:
    print("=== ASYNC CHECKS ===")
    for line in await async_checks():
        print(line)

    print("\n=== ASYNC STDIO ROUNDTRIP ===")
    result = await run_stdio_serve_then_stop()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
