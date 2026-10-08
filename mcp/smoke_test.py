# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Smoke test: import the server, list tools, and exercise a couple of tool paths."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Point at the repo-local package so we don't depend on a global install.
# The path is derived from this file's own location — it used to be a
# hard-coded "/tmp/omwga/mcp", which only worked on the machine that first
# ran it. Keep this as a bare sys.path statement: ruff's E402 carves out
# sys.path manipulation before imports, but not other module-level code.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omega_mcp import OmegaState, mcp

HERE = Path(__file__).resolve().parent


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


async def _read_until(proc, wanted_id, timeout):
    """Read stdout lines until the JSON-RPC response with `wanted_id` arrives.

    Reading until the specific id (rather than writing everything at once and
    closing stdin) is what makes this deterministic: with the old approach the
    server could see EOF and exit before answering the tool call, and the script's
    "we received at least one line" criterion called that a pass. The committed
    evidence was produced by a run that happened to win that race.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while True:
        budget = deadline - loop.time()
        if budget <= 0:
            raise TimeoutError(f"no response with id={wanted_id} within {timeout:.0f}s")
        line = await asyncio.wait_for(
            asyncio.to_thread(proc.stdout.readline), timeout=budget
        )
        if line == "":
            raise EOFError(f"server closed stdout before answering id={wanted_id}")
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("id") == wanted_id:
            return message


async def run_stdio_serve_then_stop() -> tuple[bool, str]:
    """Spin the stdio server for one tool-call round trip and report the result.

    The server runs as a second process so its event loop is real, and we speak MCP
    over stdin/stdout the way the stdio transport expects: initialize, wait for the
    initialize result, send `notifications/initialized`, then call the `ledger` tool
    and assert what comes back.

    Returns (ok, message). This used to return a message only and the script always
    exited 0, so CI stayed green across a timeout, a crashed server, or a tool call
    that was never answered — and the old criterion ("at least one line came back")
    was satisfied by the initialize response alone.
    """
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-m", "omega_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=HERE,
        text=True,
    )
    # `text=True` with explicit PIPEs always yields text streams; assert it so the
    # rest of the function can use them directly and mypy agrees.
    assert proc.stdin is not None and proc.stdout is not None

    def shutdown() -> str:
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            return proc.stderr.read() if proc.stderr else ""
        except (OSError, ValueError):
            return ""

    try:
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
        initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "ledger", "arguments": {}},
        }

        proc.stdin.write(json.dumps(init) + "\n")
        proc.stdin.flush()
        init_response = await _read_until(proc, 1, timeout=15)
        if "error" in init_response:
            return False, f"initialize failed: {init_response['error']}"

        proc.stdin.write(json.dumps(initialized) + "\n")
        proc.stdin.write(json.dumps(call) + "\n")
        proc.stdin.flush()
        call_response = await _read_until(proc, 2, timeout=15)

        stderr = shutdown()
        if "error" in call_response:
            return False, f"tools/call failed: {call_response['error']}"

        content = call_response.get("result", {}).get("content") or []
        if not content or content[0].get("type") != "text":
            return False, f"tools/call returned no text content: {call_response!r}"
        try:
            ledger = json.loads(content[0]["text"])
        except json.JSONDecodeError as exc:
            return False, f"tools/call text is not JSON: {exc}"

        expected_planes = {"sov", "use", "care", "amity", "omega"}
        missing = expected_planes - set(ledger.get("snapshot", {}))
        if ledger.get("ok") is not True or missing:
            return False, (
                f"ledger payload unexpected: ok={ledger.get('ok')!r} missing={sorted(missing)}"
            )

        server_name = init_response.get("result", {}).get("serverInfo", {}).get("name")
        planes = ",".join(sorted(ledger["snapshot"]))
        event_count = ledger["snapshot"].get("event_count")
        return True, (
            f"roundtrip_ok server={server_name} tool=ledger ok=true "
            f"planes=[{planes}] event_count={event_count}"
        )
    except TimeoutError as exc:
        stderr = shutdown()
        return False, f"timeout: {exc} stderr={stderr!r}"
    except EOFError as exc:
        stderr = shutdown()
        return False, f"{exc} stderr={stderr!r}"
    except Exception as exc:  # noqa: BLE001
        stderr = shutdown()
        return False, f"roundtrip_error={exc!r} stderr={stderr!r}"


async def main() -> int:
    print("=== ASYNC CHECKS ===")
    checks = await async_checks()
    for line in checks:
        print(line)

    print("\n=== ASYNC STDIO ROUNDTRIP ===")
    roundtrip_ok, result = await run_stdio_serve_then_stop()
    print(result)

    # Exit status is the verdict. The tool list and plane checks are asserted here so
    # that a shrunken hub cannot pass on the roundtrip alone.
    failures: list[str] = []
    if "tool_count=22" not in checks:
        failures.append("expected 22 tools")
    if any(line.endswith("=False") for line in checks if line.startswith("plane.")):
        failures.append("a state plane is missing")
    if not any(line.startswith("ledger_ok=true") for line in checks):
        failures.append("in-process ledger snapshot failed")
    if not roundtrip_ok:
        failures.append("stdio roundtrip failed")

    if failures:
        print(f"\nSMOKE TEST FAILED: {'; '.join(failures)}")
        return 1
    print("\nSMOKE TEST PASSED: 22 tools, five planes, ledger roundtrip over stdio")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
