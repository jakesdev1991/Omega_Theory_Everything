# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Smoke-test the MCP manifest, state invariants, transports, and stdio tool call."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
from pathlib import Path

# Point at the repo-local package so we don't depend on a global install.
# The path is derived from this file's own location — it used to be a
# hard-coded "/tmp/omwga/mcp", which only worked on the machine that first
# ran it. Keep this as a bare sys.path statement: ruff's E402 carves out
# sys.path manipulation before imports, but not other module-level code.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omega_mcp import OmegaState, mcp
from omega_mcp.__main__ import build_parser, configure_server

HERE = Path(__file__).resolve().parent


async def async_checks() -> list[str]:
    out: list[str] = []

    # HTTP flags must reach FastMCP settings, not just argparse.
    http_args = build_parser().parse_args(
        ["--transport", "streamable-http", "--host", "127.0.0.1", "--port", "8029"]
    )
    configure_server(http_args)
    http_settings_ok = mcp.settings.host == "127.0.0.1" and mcp.settings.port == 8029
    out.append(f"cli_http_settings_ok={http_settings_ok}")

    # Tool count
    tools = await mcp.list_tools()
    tool_names = [tool.name for tool in tools]
    manifest = json.loads((HERE / "server.json").read_text(encoding="utf-8"))
    manifest_descriptions = {
        tool["name"]: tool["description"] for tool in manifest["tools"]
    }
    runtime_descriptions = {tool.name: tool.description for tool in tools}
    descriptions_match = runtime_descriptions == manifest_descriptions and all(
        runtime_descriptions.values()
    )
    out.append(f"tool_count={len(tool_names)}")
    out.append(f"tool_descriptions_match_manifest={descriptions_match}")
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

    # Negative amounts must not mint balances by reversing a transfer.
    STATE.sov_mint("alice", 5, "smoke test")
    events_before_invalid_transfer = len(STATE.events)
    try:
        STATE.sov_transfer("alice", "bob", -1)
        negative_transfer_rejected = False
    except ValueError:
        negative_transfer_rejected = (
            STATE.sov_balances.get("alice", 0) == 5
            and STATE.sov_balances.get("bob", 0) == 0
            and len(STATE.events) == events_before_invalid_transfer
        )
    out.append(f"negative_transfer_rejected={negative_transfer_rejected}")

    balances_before_insufficient_transfer = dict(STATE.sov_balances)
    events_before_insufficient_transfer = len(STATE.events)
    try:
        STATE.sov_transfer("unknown-sender", "bob", 1)
        insufficient_transfer_rejected = False
    except ValueError:
        insufficient_transfer_rejected = (
            dict(STATE.sov_balances) == balances_before_insufficient_transfer
            and len(STATE.events) == events_before_insufficient_transfer
        )
    out.append(
        "insufficient_transfer_rejected_without_mutation="
        f"{insufficient_transfer_rejected}"
    )

    # CARE attestations must be distinct and each accepted mutation emits an event.
    claim = STATE.care_submit(
        participant="participant-1",
        service_category="support",
        consent_scope="smoke-test",
        evidence_commitment="commitment-1",
    )
    claim_id = claim["claims"][0]["claim_id"]
    STATE.care_attest(claim_id, "Verifier-1", "first test attestation")
    try:
        STATE.care_attest(claim_id, "verifier-1", "duplicate verifier")
        duplicate_verifier_rejected = False
    except ValueError:
        duplicate_verifier_rejected = (
            len(STATE.care_claims[claim_id]["attestations"]) == 1
        )
    STATE.care_attest(claim_id, "verifier-2", "second test attestation")
    attest_events = [
        event
        for event in STATE.events
        if event["plane"] == "CARE" and event["action"] == "attest"
    ]
    final = STATE.care_claims[claim_id]["status"] == "final"
    out.append(
        f"care_attest_events={len(attest_events)} final={final} "
        f"duplicate_verifier_rejected={duplicate_verifier_rejected}"
    )

    # AMITY issuance requires one final CARE claim and cannot reuse it twice.
    STATE.amity_entitle(claim_id, "participant-1", 100, "smoke-test-policy")
    amity_supply_after_issue = STATE.amity_supply
    events_before_reuse = len(STATE.events)
    try:
        STATE.amity_entitle(claim_id, "participant-1", 100, "smoke-test-policy")
        repeated_entitlement_rejected = False
    except ValueError:
        repeated_entitlement_rejected = (
            STATE.amity_supply == amity_supply_after_issue
            and len(STATE.events) == events_before_reuse
        )
    pending = STATE.care_submit(
        participant="participant-2",
        service_category="support",
        consent_scope="smoke-test",
        evidence_commitment="commitment-2",
    )
    pending_id = pending["claims"][0]["claim_id"]
    try:
        STATE.amity_entitle(pending_id, "participant-2", 100, "smoke-test-policy")
        pending_claim_rejected = False
    except ValueError:
        pending_claim_rejected = STATE.amity_supply == amity_supply_after_issue
    out.append(
        "amity_entitlement_guard="
        f"{repeated_entitlement_rejected and pending_claim_rejected}"
    )

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


async def run_http_bind_then_stop() -> tuple[bool, str]:
    """Start the HTTP transport and verify that its requested socket is bound."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omega_mcp",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    connected = False
    failure: str | None = None
    deadline = asyncio.get_running_loop().time() + 15

    try:
        while asyncio.get_running_loop().time() < deadline:
            if proc.poll() is not None:
                failure = f"HTTP server exited early with status {proc.returncode}"
                break
            try:
                connection = await asyncio.to_thread(
                    socket.create_connection, ("127.0.0.1", port), timeout=0.5
                )
                connection.close()
                connected = True
                break
            except OSError:
                await asyncio.sleep(0.1)
        if not connected and failure is None:
            failure = f"HTTP server did not bind 127.0.0.1:{port} within 15s"
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate(timeout=5)

    if connected:
        return True, f"http_bind_ok=true host=127.0.0.1 port={port}"
    return False, f"{failure}; stderr={stderr.strip()!r}"


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

    print("\n=== STREAMABLE HTTP BIND ===")
    http_bind_ok, http_result = await run_http_bind_then_stop()
    print(http_result)

    print("\n=== ASYNC STDIO ROUNDTRIP ===")
    roundtrip_ok, result = await run_stdio_serve_then_stop()
    print(result)

    # Exit status is the verdict. The tool list and plane checks are asserted here so
    # that a shrunken hub cannot pass on the roundtrip alone.
    failures: list[str] = []
    if "tool_count=22" not in checks:
        failures.append("expected 22 tools")
    if "tool_descriptions_match_manifest=True" not in checks:
        failures.append("runtime tool descriptions do not match the manifest")
    if "cli_http_settings_ok=True" not in checks:
        failures.append("HTTP CLI settings were not applied")
    if "negative_transfer_rejected=True" not in checks:
        failures.append("negative SOV transfer was not rejected without mutation")
    if "insufficient_transfer_rejected_without_mutation=True" not in checks:
        failures.append(
            "insufficient SOV transfer mutated balances or emitted an event"
        )
    if "care_attest_events=2 final=True duplicate_verifier_rejected=True" not in checks:
        failures.append("CARE attestations were not unique and auditable")
    if "amity_entitlement_guard=True" not in checks:
        failures.append(
            "AMITY entitlement was not guarded by finalized, unused CARE claims"
        )
    if any(line.endswith("=False") for line in checks if line.startswith("plane.")):
        failures.append("a state plane is missing")
    if not any(line.startswith("ledger_ok=true") for line in checks):
        failures.append("in-process ledger snapshot failed")
    if not http_bind_ok:
        failures.append(
            "Streamable HTTP transport did not bind to the requested host and port"
        )
    if not roundtrip_ok:
        failures.append("stdio roundtrip failed")

    if failures:
        print(f"\nSMOKE TEST FAILED: {'; '.join(failures)}")
        return 1
    print(
        "\nSMOKE TEST PASSED: documented tools, HTTP bind, CARE invariants, stdio roundtrip"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
