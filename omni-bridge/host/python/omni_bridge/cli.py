#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Omni-Bridge CLI — §47 command interface.

Usage: omni <command> [options]

Commands:
    doctor     Run health checks on all subsystems
    status     Display bridge daemon and component status
    build      Build all C++ and Python components
    test       Run all test suites
    verify     Verify evidence chain integrity
    benchmark  Run performance benchmarks
    audit      Query or export audit ledger
    replay     Replay a recorded trace
    skills     List, inspect, or transition skill lifecycle
    routes     Show routing table and statistics
    rollback   Rollback to a previous Btrfs snapshot
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from omni_bridge.paths import BRIDGE_SOCKET, HERMES_SOCKET, LUCIFER_SOCKET

BRIDGE_ROOT = Path(os.environ.get("OMNI_BRIDGE_ROOT", "/opt/omni-bridge"))
SCRIPTS_DIR = BRIDGE_ROOT / "source" / "omni-bridge" / "scripts"


def _run_script(name: str, extra_args: list[str] | None = None) -> int:
    script = SCRIPTS_DIR / name
    if not script.exists():
        print(f"Error: script {script} not found", file=sys.stderr)
        return 1
    cmd = ["bash", str(script)] + (extra_args or [])
    return subprocess.call(cmd)


def cmd_doctor(_args: argparse.Namespace) -> int:
    return _run_script("doctor.sh")


def cmd_status(_args: argparse.Namespace) -> int:
    """Show status of bridge daemon, IPC sockets, and running processes."""
    print("=== Omni-Bridge Status ===\n")

    # Check daemon
    sockets = {
        "bridge": BRIDGE_SOCKET,
        "hermes": HERMES_SOCKET,
        "lucifer": LUCIFER_SOCKET,
    }
    for name, sock_path in sockets.items():
        status = "UP" if Path(sock_path).exists() else "DOWN"
        print(f"  {name:12s} socket: {status:4s}  ({sock_path})")

    # Check PID
    pid_file = BRIDGE_ROOT / "runtime" / "bridge.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        try:
            os.kill(int(pid), 0)
            print(f"\n  Daemon PID:  {pid} (running)")
        except (ProcessLookupError, ValueError):
            print(f"\n  Daemon PID:  {pid} (stale)")
    else:
        print("\n  Daemon PID:  not found")

    # Kill switches
    print("\n=== Kill Switches ===\n")
    switches = [
        "OMNI_DYNAMIC_EXPERTISE_OFF",
        "OMNI_LUCIFER_OFF",
        "OMNI_HERMES_RETRIEVAL_OFF",
        "OMNI_GENERATION_OFF",
        "OMNI_LEARNING_OFF",
        "OMNI_AUDIT_READONLY",
        "OMNI_MODEL_PIN",
        "OMNI_ARTIFACT_PIN",
    ]
    for sw in switches:
        val = os.environ.get(sw, "0")
        marker = "🔴 ACTIVE" if val == "1" else "⚪ off"
        print(f"  {sw:40s} {marker}")

    # Btrfs usage
    print("\n=== Storage ===\n")
    subprocess.call(["df", "-h", str(BRIDGE_ROOT)])
    return 0


def cmd_build(_args: argparse.Namespace) -> int:
    return _run_script("build.sh")


def cmd_test(_args: argparse.Namespace) -> int:
    return _run_script("test.sh")


def cmd_verify(_args: argparse.Namespace) -> int:
    """Verify evidence ledger chain integrity."""
    sys.path.insert(0, str(BRIDGE_ROOT / "source" / "omni-bridge" / "python"))
    from omni_bridge.evidence_ledger import EvidenceLedger

    ledger_path = BRIDGE_ROOT / "audit" / "evidence.db"
    if not ledger_path.exists():
        print(f"No ledger found at {ledger_path}")
        return 1

    ledger = EvidenceLedger(str(ledger_path))
    valid, errors = ledger.verify_chain()
    latest = ledger.get_latest_hash()

    if valid:
        print(f"✅ Evidence chain VALID. Latest hash: {latest[:16]}...")
        return 0
    else:
        print("❌ Evidence chain BROKEN:")
        for err in errors:
            print(f"  - {err}")
        return 1


def cmd_benchmark(_args: argparse.Namespace) -> int:
    return _run_script("benchmark.sh")


def cmd_audit(args: argparse.Namespace) -> int:
    """Query audit ledger."""
    sys.path.insert(0, str(BRIDGE_ROOT / "source" / "omni-bridge" / "python"))
    from omni_bridge.evidence_ledger import EvidenceLedger

    ledger_path = BRIDGE_ROOT / "audit" / "evidence.db"
    if not ledger_path.exists():
        print(f"No ledger found at {ledger_path}")
        return 1

    ledger = EvidenceLedger(str(ledger_path))

    if args.trace_id:
        records = ledger.get_records_by_trace(args.trace_id)
        for r in records:
            print(
                json.dumps(
                    {
                        "trace_id": r.trace_id,
                        "actor": r.actor,
                        "route": r.route,
                        "outcome": r.outcome,
                        "hash": r.current_hash[:16],
                    },
                    indent=2,
                )
            )
    else:
        latest = ledger.get_latest_hash()
        print(f"Latest hash: {latest or '(empty ledger)'}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a recorded trace."""
    sys.path.insert(0, str(BRIDGE_ROOT / "source" / "omni-bridge" / "python"))
    try:
        from omni_bridge.replay import ReplayEngine
    except ImportError:
        print("Replay module not available", file=sys.stderr)
        return 1

    replay_path = BRIDGE_ROOT / "audit" / "replay.db"
    engine = ReplayEngine(str(replay_path))

    if args.trace_id:
        replays = engine.get_replays_for_trace(args.trace_id)
        for r in replays:
            print(
                f"  {r.replay_id[:12]}  {r.classification.value:30s}  {r.match_result}"
            )
    else:
        print("Usage: omni replay --trace-id <id>")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    """List or manage skill lifecycle."""
    sys.path.insert(0, str(BRIDGE_ROOT / "source" / "omni-bridge" / "python"))
    from omni_bridge.skill_lifecycle import SkillLifecycleManager, SkillState

    db_path = BRIDGE_ROOT / "skills" / "lifecycle.db"
    mgr = SkillLifecycleManager(str(db_path))

    if args.action == "list":
        skills = mgr.get_eligible_skills(SkillState.DISCOVERED)
        if not skills:
            print("No skills registered.")
        for s in skills:
            print(f"  {s.skill_id:30s} {s.state.value:12s} v{s.version} by {s.author}")
    elif args.action == "audit" and args.skill_id:
        trail = mgr.get_audit_trail(args.skill_id)
        for entry in trail:
            print(f"  {entry.get('transition_to', '?'):12s}  {entry.get('reason', '')}")
    else:
        print("Usage: omni skills list | omni skills audit --skill-id <id>")
    return 0


def cmd_routes(_args: argparse.Namespace) -> int:
    """Show routing configuration."""
    import yaml

    env = os.environ.get("OMNI_ENV", "development")
    config_path = (
        BRIDGE_ROOT / "source" / "omni-bridge" / "config" / env / "bridge.yaml"
    )
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1
    with open(config_path) as f:
        config = yaml.safe_load(f)
    routing = config.get("routing", {})
    print(f"=== Routing Configuration ({env}) ===\n")
    for k, v in routing.items():
        print(f"  {k:40s} {v}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Rollback to a Btrfs snapshot."""
    snapshots_dir = BRIDGE_ROOT / "snapshots"
    if args.list:
        snaps = sorted(snapshots_dir.iterdir()) if snapshots_dir.exists() else []
        if not snaps:
            print("No snapshots found.")
        for s in snaps:
            print(f"  {s.name}")
        return 0
    print("Rollback requires --list first, then manual snapshot restore.")
    print(f"Snapshot directory: {snapshots_dir}")
    print("Use: btrfs subvolume snapshot <source> <dest>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="omni",
        description="Omni-Bridge CLI — Lucifer-Hermes control interface",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("doctor", help="Run health checks")
    sub.add_parser("status", help="Show bridge status")
    sub.add_parser("build", help="Build all components")
    sub.add_parser("test", help="Run test suites")
    sub.add_parser("verify", help="Verify evidence chain")
    sub.add_parser("benchmark", help="Run benchmarks")

    audit_p = sub.add_parser("audit", help="Query audit ledger")
    audit_p.add_argument("--trace-id", help="Filter by trace ID")

    replay_p = sub.add_parser("replay", help="Replay a trace")
    replay_p.add_argument("--trace-id", help="Trace ID to replay")

    skills_p = sub.add_parser("skills", help="Manage skill lifecycle")
    skills_p.add_argument(
        "action", nargs="?", default="list", choices=["list", "audit"]
    )
    skills_p.add_argument("--skill-id", help="Skill ID for audit")

    sub.add_parser("routes", help="Show routing table")

    rollback_p = sub.add_parser("rollback", help="Rollback to snapshot")
    rollback_p.add_argument("--list", action="store_true", help="List snapshots")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "doctor": cmd_doctor,
        "status": cmd_status,
        "build": cmd_build,
        "test": cmd_test,
        "verify": cmd_verify,
        "benchmark": cmd_benchmark,
        "audit": cmd_audit,
        "replay": cmd_replay,
        "skills": cmd_skills,
        "routes": cmd_routes,
        "rollback": cmd_rollback,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
