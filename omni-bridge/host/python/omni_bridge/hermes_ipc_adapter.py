#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Hermes IPC Adapter — governance/expertise plane (§20).

Modes (first available wins):
  1. in_process  — deterministic framework engines: FTS5 retrieval, memory store,
                   evidence ledger. Always available; exercises the real modules.
  2. pybind11    — hermes_agent_lib native module if present.
  3. subprocess  — hermes_agent binary, HARD TIMEOUT (§57) as last-resort LLM path.

Request contract (JSON or raw text):
  {"op": "retrieve"|"remember"|"ledger"|"health", "query"/"content": "...", "k": 3}
  Raw text is treated as {"op": "retrieve", "query": text}.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hermes_ipc_adapter")


PY_PKG = "/opt/omni-bridge/source/omni-bridge/python"
HERMES_BUILD_DIR = "/opt/omni-bridge/source/hermes/build/"
sys.path.insert(0, PY_PKG)

from omni_bridge.paths import HERMES_SOCKET as SOCKET_PATH  # noqa: E402,F811

try:
    import hermes_agent_lib  # native Hermes core if built with pybind11

    MODE = "pybind11"
except ImportError:
    from omni_bridge.appa import ConfidentialityLevel, IntegrityLevel, SecurityLabel
    from omni_bridge.evidence_ledger import EvidenceLedger, LedgerRecord
    from omni_bridge.memory import (
        MemoryRecord,
        MemoryStore,
        MemoryType,
        ValidationStatus,
    )
    from omni_bridge.retrieval import RetrievalEngine

    MODE = "in_process"

    _retrieval = RetrievalEngine(fts5_db_path="/opt/omni-bridge/retrieval/fts5.db")
    _memory = MemoryStore(db_path="/opt/omni-bridge/memory/omni_memory.db")
    _ledger = EvidenceLedger("/opt/omni-bridge/audit/evidence.db")


def _seed_retrieval_defaults():
    """Level-0 catalog entries so governance retrieval returns real artifacts."""
    docs = [
        (
            "appa-permission-algebra",
            "APPA: Effective permissions are the intersection of System, Declared, Tenant, "
            "Tool, Context and Risk authority sets. Risk removes authority; risk never "
            "manufactures authority. Label descent is blocked.",
        ),
        (
            "skill-lifecycle-fsm",
            "Skill lifecycle states: DISCOVERED CANDIDATE TESTING AUDITED SHADOW CANARY "
            "PROMOTED MONITORED DEPRECATED RETIRED. Illegal skips are rejected. Every "
            "transition is auditable.",
        ),
        (
            "evidence-ledger-blake3",
            "Evidence ledger: append-only BLAKE3 hash chain. Records input hash, output "
            "hash, actor, route, verification result. Never records secrets.",
        ),
        (
            "qstar-selective-fallback",
            "QSTAR: confidence-gated selective fallback routing. High confidence takes the "
            "fast classical backbone; low confidence escalates to the specialized fallback "
            "head. Shadow prices raise the threshold as budget depletes.",
        ),
        (
            "rcod-swarm-phases",
            "RCOD stochastic swarm: chain overlap density Phi and reverse density mu across "
            "macro anchor and stochastic leaves. Phases: VACUUM, PRE_SHOCK, "
            "PHASE_TRANSITION_COLLAPSE.",
        ),
    ]
    for doc_id, text in docs:
        try:
            _retrieval.index_document(
                doc_id, text, {"origin": "hermes", "integrity": "TRUSTED"}
            )
        except Exception as e:
            logger.debug(f"seed skip {doc_id}: {e}")


if MODE == "in_process":
    _seed_retrieval_defaults()


def _trusted_label():
    return SecurityLabel(
        integrity=IntegrityLevel.TRUSTED,
        confidentiality=ConfidentialityLevel.INTERNAL,
        origin="hermes_adapter",
        purpose="governance",
    )


def handle_in_process(payload: str) -> str:
    """Deterministic governance-plane operations (no LLM, no network)."""
    try:
        req = json.loads(payload)
        if not isinstance(req, dict):
            raise ValueError("not an object")
    except (json.JSONDecodeError, ValueError):
        req = {"op": "retrieve", "query": payload}

    op = req.get("op", "retrieve")
    appa = {"integrity": "TRUSTED", "origin": "hermes", "plane": "governance"}

    if op in ("health", "__health__"):
        return json.dumps({"status": "ok", "agent": "hermes", "mode": MODE})

    if op == "retrieve":
        results = _retrieval.search(req.get("query", ""), int(req.get("k", 3)))
        return json.dumps(
            {
                "status": "ok",
                "agent": "hermes",
                "op": "retrieve",
                "appa_metadata": appa,
                "results": [
                    {"doc_id": r.doc_id, "text": r.content[:512], "score": r.score}
                    for r in results
                ],
            }
        )

    if op == "remember":
        now = time.time()
        rec = MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            memory_type=MemoryType.SEMANTIC,
            content=req.get("content", ""),
            source="hermes_adapter",
            confidence=float(req.get("confidence", 0.5)),
            provenance="hermes/governance",
            created_at=now,
            security_label=_trusted_label(),
            validation_status=ValidationStatus.UNVALIDATED,
            expiry=None,
            dependencies=[],
        )
        mid = _memory.store(rec)
        return json.dumps(
            {
                "status": "ok",
                "agent": "hermes",
                "op": "remember",
                "record_id": mid,
                "appa_metadata": appa,
            }
        )

    if op == "ledger":
        lrec = LedgerRecord(
            timestamp=time.time(),
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            actor="hermes",
            route=req.get("route", "hermes"),
            model="n/a",
            skill="n/a",
            tool="n/a",
            input_hash="0" * 64,
            output_hash="0" * 64,
            security_labels=_trusted_label(),
            capabilities=[],
            budget_used=0.0,
            latency_ms=0.0,
            verification_result="n/a",
            outcome=req.get("action", "NOTE"),
            previous_hash="",
        )
        saved = _ledger.record(lrec)
        return json.dumps(
            {
                "status": "ok",
                "agent": "hermes",
                "op": "ledger",
                "record_hash": saved.current_hash,
                "appa_metadata": appa,
            }
        )

    return json.dumps(
        {"status": "error", "agent": "hermes", "error": f"unknown op: {op}"}
    )


async def process_with_hermes(payload: str) -> str:
    if payload.strip() == "__health__":
        return json.dumps({"status": "ok", "agent": "hermes", "mode": MODE})

    if MODE == "in_process":
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, handle_in_process, payload), timeout=3.0
            )
        except asyncio.TimeoutError:
            return json.dumps(
                {
                    "status": "timeout",
                    "agent": "hermes",
                    "error": "in-process handler exceeded budget",
                }
            )
        except Exception as e:
            return json.dumps({"status": "error", "agent": "hermes", "error": str(e)})

    # pybind11 native path
    try:
        response = hermes_agent_lib.process(
            payload, json.dumps({"integrity": "TRUSTED"})
        )
        return response
    except Exception as e:
        logger.error(f"pybind11 processing failed: {e}")

    # Subprocess last resort — HARD TIMEOUT (§57)
    try:
        hermes_bin = os.path.join(HERMES_BUILD_DIR, "hermes_agent")
        if not os.path.exists(hermes_bin):
            return json.dumps(
                {
                    "status": "error",
                    "agent": "hermes",
                    "error": "no hermes backend available",
                }
            )
        process = await asyncio.create_subprocess_exec(
            hermes_bin,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        req = json.dumps(
            {"payload": payload, "appa_metadata": {"integrity": "TRUSTED"}}
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(req.encode()),
                timeout=float(os.environ.get("HERMES_SUBPROC_TIMEOUT", "5")),
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error("Hermes subprocess timed out — deterministic fallback")
            return json.dumps(
                {
                    "status": "timeout",
                    "agent": "hermes",
                    "error": "hermes_agent did not respond within timeout",
                    "payload": payload[:256],
                }
            )
        if process.returncode != 0:
            return json.dumps(
                {
                    "status": "error",
                    "agent": "hermes",
                    "error": "subprocess failed",
                    "details": stderr.decode()[:256],
                }
            )
        return stdout.decode()
    except Exception as e:
        logger.error(f"Subprocess fallback failed: {e}")
        return json.dumps({"status": "error", "agent": "hermes", "error": str(e)})


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        data = await asyncio.wait_for(reader.read(65536), timeout=30)
        if data:
            payload = data.decode(errors="replace").strip()
            logger.info(f"Request: {payload[:80]}{'...' if len(payload) > 80 else ''}")
            response = await process_with_hermes(payload)
            writer.write(response.encode())
            await writer.drain()
    except asyncio.TimeoutError:
        logger.warning("Client read timeout")
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o660)
    logger.info(f"Serving on {SOCKET_PATH} with mode {MODE}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
