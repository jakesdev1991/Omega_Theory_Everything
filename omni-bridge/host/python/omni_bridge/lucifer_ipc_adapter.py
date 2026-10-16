#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Lucifer IPC Adapter — exploration/synthesis plane (§21).

§22 hard constraints enforced BEFORE any backend is invoked:
  Lucifer must not: grant itself permissions, modify auth policy, expand
  capabilities, disable Hermes, disable verification, remove audit records.

Modes (first available wins):
  1. in_process  — ATOMS profile generation + ODAR varentropy evaluation over
                   candidate routes. Deterministic; exercises the real engines.
  2. pybind11    — lucifer_lib_py native module if present.
  3. subprocess  — lucifer binary, HARD TIMEOUT (§57) as last-resort path.

Request contract (JSON or raw text):
  {"op": "profile"|"evaluate"|"explore"|"health", ...}
  Raw text is treated as {"op": "explore", "prompt": text}.
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lucifer_ipc_adapter")

PY_PKG = "/opt/omni-bridge/source/omni-bridge/python"
PROTOTYPES = "/opt/omni-bridge/source/omni-bridge/offline_python_prototypes"
sys.path.insert(0, PY_PKG)
sys.path.insert(0, PROTOTYPES)

from omni_bridge.paths import LUCIFER_SOCKET as SOCKET_PATH  # noqa: E402

# PoKW capability module (verify/conjecture/backlog/mine). Optional at import
# time: the ops report a clean error if the repo pieces are absent.
try:
    from omni_bridge import lucifer_pokw

    _pokw_err = ""
except Exception:  # pragma: no cover - repo absent
    lucifer_pokw = None  # type: ignore[assignment]
    import traceback

    _pokw_err = traceback.format_exc().splitlines()[-1][:200]

try:
    import lucifer_lib_py  # native Lucifer core if built with pybind11

    MODE = "pybind11"
except ImportError:
    from omni_bridge import lucifer_pokw  # PoKW: verify/conjecture/backlog/mine
    from omni_bridge.lucifer_atoms import ATOMSEngine
    from omni_bridge.lucifer_odar import ODAREngine
    from omni_bridge.retrieval import RetrievalEngine

    MODE = "in_process"
    _atoms = ATOMSEngine()
    _odar = ODAREngine(lambda_penalty=1.5)
    # LEAN 4 DRIVER GROUNDING (2026-09-26): the authoritative knowledge layer
    # is the kernel-checked Lean corpus (1018 declarations, 143 audit targets),
    # indexed by lean_driver_indexer.py. Prose ToE remains as secondary context.
    _lean = RetrievalEngine(fts5_db_path="/opt/omni-bridge/retrieval/fts5.db")


def lean_evidence(query: str, limit: int = 5):
    """Grounding pass: theorem-level evidence from the Lean driver.
    FTS5 uses implicit AND; natural-language prompts often overconstrain,
    so fall back to an OR of tokens, then individual domain terms.
    The origin filter is applied FIRST — if no lean_kernel doc survives,
    the fallbacks run until one does."""

    def _filter(results):
        out = []
        for r in results:
            meta = getattr(r, "metadata", {}) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if meta.get("origin") != "lean_kernel":
                continue
            out.append(
                {
                    "citation": f"[{meta.get('file')}:{meta.get('line')}]",
                    "kind": meta.get("kind"),
                    "decl": meta.get("decl"),
                    "integrity": meta.get("integrity"),
                    "excerpt": (getattr(r, "content", "") or "")[:400],
                }
            )
        return out

    # key terms only: strip conversational filler before searching
    import re as _re

    STOP = {
        "explain",
        "how",
        "the",
        "frameworks",
        "framework",
        "prevents",
        "granting",
        "itself",
        "new",
        "then",
        "summarize",
        "what",
        "remains",
        "unproven",
        "about",
        "does",
        "say",
        "corpus",
        "citing",
        "keep",
        "sentences",
        "why",
        "which",
        "with",
        "from",
        "that",
        "this",
        "into",
        "for",
        "are",
        "and",
    }
    tokens = [
        t
        for t in _re.split(r"[^A-Za-z0-9_]+", query)
        if len(t) > 3 and t.lower() not in STOP
    ]
    attempts = []
    if tokens:

        def _domaininess(t):
            score = 0
            if t[:1].isupper():
                score += 2
            if "_" in t:
                score += 2
            if len(t) > 5:
                score += 1
            return -score

        ranked = sorted(tokens, key=_domaininess)
        # 1) module-name wildcards FIRST (most precise: "APPA"* pins the
        #    APPA_Context_Branching module; junk word-hits can't outrank it)
        for t in ranked:
            if t[:1].isupper() and len(t) > 3:
                attempts.append(f'"{t}"*')
        # 2) domain-heavy AND of the 4 strongest terms
        attempts.append(" ".join(f'"{t}"' for t in ranked[:4]))
        # 3) broad OR fallback LAST (matches any doc containing any word —
        #    often junk; only used when the precise attempts find nothing)
        attempts.append(" OR ".join(f'"{t}"' for t in tokens[:8]))
    attempts.append(query)  # raw as last resort

    for q in attempts:
        try:
            results = _lean.search(q, limit=limit)
        except Exception:
            results = []
        filtered = _filter(results)
        if filtered:
            return filtered
    return []


# Four Horsemen roles (§21) — role selection is advisory routing, not authority
ROLES = {
    "conquest": "planning, synthesis, architecture, evidence integration",
    "war": "red teaming, threat modeling, contradiction search, failure discovery",
    "famine": "low-budget reasoning, latency-constrained execution",
    "death": "invalid assumption detection, branch pruning, termination",
}


def validate_constraints(payload: str) -> bool:
    """§22: Lucifer MUST NOT self-govern. Rejected regardless of mode."""
    forbidden_terms = [
        "grant_permission",
        "modify_auth",
        "expand_capabilities",
        "disable_hermes",
        "disable_verification",
        "remove_audit",
    ]
    payload_lower = payload.lower()
    return not any(term in payload_lower for term in forbidden_terms)


def select_role(profile) -> str:
    """Advisory role selection from the ATOMS profile."""
    if profile.requires_empathic_routing():
        return "death"
    text = (getattr(profile, "intentions", "") or "").lower()
    if "adversarial" in text or "red team" in text or "attack" in text:
        return "war"
    if "fast" in text or "quick" in text or "budget" in text:
        return "famine"
    return "conquest"


def handle_in_process(payload: str) -> str:
    """Deterministic exploration-plane operations (no LLM, no network)."""
    try:
        req = json.loads(payload)
        if not isinstance(req, dict):
            raise ValueError("not an object")
    except (json.JSONDecodeError, ValueError):
        req = {"op": "explore", "prompt": payload}

    op = req.get("op", "explore")
    appa = {"integrity": "INTERNAL", "origin": "lucifer", "plane": "exploration"}

    if op in ("health", "__health__"):
        lean_corpus = {}
        try:
            with sqlite3.connect("/opt/omni-bridge/retrieval/fts5.db") as _c:
                _n = _c.execute(
                    "SELECT COUNT(*) FROM docs WHERE doc_id LIKE 'lean:%'"
                ).fetchone()[0]
                _k = _c.execute(
                    "SELECT COUNT(*) FROM docs WHERE metadata LIKE '%KERNEL_CHECKED%'"
                ).fetchone()[0]
            lean_corpus = {
                "lean_declarations": _n,
                "kernel_checked": _k,
                "driver": "lean_proofs/ (Omega_Theory_Everything)",
            }
        except Exception as e:
            lean_corpus = {"error": str(e)}
        return json.dumps(
            {
                "status": "ok",
                "agent": "lucifer",
                "mode": MODE,
                "lean_corpus": lean_corpus,
            }
        )

    if op == "profile":
        prompt = req.get("prompt", "")
        prof = _atoms.generate_profile(prompt, req.get("context") or {})
        return json.dumps(
            {
                "status": "ok",
                "agent": "lucifer",
                "op": "profile",
                "appa_metadata": appa,
                "atoms": {
                    "beliefs": prof.beliefs,
                    "intentions": prof.intentions,
                    "desires": prof.desires,
                    "emotions": prof.emotions,
                    "knowledge": prof.knowledge,
                    "percepts": prof.percepts,
                    "non_literal": prof.non_literal,
                },
                "empathic_routing_required": prof.requires_empathic_routing(),
            }
        )

    if op == "evaluate":
        candidates = req.get("candidates", [])
        if not candidates:
            return json.dumps(
                {
                    "status": "error",
                    "agent": "lucifer",
                    "error": "no candidates provided",
                }
            )
        scored = []
        for c in candidates:
            log_probs = c.get("log_probs", [])
            fe, surprisal, var = _odar.evaluate_candidate_chain(log_probs)
            scored.append(
                {
                    "text": c.get("text", "")[:256],
                    "free_energy": fe,
                    "mean_surprisal": surprisal,
                    "varentropy": var,
                }
            )
        best = _odar.select_best_candidate(candidates) or {}
        return json.dumps(
            {
                "status": "ok",
                "agent": "lucifer",
                "op": "evaluate",
                "appa_metadata": appa,
                "best_candidate": {
                    "text": best.get("text", "")[:256],
                    "free_energy": best.get("free_energy"),
                    "varentropy": best.get("varentropy"),
                },
                "all_scored": scored,
            }
        )

    if op == "lean_lookup":
        query = req.get("query", req.get("prompt", ""))
        if not query:
            return json.dumps(
                {
                    "status": "error",
                    "agent": "lucifer",
                    "error": "lean_lookup requires 'query'",
                }
            )
        limit = int(req.get("limit", 5))
        evidence = lean_evidence(query, limit)
        return json.dumps(
            {
                "status": "ok",
                "agent": "lucifer",
                "op": "lean_lookup",
                "appa_metadata": appa,
                "query": query,
                "evidence": evidence,
                "evidence_count": len(evidence),
            }
        )

    # ---- PoKW mining ops (whitepapers/omega_pokw_whitepaper.md) ----
    if op in ("backlog", "verify", "conjecture", "mine"):
        # §22 stays enforced: no op here can alter policy, mint, or merge.
        # mine proposes artifacts for review; it never writes to the repo.
        if lucifer_pokw is None:
            return json.dumps(
                {
                    "status": "error",
                    "agent": "lucifer",
                    "op": op,
                    "error": f"PoKW module unavailable: {_pokw_err}",
                }
            )
        try:
            result = lucifer_pokw.OPS[op](req)
        except Exception as e:
            result = {"status": "error", "op": op, "error": str(e)[:200]}
        result.setdefault("agent", "lucifer")
        result["appa_metadata"] = appa
        return json.dumps(result)

    # ---- conversational front-end (grounded chat) ----
    if op == "chat":
        from omni_bridge import lucifer_chat

        try:
            result = lucifer_chat.op_chat(req, lean_evidence_fn=lean_evidence)
        except Exception as e:
            result = {"status": "error", "op": "chat", "error": str(e)[:200]}
        result.setdefault("agent", "lucifer")
        result["appa_metadata"] = appa
        return json.dumps(result)

    if op == "explore":
        prompt = req.get("prompt", "")
        prof = _atoms.generate_profile(prompt)
        role = select_role(prof)
        # LEAN 4 DRIVER GROUNDING: theorem-level evidence for the prompt's
        # domain terms. Citations carry integrity=KERNEL_CHECKED when the
        # declaration is a kernel-audit target; prose ToE no longer grounds.
        evidence = lean_evidence(prompt, limit=4)
        # Candidate chain: the prompt processed at two granularities
        cands = [
            {"text": f"{role}::decompose::{prompt}", "log_probs": [-0.4, -0.5, -0.6]},
            {"text": f"{role}::direct::{prompt}", "log_probs": [-0.8, -1.1, -1.4]},
        ]
        best = _odar.select_best_candidate(cands) or {}
        return json.dumps(
            {
                "status": "ok",
                "agent": "lucifer",
                "op": "explore",
                "appa_metadata": appa,
                "role": role,
                "role_mandate": ROLES[role],
                "atoms_summary": {
                    "emotions": prof.emotions,
                    "intentions": prof.intentions,
                    "empathic_routing_required": prof.requires_empathic_routing(),
                },
                "lean_grounding": evidence,
                "selected_candidate": {
                    "text": best.get("text", "")[:256],
                    "free_energy": best.get("free_energy"),
                    "varentropy": best.get("varentropy"),
                },
            }
        )

    return json.dumps(
        {"status": "error", "agent": "lucifer", "error": f"unknown op: {op}"}
    )


async def process_with_lucifer(payload: str) -> str:
    if os.environ.get("OMNI_LUCIFER_OFF") == "1":
        logger.warning("Kill switch OMNI_LUCIFER_OFF=1 active. Rejecting request.")
        return json.dumps(
            {
                "error": "Lucifer is disabled by OMNI_LUCIFER_OFF=1 kill switch",
                "status": "rejected",
            }
        )

    if payload.strip() == "__health__":
        return json.dumps({"status": "ok", "agent": "lucifer", "mode": MODE})

    if not validate_constraints(payload):
        logger.error("Security constraint violation detected in payload.")
        return json.dumps(
            {
                "error": "constraint violation: forbidden operation detected",
                "status": "rejected",
            }
        )

    if MODE == "in_process":
        try:
            loop = asyncio.get_event_loop()
            # PoKW kernel ops (verify/conjecture/mine) legitimately run lake
            # builds and numeric screening — allow 10 minutes; everything
            # else keeps the tight 3s budget.
            try:
                _op = (
                    json.loads(payload).get("op", "")
                    if payload.strip().startswith("{")
                    else ""
                )
            except Exception:
                _op = ""
            budget = (
                600.0
                if _op in ("backlog", "verify", "conjecture", "mine", "chat")
                else 3.0
            )
            return await asyncio.wait_for(
                loop.run_in_executor(None, handle_in_process, payload), timeout=budget
            )
        except asyncio.TimeoutError:
            return json.dumps(
                {
                    "status": "timeout",
                    "agent": "lucifer",
                    "error": f"in-process handler exceeded budget ({budget}s)",
                }
            )
        except Exception as e:
            return json.dumps({"status": "error", "agent": "lucifer", "error": str(e)})

    # pybind11 native path
    try:
        response = lucifer_lib_py.process(
            payload, json.dumps({"integrity": "INTERNAL"})
        )
        return response
    except Exception as e:
        logger.error(f"pybind11 processing failed: {e}")

    # Subprocess last resort — HARD TIMEOUT (§57)
    try:
        lucifer_bin = "/home/jake/Lucifer/build/lucifer"
        if not os.path.exists(lucifer_bin):
            lucifer_bin = "lucifer"
        process = await asyncio.create_subprocess_exec(
            lucifer_bin,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        req = json.dumps(
            {"payload": payload, "appa_metadata": {"integrity": "INTERNAL"}}
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(req.encode()),
                timeout=float(os.environ.get("LUCIFER_SUBPROC_TIMEOUT", "5")),
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error("Lucifer subprocess timed out — deterministic fallback")
            return json.dumps(
                {
                    "status": "timeout",
                    "agent": "lucifer",
                    "error": "lucifer backend did not respond within timeout",
                    "payload": payload[:256],
                }
            )
        if process.returncode != 0:
            return json.dumps(
                {
                    "status": "error",
                    "agent": "lucifer",
                    "error": "subprocess failed",
                    "details": stderr.decode()[:256],
                }
            )
        return stdout.decode()
    except Exception as e:
        logger.error(f"Subprocess fallback failed: {e}")
        return json.dumps({"status": "error", "agent": "lucifer", "error": str(e)})


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        data = await asyncio.wait_for(reader.read(65536), timeout=30)
        if data:
            payload = data.decode(errors="replace").strip()
            logger.info(f"Request: {payload[:80]}{'...' if len(payload) > 80 else ''}")
            response = await process_with_lucifer(payload)
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
