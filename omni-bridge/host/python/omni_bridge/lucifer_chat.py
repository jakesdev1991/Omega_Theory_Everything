#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Lucifer `chat` op — the conversational front-end for the framework.

Natural language in -> intent routed across the existing deterministic ops,
with an LLM ONLY for synthesis/translation. Hallucination resistance is
structural: every factual claim in a chat response is either
  * a corpus citation (lean_lookup evidence), or
  * a kernel/numeric result (verify/conjecture), or
  * explicitly labeled unverified_generation (LLM text, §26: model output
    is never authority).

Op routing (deterministic first, generation last):
  lean_lookup / verify / conjecture / backlog / mine -> passed through
  otherwise -> LLM synthesis WITH grounding evidence attached to the prompt
"""

from __future__ import annotations

import re

from . import nim_client

SYSTEM_PROMPT = """You are the conversational interface of the Lucifer-Hermes
Omni-Bridge framework, grounded in the Omega Theory of Everything formal
corpus (kernel-checked Lean 4 proofs).

RULES (absolute):
1. Cite ONLY from the EVIDENCE block, verbatim using its [file:line] citations.
   NEVER invent a file, line, or declaration name.
2. Anything not supported by EVIDENCE must be prefixed with "UNVERIFIED:".
3. Never claim a theorem is proven unless EVIDENCE marks it KERNEL_CHECKED.
4. The corpus states its own limits: kernel-checked != physically derived.
   Preserve those limits when discussing the theory.
5. Be direct and technical. No filler."""


def _harvest_evidence(lean_evidence_fn, query: str, k: int = 6) -> list[dict]:
    try:
        ev = lean_evidence_fn(query, limit=k)
    except Exception:
        ev = []
    return ev or []


def op_chat(req: dict, lean_evidence_fn=None) -> dict:
    """req: {"op":"chat","message":"...", "history":[...optional]}"""
    message = (req.get("message") or req.get("prompt") or "").strip()
    if not message:
        return {"status": "error", "error": "chat requires 'message'"}

    low = message.lower()

    # ---- deterministic intent routing first --------------------------
    # explicit verification request containing Lean code
    if re.search(r"\btheorem\b|\blemma\b", message) and ":=" in message:
        from . import lucifer_pokw

        v = lucifer_pokw.op_verify({"mode": "standalone", "code": message})
        v["op"] = "chat-verify"
        v["note"] = (
            "routed to kernel verification; result is deterministic, not model opinion"
        )
        return v

    # backlog questions
    if re.search(r"\b(backlog|tasks? remaining|what.*work.*left|queue)\b", low):
        from . import lucifer_pokw

        return lucifer_pokw.op_backlog({"action": "list", "limit": 10})

    # corpus lookup questions ("what does the corpus say about X",
    # "look up X", "find theorem X")
    if re.search(
        r"\b(what does|find|look up|cite|evidence|theorem)\b.*\?", low
    ) or re.search(r"^(look ?up|find|cite)\b", low):
        if lean_evidence_fn is None:
            return {"status": "error", "error": "grounding unavailable in this context"}
        ev = _harvest_evidence(lean_evidence_fn, message, 8)
        return {
            "status": "ok",
            "op": "chat-lookup",
            "query": message,
            "evidence": ev,
            "evidence_count": len(ev),
            "note": "deterministic corpus retrieval; citations are "
            "kernel-status-labeled",
        }

    # ---- generation path: synthesis with grounding -----------------
    if lean_evidence_fn is not None:
        evidence = _harvest_evidence(lean_evidence_fn, message, 6)
    else:
        evidence = []
    ev_block = (
        "\n".join(
            f"[{e.get('citation', '')}] {e.get('kind', '')}: {e.get('decl', '')}"
            for e in evidence
        )
        or "(no corpus evidence found)"
    )

    history = req.get("history") or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])  # bounded context
    messages.append(
        {"role": "user", "content": f"EVIDENCE:\n{ev_block}\n\nQUESTION:\n{message}"}
    )

    model = req.get("model", nim_client.DEFAULT_MODEL)
    r = nim_client.chat(
        messages, model=model, max_tokens=int(req.get("max_tokens", 4096))
    )
    if r.get("status") != "ok":
        # generation unavailable -> degrade to deterministic-only answer
        return {
            "status": "degraded",
            "op": "chat",
            "message": message,
            "generation": r,
            "evidence": evidence,
            "note": "generation unavailable; returning corpus evidence "
            "only. Claims above are all citations.",
        }

    return {
        "status": "ok",
        "op": "chat",
        "message": message,
        "response": _check_citations(r["content"], evidence),
        "epistemic": "UNVERIFIED_GENERATION",  # §26: model output is data
        "grounding": evidence,
        "grounding_count": len(evidence),
        "model": r.get("model"),
        "latency_s": r.get("latency_s"),
        "note": "LLM synthesis. Factual claims must trace to grounding "
        "citations; everything else is UNVERIFIED per the system "
        "prompt contract. Verification available via the verify op.",
    }


_CITE_RE = re.compile(r"\[([A-Za-z0-9_]+\.lean):(\d+)\]")


def _check_citations(text: str, evidence: list[dict]) -> str:
    """Post-generation citation audit (§26 enforcement).

    Every [file:line] the model produced is checked against the evidence
    block actually retrieved. Fabricated citations are flagged inline so a
    reader cannot mistake invented grounding for real grounding.
    """
    if not text:
        return text
    valid = {e.get("citation", "") for e in evidence}
    out_lines = []
    for line in text.splitlines():
        cites = _CITE_RE.findall(line)
        bad = [f"[{f}:{n}]" for f, n in cites if f"[{f}:{n}]" not in valid]
        if bad:
            line = (
                line
                + "  <-- FABRICATED CITATION "
                + ", ".join(bad)
                + " (not in retrieved evidence)"
            )
        out_lines.append(line)
    return "\n".join(out_lines)
