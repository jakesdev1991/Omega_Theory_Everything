#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Governed NVIDIA NIM client for the Lucifer generation plane.

Boundaries (per framework §22/§45 and the PoKW discipline):
  * Keys are read at call time from ~/Downloads/keys/nvidia_api_keys.txt,
    stripped of comments/quotes; NEVER logged, NEVER written to telemetry
    or the ledger. Only key INDEX is reported in responses/logs.
  * All model output is UNVERIFIED by default. The chat layer must label
    anything not grounded in a corpus citation as unverified (§26).
  * Fail-closed: OMNI_GENERATION_OFF=1 (kill switch §72) disables every
    call. Missing/empty key pool disables too — no anonymous calls.
  * Timeouts generous (reasoning models 2-5 min per the fleet skill).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

POOL = Path.home() / "Downloads/keys/nvidia_api_keys.txt"
URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODEL = "z-ai/glm-5.3"  # verified live 2026-10-16, 3.5s warm
FAST_FALLBACK_MODEL = (
    "meta/llama-3.2-11b-vision-instruct"  # live 0.5s, weaker grounding
)
REASONING_TIMEOUT = 240
FAST_TIMEOUT = 60
# dead/stalled as of 2026-10-16: kimi-k2.6 (404), mistral-7b (404),
# deepseek-v4.1-flash (stall), glm-5.3-flash (stall), gpt-oss-20b (malformed)


def generation_disabled() -> bool:
    v = os.environ.get("OMNI_GENERATION_OFF", "")
    return v in ("1", "true", "TRUE", "yes", "on")


def load_keys() -> list[str]:
    keys: list[str] = []
    if not POOL.exists():
        return keys
    for line in POOL.read_text().splitlines():
        line = line.split("#")[0].strip().strip('"').strip("'")
        if line.startswith("nvapi-"):
            keys.append(line)
    return keys


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: int | None = None,
    extra: dict | None = None,
) -> dict:
    """One governed completion. Returns {status, content, model, key_index,
    latency_s} — or {status:'disabled'|'error', ...} fail-closed."""
    if generation_disabled():
        return {"status": "disabled", "error": "kill switch OMNI_GENERATION_OFF active"}
    keys = load_keys()
    if not keys:
        return {"status": "error", "error": "no NIM keys in pool"}

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra:
        body.update(extra)
    payload = json.dumps(body).encode()

    if timeout is None:
        timeout = (
            REASONING_TIMEOUT if "kimi" in model or "glm" in model else FAST_TIMEOUT
        )

    # simple round-robin with failover; key index only, never the key
    order = list(range(len(keys)))
    start = int(time.time()) % len(keys)
    order = order[start:] + order[:start]
    last_err = None
    for idx in order[:4]:  # try up to 4 keys
        req = urllib.request.Request(URL, data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {keys[idx]}")
        req.add_header("Content-Type", "application/json")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
                msg = (data.get("choices") or [{}])[0].get("message", {})
                content = msg.get("content") or ""
                # Reasoning models (glm-5.3 etc.) may put effort into
                # reasoning_content; if content is empty but reasoning
                # exists, surface the reasoning as the payload (labeled).
                if not content and msg.get("reasoning_content"):
                    content = msg["reasoning_content"]
                    return {
                        "status": "ok",
                        "content": content,
                        "model": model,
                        "key_index": idx,
                        "latency_s": round(time.time() - t0, 1),
                        "note": "reasoning_content surfaced (content empty)",
                    }
                return {
                    "status": "ok",
                    "content": content,
                    "model": model,
                    "key_index": idx,
                    "latency_s": round(time.time() - t0, 1),
                }
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (401, 403):
                continue  # dead key -> next
            if e.code == 429:
                time.sleep(1.5)
                continue  # throttled -> next key
            return {"status": "error", "error": f"HTTP {e.code}", "key_index": idx}
        except Exception as e:
            last_err = str(e)[:120]
            continue
    return {"status": "error", "error": f"all attempted keys failed ({last_err})"}
