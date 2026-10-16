#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Validate the NIM key fleet per the nvidia-nim-key-fleet skill.
Probes each key with a fast model (8 tokens, 20s timeout), classifies
LIVE/THROTTLED/DEAD/TIMEOUT. Paces >=2s between calls (40 RPM/account).
Reads the pool file; strips comments/quotes. Prints a table; writes
/opt/omni-bridge/config/nim_key_status.json for the router to consume.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

POOL = Path.home() / "Downloads/keys/nvidia_api_keys.txt"
OUT = Path("/opt/omni-bridge/config/nim_key_status.json")
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "deepseek-ai/deepseek-v4.1-flash"  # fast probe model per skill
TIMEOUT = 20


def load_keys():
    keys = []
    for line in POOL.read_text().splitlines():
        line = line.split("#")[0].strip().strip('"').strip("'")
        if line.startswith("nvapi-"):
            keys.append(line)
    return keys


def probe(key):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            r.read()
            return "LIVE", round(time.time() - t0, 1)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return "THROTTLED", round(time.time() - t0, 1)
        return f"DEAD({e.code})", round(time.time() - t0, 1)
    except Exception as e:
        return f"ERROR({type(e).__name__})", round(time.time() - t0, 1)


def main():
    keys = load_keys()
    print(f"probing {len(keys)} keys against {MODEL} ...")
    results = []
    for i, k in enumerate(keys):
        status, dt = probe(k)
        results.append(
            {"idx": i, "key": k[:12] + "...", "status": status, "latency_s": dt}
        )
        print(f"  [{i + 1:2d}/{len(keys)}] {k[:12]}...  {status:14s} {dt}s")
        time.sleep(2.2)  # >=2s pacing per account cap
    live = [r for r in results if r["status"] == "LIVE"]
    throttled = [r for r in results if r["status"] == "THROTTLED"]
    dead = [r for r in results if r["status"] not in ("LIVE", "THROTTLED")]
    print(
        f"\nLIVE={len(live)}  THROTTLED(valid)={len(throttled)}  DEAD/ERR={len(dead)}"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "probe_model": MODEL,
                "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "live": len(live),
                "throttled": len(throttled),
                "dead": len(dead),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
