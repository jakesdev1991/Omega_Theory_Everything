<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# Omni-Bridge host layer — the deployed control boundary

This directory is the **host implementation** of the directive: the running
system that consumes the portable policy core
([`../cpp/omni_bridge_core.hpp`](../cpp/omni_bridge_core.hpp)) as its decision
engine and enforces its decisions with real host mechanisms. What is here
compiles, runs, and is exercised by an acceptance suite on the target host
(Gentoo, Btrfs `nvme0n1p3`). This README follows the same evidence rule as
the core: no "complete/secure/proven/production-ready" language without
evidence.

## Layout

```
host/
├── cpp/            daemon + headers (C++23) — the §19 control pipeline
│   ├── include/omni_bridge/   Appa, PersistentLedger, Sandbox (Wasmtime
│   │               fuel+epoch+limiter), RequestEnvelope, ToeGrounding,
│   │               QStar, RCOD, ring buffer, Unix socket IPC
│   ├── src/        omni_bridge_daemon.cpp — parse → classify → APPA →
│   │               route (QSTAR/CBwK) → sandbox tier → IPC dispatch →
│   │               verify → persistent BLAKE3 ledger append
│   └── tests/      35-check unit suite (APPA algebra, spoof tripwire,
│                   ledger tamper detection, sandbox fuel traps, …)
├── python/omni_bridge/
│   ├── lucifer_ipc_adapter.py  exploration plane: grounding, Horsemen,
│   │                           PoKW ops, chat op
│   ├── hermes_ipc_adapter.py   governance plane: FTS5 retrieval, memory,
│   │                           evidence ledger
│   ├── lucifer_pokw.py         PoKW: backlog / verify / conjecture / mine
│   ├── lucifer_chat.py         grounded chat + citation auditor
│   ├── nim_client.py           governed generation client (kill-switched)
│   ├── lean_driver_indexer.py  corpus → FTS5 with kernel-check metadata
│   └── …                      retrieval, memory, evidence, appa, replay,
│                               skill lifecycle, audit tiers
├── scripts/        smoke + integration + benchmark suites, mine loop,
│                   backlog extractor, key-fleet validator, ops (omni CLI)
├── deploy/         systemd unit (hardened; see unit notes) + iterate timer
├── lean/           SecureRoute.lean — core-only proofs, kernel-checked
├── schemas/        route-request / route-result / security-label / tool-call
└── README.md       this file
```

## Relationship to the portable core

The portable core (one header, no dependencies) is the **policy engine**:
APPA capability algebra, governance-only caps, risk ladder, tool ABI,
skill lifecycle, replay classification. The host layer is the **enforcement
engine**: it feeds real labels and manifests into the policy and enforces
the returned tier with actual host mechanisms (Wasmtime T1 with fuel and
epoch limits, cgroups, systemd restrictions, persistent BLAKE3 JSONL
ledger).

**Unified (2026-10-16):** the daemon now authorizes every request through
the portable core's `evaluate()` via `CorePolicyAdapter` (see
`cpp/include/omni_bridge/CorePolicyAdapter.hpp` + its behavior test).
The core's type-level guarantees — `kGovernanceOnlyCaps` unreachable by
any automated path, the monotonic risk ladder, the T4 human gate, the
audit fail-closed rule, in-pipeline skill eligibility — now govern the
live system. The daemon keeps its spoof tripwire as defense-in-depth and
its ledger/sandbox/dispatch as enforcement only. A vendored copy of the
core lives in `cpp/vendor/` under namespace `omni_core` (identical to
`../cpp/omni_bridge_core.hpp` except the namespace rename, which avoids
colliding with the daemon's own `omni::bridge` types).

## Measured evidence (2026-10-16, on the target host)

| Check | Result |
|---|---|
| C++ unit suite | 35/35 pass |
| Adversarial integration (§57–§60 scenarios, isolated sockets) | 12/12 pass |
| Live smoke (typed envelopes, injection downgrade+spoof tripwire, ledger growth, ToE grounding) | 10/10 pass |
| Bridge latency, full pipeline incl. persistent ledger append | p50 0.755 ms / p99 2.4 ms (n=300, production sockets) |
| Boundary-internal paths | p50 0.046–0.083 ms |
| Lean `SecureRoute.lean` | kernel-checked, 0 errors, 0 sorry (core-only; the host's default Lean is non-Mathlib) |
| Local kernel verification (`lake build ToE` against the main corpus) | 8723 jobs green, ~2 s incremental |
| PoKW mine loop (advisory) | 22/22 backlog tasks processed to reviewable proposals; numeric falsifier produced live refutations with counterexamples |
| Grounded chat | glm-5.3 synthesis; 6/6 correct corpus citations in final verification; fabricated-citation auditor verified catching invented `[file:line]` |

## Known deviations (documented, not hidden)

- **Generation plane requires network.** The unit's earlier
  `RestrictAddressFamilies=AF_UNIX` blocked DNS inside the service; the
  unit now permits AF_INET/AF_INET6, with generation still gated by the
  `OMNI_GENERATION_OFF` kill switch (fail-closed) and model output
  carrying `epistemic: UNVERIFIED_GENERATION` (§22/§26: model output is
  never authority).
- **MemoryDenyWriteExecute is NOT set** — Wasmtime's Cranelift JIT needs
  W^X pages (sandbox tests fail under that flag); wasm containment is
  enforced by fuel + epoch + store limiter instead.
- **Ledger append uses flush(), not fsync()** — millisecond durability
  window; Btrfs + backup-staging is the recovery path.
- **Claimed TRUSTED envelopes rest on socket ownership** (0600 runtime
  dir, owner-only). Envelope attestation (ed25519 signatures) is the
  planned upgrade; until then raw/text input is clamped to UNTRUSTED and
  label spoofing is rejected.
- **The injection classifier is a downgrade heuristic, not an immunity
  claim** (§71); containment comes from the child-branch (write/exec
  dropped, absorbed label, T1 tier), not from detection.

## Files of note

- `cpp/include/omni_bridge/ToeGrounding.hpp` — per-stage thinking
  directives bound to kernel-checked Lean citations (APPA declassification
  iff theorems, CBwK conservation, Vol18 probability bounds, Born rule,
  Nash equilibrium); a missing corpus degrades to an explicit "ungrounded"
  marker, never fabricated citations.
- `python/omni_bridge/lucifer_chat.py` — the anti-hallucination stack:
  evidence block → discipline prompt → post-generation citation auditor
  that flags any `[file:line]` not present in the retrieved evidence.
- `python/omni_bridge/lucifer_pokw.py` — Proof-of-Kernel-Verified-Work
  ops (see [`../../whitepapers/omega_pokw_whitepaper.md`](../../whitepapers/omega_pokw_whitepaper.md)):
  the mining loop proposes artifacts for review; it never merges (§22).
- `scripts/integration_test.sh` — full adversarial matrix on isolated
  sockets (env-overridable paths; does not touch the live service).

## Deployment

See `deploy/omni-bridge.service` (hardened unit) and the sibling timers
(omni-iterate: 30-min build/test/doctor cycle; omni-mine: PoKW loop). The
unit expects the layout at `/opt/omni-bridge` on the target host (Btrfs
subvolumes per the directive's §6), which is a deployment concern, not a
repository concern.
