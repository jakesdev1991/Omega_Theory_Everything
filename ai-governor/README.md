<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# AI Governor — the AI that powers the Lucifer framework (prototype)

The C.A.R.E. economy is designed to be governed by an AI. Lucifer is the
framework (Four Horsemen, APPA capability algebra, sandbox tiers); this
package is the **AI that powers it** — in testing now, and staying in
testing far longer than everything else, per the repository's own rule that
nothing becomes valid merely because it is written.

**Everything here is C++23.** No Python in the runtime path (a Python
reference implementation of the governor exists in `rcod/` for
cross-checking; it is research tooling, not product code).

## The loop: training IS the useful work

```
        ┌────────────────────────────────────────────────────────┐
        │                   Lean 4 curriculum                    │
        │   54 volumes, kernel-checked (0 sorry, 0 trivial)      │
        └───────────────────────┬────────────────────────────────┘
                                │ bytes
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │  ByteModel (byte-level next-token model, C++23)        │
        │  one-hot context × context → tanh MLP → softmax        │
        └───────────────────────┬────────────────────────────────┘
                                │ gradients
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │  RCOD governor (multi-window Φ/μ, FLOW/VISC/SHOCK,     │
        │  Reverse-With-Matter) — with the two fixes the rcod/    │
        │  benchmark mandated: loss-gated checkpoints + EMA μ̄    │
        └───────────────────────┬────────────────────────────────┘
                                │ gated update
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │  Work receipt (kind 31331 shape): sha256(weights+step+ │
        │  loss), regime counts, mean α — minted every N steps    │
        │  → published to the Nostr backplane by nostr-client    │
        │  → settles into the TWC ledger as Proof of Useful Work │
        └────────────────────────────────────────────────────────┘
```

The Four Horsemen cycle is the trainer's control loop:

| Horseman | Role | Where |
|---|---|---|
| Conquest | PERCEIVE | batch gradient + telemetry collection |
| War | UNDERSTAND | multi-window Φ/μ computation, regime classification |
| Famine | DECIDE | α (reversal strength) decision, checkpoint policy |
| Death | ACT | apply update, mint receipt |

## Why this is "useful work"

The economy's USE plane issues **non-transferable receipts** for verified
work. The governor's training runs produce exactly that: a cryptographic
commitment (sha256 of weights + step + loss) to a reproducible computation
(the training script + corpus + seed). This is the honest version of
"proof of useful work" — the work is the training; the proof is the hash
chain of receipts; the usefulness is the resulting model's eval loss on
held-out curriculum bytes. No formula becomes valid merely because it is
written: the verdict line in `governor_train` reports whether the governor
helped, hurt, or was indistinguishable at prototype scale.

## Build & run

```bash
g++ -std=c++23 -O2 -march=native -Iinclude governor_train.cpp -lcrypto -o governor_train
./governor_train "lean_proofs/*.lean" 4000
```

Requires: C++23 compiler (GCC 15.2+), OpenSSL (libcrypto) for SHA-256.
No other dependencies. CPU-first by design — the AMD Radeon 890M
(gfx1100) + XDNA 2 NPU path lands when the telemetry crate (blueprint
§10) is in place; the header is written so the kernel loops can be
swapped for HIP kernels without touching the governor.

## Layout

```
ai-governor/
├── include/omega/omega_governor.hpp   # model + governor + receipts + trainer
├── governor_train.cpp                 # entry point: governed vs baseline
└── README.md
```

## Honest status (2026-09-26)

- Prototype scale only: ~164k parameters, byte-level, 333KB corpus.
- The governor ENGAGES on this workload (unlike the Python spec-as-written
  inert case) because the state vector is the *update* direction.
- Single seed, no significance tests — this is a smoke test of the loop,
  not a research result. Multi-seed + significance land in `RESULTS.md`
  before any claim is made.
- The next real curriculum step is Lean 4 AST-level tokens (via the
  LeanDojo-style extraction in `lean_proofs/lean_source.py`), not raw
  bytes; the model interface is ready for it.

## License

PolyForm-Noncommercial-1.0.0 (product side of the repository). The AI
governor is product code for the C.A.R.E. economy, not science tooling.
