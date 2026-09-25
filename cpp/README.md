<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# CBwK Primal-Dual Shadow-Price Pacer (C++23)

Reference implementation of the Contextual-Bandit-with-Knapsacks (CBwK)
primal-dual **shadow-price pacer** for the Lucifer–Hermes Omni-Bridge
agentic-routing stack: budget-gated dispatch across model backends
(USD / latency / memory / token dimensions) with Lagrange-multiplier
("shadow price") feedback.

**Scope:** the pacer only. It consumes per-action scores from any external
model or bandit, penalizes them with current prices, enforces per-resource
budgets via pre-execution reservations, and updates prices by projected
dual subgradient ascent at interval boundaries. Reward learning is out of
scope. Lineage: Badanidiyuru et al., *Bandits with Knapsacks* (2013);
budget-pacing duals per Balseiro et al.

## Files

| File | Purpose |
|---|---|
| `cbwk_shadow_pacer.hpp` | Header-only pacer (`pac::CBwKShadowPacer<K>`) |
| `test_cbwk_shadow_pacer.cpp` | Validation suite T0–T6 (unit, concurrency, alloc, seqlock, simulation) |
| `build.sh` | Strict build + TSan + ASan/UBSan + fallback-`expected` build |

```bash
./build.sh          # full validation (4 build configurations)
./build.sh --quick  # strict build + run only
```

## Claims audit — what the design doc promised vs. what was verified

**C1 — "Pre-execution atomic reservation invariant
`spent + outstanding + c ≤ B` across all K dimensions, race-free."**

Verified with one precision the spec glosses over. Per resource the guard
is a single atomic CAS on `held_k = spent_k + outstanding_k ≤ B_k` — that
part is genuinely race-free (ThreadSanitizer-clean under an 8-thread
hammer with reconciliation of every ledger). But **no CAS can span K
resources in one atomic instant**: acquisition is two-phase (acquire
resource 0, 1, …; roll back all on first failure). The guarantee is
*no dispatch occurs unless all K reservations succeeded*, with transient
outstanding reservations on a subset while contending — not a globally
atomic multi-dimensional reservation. A global lock or packed-word CAS
would be needed for the stronger claim; neither is worth it here.

**C2 — "Dual ascent subgradient update
`λ ← max(0, λ + η(u − b))`, prices penalize expensive actions."**

Implemented exactly as specified and unit-tested (rise on overspend,
decay on underspend, clamp at zero; price-gated scoring flip verified).
Units are now explicit: `λ` in [score]/[unit], `η` in [score]/[unit]².
One honest caveat validated by simulation: **fixed η oscillates** — see
results below (η=3.0 over-throttles and strands 7.6% of budget). η is a
workload parameter, not a constant; there is no universal value.

**C3 — "Zero heap allocations on the hot path."**

Measured, not assumed: a global `operator new` counter observes **0
allocations across 10,000 select → try_reserve → commit → settle
cycles** (T4). This holds because K and the action table are
compile-time bounded (`kMaxActions = 64`) and all state is inline.
Trade-off: dynamic action sets and runtime-K would reintroduce
allocation or indirection.

**C4 — `alignas(64)` cache-line isolation.**

Done, with one deviation: GCC's own
`std::hardware_destructive_interference_size` triggers a warning that its
value varies with `-mtune`/`-mcpu` (an ABI hazard), so the header defines
its own `kPacerCacheLine = 64` with the rationale documented in place.

**C5 — `std::expected<ReservationToken<K>, PacingError>`.**

Real C++23 `std::expected` on this toolchain (GCC 12.2+). Because the
claim silently assumes a recent libstdc++/libc++, the header also ships a
minimal API-compatible fallback for older toolchains (force-tested via
`-DPAC_NO_STD_EXPECTED` in `build.sh`).

**C6 — a gap the spec did not address: estimates vs. actuals.**

The pre-execution invariant guards the *estimated* cost `c_k(a)`. For
output tokens / memory / latency, actuals are known only after execution.
`commit()` returns a `Settlement`; `settle(actual)` trues up the ledgers,
refunding over-estimates and charging under-estimates — the latter can
push `held_k` above `B_k`, which is **counted as an overshoot event and
reported, never silently absorbed**. The invariant is exactly as good as
the cost model; the pacer makes that visible instead of hiding it.

## Validation results (2026-09-23, GCC 12.2, -O2)

All checks pass in four configurations: strict (`-Wall -Wextra -Werror`),
ThreadSanitizer, AddressSanitizer+UBSan, and fallback-`expected`.

- T0 RAII token: uncommitted reservations release budget on destruction;
  commit/settle/refund reconcile exactly.
- T1 dual law: `λ` rises `η(u−b)` on overspend, decays, clamps at 0.
- T2 scoring: at λ=0 the premium backend wins; at λ=3 (score/$) traffic
  flips to the cheap local backend.
- T3 hammer: 8 threads × 3,000 reservations on 2 tight budgets — ledgers
  reconcile to the micro-unit; `spent ≤ B` always; 17,189 blocked
  reservations (contention was real); TSan-clean.
- T4: 0 hot-path allocations (10,000 cycles).
- T5 seqlock: ~10k concurrent price reads across 2,000 publishes — no
  torn vectors; TSan-clean.

**T6 end-to-end pacing simulation** (5,000 requests; budget $200;
premium backend $0.10/request scoring ~0.90; local NPU $0.005 scoring
~0.75; premium-only demand would spend $500; 3 seeds averaged):

| Policy | Reward | Spent | Premium / Local / Declined | Final λ |
|---|---|---|---|---|
| greedy (no prices) | 1,835 | $200.00 | 1,998 / 39 / 2,961 | — |
| pacer η=0.5 | **4,099** | $200.00 | 1,845 / 3,078 / 72 | 1.65 |
| pacer η=1.0 | **4,090** | $200.00 | 1,844 / 3,106 / 48 | 1.49 |
| pacer η=3.0 | 4,009 | $184.73 | 1,681 / 3,317 / 0 | 8.72 |

Reading: both greedy and the pacer spend the same $200, but the pacer
delivers **2.2× the reward** — greedy burns the budget in the first 2,000
requests and then declines everything, while the pacer throttles premium
traffic to roughly the budget rate and serves the remainder locally.
η=3.0 shows the fixed-step-size failure mode: over-throttling strands
7.6% of budget (the documented C2 caveat). Zero budget breaches, zero
overshoots (actuals = estimates in this sim).

Threats to validity: stationary Gaussian scores, two-service world,
single resource dimension in the sim (the hammer test exercises K=2),
3 seeds, no arrival-rate variation. The numbers characterize the
mechanism, not production traffic.

## Concurrency contract (precise form)

- `select / try_reserve / commit / settle / prices / stats`:
  multi-thread safe.
- `on_interval` and all setters: single writer (setup/pacer thread).
- Prices publish through a seqlock (lock-free `atomic<double>` elements,
  version double-step); readers never observe a torn price vector.
- Budgets are immutable after `start()`.

## License

Product materials, all rights reserved, `LicenseRef-Omega-Product-Proprietary`
(same scope as `rust/`, `app/`, and `whitepapers/`; see [`../LICENSE`](../LICENSE)).
This is routing infrastructure for the Lucifer–Hermes Omni-Bridge stack,
not open-source science tooling — do not re-scope without a deliberate
licensing decision.
