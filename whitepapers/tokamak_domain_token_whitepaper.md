<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# TOKAMAK Reality-to-Measurement Token Whitepaper

**Version:** 0.3 research draft (historical scope; see the naming status below)  
**Status:** Experimental compute and measurement design; not a claim of fusion-reactor safety or a financial instrument.

> **Naming status (2026-09-23, restated 2026-09-25):** `TOKAMAK` is retained here as the historical
> name of this research draft and is **retired as the Solana token identity**. The selected
> canonical Solana identity is **Token of the World Citizen (`TWC`)**; its Devnet-only, valueless
> pilot symbol is `tTWC`. This document does not authorize a token deployment or a value-bearing
> use.

> **Where the live specification is now.** This paper's *methods* — reproducible artifacts,
> uncertainty and provenance recording, work-class-specific verification, safety governors — survive
> into the TWC work-receipt protocol, which is specified in
> [`../docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md) §3 and implemented as
> a Devnet-only issuer/verifier in [`../solana/README.md`](../solana/README.md). This paper's
> *token identity and issuance* are not carried forward. Read §Emissions below as a historical
> proposal, not as the TWC emissions policy.

## Abstract

TOKAMAK is the proposed reality-to-measurement utility asset for work that decomposes real-world phenomena into numbers, models, simulations, measurements, and testable predictions. Plasma diagnostics and fusion research are an important domain, but not the boundary of the token. Climate, biology, infrastructure, economics, astronomy, materials, software observability, and other domains may qualify when the contribution is reproducible, useful, and open to correction.

The token rewards independently reproducible scientific and technical artifacts. It does not certify that a theory is true, grant authority to operate hazardous equipment, or replace qualified engineers, instrumentation, scientific review, or regulatory controls.

## Scope and correction standard

A contributor may submit a theory-derived model, measurement pipeline, dataset, benchmark, simulation, or correction. The theory's origin does not create automatic legitimacy. Reviewers evaluate the artifact against evidence and competing explanations.

A qualifying contribution should make clear:

- what reality-facing question is being measured;
- which quantities and units are used;
- what assumptions convert observations into numbers;
- what predictions or classifications follow;
- how another person can reproduce the result;
- what would falsify or correct it;
- where uncertainty, bias, and missing data remain.

A correction is valuable work when it improves measurement, exposes an invalid assumption, or makes a result less misleading—even when it does not confirm the original theory.

The repository's own physics track follows this ladder explicitly: hypothesis, then mathematical
model, then formal theorem relative to stated assumptions, then simulation, then falsifiable
prediction, then independent observation. A simulation result is not an empirical confirmation, and
a kernel-checked theorem is not evidence that the theorem's assumptions describe nature. The
per-volume status of the formal work is in [`../lean_proofs/README.md`](../lean_proofs/README.md).

## Utility

TOKAMAK may pay for:

- reproducible plasma-data preprocessing;
- Physics-Informed Neural Network training and evaluation;
- turbulence and stability benchmark workloads;
- signed diagnostic reports;
- independent replication of published analyses;
- simulation jobs with declared resource and accuracy requirements.

A model result is useful only when its data provenance, uncertainty, validation set, and failure conditions are recorded.

## Telemetry boundary

Telemetry is treated as noisy evidence:

```text
measurement = value + calibration_error + environmental_noise
```

Each record must include source, unit, timestamp, uncertainty, calibration version, quality flags, and provenance commitment. Raw telemetry must not be placed on a public ledger.

The proposed Reverse Chain Overlap Density metric and any shock threshold are research hypotheses. Values such as `0.7`, `0.87`, or `72.0` must be versioned test parameters, not universal constants.

## Proof of Useful Work

A TOKAMAK work claim requires:

1. a declared scientific task;
2. a reproducible input commitment;
3. a containerized or otherwise replayable execution;
4. output hashes and metrics;
5. independent validation;
6. an uncertainty and failure report;
7. an appeal path for rejection.

PoUW classes may include combinatorial optimization, PINN training, and zero-knowledge proving, but each class requires separate quality metrics. A WalkSAT solution, for example, cannot be valued solely by runtime; correctness and reproducibility are mandatory.

The successor specification generalizes this list into a single receipt schema with
work-class-specific verifier adapters, so that engineering work, Lean formalization, ZK proving,
infrastructure uptime, and physics research are each verified by their own evidence standard rather
than by one universal algorithm
([`../docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md) §3). One distinction
carried forward without compromise: an artifact may be a *research artifact*, a *formalization
attempt*, or a *kernel-checked theorem*, and only the last may be described as a verified proof.

## Safety governors

The token protocol may pause job issuance or route work for review when telemetry quality degrades. It must never directly command physical equipment from an unreviewed model output. Safety decisions remain under qualified human and institutional control.

Suggested state machine:

```text
NOMINAL -> REVIEW -> DEGRADED -> PAUSED -> RECOVERY
```

State transitions require hysteresis, source diversity, audit logs, and manual override. False alarms, missed events, and recovery time must be published.

## Emissions

**Historical proposal — not the current policy.** Issuance should depend on bounded, validated work units:

```text
reward = min(category_cap, quantity × quality × reproducibility × reviewer_confidence)
```

The factors are normalized and versioned. No reward is issued merely for producing more telemetry, longer model output, or a lower training loss on an unvalidated dataset.

The principle that survives into the current design is that issuance is bounded, versioned, and
reproducibility-weighted. The formula itself is superseded: the TWC receipt protocol defines the
verification path, and the Solana Devnet pilot issues a fixed one-time supply with null mint
authority, so **no live emission schedule exists for `TWC`**.

## Non-goals

TOKAMAK is not:

- a reactor-control credential;
- a safety certification;
- proof that a fusion claim is true;
- a substitute for scientific peer review;
- a guarantee of liquidity or price stability.

## Research and deployment

Begin with synthetic datasets and historical public benchmarks. Require independent replication before accepting live telemetry. Use valueless test tokens until the complete audit, safety, and legal review is complete.

## Revision history

- **0.3 (2026-09-25)** — Legacy framing made explicit; added the pointer to the TWC receipt protocol
  and the artifact-classification distinction; marked §Emissions as historical. No scientific claim
  changed.
- **0.2 (2026-09-23)** — Scope broadened beyond plasma; `TOKAMAK` retired as the Solana identity in
  favor of `TWC`.
