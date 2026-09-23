<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# TOKAMAK Reality-to-Measurement Token Whitepaper

**Version:** 0.2 research draft  
**Status:** Experimental compute and measurement design; not a claim of fusion-reactor safety or a financial instrument.

## Abstract

TOKAMAK is the proposed reality-to-measurement utility asset for work that decomposes real-world phenomena into numbers, models, simulations, measurements, and testable predictions. Plasma diagnostics and fusion research are an important domain, but not the boundary of the token. Climate, biology, infrastructure, economics, astronomy, materials, software observability, and other domains may qualify when the contribution is reproducible, useful, and open to correction.

The token rewards independently reproducible scientific and technical artifacts. It does not certify that a theory is true, grant authority to operate hazardous equipment, or replace qualified engineers, instrumentation, scientific review, or regulatory controls.

## Scope and correction standard

A contributor may submit a theory-derived model, measurement pipeline, dataset, benchmark, simulation, or correction. The theory’s origin does not create automatic legitimacy. Reviewers evaluate the artifact against evidence and competing explanations.

A qualifying contribution should make clear:

- what reality-facing question is being measured;
- which quantities and units are used;
- what assumptions convert observations into numbers;
- what predictions or classifications follow;
- how another person can reproduce the result;
- what would falsify or correct it;
- where uncertainty, bias, and missing data remain.

A correction is valuable work when it improves measurement, exposes an invalid assumption, or makes a result less misleading—even when it does not confirm the original theory.

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

## Safety governors

The token protocol may pause job issuance or route work for review when telemetry quality degrades. It must never directly command physical equipment from an unreviewed model output. Safety decisions remain under qualified human and institutional control.

Suggested state machine:

```text
NOMINAL -> REVIEW -> DEGRADED -> PAUSED -> RECOVERY
```

State transitions require hysteresis, source diversity, audit logs, and manual override. False alarms, missed events, and recovery time must be published.

## Emissions

Issuance should depend on bounded, validated work units:

```text
reward = min(category_cap, quantity × quality × reproducibility × reviewer_confidence)
```

The factors are normalized and versioned. No reward is issued merely for producing more telemetry, longer model output, or a lower training loss on an unvalidated dataset.

## Non-goals

TOKAMAK is not:

- a reactor-control credential;
- a safety certification;
- proof that a fusion claim is true;
- a substitute for scientific peer review;
- a guarantee of liquidity or price stability.

## Research and deployment

Begin with synthetic datasets and historical public benchmarks. Require independent replication before accepting live telemetry. Use valueless test tokens until the complete audit, safety, and legal review is complete.
