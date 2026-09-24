<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# CARE / TWC / OMEGA / AMITY Integration v1

**Status:** Research architecture and testnet sequencing draft
**Purpose:** Keep the four economic planes coherent while their algorithms are still being defined.

This document prevents one subsystem from becoming “complete” under assumptions that the other subsystems cannot support. No section authorizes a mainnet deployment or a value-bearing issuance.

## 1. Shared protocol primitives

All planes use the same versioned primitives:

```text
ParticipantCredential
ConsentRecord
ResourceBudget
WorkProposal
EvidenceCommitment
VerificationResult
Entitlement
SettlementEpoch
Case
Appeal
PolicyVersion
AuditEvent
```

Each mutation must identify:

```text
policy_version
actor_commitment
input_commitments
output_commitments
evidence_references
review_state
appeal_state
```

The planes may have different rules, but they must not invent incompatible identities, clocks, evidence formats, or appeal semantics.

## 2. Plane responsibilities

| Plane | Primary question | Does it issue the other planes automatically? |
|---|---|---|
| CARE | What social participation, support, reach, hardship, and solidarity records exist? | No AMITY is created automatically |
| TWC | What useful engineering, infrastructure, proving, or resource work was completed? | No CARE claim is created automatically |
| OMEGA | Who may change a policy, and under what delayed governance process? | No direct personal payout from enforcement |
| AMITY | Who voluntarily converts protected CARE into a market-facing asset? | No obligation beyond published gate taxes and market policy |

The system is symbiotic through published budgets, fees, and resource flows—not through hidden automatic conversions.

## 3. TWC Proof-of-Useful-Work

There should not be one universal algorithm that pretends software engineering, Lean proofs, ZK proving, node uptime, and physics research are the same activity. TWC uses a common receipt protocol with work-class-specific verifier adapters.

### 3.1 Work lifecycle

```text
proposal
→ acceptance specification
→ bounded execution
→ artifact and resource receipts
→ deterministic checks
→ independent review
→ appeal window
→ TWC settlement receipt
```

A work proposal contains:

```text
work_id
contributor_commitment
work_class
objective
acceptance_tests
input_commitments
resource_budget
review_policy
expiry
```

A completed work receipt contains:

```text
artifact_hash
repository_or_dataset_reference
execution_attestation
test_report
resource_receipts
verifier_set
verification_result
policy_version
```

TWC is not issued for a model's confidence, a person's title, or raw device time alone. A useful-work adapter must state what was accepted and how another party can reproduce the result.

### 3.2 Initial work classes

#### Engineering and protocol work

Required evidence may include:

- Reproducible build
- Tests and benchmark results
- Reviewable code or configuration diff
- Security impact assessment
- Independent reviewer attestations

#### Lean formalization

A Lean artifact is categorized separately as:

```text
research_artifact
formalization_attempt
kernel_checked_theorem
```

Only a successful, reproducible kernel check may be described as a checked theorem. A theorem that depends on explicit axioms remains a theorem relative to those axioms. An uncompiled file, `sorry`, unverified tactic trace, or informal proof is not a completed formal proof.

The Lean proof pipeline must record:

```text
Lean version
mathlib/dependency lock
build command
axiom inventory
sorry count
proof artifact hash
kernel-check result
```

TWC may reward a research artifact or debugging contribution under a different work class, but it must not represent that artifact as a kernel-verified proof.

#### ZK circuits and proving

Required evidence may include:

- Circuit source hash
- Constraint count
- Trusted setup or transparent-proof configuration
- Known test vectors
- Verifier result
- Soundness assumptions
- Private input commitment

A ZK proof proves a statement about committed inputs and a computation. It does not independently prove that a sensor, oracle, or physical-world input was truthful.

#### Infrastructure and resource work

Required evidence may include:

- Signed uptime or relay receipts
- CPU, storage, bandwidth, or proving receipts
- User-selected resource budget
- Hardware or service attestation where available
- Availability and failure records

Wallet users may contribute through a visible quota. They do not have to run a full node, and no background resource use may occur without consent.

#### Physics and simulation research

The research oracle must preserve this ladder:

```text
hypothesis
→ mathematical model
→ formal theorem relative to assumptions
→ simulation
→ falsifiable prediction
→ independent observation
→ empirical support or rejection
```

An AI-generated physics expansion is not a physical law merely because it has a formula or a Lean proof. The formal proof, assumptions, simulation, and observation must remain separately labeled.

## 4. CARE integration

CARE can consume TWC receipts without becoming a TWC-only economy.

Examples:

```text
TWC resource receipt → satisfies a CARE wallet quota
TWC infrastructure work → contributes to CARE service capacity
TWC engineering work → improves the oracle or social application
CARE reach record → CARE participation allocation
CARE conversion request → optional AMITY settlement
```

A TWC engineer does not need to participate in CARE to earn TWC. A CARE participant may earn TWC by contributing approved infrastructure or engineering work.

CARE remains independently meaningful when no AMITY conversion occurs.

## 5. AI oracle relationship

The AI oracle is a shared evidence and research service, not a sovereign decision-maker.

It may:

- Propose hypotheses and work plans
- Generate proof candidates
- Run reproducible simulations
- Check evidence completeness
- Detect anomalies
- Produce a signed evidence bundle
- Route a result to independent reviewers

It may not:

- Create a new offense category
- Decide moral worth
- Be the sole verifier of a high-impact CARE claim
- Promote itself or its operator
- Mint, slash, or transfer value by model output alone
- Change Lean assumptions silently
- Treat a model result as an empirical fact without evidence

Oracle authority increases in stages:

```text
research-only
→ shadow testnet
→ bounded reversible testnet action
→ audited production authority
```

## 6. Weekly settlement and funding flow

The weekly CARE pool is a fixed policy amount, distributed according to the published reach/contribution policy. TWC, OMEGA, AMITY taxes, and external grants may contribute to the funding budget, but expected market value is not reserve value.

```text
OMEGA treasury commitment
+ TWC service/resource allocation
+ settled AMITY gate taxes
+ optional external funding
= funded CARE budget
```

If the funded budget is insufficient, the system records the shortfall and pauses unfunded conversion or settlement. It does not invent reserves or silently rewrite a person's history.

## 7. Integration test slices

The system should be developed through vertical slices rather than completing one token in isolation.

### Slice A — Resource to CARE status

```text
wallet quota
→ resource receipt
→ TWC work receipt
→ CARE active/grace/under-quota state
```

### Slice B — CARE social reach

```text
circle activity
→ raw reach/attention record
→ fixed weekly allocation calculation
→ audit event
```

### Slice C — Hardship solidarity

```text
hardship request
→ uncovered quota
→ sponsor resource receipt
→ sponsor bonus calculation
→ no debt and no hardship expiry
```

### Slice D — CARE to AMITY

```text
CARE conversion request
→ CARE lock/redeem record
→ conversion dock and unissued holdback
→ funded settlement check
→ optional AMITY issuance
```

### Slice E — TWC engineering work

```text
work proposal
→ reproducible artifact
→ verifier adapter
→ TWC receipt
→ contribution to network resource budget
```

### Slice F — Governance and role control

```text
policy proposal
→ simulation
→ independent review
→ Archangel/community vote
→ timelock
→ activation or rejection
```

## 8. Immediate build order

1. Implement the shared TypeScript domain types and state-machine transitions.
2. Add the local CARE API behind the existing social prototype.
3. Add a TWC work-receipt simulator with adapters for code, Lean, ZK, infrastructure, and research artifacts.
4. Add cross-plane integration tests for the six slices above.
5. Add the wallet resource budget and hardship sponsorship ledger.
6. Add a shadow AI oracle that has no authority to mutate balances.
7. Keep AMITY conversion in a disabled, local test mode until CARE and TWC receipts are stable.

No production token, physical attestation, AI oracle, or economic conversion should be wired before the cross-plane state transitions and audit events are reproducible.
