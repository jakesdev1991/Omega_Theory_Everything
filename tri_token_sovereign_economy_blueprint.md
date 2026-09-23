<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Tri-Token Sovereign Economy Blueprint

**Status:** Research and systems-design specification  
**Version:** 0.1  
**Repository:** Omega Theory Everything  
**Implementation status:** Research architecture. This document does not imply a production smart contract, custody system, consensus network, or policy deployment. The repository contains only bounded, valueless local/testnet pilot tooling for `$OMEGA` (Sepolia) and `tTWC` (Solana Devnet), neither deployed nor approved for mainnet/value-bearing use.

> This document integrates the Omega mathematical-physics research program with a proposed three-token economic coordination system. The physics and psychosocial sections are hypotheses and design inputs, not experimentally validated facts. Any deployment involving money, identity, healthcare, employment, or public services requires independent legal, security, safety, and ethics review.

## 1. Purpose and design principles

The system is intended to coordinate three distinct functions without collapsing them into one speculative asset:

1. **Accounting:** a stable unit for public and community accounting.
2. **Contribution:** a token representing verified useful work and service.
3. **Governance and resilience:** a constrained token for participation, stewardship, and long-horizon capital.

Core principles:

- Separate money, contribution, and governance powers.
- Never use a psychosocial score as a universal credit score or an automatic benefit gate.
- Minimize personally identifiable data; use selective disclosure and revocation.
- Make every economic rule auditable, replayable, and parameter-versioned.
- Treat telemetry as noisy evidence with confidence intervals, not as truth.
- Do not let an oracle, validator, administrator, or model unilaterally mint value.
- Prefer reversible pilots, hard issuance caps, circuit breakers, and human appeal.

## 2. Token model

The names below are placeholders and must be reviewed by counsel before launch.

### 2.1 `SOV` — accounting and settlement unit

`SOV` is the nominal unit for budgets, invoices, grants, and internal accounting. A deployment must specify whether it is:

- an off-chain accounting credit;
- a permissioned settlement token; or
- a regulated financial instrument.

This document does **not** assert that `SOV` is a stablecoin. If stability is required, the collateral, redemption, reserve attestations, insolvency treatment, and circuit breakers must be specified separately.

Suggested invariant:

```text
accounting_balance(user) = received + earned + grants - spent - valid_disputes
```

The invariant is ledger-local and does not imply that balances have external monetary value.

### 2.2 `USE` — Proof of Useful Work contribution receipt

`USE` records accepted contributions. It is non-transferable by default and should decay or be revalidated when the underlying contribution becomes stale. A contribution receipt contains:

```text
work_id
contributor_commitment
work_class
issuer_commitment
quality_attestation
quantity_and_units
created_at
expiry_or_review_at
appeal_state
protocol_version
```

`USE` should not be exchangeable for unrestricted governance power. Weighting must be bounded, sybil-resistant, and robust against issuer collusion.

### 2.3 `CARE` — stewardship and governance token

`CARE` represents participation in stewardship processes, not a measure of human worth. It may be delegated, time-locked, or earned through service, but it must not encode protected health or psychological attributes.

Recommended constraints:

- quadratic or capped voting for low-stakes decisions;
- one-person-one-reviewer limits in arbitration panels;
- time-weighted delegation with easy revocation;
- no automatic access denial based solely on `CARE`;
- independent accessibility and anti-discrimination audits.

## 3. Ledger and identity boundaries

The reference architecture uses three planes:

### 3.1 Public settlement plane

Contains token movements, commitments, protocol versions, aggregate reserve reports, and dispute outcomes. It must not contain raw telemetry, clinical information, private narratives, or identifiable case files.

### 3.2 Private evidence plane

Stores encrypted evidence under the control of the contributor or consenting organization. The public ledger receives only commitments, proofs, and minimum necessary disclosures.

### 3.3 Governance plane

Stores proposals, quorum calculations, conflict-of-interest declarations, arbitration decisions, and parameter changes. Governance actions must be delayed, observable, and reversible where possible.

An identity layer should support multiple credentials and selective disclosure. No single identifier should link economic activity, psychosocial information, and public voting by default.

## 4. Proof of Useful Work

Proof of Useful Work (PoUW) is a claim-verification protocol, not a claim that all socially valuable work can be reduced to one scalar score.

### 4.1 Contribution lifecycle

```text
proposed -> accepted_for_review -> independently attested
         -> quality_checked -> issued -> challenged -> final or reversed
```

A work claim should require at least two independent attestations for high-impact issuance. Attestors must stake reputation or a bounded bond, but penalties must be proportional and appealable.

### 4.2 Claim schema

```rust
struct WorkClaim {
    claim_id: [u8; 32],
    contributor: Commitment,
    category: WorkCategory,
    quantity: FixedPoint,
    evidence_root: [u8; 32],
    attestations: Vec<Attestation>,
    risk_class: RiskClass,
    protocol_version: u32,
    created_at: Timestamp,
}
```

### 4.3 Acceptance rule

An implementation should calculate:

```text
quality = bounded(weighted_attestations)
reliability = historical_attestor_reliability with decay
risk_adjustment = category_specific bounded factor
issuance = min(category_cap, quantity × quality × reliability × risk_adjustment)
```

Every factor must be inspectable. Model-generated recommendations may assist review but may not be the sole basis for issuance, reversal, or exclusion.

### 4.4 Anti-gaming controls

- duplicate-evidence detection;
- rotating review committees;
- conflict-of-interest declarations;
- delayed issuance for high-risk categories;
- random audits;
- public aggregate error rates;
- contributor and attestor appeals;
- rate limits and category-specific caps;
- no reward for merely maximizing telemetry volume.

## 5. Plasma telemetry and disruption research

The Omega research may use plasma or other environmental telemetry as an experimental signal. It must remain isolated from monetary finality until independently validated.

### 5.1 Telemetry record

```text
source_id
measurement_type
value
unit
uncertainty
sample_time
collection_time
calibration_version
provenance_commitment
quality_flags
```

### 5.2 Threshold policy

A threshold must be represented as a versioned policy, not a hard-coded universal constant:

```text
threshold = baseline + sensitivity × uncertainty + context_adjustment
```

A disruption signal can trigger observation, rate reduction, or a safety pause. It must never directly confiscate balances or determine a person’s eligibility.

Suggested states:

```text
NORMAL -> WATCH -> DEGRADED -> PAUSED -> RECOVERY
```

Transitions require hysteresis, a minimum duration, multiple sources where possible, and a manual override with an audit trail. False-positive and false-negative rates must be published before using the signal operationally.

## 6. C.A.R.E. protocol

C.A.R.E. is a proposed psychosocial coordination and support protocol. It is not a diagnostic system and must not be used to infer protected traits or rank human value.

### 6.1 Privacy requirements

- collect only purpose-specific data;
- obtain informed, revocable consent;
- encrypt data at rest and in transit;
- separate case data from ledger identity;
- use short retention periods;
- log every access;
- support correction, deletion, and export where legally required;
- publish a data-use register;
- provide a non-digital and human-accessible route.

### 6.2 Arbitration

A dispute process should provide:

1. notice of the disputed claim;
2. access to the evidence used, subject to safety and privacy limits;
3. an independent reviewer;
4. a reasoned decision;
5. a correction period;
6. escalation to a panel with no conflicted members;
7. emergency safeguarding without permanent economic punishment.

Arbitrators must not be able to alter historical ledger records. Corrections are append-only reversals linked to the original decision.

### Randomized Proof-of-Care audits

High-impact care claims are sent through auditable random assignment to one conflict-screened Archangel and a heterogeneous AI audit panel drawn from independent providers and model families. An Archangel’s own agents cannot form the whole panel. Selection uses a verifiable randomness commitment, a case-specific nonce, and a published algorithm. Evidence is blinded or minimized, and reviewer conflicts, dissent, reason codes, re-audits, and appeals are recorded.

AI agents may check completeness, reproduce calculations, identify duplicate evidence, and flag disagreement. They cannot independently finalize a high-impact care decision. No design can honestly guarantee that collusion is impossible; the goal is to distribute authority, reduce payoff, detect correlated behavior, and preserve a practical correction path.

## 7. Human Archangels and relational safety

The agentic infrastructure is governed by accountable humans. Within C.A.R.E., senior stewards are called **Archangels**. This is a role title for trained, reviewable human governors—not supernatural authority or infallibility.

Archangels supervise verifier legitimacy, high-impact appeals, privacy, safeguarding, agent suspension, and standards changes. They cannot unilaterally mint AMITY or override the separation between CARE evidence and exchange settlement.

A proposed advancement rule gives 80% weight to demonstrated care integrity—empathy in action, non-judgmental listening, truthful disclosure, repair, safeguarding, and independent-review outcomes—and 20% to transparent economic stewardship. Wealth alone cannot raise a person’s level. Levels are bounded, non-transferable, periodically reviewed, and appealable.

C.A.R.E. distinguishes people from behaviors. A distinction assessment is a survival and safeguarding tool: it checks observable behavior, evidence, uncertainty, and immediate risk. It is not a permanent judgment of identity or worth. The design recognizes that reflexive judgment can be insecurity or status defense, while still allowing proportionate evidence-based boundaries.

A **thought virus** is the protocol’s name for a relational communication failure in which two or more people develop mutually reinforcing, skewed interpretations and use them as faulty foundational assumptions. It is not a psychiatric diagnosis. The response is to separate observations from interpretations, restore shared facts, mediate where safe, repair the misunderstanding, and document consent. Immediate danger permits protective action, with human oversight and later review.

## 8. Prevention, public-health partnerships, and grants

C.A.R.E. can be proposed to governments, foundations, universities, and community organizations as voluntary prevention and connection infrastructure. The scope is broader than rehabilitation or substance-use services: trauma processing, isolation, grief, family conflict, domestic violence, coercive control, and early connection to qualified help are all relevant use cases.

The policy case must remain evidence-seeking. Domestic violence and untreated trauma can create major human, health, housing, legal, and economic costs, but the platform must not claim that a token or social feed alone prevents violence or reduces GDP costs. Grant funds should support trauma-informed design, survivor-led governance, qualified moderation, accessibility, privacy engineering, crisis referrals, and independent evaluation—not speculative token appreciation or engagement maximization.

A public pilot should separate:

- anonymous reflection and first-person storytelling;
- moderated peer-support rooms;
- resource and referral rooms;
- consensual mediation;
- verified Proof-of-Care claims.

People may speak about their own experiences without courtroom-level proof in a support space. That does not make an allegation an established fact. Public posts should distinguish personal narrative, opinion, corroborated evidence, and unresolved allegation. Naming another person, publishing identifying details, doxxing, threats, targeted harassment, or coordinated retaliation is prohibited without explicit consent or a narrowly scoped lawful safeguarding process. A named person can request privacy review, correction, or interjection without controlling someone else’s account of their own experience.

Unverified stories may receive empathy and resource support, but they must not automatically mint tokens, punish an identifiable person, affect employment, or trigger public retaliation. Immediate safety risks and legal reporting duties require trained human safeguarding channels, minimum-necessary disclosure, fast exit tools, and device-safety guidance. Agents may flag possible risk; they may not diagnose, confront an alleged abuser, contact authorities, or publish a safety plan autonomously.

Grant evaluation should measure connection to qualified support, voluntary safety-plan completion, reduced isolation, participant-reported stability, time to human help, harmful escalation, harassment, false-report outcomes, privacy retention, and subgroup accessibility effects.

## 9. Wallet and social application

The primary user surface is a wallet-first social application. It should let people hold simulated or regulated balances, publish useful work, submit Proof of Care through consent-controlled flows, discover projects, and see exactly which human and deterministic checks sit between a contribution and a settlement event.

The app must separate:

- **wallet:** balances, permissions, transfers, and transaction history;
- **social layer:** posts, projects, collaboration, and creator profiles;
- **contribution layer:** PoUW submissions, evidence, reviewers, and appeals;
- **care layer:** protected CARE proofs, consent, human verifiers, and Archangel oversight;
- **research layer:** reality-to-measurement claims, models, reproducibility, and corrections;
- **privacy layer:** anonymity, visibility controls, local agents, and scoped disclosures.

### Anonymous-by-default participation

A user is anonymous or pseudonymous by default. Visibility is granular and independently opt-in for identity, profile, location, work history, care history, contactability, and agent capabilities. A suggested ladder is:

```text
L0 private local participation
L1 pseudonymous public handle
L2 approved contribution visibility
L3 community profile and collaboration
L4 identity disclosure to a named authorized party
```

Greater visibility may improve earning and collaboration opportunities, but it cannot be required for basic access and cannot be treated as a measure of human worth. Each disclosure must explain who receives it, why, for how long, and how to withdraw it.

Every user’s agent or node runs on their own hardware, or on a custodian they explicitly choose. Other users cannot see, discover, call, or inspect it unless the owner publishes a narrowly scoped capability or temporary proof endpoint. Raw prompts, private memories, credentials, wallet secrets, local files, and sensor streams remain local by default.

Creators may earn TWC/AMITY only under published, independently reviewed policies. CARE eligibility is not automatic: it requires a qualifying care action, consent, human verification, safeguarding review where necessary, and an appeal path. The first release must remain local/testnet and valueless, with no private-key collection and no automatic exchange settlement.

## 9. Active inference and cadCAD simulation

Active inference is used here as a modeling vocabulary for agents that update beliefs and select actions under uncertainty. It is not evidence that an individual’s mental state can be reliably inferred from economic or biometric data.

### 7.1 Agent state

```python
agent = {
    "resources": {"sov": 0, "use": 0, "care": 0},
    "beliefs": {"task_success": 0.5, "institution_trust": 0.5},
    "preferences": {"income": 1.0, "care": 1.0, "privacy": 1.0},
    "consent": {"telemetry": False, "case_support": False},
    "risk_flags": [],
}
```

### 7.2 Required simulation parameters

All parameters must be versioned and swept over ranges:

- population size and agent heterogeneity;
- initial resource distribution;
- contribution categories and caps;
- attestation accuracy and collusion rate;
- dispute probability and resolution latency;
- AMM liquidity and fee schedule;
- issuance and decay rates;
- telemetry noise, latency, and false-alarm rate;
- governance turnout and delegation concentration;
- privacy cost and consent withdrawal rate;
- extreme-event frequency.

### 7.3 Minimum outcomes

A simulation is not successful merely because token price or throughput rises. Report:

- Gini and percentile resource distributions;
- concentration of governance power;
- unpaid or rejected claim rates;
- appeal success rates;
- privacy leakage proxies;
- false-positive and false-negative safety events;
- system recovery time;
- dependency on individual validators or data sources;
- distributional effects across vulnerable groups.

## 8. Asymmetric AMM and FINN clearing engine

The AMM is a proposed internal liquidity mechanism, not a guarantee of liquidity or price stability.

### 8.1 Bounded invariant

A deployment must publish the exact invariant and prove that fees, rounding, emergency exits, and asset insolvency cannot silently violate it. A generic asymmetric curve may be represented as:

```text
I(x, y) = w_x · f(x) + w_y · g(y)
```

where weights and functions are governance parameters subject to bounds. No production implementation should use this notation without a complete numerical specification, overflow analysis, and adversarial testing.

### 8.2 FINN clearing

The clearing engine should:

- batch orders to reduce priority manipulation;
- reveal a deterministic price rule before settlement;
- reject orders that exceed risk and slippage limits;
- isolate failed batches;
- publish all fee and reserve movements;
- support a pause and withdrawal path;
- never represent simulated liquidity as actual reserves.

### 8.3 Safety limits

```text
max_trade_size
max_price_impact
max_oracle_deviation
minimum_reserve_ratio
withdrawal_queue_limit
per-account rate limit
```

## 9. Smart-contract specification

Contracts should initially be deployed only to a local test network and audited before any value is accepted.

Required modules:

```text
SovereignAccounting
UsefulWorkRegistry
AttestationRegistry
CareGovernance
DisputeAndAppeal
TreasuryAndReserves
ParameterController
EmergencyPause
```

Contract invariants:

- total supply changes only through authorized, logged paths;
- every issuance references a finalized claim;
- every reversal references an original issuance;
- paused modules cannot transfer or mint unexpectedly;
- parameter changes have timelocks and version numbers;
- administrators cannot bypass appeals through a hidden path;
- upgrade authority is multisignature and publicly documented.

A **Sepolia-only, valueless `$OMEGA` pilot suite** now lives in [`evm/`](evm/). It contains a fixed-supply `tOMEGA` ERC-20, non-transferable vote escrow, bounded time locking, an OpenZeppelin Governor + TimelockController, and a public claim-receipt gate. A separate **Solana Devnet-only, valueless `tTWC` pilot issuer/verifier** now lives in [`solana/`](solana/): it uses the standard SPL Token Program, fixed one-time supply, null mint/freeze authorities, and immutable Metaplex fungible metadata. Neither pilot is deployed. Neither implements a mainnet token, Taproot Asset, custody system, or sensitive CARE/PoUW flow. Their assets, authorities, residual risks, and required drills are documented in [`evm/THREAT_MODEL.md`](evm/THREAT_MODEL.md) and [`solana/THREAT_MODEL.md`](solana/THREAT_MODEL.md). Both still require independent audit, legal review, adversarial drills, and explicit approval before any value-bearing deployment.

Before any production implementation, define and independently review storage layout, access control, event schemas, upgrade or redeployment policy, formal invariants, and adversarial test vectors.

## 10. Rust workspace specification

A future Rust workspace may use the following crate boundaries:

```text
crates/
  protocol-types       # versioned wire and ledger types
  commitments          # hashes, Merkle proofs, selective disclosure
  work-claims          # PoUW claim lifecycle
  attestations         # reviewer signatures and reliability
  ledger               # deterministic accounting state machine
  clearing             # AMM and FINN batch clearing
  telemetry            # signed measurements and quality flags
  privacy              # encryption and access-control adapters
  governance           # proposals, quorum, timelocks, delegation
  simulation           # cadCAD-compatible state transitions
  audit-cli             # replay, invariant checks, export
```

The core state machine should be deterministic and independent of network transport, database choice, or machine-learning inference. Models may propose actions; only deterministic policy code and authorized human governance may commit state.

## 11. Governance and public policy

A public-sector pilot must begin as a voluntary, reversible program with an independent ombuds function. It must not replace legal tender, public benefits, labor protections, medical care, or due process.

Before deployment, publish:

- jurisdictional legal analysis;
- economic impact assessment;
- privacy and data-protection impact assessment;
- accessibility assessment;
- algorithmic impact assessment;
- threat model and incident response plan;
- reserve and insolvency policy;
- procurement and conflict-of-interest disclosures;
- sunset and rollback criteria.

## 12. Threat model

Primary threats include sybil identities, attestor collusion, fabricated evidence, oracle manipulation, governance capture, liquidity attacks, insider abuse, privacy linkage, model drift, denial of service, and coercive participation.

The system must assume:

- telemetry can be wrong;
- validators can collude;
- administrators can make mistakes;
- users can be coerced;
- markets can become illiquid;
- a smart contract bug can be permanent;
- a model can amplify historical discrimination.

Security controls must be tested against these assumptions rather than against an ideal participant.

## 13. Delivery plan

### Phase 0 — falsification and review

Formalize terms, identify unsupported claims, define measurable hypotheses, and obtain independent review.

### Phase 1 — local simulation

Implement deterministic ledger types, PoUW claim replay, dispute flows, AMM stress tests, and active-inference parameter sweeps with no real tokens.

### Phase 2 — testnet pilot

Use synthetic identities and valueless test tokens. Run adversarial audits, accessibility testing, privacy attacks, and rollback drills.

### Phase 3 — bounded community pilot

Use voluntary participation, strict issuance caps, human review, transparent reporting, and a sunset date.

### Phase 4 — independent evaluation

Compare outcomes against a control or baseline. Publish failures, subgroup effects, and unresolved risks before expanding scope.

## 14. Acceptance criteria

The blueprint is ready for implementation only when:

- every protocol term has a normative definition;
- all token issuance and reversal paths are enumerated;
- state-machine invariants have executable tests;
- privacy and threat models are reviewed independently;
- simulations include adversarial and distributional outcomes;
- the physics telemetry is isolated from economic finality;
- appeals and exit work during outages;
- contracts pass audits and formal invariant checks;
- legal and policy approvals are documented;
- no participant must surrender unnecessary personal data to leave.

## 15. Immediate repository work

The next implementation artifacts should be produced in this order:

1. versioned protocol types and a deterministic ledger state machine;
2. a synthetic PoUW claim generator and dispute simulator;
3. invariant/property tests;
4. cadCAD-compatible simulation adapters;
5. a Sepolia-only, valueless `$OMEGA` pilot suite (implemented in [`evm/`](evm/));
6. a Solana Devnet-only, valueless `tTWC` SPL/Metaplex pilot issuer and verifier (implemented in [`solana/`](solana/));
7. threat-model documentation and adversarial test vectors for those pilots and the remaining chain legs.

This sequence intentionally postpones live deployment, monetary value, psychosocial inference, and telemetry-triggered economic action until the underlying claims and controls have been independently evaluated.
