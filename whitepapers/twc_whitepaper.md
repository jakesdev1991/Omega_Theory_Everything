<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# Token of the World Citizen (TWC) Whitepaper

**Version:** 0.1 research draft  
**Status:** Design specification; Devnet pilot only; not a token sale, investment product, or promise of value.  
**Revision date:** 2026-09-25

## Abstract

Token of the World Citizen (TWC) is the work-and-resource coordination plane in the four-plane Omega ecosystem. It is intended to record and settle independently verifiable useful work: engineering, infrastructure, formal verification, scientific computation, and other bounded contributions whose acceptance criteria can be published and reproduced.

TWC is not a measure of a person's worth, citizenship, morality, care, wealth, or device ownership. A token balance is not proof that work was useful. The protocol's central claim is narrower: a properly verified work receipt can support a bounded settlement record without relying on an issuer's subjective confidence alone.

The current implementation is a Solana Devnet-only, deliberately valueless pilot named **`tTWC`**. The proposed production identity is **TWC**. No mainnet launch, exchange listing, yield, redemption, or monetary value is authorized by this paper.

## 1. Scope and design principles

TWC follows six principles:

1. **Evidence before issuance.** No receipt, no settlement.
2. **Class-specific verification.** Software, Lean proofs, scientific research, uptime, and physical infrastructure require different acceptance tests.
3. **Reproducibility over reputation.** A contributor's title, identity, or model confidence is not an acceptance result.
4. **Bounded budgets.** A proposal states its resource and settlement limits before execution.
5. **Independent review.** The submitting party cannot be the sole verifier of its own work.
6. **No automatic social conversion.** TWC does not create CARE records, OMEGA voting power, or AMITY balances merely because TWC was issued.

These principles are coordinated with [`docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md), which is the interim integration specification, and with the machine-readable CARE policy where social-plane parameters are involved.

## 2. Position in the four-plane model

| Plane | Responsibility | TWC boundary |
|---|---|---|
| CARE | Social participation, support, reach, hardship, and solidarity records | TWC cannot certify care or determine human legitimacy |
| TWC | Useful work, infrastructure, proving, and resource receipts | TWC cannot unilaterally change protected CARE rules |
| OMEGA | Delayed governance, policy authority, and treasury controls | TWC holdings do not automatically confer governance authority |
| AMITY | Voluntary exchange-facing representation of approved CARE entitlements | TWC is not automatically convertible into AMITY |

The planes may exchange explicitly versioned receipts, but there are no hidden conversions or implied guarantees between them.

## 3. What counts as useful work

TWC does not define one universal proof algorithm. It defines a common receipt envelope and requires a work-class adapter to explain how acceptance is determined.

Initial work classes are:

- software and security engineering;
- formal verification and Lean proof artifacts;
- zero-knowledge or cryptographic proving;
- public infrastructure and service uptime;
- scientific or physics computation;
- data, benchmark, and reproducibility work; and
- other classes approved through a versioned governance process.

A class adapter must specify the artifact, input commitments, acceptance tests, resource bounds, verifier set, failure conditions, and appeal process. Raw CPU/GPU time, unverifiable claims, popularity, social status, or a model's self-reported confidence are insufficient.

## 4. Work lifecycle and receipt format

The canonical lifecycle is:

```text
proposal
→ acceptance specification
→ bounded execution
→ artifact and resource receipts
→ deterministic checks
→ independent review
→ appeal window
→ settlement receipt
```

A proposal includes at least:

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

A completed receipt includes at least:

```text
artifact_hash
repository_or_dataset_reference
execution_attestation
test_report
resource_receipts
verifier_set
verification_result
policy_version
appeal_state
```

Receipts should be content-addressed where practical. A verifier must be able to distinguish **accepted**, **rejected**, **inconclusive**, **under appeal**, and **simulated/counterfactual** results. A research artifact, formalization attempt, and kernel-checked theorem must not be presented as equivalent evidence.

## 5. Issuance and settlement

TWC issuance is a settlement consequence, not a mining reward for unexamined effort. A proposed settlement must be bounded by the approved work proposal and its policy version. The receipt records who accepted the work, which verifiers participated, what evidence was examined, and whether an appeal remains open.

The interim architecture uses versioned `ParticipantCredential`, `WorkProposal`, `EvidenceCommitment`, `VerificationResult`, `Entitlement`, `SettlementEpoch`, `Appeal`, and `PolicyVersion` primitives. Mutations identify actor commitments, input/output commitments, evidence references, review state, and appeal state.

No production emission curve, supply cap, treasury allocation, or redemption schedule is established by this v0.1 paper. Those are open governance decisions and must be published in a later version before any value-bearing deployment is considered. The Devnet pilot is intentionally separate from that decision.

## 6. Current Devnet pilot

The repository's current pilot uses the standard Solana SPL Token Program and Metaplex Token Metadata. It creates no bespoke Solana program and uses no Token-2022 extensions.

Pilot properties:

- network: Solana Devnet only;
- visible symbol: `tTWC`;
- proposed name: Token of the World Citizen;
- decimals: 9;
- default test supply: 1,000,000,000 `tTWC`;
- mint authority: removed after the one-time pilot mint;
- freeze authority: `None`;
- metadata: immutable after creation; and
- value: deliberately none.

The implementation, guarded deployment procedure, metadata hash checks, and independent verifier are documented in [`solana/README.md`](../solana/README.md). A Devnet balance is not evidence of useful work, an entitlement, citizenship, or future mainnet allocation.

## 7. Verification, appeals, and anti-gaming controls

A production design must address at least:

- duplicate submissions and copied artifacts;
- unverifiable or non-reproducible results;
- collusion among contributors and reviewers;
- fabricated execution or resource receipts;
- benchmark gaming and acceptance-test overfitting;
- reviewer capture and conflicts of interest;
- identity concentration without making sensitive identity public;
- denial-of-service through appeal volume; and
- work whose social or physical externalities exceed its stated benefit.

Controls should include independent verifier selection, deterministic replay where possible, delayed settlement, bounded reviewer authority, conflict declarations, evidence commitments, audit trails, and a reviewable appeal path. A risk control may reject or suspend a receipt; it must not silently rewrite the historical evidence.

The protocol should preserve uncertainty. When evidence is insufficient, the correct result is **inconclusive**, not an invented confidence score or forced payout.

## 8. Governance and authority

TWC policy changes require versioned governance and a delayed effective date. Governance may approve work classes, verifier requirements, settlement limits, and emergency pauses within published authority. It must not silently convert TWC into CARE, OMEGA, or AMITY.

Protected boundaries include:

- no unilateral change to CARE's protected invariants by TWC operators;
- no automatic OMEGA voting power from TWC balances;
- no automatic AMITY conversion from work receipts;
- no personal enforcement percentage for founders, reviewers, agents, or stewards; and
- no mint-or-move-value authority hidden inside an ordinary work verifier.

The agentic control boundary described in [`lucifer_hermes_omni_bridge_whitepaper.md`](lucifer_hermes_omni_bridge_whitepaper.md) is relevant to capability separation, but it is not a substitute for a production TWC governance system.

## 9. Privacy and public audit

Public auditability must not require publishing unnecessary personal or sensitive data. The preferred record is a commitment to the contributor, artifact, evidence, and policy version, with selective disclosure or privacy-preserving credentials where appropriate.

Privacy cannot be used to make acceptance unverifiable. A verifier must receive enough evidence to reproduce or independently assess the stated result. Sensitive data should remain off-chain by default, with explicit consent, retention limits, and an appeal process for disputed disclosure.

## 10. Security and operational status

The Devnet pilot is not audited and is not production software. Mainnet use would require, at minimum:

1. an approved token identity and legal/trademark review;
2. a pinned supply and issuance policy;
3. independent smart-contract and infrastructure audits;
4. a threat model covering issuers, reviewers, treasuries, bridges, metadata, and key compromise;
5. reproducible verifier implementations and test vectors;
6. a functioning appeal and incident-response process;
7. privacy and data-protection review;
8. governance approval with a delayed activation window;
9. documented treasury and custody controls; and
10. an explicit decision that any deployment may carry value or redemption rights.

Until these gates are met, TWC remains a research and valueless testnet component.

## 11. Open questions for v0.2

This paper deliberately leaves the following unresolved:

- whether TWC should ever be value-bearing;
- the production supply and issuance schedule, if any;
- whether settlement is continuous or epoch-based;
- how physical work and infrastructure uptime are independently verified;
- the acceptable relationship between pseudonymous contributors and accountable legal entities;
- verifier compensation and conflict handling;
- cross-chain representation, if any; and
- the conditions under which a disputed receipt is finally closed.

A later paper must answer these questions with testable policy, not marketing language.

## 12. Summary

TWC is a proposed evidence-and-settlement plane for useful work. Its unit of trust is not the token; it is the reproducible receipt behind the token. The current `tTWC` asset is a Devnet-only pilot with no value. Any future deployment must preserve class-specific verification, bounded issuance, independent review, privacy by default, explicit appeals, and strict separation from CARE, OMEGA, and AMITY.

## License

This whitepaper is proprietary product documentation. See [`../docs/LICENSING.md`](../docs/LICENSING.md) for the repository's licensing and rights notices.
