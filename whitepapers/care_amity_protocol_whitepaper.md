# C.A.R.E. / AMITY Protocol Whitepaper

**Version:** 0.1 research draft  
**Status:** Social-infrastructure design; not a clinical system, benefits determination system, or promise of exchange value.

## Abstract

C.A.R.E. is the protected social and care-coordination layer of the ecosystem. AMITY is the exchange-eligible representation of approved, auditable value associated with that layer. They are related but intentionally not interchangeable:

- **CARE:** non-market proof-of-care records, attestations, service history, and community governance rights.
- **AMITY:** a constrained, market-facing token or wrapped representation that may be exchanged only under applicable compliance, reserve, custody, and consumer-protection rules.

This separation protects care relationships from direct market speculation while allowing approved providers and participants to settle eligible services.

## Proof of Care

Proof of Care is evidence that an agreed service, support action, safeguarding action, or community contribution occurred. It is not a score of empathy and must not rank a person’s moral worth.

A claim contains:

```text
claim_id
participant_commitments
service_category
consent_scope
quantity_and_time_window
evidence_commitment
reviewer_attestations
quality_and_safety_flags
appeal_state
protocol_version
```

Sensitive narratives, health information, and identifying details remain off-ledger under consent-controlled storage.

## Legitimacy verifiers

The people who verify Proof of Care are **CARE Verifiers**. Their role is to check legitimacy, not to make medical diagnoses or decide a person’s general eligibility for life opportunities.

A verifier must:

- be trained for the claim category;
- disclose conflicts of interest;
- review the minimum necessary evidence;
- sign a reasoned attestation;
- receive bounded reputation impact for repeated error;
- have their decision audited;
- provide correction and appeal routes;
- be prevented from unilaterally minting AMITY.

High-risk claims require multiple independent verifiers. Automated agents may identify missing fields or duplicate evidence, but cannot be the sole verifier of care legitimacy.

## CARE to AMITY boundary

A finalized Proof-of-Care record may create an eligibility or settlement entitlement under a published conversion policy. It does not automatically create freely transferable AMITY.

```text
CARE evidence -> multi-party verification -> finalized entitlement
             -> compliance / reserve checks -> eligible AMITY representation
```

The bridge must support rejection, expiration, reversal, and appeal. It must never expose protected care data to an exchange.

## Fees and reserves

Any fee schedule is a governance parameter, not an immutable promise. Proposed parameters such as a base transfer fee, a Sanctuary Gate fee, or a care-reserve allocation must be simulated, disclosed, legally reviewed, and change-controlled.

The treasury may absorb a bounded portion of volatility only if reserves are real, segregated, independently attested, and sufficient. An “80/20 loss split” is not protection unless the reserve actually covers the stated obligation.

Whitelisted vendors require due diligence, a public policy, expiration, monitoring, and an appeal path. A fee exemption must not become a covert surveillance or exclusion list.

## Arbitration

A dispute process should use randomized reviewer assignment only when the randomization source is auditable and the process is accessible. A three-disapproval rule or penalty rate must be tested for reviewer collusion, minority bias, false positives, and coercion before adoption.

Slashing must be:

- proportionate;
- evidence-based;
- reversible after appeal;
- bounded by a maximum;
- unavailable to a single administrator;
- recorded as a reasoned event.

## Privacy and safety

C.A.R.E. requires informed and revocable consent, purpose limitation, encryption, short retention, access logs, correction, deletion where required, and non-digital access. Participation must not be required to access ordinary legal rights or essential services.

## Governance split

CARE governance may oversee verification standards, ethics, privacy, accessibility, and safeguarding. AMITY market governance may oversee approved exchange parameters, liquidity, custody, and settlement. Neither group should be able to unilaterally change the other’s protected rules.

## Deployment stages

1. Synthetic claims and verifier training.
2. Human-reviewed, valueless testnet.
3. Privacy and discrimination impact assessments.
4. Voluntary pilot with an ombuds process.
5. Independent evaluation of false approvals, false rejections, appeals, and subgroup effects.
6. Only then consider regulated exchange eligibility.
