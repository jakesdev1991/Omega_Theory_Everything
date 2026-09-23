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

## Privacy, anonymity, and safety

C.A.R.E. is anonymous or pseudonymous by default. A participant may choose a disclosure level independently for identity, profile, work history, care history, location, and agent availability. Greater visibility may unlock more collaboration or earning opportunities, but it is never required for basic participation and never proves that a person is more valuable.

Suggested visibility levels are:

```text
L0  fully private: local wallet and local agent only
L1  pseudonymous: public handle, no legal identity
L2  contribution-visible: approved work and aggregate outcomes
L3  community-visible: profile, projects, and contact preferences
L4  verified disclosure: identity shared only with a specific authorized party
```

Each level requires a separate, plain-language opt-in. Opting out must be easier than opting in. Consent must be revocable where legally and operationally possible, and a withdrawal must not erase an immutable public event; instead, public records should be minimized, detached, or replaced with a privacy-preserving revocation marker.

A user’s personal agent and node run on hardware controlled by that user or a clearly designated custodian. They are not discoverable, callable, or visible to other users by default. A user may selectively publish an agent capability, accept a task, or expose a proof endpoint for a limited period. Private keys, raw prompts, private memories, sensor streams, and local files never belong in the public social layer.

C.A.R.E. requires informed consent, purpose limitation, encryption, short retention, access logs, correction, deletion where required, and non-digital access. Where the system handles protected health information for a HIPAA-covered entity or business associate, it must be designed and operated to meet HIPAA’s applicable Privacy, Security, and Breach Notification requirements. HIPAA may not apply to every deployment, but health information still requires strong protection under applicable state, federal, and sector-specific laws. Privacy is not a mechanism for evading lawful health, tax, financial, labor, or public-safety duties. Where a legally required disclosure exists, the protocol should disclose the minimum necessary information through a scoped, audited channel rather than expose a user’s entire identity or agent state.

## Human governance: Archangels

The agentic framework is governed by accountable humans. Within C.A.R.E., senior human stewards are called **Archangels**. This is a role title, not a claim of supernatural authority, infallibility, or personal superiority.

Archangels may:

- supervise CARE Verifier training and accreditation;
- review high-impact or disputed Proof-of-Care decisions;
- protect consent, privacy, accessibility, and non-coercion;
- approve or reject changes to care-verification standards;
- appoint independent ombuds and appeal panels;
- audit agent behavior and suspend unsafe automation;
- publish conflict-of-interest and stewardship reports.

An Archangel cannot unilaterally mint AMITY, erase a record, deny an appeal, or bypass the separation between CARE evidence and the exchange layer. Every high-impact action requires a documented reason, an audit event, and an independent review route.

### Stewardship progression and the 80/20 rule

Archangel progression is based on demonstrated stewardship rather than wealth. A proposed level score is:

```text
level_evidence = 0.80 × care_integrity_score
               + 0.20 × economic_stewardship_score
```

The 80% component measures sustained empathy in action, non-judgmental listening, reliable follow-through, safeguarding, truthful disclosure, repair after mistakes, and the outcomes of independent reviews. The 20% component measures responsible management of resources, transparent budgeting, and contribution to sustainable infrastructure. Economic value alone cannot compensate for failures of care integrity.

This formula is a governance proposal, not an objective measurement of a human being. Levels must be bounded, periodically re-evaluated, non-transferable, and subject to appeal. A person’s level must never determine access to essential rights, healthcare, legal protection, or basic services.

The protocol treats genuine, non-coercive care as more important than accumulated value. It must therefore reject leaderboard incentives that reward performative kindness, dependency creation, surveillance, or financial extraction. Advancement requires evidence from multiple perspectives and must include a path back after a substantiated mistake.

### Distinction, judgment, and safety

C.A.R.E. distinguishes **behavior, evidence, and risk** for survival and safeguarding purposes. This is not the same as judging a person’s identity, dignity, motives, or permanent worth.

The design principle is that reflexive judgment often expresses fear, insecurity, status defense, or incomplete information. That principle does not prohibit evidence-based boundaries. A verifier may still say that a behavior is unsafe, pause a process, seek help, or protect a participant. The required standard is:

```text
observe behavior -> check evidence -> assess immediate risk
-> communicate uncertainty -> choose the least harmful safeguard
-> provide notice, repair, and appeal
```

No agent or Archangel may convert an inferred motive into a permanent label.

### Thought-virus communication failure model

A **thought virus** is a C.A.R.E. communication model for a mutually reinforcing misunderstanding: two or more people receive incomplete or ambiguous information, form skewed interpretations of reality, and then use those interpretations as faulty foundational assumptions in later decisions. The term describes a relational information failure; it is not a psychiatric diagnosis, proof of bad character, or evidence that a participant is dangerous.

A thought-virus report must therefore identify:

- the original observable communication;
- each participant’s interpretation;
- the uncertainty and missing context;
- the point where interpretations began reinforcing one another;
- the concrete harm or risk, if any;
- the proposed repair conversation or mediation.

The response protocol is:

```text
pause escalation -> separate observation from interpretation
-> invite each perspective -> verify shared facts
-> repair the misunderstanding -> document consent and next steps
```

Safety exceptions are permitted when there is an immediate risk of violence, exploitation, or serious harm. Even then, the least-restrictive protective response, human oversight, notice, and later review are required. Agents may flag possible communication divergence, but must not diagnose, shame, isolate, or punish people based on a model inference.

## Governance split

CARE governance may oversee verification standards, ethics, privacy, accessibility, safeguarding, Archangel accreditation, and thought-virus mediation standards. AMITY market governance may oversee approved exchange parameters, liquidity, custody, and settlement. Neither group should be able to unilaterally change the other’s protected rules.

## Deployment stages

1. Synthetic claims and verifier training.
2. Human-reviewed, valueless testnet.
3. Privacy and discrimination impact assessments.
4. Voluntary pilot with an ombuds process.
5. Independent evaluation of false approvals, false rejections, appeals, and subgroup effects.
6. Only then consider regulated exchange eligibility.
