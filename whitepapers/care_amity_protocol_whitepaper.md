<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# C.A.R.E. / AMITY Protocol Whitepaper

**Version:** 0.2 research draft  
**Status:** Social-infrastructure design; not a clinical system, benefits determination system, or promise of exchange value.  
**Supersedes:** 0.1 (claims unchanged; this revision aligns the paper with the v0.2 architecture and
records the parameters and invariants that were settled after 0.1 was written).

> **Authoritative companion.** [`../docs/care-architecture-v2.md`](../docs/care-architecture-v2.md)
> is the current contract for the plane, and [`../docs/care-policy-v2.json`](../docs/care-policy-v2.json)
> is its machine-readable parameter set. Where this paper and those two disagree on a number, the
> machine-readable file governs. This paper carries the reasoning; it is not the parameter store.

> **Expansion.** C.A.R.E. stands for **Call About Resuscitating Everyone**.

> **Implementation status.** A local social prototype exists in
> [`../web/`](../web/README.md), a bounded CARE-to-AMITY conversion policy is encoded in
> [`../amity/`](../amity/README.md), and a Devnet-only, valueless `tTWC` issuer/verifier exists in
> [`../solana/`](../solana/README.md). None of it is deployed, audited, or approved for
> value-bearing use.

## Abstract

C.A.R.E. is the protected social and care-coordination layer of the ecosystem. AMITY is the exchange-eligible representation of approved, auditable value associated with that layer. They are related but intentionally not interchangeable:

- **CARE:** non-market proof-of-care records, attestations, service history, and community governance rights.
- **AMITY:** a constrained, market-facing token or wrapped representation that may be exchanged only under applicable compliance, reserve, custody, and consumer-protection rules.

This separation protects care relationships from direct market speculation while allowing approved providers and participants to settle eligible services.

CARE is described as protected, not as safe or stable. Support is protected only to the extent that real reserves, services, or sponsor commitments can cover it, and CARE is not called a stablecoin until its reference value, redemption, insolvency treatment, and audit process are defined.

## Non-negotiable invariants

The plane is bounded by ten invariants. They are constraints on implementation, not aspirations:

1. **No permanent moral exile.** Community access may be restricted indefinitely for safety, but every person retains a reviewable path to restoration.
2. **No vague economic offenses.** A sanction requires a published `rule_id`, reproducible evidence, an affected amount, and an appeal route.
3. **No punishment for hardship.** Hardship has no maximum duration, frequency, or lifetime allowance.
4. **No hidden device use.** Wallet resource contribution is visible, user-controlled, and capped.
5. **No automatic clinical or psychological judgment.** Agents may assist with completeness and anomaly checks; they cannot diagnose, decide moral worth, or independently punish.
6. **No personal enforcement profit.** No founder, Archangel, reviewer, or agent receives a percentage of a user's sanction.
7. **No unbacked value creation.** A pending entitlement may not become a market-facing asset unless the applicable policy and reserve checks pass.
8. **No unrelated balance confiscation.** A proven exploit can reverse the exact fraud-derived delta; legitimate unrelated CARE is not automatically seized.
9. **No sensitive data by default.** Raw conversations, trauma narratives, health information, and precise location histories remain private and off-chain unless separately consented.
10. **No required government support.** Government grants may supplement the system but are not a core invariant.

An invariant that cannot be enforced by the deployed system must be documented as unmet rather than
assumed.

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

## Social application first

The first C.A.R.E. product is a social application, not a token exchange: pseudonymous or verified
profiles, user-controlled circles, care requests and offers, posts and group activity,
consent-controlled sharing, blocking and boundary controls, safety reports and appeals, weekly
participation views, and voluntary in-person attestations.

Ordinary participation is **not** gated by AMITY, OMEGA, TWC, legal-name disclosure, or a high
privacy tier. Any design that makes basic participation conditional on holding or disclosing
something has failed this section.

## Participation, reach, and weekly distribution

The protocol may reward observable social reach, because reach and attention are measurable proxies
for reducing isolation. It must not claim that raw reach proves a person's private motives or
therapeutic outcomes, and it must publish the definition rather than operate a hidden authenticity
score:

```text
raw reach units = unique audience observations + attention events
```

Coefficients and event definitions are versioned and public. Automated-account and duplicate-event
limits protect the ledger; they are not judgments about a person's character.

Issuance is a **fixed weekly pool**, not an uncapped per-person mint:

```text
participant allocation = weekly pool × participant reach units / total eligible reach units
```

The pool and policy version are published before the epoch. Early participation may produce larger
individual allocations because fewer participants share the pool; the design does not need a burn to
imitate early Bitcoin distribution, because unissued holdbacks simply never enter circulation.

## Wallet resource contribution

A wallet may contribute a small, user-defined resource quota while the device is powered on. The
initial design target is a **2% device-resource budget** — a policy target, not a verified claim of
exact electrical consumption. Receipts may account for CPU time, network bandwidth, storage, relay
uptime, memory or service capacity, and optional operating-system energy estimates.

```text
protocol minimum <= user quota <= device-safe maximum
```

The wallet must show the quota, current usage, remaining requirement, and pause controls, and must
pause for low battery, overheating, battery-saver mode, or the user's own limits. A quota may be
satisfied through the user's own device, a delegated community node, an approved hosted provider, or
a sponsored contribution. A wallet is not automatically a full blockchain node; light-client use is
supported.

A missed quota changes activity status — `active -> grace -> under-quota` — and never destroys a
person's history. When funded settlement capacity is unavailable, conversion requests may be queued;
the protocol must not create unbacked AMITY to conceal an uncovered obligation.

## Hardship and solidarity

Hardship is unbounded in every direction:

```text
hardship duration cap = none
hardship count cap = none
hardship assistance cap = none
forced hardship expiry = forbidden
```

A person may request indefinite hardship status without publicly disclosing the reason, and hardship
creates no debt. Other holders may voluntarily cover the deficit with verified resource
contributions. A sponsor receives a published, bounded bonus based on actual contribution — never on
the recipient's private hardship details — and receives no control, ownership, contact rights, or
governance power over the person receiving support.

If no sponsor is available: the hardship status remains valid, the person is not expelled or
punished, CARE is not destroyed, the uncovered need remains an open support request, and settlement
may be delayed rather than minted against no reserve.

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

Before any conversion, a weekly settlement record must show:

```text
CARE/reach entitlement
gross AMITY calculation
participant dock
reserve allocation
unissued holdback
net AMITY
policy version
resource/quota status
```

The initial conversion schedule is a dock and holdback, not a burn:

```text
initial:   5% participant dock
maximum:  20% participant dock
extreme:  20% dock + 15% unissued holdback
```

The base design does not burn supply. The extreme 15% is **not issued** rather than destroyed after
circulation, and it can never be applied as a hidden tax outside the published policy. The total
reduction is capped, and conversion is available whenever funded settlement capacity exists — the
policy queues the request rather than minting an uncovered entitlement.

## Roles and promotion

The role ladder exists to distribute responsibility, not to rank human worth:

```text
Level 0: Participant
Level 1: X1
Level 2: X2
Level 3: Archangel
```

- **Participant** — may use the app, join communities, request support, contribute resources, and appeal decisions.
- **X1** — requires safety and consent training, supervised participation, and no unresolved serious integrity case; may assist with low-risk activities.
- **X2** — requires independent review, supervised attestations, conflict disclosure, and additional safeguarding training; may serve on limited verification panels.
- **Archangel** — requires nomination or selection, independent conflict review, senior safeguarding training, a fixed term, and a reviewable appointment.

No role may be purchased with AMITY, TWC, or OMEGA. Draft capacity for the top of the ladder is:

```text
maximum local Archangels = floor(active participants / 500)
```

Small communities use a shared regional pool rather than appointing a single local authority.
Archangels receive a bounded, published governance allocation rather than unlimited control; within
it, active Archangels receive equal shares. Economic and treasury decisions require broader
participation than the Archangel council, and conflicted Archangels recuse themselves. An Archangel
cannot unilaterally mint, slash, erase records, or override an appeal.

## Integrity sanctions

Community safety and protocol integrity are separate systems. Safety actions may restrict access to a
person, circle, venue, or community; they do not automatically confiscate balances. Only enumerated
technical or economic events — forged signatures, duplicate finalized claims or nonces, double
redemption, unauthorized state transitions, deliberate oracle manipulation, deliberate contract
exploitation, falsified verifier attestations, coordinated issuance or identity attacks — count as
protocol-integrity offenses. A normal mistake, a good-faith overclaim, a hardship request, or a bug
report is not one.

The three-offense rule applies only to final, adjudicated violations, and one exploit is not counted
multiple times for affecting multiple records.

**First confirmed violation:** freeze the affected pending path; recover the exact fraud-derived or
unauthorized delta; remove relevant trusted permissions; preserve unrelated legitimate CARE; provide
an appeal.

**Second confirmed violation**, for the exact illicit delta:

```text
75% permanently removed or left unissued
15% security, restitution, or protocol reserve
10% independent audit and bug-bounty fund
```

There is no founder or individual enforcement share.

**Third confirmed violation:** apply the second-offense financial rule to the new illicit delta,
impose indefinite temporary exclusion from the affected community, remove trusted roles, require
periodic independent review, and preserve a restoration path — never permanent moral exile.

Responsible vulnerability disclosure receives safe-harbor treatment and may receive a fixed bounty.

## Fees and reserves

Any fee schedule is a governance parameter, not an immutable promise. Proposed parameters such as a base transfer fee, a Sanctuary Gate fee, or a care-reserve allocation must be simulated, disclosed, legally reviewed, and change-controlled.

The treasury may absorb a bounded portion of volatility only if reserves are real, segregated, independently attested, and sufficient. An “80/20 loss split” is not protection unless the reserve actually covers the stated obligation.

Whitelisted vendors require due diligence, a public policy, expiration, monitoring, and an appeal path. A fee exemption must not become a covert surveillance or exclusion list.

## Randomized Proof-of-Care audits and arbitration

Finalized Proof-of-Care claims and high-impact disputes are assigned through an auditable random process. A case is sent to:

1. one Archangel selected from an eligible, conflict-screened pool; and
2. a panel of AI audit agents selected from independently governed providers, model families, and evaluation profiles.

The Archangel is the accountable human decision-maker, not the owner of the entire audit process. Their own agents cannot constitute the whole panel or provide a hidden override. At least one panel member must be independent of the assigned Archangel’s organization, and high-risk cases require a second human reviewer or appeal panel.

Assignment should use a public commitment to a verifiable randomness source, a case-specific nonce, and a published selection algorithm. The system must record the selection event without exposing sensitive case data. Evidence should be blinded or minimized so reviewers do not see unnecessary identity, status, wealth, or protected health information.

Anti-collusion controls include:

- independent provider and model-family diversity;
- concealed reviewer identities until required for due process;
- conflict-of-interest declarations;
- reviewer rotation and rate limits;
- pairwise agreement and suspicious-pattern monitoring;
- deterministic audit logs and signed reason codes;
- quorum decisions rather than one-model approval;
- random re-audits of approved and rejected cases;
- an independent appeal panel with no conflicted members;
- sealed evidence commitments that prevent after-the-fact editing.

AI agents may identify missing evidence, compare the claim with policy, reproduce calculations, and flag disagreement. They may not diagnose a person, infer moral worth, or independently finalize a high-impact care decision. The human Archangel must record the reasoning, uncertainty, dissenting signals, and appeal route.

No system can honestly promise that collusion is impossible. The security goal is to reduce the payoff, distribute authority, detect correlated behavior, preserve evidence, and make correction practical.

Slashing or reversal must be:

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

These visibility levels are separate from the accountability proof levels, which run in the other
direction — from pseudonymous participation to verified person, verified service or community
history, and senior accountable steward:

```text
P0: pseudonymous participation
P1: verified person credential
P2: verified service or community history
P3: senior accountable steward
```

Higher proof levels may unlock greater payout limits, verifier eligibility, or governance
responsibilities. They do not make a person more valuable, and P0 is always sufficient for basic
participation.

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

The protocol has four planes, and CARE governance is not the governance of the others:

| Plane | Owns |
|---|---|
| CARE | Care records, consent, privacy, safeguarding, verifier standards, the weekly pool policy |
| TWC | Useful-work receipt classes and their verifier adapters |
| OMEGA | Protocol parameters, treasury, and long-horizon security commitments under timelock |
| AMITY | Exchange eligibility, custody, reserves, and settlement parameters |

CARE consumes TWC receipts — a resource receipt can satisfy a wallet quota, infrastructure work can
contribute service capacity, engineering work can improve the oracle or the social application — but
CARE does not become a TWC-only economy, and a TWC contributor does not need to participate in CARE
to earn TWC. The cross-plane primitives and the settlement funding flow are specified in
[`../docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md).

## Public-health prevention and government partnership

C.A.R.E. can be proposed to governments, foundations, universities, and community organizations as a voluntary prevention and connection infrastructure. The purpose is broader than substance-use or rehabilitation programs. It may support people who are processing trauma, isolation, grief, family conflict, domestic violence, coercive control, or other destabilizing experiences by giving them a safer way to speak, listen, find resources, and seek human help earlier.

The policy case should be framed as a hypothesis to evaluate, not a guaranteed GDP intervention. Domestic violence and untreated trauma can impose substantial human, health, housing, legal, and economic costs, but the project must not claim that a token or social feed alone prevents violence. Government grants should fund trauma-informed design, survivor-led governance, qualified moderation, crisis referral, accessibility, privacy engineering, and independent evaluation—not speculative token price or engagement growth.

Grant applications should measure outcomes such as:

- successful connection to qualified support;
- safety-plan completion where voluntarily offered;
- reduced isolation and participant-reported distress;
- time from disclosure to human support;
- rates of harmful escalation, harassment, and retraumatization;
- appeal, moderation, and false-report outcomes;
- retention of privacy and consent;
- subgroup and accessibility effects.

No participant should be required to disclose trauma to receive ordinary services. Participation must remain voluntary and must not replace shelters, emergency services, medical care, legal aid, child protection, or domestic-violence specialists.

### Storytelling and peer-support spaces

The platform may offer separate spaces for:

1. **Reflection rooms:** people speak about their own experiences, anonymously or pseudonymously.
2. **Peer-support rooms:** trained facilitators help participants listen without diagnosing or judging.
3. **Resource rooms:** qualified organizations publish support, safety-planning, housing, legal, and crisis resources.
4. **Mediation rooms:** only when all relevant participants consent and a trained human facilitator determines it is safe.
5. **Verified claim rooms:** evidence-bearing claims are handled by the Proof-of-Care and appeal process, not by a general feed.

A participant may describe their own experience without having to prove it to deserve compassion. That is different from presenting an allegation as an established fact. The product must clearly label personal narrative, opinion, corroborated evidence, and unresolved allegation.

### Consent before naming another person

Users should not identify another person by legal name, contact details, workplace, address, image, or uniquely identifying story without that person’s explicit consent, except through narrowly scoped lawful safeguarding channels. A person who is named may request an interjection, correction, redaction, or removal review. Interjection does not give the named person control over another person’s lived experience, but it does trigger privacy, safety, and defamation review.

The safer default is to describe conduct and impact without identifying details. Doxxing, threats, targeted harassment, coordinated retaliation, fabricated evidence, and attempts to expose a survivor are prohibited. A private moderation or safeguarding channel may preserve relevant evidence without making it public.

### Compassion is not adjudication

An unverified story can still help a participant process experience and connect with support; the system should not demand courtroom-level proof before offering empathy. However, false or malicious allegations are not harmless merely because some readers found them useful. Unverified stories must not automatically trigger token issuance, public punishment, employment consequences, financial sanctions, or safety action against an identifiable person.

Moderation should focus on immediate harm, privacy, coercion, and threats. Factual disputes belong in a separate review process with notice, evidence standards, human oversight, correction, and appeal. This preserves a non-judgmental support culture without turning the platform into an accusation marketplace.

### Safeguarding boundaries

The service must display location-appropriate crisis and domestic-violence resources, provide a fast exit and device-safety guidance, and avoid notifications that could expose a participant to an abuser. Where there is an imminent risk of serious harm or a legally required reporting duty, trained human safeguarding personnel follow the applicable law and disclose the minimum necessary information. Agents may flag risk signals, but they must not independently diagnose, contact authorities, confront an alleged abuser, or make a survivor’s safety plan public.

## Deployment stages

1. Synthetic claims and verifier training.
2. Human-reviewed, valueless testnet.
3. Trauma-informed design, privacy, and discrimination impact assessments.
4. Survivor-led and community-led pilot with an ombuds process.
5. Independent evaluation of support connection, safety, false approvals, false reports, appeals, and subgroup effects.
6. Only then consider regulated exchange eligibility.

## Mainnet gate

No value-bearing mainnet deployment should occur until every one of the following holds. This is a
gate, not a checklist to be waived by schedule pressure:

1. The social application exists and is usable without token pressure.
2. The C.A.R.E. rules are machine-readable and tested.
3. The hardship and solidarity flow has been piloted.
4. Physical attestations have been tested voluntarily and privately.
5. The role ladder and Archangel ratio have been stress-tested.
6. Protocol-integrity cases have reproducible evidence and appeal paths.
7. Weekly payout and reserve simulations pass severe drawdown scenarios.
8. Wallet resource use is opt-in, visible, and bounded.
9. Independent security, privacy, safeguarding, and legal reviews are complete.
10. The system does not rely on unverified clinical, economic, or performance claims.

Until then, C.A.R.E., AMITY, OMEGA, and TWC remain research or valueless testnet components. The
repository contains local and Devnet-only tooling for the mechanical parts of this work; that tooling
does not satisfy the gate, and its existence must not be presented as compliance.

## Revision history

- **0.2 (2026-09-25)** — Aligned with `docs/care-architecture-v2.md` and `care-policy-v2.json`:
  added the ten invariants, social-application-first scope, reach and fixed weekly pool, the 2%
  wallet-resource budget, unbounded hardship and solidarity, the role ladder and Archangel ratio,
  progressive integrity sanctions, the conversion dock/holdback schedule, the four-plane governance
  split, proof levels P0–P3, and the mainnet gate. No safety commitment was weakened.
- **0.1 (2026-09-23)** — Initial research draft.
