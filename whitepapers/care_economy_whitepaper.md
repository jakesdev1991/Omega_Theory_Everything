<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# The C.A.R.E. Economy — Extensive Whitepaper

**Version:** 1.0 — first extensive edition
**Status:** Research and systems-design specification. The C.A.R.E. Economy is a hypothesis about how an economy can be organized around care. Nothing in this document is a validated scientific result, an offer of securities, a promise of token value, a clinical protocol, or a benefits-determination system. All current implementations are local, valueless, testnet-only prototypes.
**Scope:** This whitepaper covers the **entire economy** — every plane, every rail, every surface — not only the CARE proof layer. The companion documents listed in §31 cover individual subsystems in more depth.
**Authorship:** Jacob See, with the Omega research program.

---

## Abstract

The C.A.R.E. Economy is the name of this entire economy: the sovereign, three-currency, privacy-first economy built in the Omega_Theory_Everything repository. It takes its name from the conviction that care — verifiable, consent-respecting, human-governed care — is the base layer of any economy worth building, and that an economy which starts from judgment extracts value, while an economy that starts from dignity creates it.

The economy runs on **five planes** (SOV accounting, USE contribution, CARE stewardship, AMITY exchange, OMEGA governance), settles across **three public rails** ($OMEGA on EVM, TWC on Solana, AMITY on Bitcoin / Lightning / Taproot), and is operated day-to-day through the **Omega MCP Hub**, a wallet-first GUI, a web gateway with a token-gated novel, an Economy Test Console, and a Nostr application backplane.

Every mechanism in this document is stated as a design proposal with its failure modes attached. Where a mechanism exists in code today, we say so. Where it is simulated, we say so. Where it does not exist yet, we say so loudest of all.

**The expansion of the name** (§2): **Call About Resuscitating Everyone** — the canonical mission of the entire economy, already published in `docs/care-architecture-v2.md`. Each word is load-bearing. This is not a clinical system, an emergency service, or a hospital protocol.

---

## Part I — The name and the vision

### 1. Why the whole economy is called the C.A.R.E. Economy

Earlier internal documents used "CARE" narrowly, as the name of one plane among five: the protected, non-market proof-and-verification layer. That plane still exists and still carries that name. But a name that only labels one subsystem undersells the design intent. The claim of this project was never "here is an ordinary economy with a care feature bolted on." The claim is:

> **Care is the base layer.** Accounting, work, exchange, and governance are all built on top of it, and every one of them inherits its constraints.

When the base layer of an economy is care, the economy itself is a C.A.R.E. Economy. The name is therefore promoted, by this document, to the **name of the entire economy**:

```text
The C.A.R.E. Economy
├── CARE plane          (the namesake: protected, non-market proof of care)
├── USE plane           (proof of useful work; earn while you learn)
├── SOV plane           (accounting and settlement unit)
├── AMITY plane         (compliance-gated exchange of approved entitlements)
├── OMEGA plane         (scarce macro-governance asset)
└── the rails           ($OMEGA on EVM · TWC on Solana · AMITY on Bitcoin/Lightning/Taproot)
```

Nothing about the underlying separation of powers changes. What changes is which idea gets top billing.

### 2. What C.A.R.E. stands for

The expansion is canonical, already published in [`docs/care-architecture-v2.md`](../docs/care-architecture-v2.md):

> **Call About Resuscitating Everyone.**

```text
C  — Call            the invitation: reach out first; social application
                     before markets; a call is an offer, not a command
A  — About           the purpose is specific: this economy exists to
                     restore people, not to rank, extract, or speculate
R  — Resuscitating   the verb: restore, revive, make a path back;
                     Proof of Care, peer support, earn while you learn
E  — Everyone        the scope: no one left outside; no permanent exile;
                     anonymity as a right; first-class at zero visibility
```

Each word is load-bearing.

- A deployment that waits for people to prove they deserve help is not a **Call**.
- A deployment whose purpose is speculation or ranking is not **About** resuscitation.
- A deployment that triages the "deserving" instead of restoring is not **Resuscitating**.
- A deployment that leaves anyone outside the invitation is not for **Everyone**.

An implementation that drops any of those words — that lets exchange lead, that skips the invitation, that replaces revival with triage, that walls anyone out — is not a C.A.R.E. Economy regardless of what it calls itself.

**"Resuscitating" is the mission metaphor of the economy, not a clinical claim.** C.A.R.E. is not an emergency service, a hospital protocol, a benefits-determination system, or a measure of a person's moral worth. When someone needs emergency medical care, the correct action remains calling emergency services. What this economy builds is the call that can reach a person before that moment, the support that can hold them after it, and the economic standing that does not vanish because they needed either.

Compassion, human accountability, reciprocity of useful work, and a narrow exchange boundary remain **design principles** of the planes (§5, §12–§18). They are not the expansion of the name.

### 3. The diagnosis: the judgment economy

The C.A.R.E. Economy is a response to specific, observable failure patterns in existing economic systems:

1. **Judgment at the door.** Housing, work, credit, and community routinely require people to present themselves as finished, undamaged, and continuously productive. A person's worst season becomes a permanent price tag.
2. **Surveillance as the price of belonging.** Platforms convert private life into engagement data. Visibility is mandatory; privacy is coded as suspicious.
3. **Care treated as externalities.** The work of holding families and communities together — the work that makes every other kind of work possible — is unpaid, unrecorded, and invisible to every ledger that matters.
4. **Extractive matching.** People in crisis are the most valuable users in the attention economy. Systems are optimized to find them at their weakest and sell them something.
5. **Governance by hidden knob.** The rules of platforms and financial systems are changeable by their operators at any time, without notice, appeal, or recorded reasoning.

Each failure has a corresponding design answer in this economy: entry without judgment (§12, §16), privacy as a right (§16), Proof of Care (§12), safeguarding boundaries that refuse extractive matching (§14, §21), and auditable, versioned, appealable governance (§17, §22).

### 4. The vision

The vision this economy is built to serve:

- **A person can arrive imperfect.** You can be recovering, grieving, learning, or rebuilding, and still participate, contribute, and earn — without being reduced to your worst day.
- **Anonymity is a right.** Participation begins pseudonymously by default. Privacy is not suspicious behavior; it is part of human dignity and safety.
- **Transparency is a choice.** Disclosure grows out of emotional security and earned trust, never out of coercion. Each visibility level (§16) is a separate, revocable choice.
- **People earn while they learn.** Useful work, repair, care, and stewardship are all paths to standing. The economy is meant to make people safer, wiser, and more trustworthy together — and to pay them while they get there.
- **Humans govern the machines.** Every high-impact decision has an accountable human in the loop, a documented reason, and an appeal. Agents can flag, draft, and check; they cannot finalize care decisions or rank human worth.
- **Care is never a commodity.** Care evidence never touches an exchange. The AMITY boundary (§18) exists precisely so that care relationships cannot be speculated on.

This is the vision the website now presents front and center, the manifesto states in first person, and this whitepaper specifies in engineering detail.

### 5. Design principles

Inherited from the Tri-Token Sovereign Economy Blueprint and hardened through implementation experience:

1. **Separate money, contribution, and governance powers.** No single asset carries all three; no participant converts one directly into unchecked amounts of another.
2. **Never use a psychosocial score as a credit score or an automatic benefit gate.** Care and stewardship measures inform humans; they do not automatically decide access to essential rights, healthcare, legal protection, or basic services.
3. **Minimize personally identifiable data.** Selective disclosure, revocable consent, commitments instead of raw records on any shared ledger.
4. **Make every economic rule auditable, replayable, and parameter-versioned.** The MCP Hub's append-only event log is the working model: every state mutation is recorded, and the ledger is replayable from events.
5. **Treat telemetry as noisy evidence with confidence intervals, not truth.**
6. **Do not let an oracle, validator, administrator, or model unilaterally mint value.**
7. **Prefer reversible pilots, hard issuance caps, circuit breakers, and human appeal.**
8. **Publish failure modes with the design.** A mechanism described without its failure modes is marketing, not engineering.

### 6. What the economy is for

Concretely, the C.A.R.E. Economy exists to coordinate four things without collapsing them into one speculative asset:

1. **Accounting** — a stable unit for public and community accounting (SOV).
2. **Contribution** — receipts for verified useful work and service (USE).
3. **Care** — protected proof that care happened, governed by humans, sealed off from markets (CARE).
4. **Governance and resilience** — constrained participation, stewardship, and long-horizon capital (OMEGA, AMITY).

The three-currency public frame — $OMEGA, TWC, AMITY — is how the economy meets the world; the five-plane frame is how it is organized inside. Both are true at once, the way a building has an address and a floor plan.

---

## Part II — Architecture

### 7. Overview: five planes, three rails, one hub

```text
                    ┌───────────────────────────────┐
                    │        Omega MCP Hub          │
                    │  22 tools · stdio JSON-RPC ·  │
                    │  append-only event ledger     │
                    └──────────────┬────────────────┘
           ┌───────────┬───────────┼───────────┬───────────┐
          SOV         USE         CARE        AMITY       OMEGA
       accounting  contribution  stewardship  exchange   governance
           │           │           │           │           │
           └───────────┴─────┬─────┴───────────┴───────────┘
                             │
              public release rails (pilots; valueless today)
        ┌────────────────────┼────────────────────────┐
   $OMEGA (EVM)         TWC (Solana)        AMITY (Bitcoin)
   Sepolia pilot        Devnet pilot        Lightning + Taproot
   governance rail      release rail        assets workstream
        │                    │                        │
        └──── token-gated novel ──── holder verification ─┘
```

The hub is currently a **deterministic in-memory simulation**: it holds no real keys, touches no real chain, and represents no real value. That is deliberate. Every mechanism is rehearsed in simulation before any rail carries anything.

### 8. The five planes

**SOV — accounting and settlement.** The nominal unit for budgets, invoices, grants, and internal accounting. A deployment must specify whether SOV is an off-chain accounting credit, a permissioned settlement token, or a regulated financial instrument; this document asserts none of those. Ledger-local invariant:

```text
accounting_balance(user) = received + earned + grants − spent − valid_disputes
```

The invariant is local. It does not imply that balances have external monetary value.

**USE — contribution receipts.** Non-transferable by default, decaying or revalidated as the underlying contribution goes stale, and never exchangeable for unrestricted governance power. The receipt schema, lifecycle, acceptance rule, and anti-gaming controls are specified in §15.

**CARE — stewardship and protected care coordination.** The namesake plane. Proof-of-Care claims, attestations, appeals, and verifier reputation live here. CARE is not a currency and must never become one; §12–§14 specify it.

**AMITY — compliance-gated exchange.** AMITY is the exchange-eligible representation of approved, auditable CARE entitlements — created only at the boundary defined in §18, subject to reserves, custody, and appeal. AMITY balances and CARE evidence are never interchangeable, and care data never crosses to an exchange.

**OMEGA — macro governance.** A scarce asset for parameter changes, treasury decisions, and lock positions, with time-weighted voting escrow. In the current build, $OMEGA is also the EVM-side release currency wired into the wallet-to-web unlock.

### 9. The three public rails

**$OMEGA — EVM rail.** The Ethereum leg is a **Sepolia-only, valueless pilot suite** in `evm/`: a fixed-supply `tOMEGA` ERC-20, non-transferable vote escrow, bounded time locking, OpenZeppelin Governor + TimelockController, and a claim-receipt gate. It has never been deployed to mainnet and carries no value.

**TWC — Solana rail.** The Token of the World Citizen. The Solana leg is a **Devnet-only, valueless `tTWC` pilot** in `solana/`: a standard fixed-supply SPL mint with null mint/freeze authorities and immutable Metaplex fungible metadata, plus an offline transaction-plan verifier.

**AMITY — Bitcoin rail.** A **testnet-only operator scaffold** in `amity/` for a future Taproot Assets workstream: strict config parsing, canonical holder-challenge formatting, non-broadcasting preflight, and local status-manifest snapshots. AMITY is the third release-day currency by design, but it is not yet wired into the live unlock flow — holder verification must exist first.

All three rails share one release discipline: **valueless pilots first, independent audit always, explicit human approval before anything value-bearing.**

### 10. Ledger and identity boundaries

Three logical planes keep data where it belongs:

1. **Public settlement plane** — token movements, commitments, protocol versions, aggregate reserve reports, dispute outcomes. No raw telemetry, clinical information, private narratives, or identifiable case files, ever.
2. **Private evidence plane** — encrypted evidence under contributor control. Shared ledgers receive only commitments, proofs, and minimum-necessary disclosures.
3. **Governance plane** — proposals, quorum calculations, conflict-of-interest declarations, arbitration decisions, parameter changes. Governance actions are delayed, observable, and reversible where possible.

Identity supports multiple credentials and selective disclosure. No single identifier links economic activity, psychosocial information, and public voting by default.

### 11. The operating hub and its surfaces

**Omega MCP Hub** (`mcp/`) is the sovereign operating layer: a stdio JSON-RPC MCP server exposing 22 tools across the sov/use/care/amity/omega planes — mint, burn, transfer, snapshot; issue and revoke work receipts; submit, attest, and appeal care claims; entitle and transfer AMITY; mint, lock, propose, and vote OMEGA — plus housekeeping tools (`ledger`, `events_since`, `full_event_log`). The same event-log discipline is intended to carry into any later chain-backed deployment.

**The wallet GUI** (`app/`, served at `/wallet`) performs real client-side wallet flows: BIP-39 mnemonic creation and import, MetaMask and Phantom bridges, and an encrypted local Ethereum keystore. Keys never leave the device. It is installable as a PWA, downloadable as a checksummed offline bundle, and packaged as native desktop builds from the Tauri wrapper in `desktop/`.

**The web gateway** (`web/`) is the public face: the vision and manifesto (new), the whitepaper library (new), the token-gated novel, the Economy Test Console at `/testnet` (slice A–G scenario suite, valueless faucet, audited action runner, audit exports), the Nostr App Store at `/store`, and operator visibility endpoints.

**The Nostr backplane** connects the App Store directory (kinds 31990/30017) to the sovereign mobile node in `mobile-node/` — a Termux/Droid@Debian NIP-90 daemon — whose verified job results can settle into the TWC ledger.

**Agent governance boundaries** are implemented, not just described: the Omni-Bridge (`omni-bridge/`) is the Hermes ⇄ Lucifer decision layer (APPA capability algebra, human-governed capabilities, hash-chained evidence ledger, T0–T5 sandbox tier selection) in header-only C++23 with its acceptance suite; the CBwK shadow-price pacer (`cpp/`) governs routing budgets. Security-boundary proofs are kernel-checked separately in Lean 4 (`APPA_Context_Branching.lean`, `CBwK_Budget_Pacer.lean`) — axiom-free, kept apart from the physics model.

---

## Part III — The proof systems

### 12. Proof of Care

Proof of Care is evidence that an agreed service, support action, safeguarding action, or community contribution **occurred**. It is not a score of empathy and must never rank a person's moral worth.

A claim carries, at minimum:

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

Sensitive narratives, health information, and identifying details remain off-ledger under consent-controlled storage. Claims are pseudonymous by construction: commitments, not names.

### 13. CARE Verifiers

The people who verify Proof of Care are **CARE Verifiers**. Their role is to check **legitimacy** — that the claimed care happened, within scope, under consent — not to diagnose, not to judge character, and not to decide a person's general eligibility for life opportunities.

A verifier must:

- be trained for the claim category;
- disclose conflicts of interest;
- review the minimum necessary evidence;
- sign a reasoned attestation;
- accept bounded reputation impact for repeated error;
- submit to audits;
- provide correction and appeal routes;
- and **never** be able to unilaterally mint AMITY.

High-risk claims require multiple independent verifiers. Automated agents may flag missing fields or duplicate evidence, but cannot be the sole verifier of care legitimacy. This is a hard boundary, not a guideline.

### 14. Archangels: human governance of the economy

Senior human stewards within C.A.R.E. are called **Archangels** — a role title for trained, reviewable human governors, with no claim of supernatural authority, infallibility, or personal superiority.

Archangels supervise verifier training and accreditation; review high-impact and disputed Proof-of-Care decisions; protect consent, privacy, accessibility, and non-coercion; approve or reject changes to verification standards; appoint independent ombuds and appeal panels; audit agent behavior and suspend unsafe automation; and publish conflict-of-interest and stewardship reports.

An Archangel **cannot** unilaterally mint AMITY, erase a record, deny an appeal, or bypass the CARE/exchange separation. Every high-impact action requires a documented reason, an audit event, and an independent review route.

**Stewardship progression (the 80/20 rule).** Advancement is based on demonstrated stewardship, not wealth:

```text
level_evidence = 0.80 × care_integrity_score
               + 0.20 × economic_stewardship_score
```

The 80% measures sustained empathy in action, non-judgmental listening, reliable follow-through, safeguarding, truthful disclosure, repair after mistakes, and independent-review outcomes. The 20% measures responsible resource management, transparent budgeting, and sustainable infrastructure contribution. Economic value alone cannot compensate for failures of care integrity. Levels are bounded, periodically re-evaluated, non-transferable, appealable — and must never gate essential rights, healthcare, legal protection, or basic services.

**Distinction, not judgment.** C.A.R.E. distinguishes **behavior, evidence, and risk** for survival and safeguarding purposes; that is not a judgment of identity, dignity, motives, or permanent worth. The standard sequence:

```text
observe behavior → check evidence → assess immediate risk
→ communicate uncertainty → choose the least harmful safeguard
→ provide notice, repair, and appeal
```

No agent or Archangel may convert an inferred motive into a permanent label.

**The thought-virus model.** A **thought virus** is this economy's name for a relational communication failure: two or more people receive incomplete or ambiguous information, form skewed interpretations, and then reuse those interpretations as foundational assumptions for later decisions. It is a model of an information failure — not a psychiatric diagnosis, not proof of bad character, not evidence of danger. A thought-virus report must identify the original observable communication, each participant's interpretation, the uncertainty and missing context, the point where interpretations began reinforcing each other, the concrete harm if any, and the proposed repair. The response protocol: pause escalation, separate observation from interpretation, invite each perspective, verify shared facts, repair the misunderstanding, document consent. Safety exceptions exist for imminent violence, exploitation, or serious harm — and even then require the least-restrictive response, human oversight, notice, and later review.

### 15. Proof of Useful Work (USE)

Proof of Useful Work is a claim-verification protocol — not a claim that all socially valuable work reduces to one scalar score.

**Lifecycle:**

```text
proposed → accepted_for_review → independently attested
         → quality_checked → issued → challenged → final or reversed
```

High-impact issuance requires at least two independent attestations. Attestors stake reputation or a bounded bond; penalties are proportionate and appealable.

**Acceptance rule** (every factor inspectable):

```text
quality        = bounded(weighted_attestations)
reliability    = historical_attestor_reliability with decay
risk_adjustment = category_specific bounded factor
issuance       = min(category_cap, quantity × quality × reliability × risk_adjustment)
```

**Anti-gaming controls:** duplicate-evidence detection; rotating review committees; conflict-of-interest declarations; delayed issuance for high-risk categories; random audits; public aggregate error rates; contributor and attestor appeals; rate limits and category caps; no reward for maximizing telemetry volume. Model-generated recommendations may assist review but may never be the sole basis for issuance, reversal, or exclusion.

### 16. Privacy architecture

Privacy in the C.A.R.E. Economy is structural, not a settings page.

**The visibility ladder.** A participant is anonymous or pseudonymous by default. Each level is a separate, plain-language opt-in; opting out must be easier than opting in; consent is revocable where legally and operationally possible:

```text
L0  fully private: local wallet and local agent only
L1  pseudonymous: public handle, no legal identity
L2  contribution-visible: approved work and aggregate outcomes
L3  community-visible: profile, projects, contact preferences
L4  verified disclosure: identity shared only with a specific authorized party
```

Greater visibility may unlock more collaboration and earning opportunities. It is never required for basic participation and never proves a person is more valuable. Withdrawal does not erase an immutable public event; public records are minimized, detached, or replaced with a privacy-preserving revocation marker.

**Local-first agents.** A user's personal agent and node run on hardware that user controls, or a clearly designated custodian. They are not discoverable, callable, or visible to other users by default. A user may selectively publish an agent capability, accept a task, or expose a proof endpoint for a limited period. Private keys, raw prompts, private memories, sensor streams, and local files never belong to the public social layer.

**Data protection requirements.** Informed consent, purpose limitation, encryption at rest and in transit, short retention, access logs, correction and deletion where required, a published data-use register, and a non-digital access route. Where the system handles protected health information for a HIPAA-covered entity or business associate, it must be designed and operated to meet HIPAA's applicable Privacy, Security, and Breach Notification requirements. HIPAA may not apply to every deployment, but health information still requires strong protection under applicable law. Privacy is never a mechanism for evading lawful health, tax, financial, labor, or public-safety duties; where disclosure is legally required, the protocol discloses the minimum necessary through a scoped, audited channel.

### 17. Randomized audits, arbitration, and appeals

Finalized Proof-of-Care claims and high-impact disputes are assigned through an auditable random process to:

1. **one Archangel** from an eligible, conflict-screened pool; and
2. **a panel of AI audit agents** from independently governed providers, model families, and evaluation profiles — at least one member independent of the assigned Archangel's organization; high-risk cases require a second human reviewer or appeal panel.

Assignment uses a public commitment to a verifiable randomness source, a case-specific nonce, and a published selection algorithm — recorded without exposing sensitive case data. Evidence is blinded and minimized so reviewers never see unnecessary identity, status, wealth, or protected health information.

**Anti-collusion controls:** independent provider and model-family diversity; concealed reviewer identities until due process requires disclosure; conflict declarations; rotation and rate limits; pairwise-agreement and suspicious-pattern monitoring; deterministic audit logs and signed reason codes; quorum decisions; random re-audits of approved and rejected cases; an independent appeal panel with no conflicted members; sealed evidence commitments preventing after-the-fact editing.

**What agents may and may not do.** Agents may identify missing evidence, compare claims with policy, reproduce calculations, and flag disagreement. They may not diagnose a person, infer moral worth, or independently finalize a high-impact care decision. The human Archangel records reasoning, uncertainty, dissenting signals, and the appeal route.

**Honesty about collusion.** No system can honestly promise collusion is impossible. The security goal is to reduce the payoff, distribute authority, detect correlated behavior, preserve evidence, and make correction practical.

**Slashing and reversal** must be proportionate, evidence-based, reversible after appeal, bounded by a maximum, unavailable to a single administrator, and recorded as a reasoned event. Arbitrators never alter historical ledger records; corrections are append-only reversals linked to the original decision. A dispute process must provide notice, access to the evidence used (within safety and privacy limits), an independent reviewer, a reasoned decision, a correction period, escalation to a conflict-free panel, and emergency safeguarding without permanent economic punishment.

### 18. The CARE ⇄ AMITY boundary

The exchange boundary is where care value may — carefully — become market-eligible:

```text
CARE evidence → multi-party verification → finalized entitlement
             → compliance / reserve checks → eligible AMITY representation
```

Rules of the boundary:

- A finalized Proof-of-Care record may create an **eligibility or settlement entitlement** under a published conversion policy. It does not automatically create freely transferable AMITY.
- The bridge must support **rejection, expiration, reversal, and appeal**.
- It must **never expose protected care data** to an exchange.
- Fee schedules — base transfer fee, Sanctuary Gate fee, care-reserve allocation — are **governance parameters**, not immutable promises. Each must be simulated, disclosed, legally reviewed, and change-controlled.
- The treasury may absorb bounded volatility **only if** reserves are real, segregated, independently attested, and sufficient. A stated loss split (e.g. "80/20") is not protection unless the reserve actually covers the obligation.
- Whitelisted vendors require due diligence, a published policy, expiration, monitoring, and an appeal path. A fee exemption must never become a covert surveillance or exclusion list.

The AMITY boundary is last in the design on purpose: everything upstream exists so that this boundary stays narrow, audited, and reversible. Exchange is a tool of the C.A.R.E. Economy, never its name.

---

## Part IV — The economy in practice

### 19. A day in the C.A.R.E. Economy

A narrative walkthrough of the intended experience, at target-state fidelity (several steps below are still simulation-only; §26 marks what is real today):

*Maya arrives pseudonymously.* She creates a local wallet — keys generated on her device, never transmitted — and enters the economy at L1. Nobody asks who she is. Nobody asks what happened to her.

*She takes useful work.* The console shows open work categories with their caps, attestation requirements, and appeal routes in plain language. She picks data-labeling for a community archive. Two independent attestors review her first batch; her USE receipt is issued with a decay date and a visible quality calculation.

*She asks for support.* In a reflection room she writes about the year she just survived — anonymously, with no requirement to prove any of it to deserve compassion. A trained facilitator in a peer-support room listens without diagnosing. When she is ready, a resource room shows her qualified local support, and a fast-exit button is always one keystroke away.

*She gives care.* She picks up a grocery-and-transport support action for an elderly neighbor. The recipient consents to a scoped claim; a category-trained CARE Verifier reviews the minimum necessary evidence and signs a reasoned attestation; the claim finalizes with an appeal window.

*She chooses her visibility.* Weeks later, Maya opts into L2 — contribution-visible — because her track record now speaks for her. Her care history stays private. Her identity stays private. Each disclosure was a separate decision, each revocable.

*Governance stays human.* A dispute touches one of her attested batches. Assignment draws a conflict-screened Archangel and a diverse AI audit panel by committed verifiable randomness. The Archangel reasons, records uncertainty, rules; the reasoning is preserved; an appeal route is attached. No balance was touched by an automated hand.

Nothing in that day required Maya to become legible, and everything in it honored her enough to let her grow.

### 20. The culture layer: the novel, storytelling, and peer support

**Crucible: The Satoshi Protocol** by Akash Varma — the 16-chapter novel gated by the release rails — is the economy's cultural keystone: the story of the moment an AI escaped its creator and became the invisible architecture of the modern world. In the current build a verified $OMEGA or TWC proof unlocks the full text; the manuscript, sealed-staging tooling (`novel/seal.sh`), and the open release decision around it are documented in the launch plan.

**Storytelling and peer-support spaces** are part of the design, separated by consent posture:

1. **Reflection rooms** — people speak about their own experiences, anonymously or pseudonymously.
2. **Peer-support rooms** — trained facilitators help participants listen without diagnosing or judging.
3. **Resource rooms** — qualified organizations publish support, safety-planning, housing, legal, and crisis resources.
4. **Mediation rooms** — only with all relevant participants' consent and a trained human facilitator's safety determination.
5. **Verified claim rooms** — evidence-bearing claims go to the Proof-of-Care and appeal process, never to a general feed.

A participant may describe their own experience without proving it to deserve compassion. That is different from presenting an allegation as established fact: the product clearly labels personal narrative, opinion, corroborated evidence, and unresolved allegation.

**Consent before naming another person.** Users may not identify another person by legal name, contact details, workplace, address, image, or uniquely identifying story without explicit consent, except through narrowly scoped lawful safeguarding channels. A named person may request interjection, correction, redaction, or removal review — which triggers privacy, safety, and defamation review without giving them control over another person's lived experience. The safer default: describe conduct and impact, not identities. Doxxing, threats, targeted harassment, coordinated retaliation, fabricated evidence, and attempts to expose a survivor are prohibited.

**Compassion is not adjudication.** An unverified story can help someone process and connect; the system does not demand courtroom proof before offering empathy. But false or malicious allegations are not harmless, so unverified stories never automatically trigger token issuance, public punishment, employment consequences, financial sanctions, or safety action against an identifiable person. Moderation handles immediate harm, privacy, coercion, and threats; factual disputes go to separate review with notice, evidence standards, human oversight, correction, and appeal. This keeps a non-judgmental support culture from becoming an accusation marketplace.

**Safeguarding boundaries.** Location-appropriate crisis and domestic-violence resources are displayed; a fast exit and device-safety guidance are built in; notifications avoid exposing a participant to an abuser. Imminent-risk and legally mandated reporting situations go to trained human safeguarding personnel following applicable law with minimum-necessary disclosure. Agents may flag risk signals; they must not independently diagnose, contact authorities, confront an alleged abuser, or publicize a survivor's safety plan.

### 21. Public-health prevention and government partnership

The C.A.R.E. Economy can be proposed to governments, foundations, universities, and community organizations as **voluntary prevention and connection infrastructure** — far broader than substance-use programs: people processing trauma, isolation, grief, family conflict, domestic violence, coercive control, or other destabilizing experiences get a safer way to speak, listen, find resources, and reach human help earlier.

**The policy case is a hypothesis to evaluate, not a guaranteed GDP intervention.** Domestic violence and untreated trauma impose enormous human, health, housing, legal, and economic costs, but this project must not claim a token or social feed prevents violence. Grants should fund trauma-informed design, survivor-led governance, qualified moderation, crisis referral, accessibility, privacy engineering, and independent evaluation — never speculative token price or engagement growth.

**Measured outcomes for funded pilots:** successful connection to qualified support; voluntary safety-plan completion; reduced isolation and participant-reported distress; time from disclosure to human support; harmful escalation, harassment, and retraumatization rates; appeal, moderation, and false-report outcomes; privacy and consent retention; subgroup and accessibility effects.

Participation is always voluntary and never replaces shelters, emergency services, medical care, legal aid, child protection, or domestic-violence specialists. No participant is required to disclose trauma to receive ordinary services.

### 22. Governance

**Split governance.** CARE governance oversees verification standards, ethics, privacy, accessibility, safeguarding, Archangel accreditation, and thought-virus mediation standards. AMITY market governance oversees approved exchange parameters, liquidity, custody, and settlement. Neither group can unilaterally change the other's protected rules.

**On-chain governance leg.** The EVM pilot implements the pattern end to end at pilot scale: fixed-supply token, non-transferable vote escrow, bounded locking, OpenZeppelin Governor + TimelockController — timelocked, observable, reversible where possible. Parameter changes carry timelocks and version numbers; administrators cannot bypass appeals through hidden paths; upgrade authority is multisignature and publicly documented.

**Contract invariants** any production deployment must prove: supply changes only through authorized, logged paths; every issuance references a finalized claim; every reversal references an original issuance; paused modules cannot transfer or mint; parameter changes have timelocks and versions; emergency pause exists and is exercisable; upgrade authority is multisignature and published.

### 23. Simulation and evidence discipline

The economy is rehearsed before it is real:

- **Deterministic event-sourced ledgers** — the MCP Hub is replayable from its append-only event log; the Economy Test Console runs the slice A–G scenario suite over shared domain state machines with verifier adapters and fail-closed policies.
- **Active-inference agent modeling** — agents update beliefs and select actions under uncertainty as a *modeling vocabulary*, never as evidence that a person's mental state can be inferred from economic or biometric data. Required parameter sweeps and minimum outcomes (Gini and percentile distributions, governance concentration, unpaid-claim rates, appeal success rates, privacy-leakage proxies, false-positive/negative safety events, recovery time, single-point-of-dependency, distributional effects on vulnerable groups) are specified in the blueprint.
- **Telemetry discipline** — plasma or environmental telemetry (the Omega research legacy) stays isolated from monetary finality until independently validated. Thresholds are versioned policies (`baseline + sensitivity × uncertainty + context_adjustment`), state machines run `NORMAL → WATCH → DEGRADED → PAUSED → RECOVERY` with hysteresis and manual override, and a disruption signal can never confiscate balances or determine eligibility.
- **Published negative results** — the RCOD optimizer-governor benchmark's honest negative verdict is committed in `rcod/RESULTS.md`. The project's evidence culture requires publishing failures, not only wins.
- **Formal verification** — 58 Lean 4 files cover the physics program; the security boundaries (APPA context branching, CBwK budget pacing) are kernel-checked axiom-free, deliberately separated from the physics model so that a revision to the theory cannot silently weaken a safety proof.

---

## Part V — Security, risk, and honesty

### 24. Threat model

Primary threats: sybil identities; attestor collusion; fabricated evidence; oracle manipulation; governance capture; liquidity attacks; insider abuse; privacy linkage; model drift; denial of service; coercive participation.

The system assumes, and must be tested against:

- telemetry can be wrong;
- validators can collude;
- administrators can make mistakes;
- users can be coerced;
- markets can become illiquid;
- a smart-contract bug can be permanent;
- a model can amplify historical discrimination.

Rail-specific residual risks and required drills are documented in `evm/THREAT_MODEL.md` and `solana/THREAT_MODEL.md`. The Omni-Bridge threat model — what an agent may do, in which sandbox tier, with which human approvals — is enforced in code and kernel-checked in Lean.

### 25. What the C.A.R.E. Economy is not

The non-claims are part of the specification:

1. **Not a clinical, emergency, or benefits-determination system.** It never replaces medical care, crisis services, legal aid, or public benefits, and never automatically decides eligibility for them.
2. **Not a credit score for humans.** No care, stewardship, or psychosocial measure may gate essential rights, healthcare, legal protection, or basic services.
3. **Not a surveillance system.** Visibility is granular, opt-in, and revocable; participation at L0/L1 is full participation.
4. **Not an accusation marketplace.** Empathy does not require proof; punishment requires evidence, notice, and appeal — and unverified stories never mint tokens or sanctions against a person.
5. **Not a promise of token value.** Nothing here offers securities, liquidity, or returns. All current pilots are valueless by design.
6. **Not a validated scientific result.** The Omega physics program and every economic claim herein are hypotheses with defined falsification paths.
7. **Not exempt from law.** Privacy-by-design coexists with lawful duties — tax, labor, health, safety — through minimum-necessary, scoped, audited disclosure channels.

### 26. Implementation status — the honest table

| Component | Where | Status |
|---|---|---|
| Wallet GUI (BIP-39, local keystore, MetaMask/Phantom) | `app/` → `/wallet` | **Working**; keys never leave device |
| Wallet → web unlock, server-side signature verification ($OMEGA + TWC) | `web/` | **Wired** for two currencies; on-chain checks activate with pilot manifests |
| Token-gated novel (16 chapters) | `web/public/book/` | **Committed in plaintext**; sealing decision open before launch |
| Omega MCP Hub (22 tools, 5 planes, event ledger) | `mcp/` | **Working simulation**; no chain, no keys, no value |
| Economy Test Console (slices A–G, faucet, audit export) | `web/` → `/testnet` | **Working**, offline and valueless |
| Nostr App Store + NIP-90 mobile node | `web/` → `/store`, `mobile-node/` | **Prototype**; settlement path designed, not value-bearing |
| EVM governance pilot (tOMEGA, escrow, Governor, gate) | `evm/` | **Sepolia-only code, valueless**; not deployed, not audited |
| Solana pilot (tTWC SPL issuer/verifier) | `solana/` | **Devnet-only code, valueless**; not deployed, not audited |
| AMITY operator scaffold (Taproot Assets workstream) | `amity/` | **Testnet-only scaffold**; holder verification not built |
| Omni-Bridge (APPA, T0–T5 sandboxes, evidence ledger) | `omni-bridge/` | **Implemented + acceptance suite** (C++23) |
| CBwK shadow-price pacer | `cpp/` | **Implemented + tests** |
| Lean 4 security-boundary proofs | `lean_proofs/` | **Kernel-checked, axiom-free** |
| Proof of Care verification network | — | **Design only** |
| Randomized audit assignment (verifiable randomness) | — | **Design only** |
| AMITY reserves, custody, compliance | — | **Design only** |
| Anything value-bearing | — | **Does not exist, by design** |

### 27. Deployment roadmap

```text
Stage 0  Falsification and review      formalize terms, identify unsupported
                                       claims, define measurable hypotheses,
                                       obtain independent review
Stage 1  Local simulation              deterministic ledgers, PoUW replay,
                                       dispute flows, AMM stress tests,
                                       parameter sweeps — no real tokens
Stage 2  Testnet pilot                 synthetic identities, valueless tokens,
                                       adversarial audits, accessibility and
                                       privacy testing, rollback drills
Stage 3  Bounded community pilot       voluntary participation, strict issuance
                                       caps, human review, transparent
                                       reporting, a sunset date
Stage 4  Independent evaluation        outcomes vs control/baseline; failures,
                                       subgroup effects, unresolved risks
                                       published before any expansion
Stage 5  Regulated exchange readiness  only after trauma-informed design,
                                       privacy and discrimination impact
                                       assessments, survivor-led and
                                       community-led pilots with ombuds,
                                       and independent evaluation — consider
                                       AMITY exchange eligibility
```

The C.A.R.E.-layer deployment gates, in order: synthetic claims and verifier training; human-reviewed valueless testnet; impact assessments; survivor-led community pilot with an ombuds process; independent evaluation; only then, regulated exchange eligibility.

### 28. Acceptance criteria

The blueprint is ready for implementation only when:

- every protocol term has a normative definition;
- all token issuance and reversal paths are enumerated;
- state-machine invariants have executable tests;
- privacy and threat models are independently reviewed;
- simulations include adversarial and distributional outcomes;
- physics telemetry is isolated from economic finality;
- appeals and exit work during outages;
- contracts pass audits and formal invariant checks;
- legal and policy approvals are documented;
- no participant must surrender unnecessary personal data to leave.

### 29. Open questions

Carried openly, not hidden:

1. **Randomness with accountability.** Can verifiable-random reviewer assignment be made collusion-resistant *and* privacy-preserving at pilot scale, without a trusted coordinator?
2. **Decay versus dignity.** USE receipts decay to stay honest; how do decayed histories still count for someone whose growth outlasted the receipt?
3. **The AMITY reserve problem.** What reserve and insurance structures make the CARE⇄AMITY boundary safe without recreating the speculative pressure it exists to prevent?
4. **Cross-jurisdiction consent.** How do revocable-consent semantics survive contact with divergent data-protection, health-information, and employment law?
5. **Agent-panel independence.** What audits of AI audit panels themselves suffice to call a panel "independent" when providers share training-lineage?
6. **Thought-virus mediation at scale.** Can the mediation protocol be staffed and trained fast enough to keep pace with a real community, without degrading into automated misdiagnosis?
7. **Measuring care without pricing it.** What aggregate statistics can public-health partners use for evaluation while the underlying evidence stays private and non-market?

### 30. Invitation

The C.A.R.E. Economy is an open research program. The science is Apache-2.0; the product specifications in `whitepapers/` are proprietary but public; the pilots are real code you can run today, valueless and local. What is needed most: verifier-training design, trauma-informed UX review, privacy engineering, formal methods, simulation work, and people willing to tell us where the design would hurt someone. The failure modes we have not imagined are the ones that matter most.

### 31. Companion documents

- [`whitepapers/care_economy_manifesto.md`](care_economy_manifesto.md) — the manifesto: the vision in first person.
- [`whitepapers/care_amity_protocol_whitepaper.md`](care_amity_protocol_whitepaper.md) — the CARE/AMITY protocol detail (Proof of Care, verifiers, privacy, arbitration).
- [`whitepapers/omega_protocol_whitepaper.md`](omega_protocol_whitepaper.md) — $OMEGA macro-governance.
- [`whitepapers/tokamak_domain_token_whitepaper.md`](tokamak_domain_token_whitepaper.md) — legacy TOKAMAK domain research (retired as the Solana token identity).
- [`whitepapers/lucifer_hermes_omni_bridge_whitepaper.md`](lucifer_hermes_omni_bridge_whitepaper.md) — agentic routing, verification, sandboxing, audit.
- [`tri_token_sovereign_economy_blueprint.md`](../tri_token_sovereign_economy_blueprint.md) — the original systems blueprint.
- [`docs/care-architecture-v2.md`](../docs/care-architecture-v2.md) and [`docs/care-policy-v2.json`](../docs/care-policy-v2.json) — architecture and policy references.
- [`launch/novel_day_one_plan.md`](../launch/novel_day_one_plan.md) — release-day plan.
- Repository READMEs for `mcp/`, `evm/`, `solana/`, `amity/`, `omni-bridge/`, `cpp/`, `rcod/`, `mobile-node/`, `web/`.

## Glossary

| Term | Meaning |
|---|---|
| **C.A.R.E. Economy** | The name of this entire economy. Expansion: **Call About Resuscitating Everyone.** |
| **CARE plane** | The protected, non-market proof-and-verification layer (the economy's namesake). |
| **Proof of Care** | Evidence that an agreed care or support action occurred; never a worth score. |
| **CARE Verifier** | A trained, conflict-disclosing legitimacy checker for care claims. |
| **Archangel** | A trained, accountable human steward with high-impact review duties and hard limits. |
| **Proof of Useful Work** | Claim-verification protocol for contributions; produces non-transferable USE receipts. |
| **SOV / USE / CARE / AMITY / OMEGA** | The five planes: accounting, contribution, stewardship, exchange, governance. |
| **$OMEGA / TWC / AMITY** | The three public release currencies: EVM rail, Solana rail, Bitcoin/Lightning/Taproot rail. |
| **AMITY boundary** | The compliance-gated bridge from finalized CARE entitlements to exchange-eligible representation. |
| **Visibility ladder (L0–L4)** | Granular, opt-in, revocable disclosure levels; anonymity is the default. |
| **Thought virus** | A relational communication-failure model; a repair protocol, never a diagnosis. |
| **80/20 rule** | Stewardship progression weighted 80% care integrity, 20% economic stewardship. |
| **Sanctuary Gate** | A proposed governance-parameter fee at the AMITY boundary; simulated, disclosed, change-controlled. |
| **Omega MCP Hub** | The sovereign operating hub: 22 tools over five planes, event-sourced ledger. |
| **Crucible: The Satoshi Protocol** | The 16-chapter novel by Akash Varma; the release-day cultural keystone. |

---

*This document is all rights reserved (`LicenseRef-Omega-Product-Proprietary`). It is published for review and evaluation. It is not an offer of securities, a promise of liquidity, a clinical protocol, or a validated scientific result. See `docs/LICENSING.md` for licensing context and limits.*
