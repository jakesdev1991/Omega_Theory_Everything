<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# C.A.R.E. Protocol Architecture v2

**Expansion:** Call About Resuscitating Everyone
**Status:** Research architecture / testnet design draft
**Version:** 0.2
**Mainnet status:** Not approved; no value-bearing deployment is authorized by this document.

> C.A.R.E. is a voluntary social and care-coordination system. It is not a clinical system, emergency service, benefits determination system, or measure of a person's moral worth.

## 1. Purpose

C.A.R.E. is the social foundation for a coupled economy in which people may contribute attention, useful work, care, and network resources. The system is designed around a simple premise:

> Make participation, care, cooperation, correction, and repair easier than isolation, exploitation, or retaliation, while preserving each person's control over their own life.

The protocol does not attempt to identify who is inherently good or evil. It records explicit actions, consent, resource contributions, claims, decisions, and boundaries.

## 2. Non-negotiable invariants

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

## 3. Economy: Sanctuary and Casino

The ecosystem separates protected internal support from market risk.

| Plane | Function | Default risk |
|---|---|---|
| CARE | Internal care/support entitlement, contribution record, and community coordination | Protected from ordinary market speculation |
| AMITY | External, market-facing representation that may be traded under policy | Speculative and risk-bearing |
| TWC | Useful-work, service, and resource-contribution settlement | Contribution/liquidity layer |
| OMEGA | Long-term governance, treasury, and protocol accountability | Governance/treasury risk |

CARE is not called a stablecoin until its reference value, reserves, redemption, insolvency treatment, and audit process are defined. It may begin as an internal accounting and support entitlement.

AMITY is not a guarantee of value. It may absorb market risk, while CARE support is protected only to the extent that actual reserves, services, or sponsor commitments can cover it.

## 4. Social application first

The first C.A.R.E. product is a social application, not a token exchange. It should provide:

- Pseudonymous or verified profiles
- User-controlled circles and communities
- Care requests and offers
- Posts, projects, and group activities
- Consent-controlled sharing
- Blocking, muting, and boundary controls
- Safety reports and appeals
- Weekly participation/accounting views
- Voluntary in-person events and attestations

Ordinary participation is not gated by AMITY, OMEGA, TWC, legal-name disclosure, or a high privacy tier.

### 4.1 Attention and reach

The protocol may reward observable social reach because reach and attention are measurable proxies for reducing isolation and distributing support. It must not claim that raw reach perfectly proves a person's private motives or therapeutic outcomes.

The initial policy may publish:

```text
raw reach units = unique audience observations + attention events
```

The coefficients and event definitions must be versioned and public. There is no hidden authenticity score, moral score, or requirement that the system decide whether a person is sincere.

Normal economic safeguards may limit automated account floods, duplicate events, and impossible activity rates. Those controls protect the ledger; they are not judgments about a person's character.

### 4.2 Fixed weekly distribution

The base issuance model is a fixed weekly pool rather than an uncapped per-person mint:

```text
participant allocation = weekly pool × participant reach units / total eligible reach units
```

The weekly pool and policy version are published before the epoch. Early participation may produce larger individual allocations because fewer participants share the pool. The system does not need a burn to imitate early Bitcoin distribution; unissued holdbacks simply never enter circulation.

## 5. Wallet resource contribution

A wallet may contribute a small user-defined resource quota while the device is powered on. The initial design target is a **2% device-resource budget**, not an unverified claim of exact electrical consumption.

Resource receipts may account for:

- CPU time
- Network bandwidth
- Storage
- Relay uptime
- Memory or service capacity
- Optional energy estimates supplied by the operating system

The user chooses a quota, subject to:

```text
protocol minimum <= user quota <= device-safe maximum
```

The wallet must show the quota, current usage, remaining requirement, and pause controls. It must pause for low battery, overheating, battery-saver mode, or the user's configured limits.

Users may satisfy a quota through:

1. Their own wallet device
2. A delegated community node
3. An approved hosted resource provider
4. A sponsored resource contribution

A wallet is not automatically a full blockchain node. Light-client use is supported.

### 5.1 CARE activity status

A missed quota changes activity status; it does not destroy a person's history:

```text
active -> grace -> under-quota
```

CARE remains recorded and owned. Conversion requests may be queued when funded settlement capacity is unavailable; the protocol must not create unbacked AMITY merely to conceal an uncovered quota or reserve obligation.

## 6. Hardship and solidarity

Hardship is unbounded:

```text
hardship duration cap = none
hardship count cap = none
hardship assistance cap = none
forced hardship expiry = forbidden
```

A person may request an indefinite hardship status without publicly disclosing the reason. Hardship does not create debt.

For each epoch:

```text
hardship deficit = minimum quota - personal contribution
```

Other holders may voluntarily cover the deficit with verified resource contributions. A sponsor receives a published, bounded bonus based on actual contribution, not on the recipient's private hardship details.

If no sponsor is available:

- The hardship status remains valid.
- The person is not expelled or punished.
- CARE is not destroyed.
- The uncovered need remains an open support request.
- The system may delay unfunded AMITY settlement rather than mint unbacked assets.

No sponsor receives control, ownership, contact rights, or governance power over the person receiving support.

## 7. Privacy and proof levels

Identity disclosure and care evidence are separate, user-controlled dimensions. A person may maximize proof strength by disclosing more accountability information, but basic participation remains available with less disclosure.

```text
P0: pseudonymous participation
P1: verified person credential
P2: verified service or community history
P3: senior accountable steward
```

Higher levels may unlock greater payout limits, verifier eligibility, or governance responsibilities. They do not make a person more valuable.

The strongest proof of care should use selective disclosure, encrypted evidence, short-lived credentials, and commitments rather than publishing raw personal histories.

## 8. Physical and social attestations

Real-world interactions may use voluntary:

- Ephemeral NFC handshakes
- Short-lived QR credentials
- Venue or event attestations
- Bilateral completion reports
- Zero-knowledge or coarse presence proofs where technically justified

Physical presence alone is not proof that care occurred. It is one possible evidence component. Continuous GPS surveillance is not required.

## 9. Roles and promotion

The role ladder is:

```text
Level 0: Participant
Level 1: X1
Level 2: X2
Level 3: Archangel
```

Role progression represents responsibility and accountability, not human rank.

### Participant

May use the app, join communities, request support, contribute resources, and appeal decisions.

### X1

Requires safety and consent training, supervised participation, and no unresolved serious integrity case. X1 participants may assist with low-risk activities.

### X2

Requires independent review, supervised attestations, conflict disclosure, and additional safeguarding training. X2 participants may serve on limited verification panels.

### Archangel

Requires nomination or selection, independent conflict review, senior safeguarding training, a fixed term, and reviewable appointment. Archangels supervise standards, appeals, and safety; they cannot unilaterally mint, slash, erase records, or override an appeal.

No role may be purchased with AMITY, TWC, or OMEGA.

### 9.1 Archangel ratio and voting

The draft capacity parameter is:

```text
maximum local Archangels = floor(active participants / 500)
```

Small communities may use a shared regional pool rather than appointing a single local authority.

Archangels receive a bounded governance allocation rather than unlimited control. Within that allocation, active Archangels receive equal shares. The final governance percentage remains a published, simulated parameter and must not be changed by Archangels voting only for themselves.

Economic and treasury decisions require broader participation than the Archangel council. Conflicted Archangels recuse themselves.

## 10. Safety and integrity enforcement

Community safety and protocol integrity are separate systems.

### 10.1 Community safety

The safety system may address:

- Threats
- Stalking
- Doxxing
- Coercion
- Repeated unwanted contact
- Non-consensual disclosure
- Defined boundary violations

These actions may restrict access to a person, circle, venue, or community. They do not automatically confiscate economic balances.

### 10.2 Protocol integrity

Only enumerated technical or economic events qualify:

- Forged signature
- Duplicate finalized claim or nonce
- Double redemption
- Unauthorized state transition
- Deliberate oracle manipulation
- Deliberate smart-contract exploitation
- Falsified verifier attestation
- Coordinated issuance or identity attack

A normal mistake, good-faith overclaim, hardship request, or bug report is not a protocol-integrity offense.

Every case requires:

```text
case_id
rule_id
policy_version
evidence references
reproducible facts
affected delta
reviewers and conflicts
decision
appeal route
```

### 10.3 Progressive integrity sanctions

The three-offense rule applies only to final, adjudicated integrity violations. One exploit is not counted multiple times because it affected multiple records.

**First confirmed violation**

- Freeze the affected pending path
- Recover the exact fraud-derived or unauthorized delta
- Remove relevant trusted permissions
- Preserve unrelated legitimate CARE
- Provide an appeal

**Second confirmed violation**

For the exact illicit delta:

```text
75% permanently removed or left unissued
15% security, restitution, or protocol reserve
10% independent audit and bug-bounty fund
```

There is no founder or individual enforcement share.

**Third confirmed violation**

- Apply the second-offense financial rule to the new illicit delta
- Indefinite temporary exclusion from the affected network/community
- Removal from trusted roles
- Periodic independent review
- No permanent moral exile

Responsible vulnerability disclosure receives safe-harbor treatment and may receive a fixed bounty.

## 11. Governance and agent boundaries

Agents may:

- Summarize evidence
- Reproduce deterministic calculations
- Detect duplicate or anomalous records
- Route cases to authorized humans
- Run resource and payout simulations

Agents may not:

- Decide that a person is morally bad
- Diagnose a person
- Create new offense categories
- Finalize a high-impact sanction alone
- Mint or transfer value by model output
- Promote themselves or an ally
- Serve as the sole Archangel or appeal authority

Governance changes require a versioned proposal, simulation, independent review, timelock, and rollback plan.

## 12. Bootstrap and settlement

The initial economy may be supported by bounded OMEGA and TWC commitments. Those commitments must be:

- Explicit
- Time-limited or budgeted
- Publicly accounted for
- Segregated from personal funds
- Tested against drawdowns and demand spikes

No government grant is assumed. Government, foundation, or partner funding may later supplement the system.

Before AMITY conversion, a weekly settlement record must show:

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

The initial conversion schedule is:

```text
initial:   5% participant dock
maximum:  20% participant dock
extreme:  20% dock + 15% unissued holdback
```

The base design does not burn supply. The extreme 15% is not issued; it is not destroyed after circulation.

## 13. Mainnet gate

No value-bearing mainnet deployment should occur until:

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

Until then, C.A.R.E., AMITY, OMEGA, and TWC remain research or valueless testnet components.
