<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Lucifer–Hermes Omni-Bridge Prime Whitepaper

**Version:** 0.2 research draft  
**Status:** Agentic infrastructure architecture; no autonomy, security guarantee, or performance claim is implied.  
**Supersedes:** 0.1 (adds the four-plane scope, the capability algebra, the sandbox-tier model, the
evidence-ledger format, the routing pacer, and the implementation's measured status and deviations).

> **Implementation status.** A header-only C++23 control-boundary implementation of the decision
> layer — APPA capability algebra, human-governed capabilities, a hash-chained evidence ledger, and
> T0–T5 sandbox-tier selection — lives in [`../omni-bridge/`](../omni-bridge/README.md), with its
> acceptance suite committed alongside it. That README lists, without hedging, what is implemented,
> what deviates from this paper, and what is host-side work that does not exist yet. The routing
> pacer is in [`../cpp/`](../cpp/README.md) (header-only, with its own validation suite). This
> document does not claim production readiness for either.

## Abstract

Lucifer–Hermes Omni-Bridge Prime is the proposed agentic control plane for the ecosystem. Its job is to route work, enforce typed permissions, coordinate verification, execute deterministic policies, and preserve an audit trail. It is not a sovereign replacement for lawful governance, human review, or scientific validation.

Earlier revisions of this paper described a "Tri-Token ecosystem." The repository now models
**four planes** — CARE, TWC, OMEGA, and AMITY — and the bridge's authority is defined separately
against each one. Read the older phrase as historical framing, not as a scope statement.

## Plane separation

### Lucifer — exploration and synthesis

Lucifer decomposes tasks, proposes hypotheses, performs red-team analysis, and identifies failing branches. Role profiles such as Conquest, War, Famine, and Death are operational labels, not independent authorities.

### Hermes — expertise and governance

Hermes manages versioned knowledge, procedures, constraints, tool adapters, and composite workflows. It uses progressive disclosure so agents load only the artifacts needed for a task.

### Omni-Bridge — enforcement boundary

The bridge controls tool capabilities, context labels, evidence provenance, deterministic state transitions, and approvals. No language-model output directly changes token state.

### What each plane may ask the bridge to do

| Plane | The bridge may | The bridge may not |
|---|---|---|
| CARE | Route a care claim to authorized reviewers; check completeness, duplicates, and consent; reproduce deterministic calculations | Infer care legitimacy from a model score, diagnose, or finalize a high-impact case |
| TWC | Validate a work receipt against its class adapter; check artifact and resource commitments | Treat a model's confidence, a title, or raw device time as evidence of useful work |
| OMEGA | Execute an authorized, timelocked parameter change and append the audit event | Originate an issuance, burn, or reversal without governance authorization |
| AMITY | Enforce the published conversion policy's mechanical checks | Create an entitlement, bypass a reserve check, or reveal care data to an exchange |

## Capability algebra (APPA)

Tool and agent permissions are computed, not granted informally. A capability set is the intersection

```text
System ∩ Declared ∩ Tenant ∩ Tool ∩ Context ∩ Risk
```

with two properties that are checked rather than promised:

1. **Risk only removes.** Increasing a task's assessed risk can shrink the permitted set; it cannot
   expand it.
2. **Governance-only capabilities are masked from every automated level.** `PolicyWrite`,
   `SkillPromote`, `MintOrMoveValue`, `SecretRead`, and `CredentialRead` can only be converted from
   *requested* to *authorized* by an explicit human-governance approval step, never by a model call.

## Evidence and verification

All high-impact decisions pass through:

```text
proposal -> typed schema validation -> deterministic checks
         -> independent review -> human or governance authorization
         -> append-only event -> replayable audit
```

The audit trail is a **hash-chained, append-only ledger** with length-prefixed canonicalization,
gap counting, and tamper detection. Replay is classified rather than assumed: an event is exactly
replayable, deterministically replayable, simulated-counterfactual, approximate-counterfactual, or
non-replayable, and the class is recorded with the event.

Lean, property tests, and contract tests may verify formal invariants. They do not establish that an
empirical claim is true, and "zero hallucination" is not a valid blanket system guarantee. In this
repository, the build is the source of truth for which statements are kernel-checked; see
[`../lean_proofs/README.md`](../lean_proofs/README.md).

## Risk tiers and sandboxing

Tasks are assigned a sandbox tier T0–T5, selected as the strictest of the tool's floor and the
policy's floor, so that neither a tool nor a caller can lower the requirement:

```text
selected tier = max(tool floor, policy floor)
```

Tier selection is implemented in the control boundary; actually spawning the corresponding
Wasmtime, container, or microVM sandboxes is **host-side work that is not implemented here**. A
deployment that treats tier selection as equivalent to enforcement has not implemented this section.

## Privacy-first node model

Each participant’s agent is local-first: it runs on hardware controlled by the participant or an explicitly chosen custodian. The node is private and non-discoverable by default. The public network receives only the minimum signed commitment or proof needed for a chosen task.

Agent exposure is capability-scoped and opt-in:

```text
private local agent -> temporary proof endpoint -> selected capability
                    -> selected counterpart -> explicit revocation
```

The owner may choose whether to reveal a public handle, contribution history, agent capabilities, legal identity to a designated party, or nothing beyond a zero-knowledge-style commitment. More visibility can improve collaboration and earning opportunities, but the protocol must not pressure users to reveal more than they need.

A local agent must not upload raw prompts, private memories, local files, credentials, sensor streams, or wallet secrets merely to participate. Remote tasks must be sandboxed, time-limited, and auditable. A user can opt out of discovery without losing access to basic wallet, contribution, or appeal functions.

## Security model

The design may use control/data separation, capability-based tools, isolated child contexts, and WASI or stronger sandbox tiers. The parent process must not receive raw untrusted content when a lower-trust branch can safely sanitize it first.

Parent and child contexts are related by restriction, not inheritance-by-default: a child context can
never hold capabilities its parent does not hold, and it cannot escape the parent's risk ceiling.

Privacy architecture does not provide immunity from lawful health, labor, tax, financial, or regulatory duties. If a deployment handles protected health information for a HIPAA-covered entity or business associate, HIPAA Privacy, Security, and Breach Notification requirements must be addressed in the operating model. When disclosure is legally required, the system should provide a narrowly scoped, audited disclosure channel rather than broad visibility into the user or node.

The system must explicitly defend against:

- prompt injection;
- tool argument confusion;
- credential exfiltration;
- supply-chain compromise;
- model collusion;
- data poisoning;
- governance capture;
- denial of service;
- unsafe autonomous actions.

No label system can by itself prove immunity to indirect prompt injection. Security claims require adversarial testing and operational controls. The acceptance suite includes prompt-injection payloads and a capability-escalation scenario; passing them is evidence about those cases, not a general proof of safety.

## Routing and active inference

ToMA, ODAR, EquiRouter, ROUTERHGC, and TARo are candidate components. They should first be evaluated as replaceable modules against simple baselines. Routing objectives must include quality, cost, latency, privacy, safety, and uncertainty—not only token efficiency.

A candidate route can be scored with a documented multi-objective function:

```text
score = quality - cost - latency - privacy_risk - safety_risk - uncertainty_penalty
```

Weights and thresholds must be versioned, tested, and reviewable.

### Budget pacing

Dispatch across model backends is budget-gated. The CBwK primal-dual shadow-price pacer
([`../cpp/README.md`](../cpp/README.md)) consumes per-action scores from any external model or
bandit, penalizes them with current shadow prices, enforces per-resource budgets (USD, latency,
memory, tokens) through pre-execution reservations, and updates prices by projected dual subgradient
ascent at interval boundaries. Reward learning is explicitly out of scope: the pacer enforces
budgets and does not decide what is worth doing.

A route table that fails repeatedly is treated as unstable and pinned back to a baseline rather than
retried indefinitely.

## Human governors and Archangel oversight

The agentic framework is subordinate to accountable human governance. In the C.A.R.E. protocol, qualified senior stewards are called Archangels. The title identifies a level of responsibility; it grants no supernatural authority and does not make a person infallible.

For Proof-of-Care audits, the bridge uses auditable random assignment: one conflict-screened Archangel plus a heterogeneous AI panel selected from independent providers and model families. The Archangel is accountable for the human decision, but cannot choose the entire panel, use only their own agents, or silently override dissent. High-impact cases require independent human review. Selection, dissent, reason codes, and re-audit events are logged without exposing unnecessary care data.

Archangels govern verifier standards, high-impact appeals, safety boundaries, privacy, accessibility, and agent suspension. They must be selected through transparent criteria, publish conflicts of interest, rotate where practical, and remain subject to independent review. The framework must preserve a complete distinction between:

```text
agent recommendation -> human verifier judgment -> authorized governance action
```

An agent cannot promote an Archangel, assign a moral score, or infer genuineness from facial, linguistic, biometric, or economic data. Progression evidence must be reviewed by humans and include successful correction, repair, and appeal behavior.

The single-process implementation models the governance gate with a private-constructor approval
type. In production, the approval must arrive from a separate service or account the model cannot
reach; the type system alone does not enforce the boundary across a deployment.

## Economic role

The framework can verify a PoUW claim, calculate a bounded issuance recommendation, or route a care claim to authorized reviewers. It may not mint, slash, freeze, or transfer value without the protocol’s deterministic authorization and an append-only audit event.

For CARE/AMITY specifically:

- agents may check schemas, duplicate evidence, and missing consent;
- CARE Verifiers determine legitimacy under human-governed procedures;
- the exchange boundary performs separate compliance and reserve checks;
- no agent may infer care legitimacy solely from a model confidence score.

Cross-plane authority is bounded by the shared primitives in
[`../docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md) §1, so an agent action
is attributable to a `policy_version`, an actor commitment, and an evidence reference.

## Evolution plane

GEPA-like reflection may propose new skills or policies from failure traces. Promotion requires deterministic tests, sandbox execution, schema review, security review, and human authorization. Novelty metrics such as Fisher-Rao distance are research diagnostics, not proof that a skill is useful or safe.

The skill lifecycle is an explicit state machine (discovered, evaluated, proposed, promoted,
deprecated, retired) with append-only version history, governance-gated promotion, and rollback.

## Kill switches

Emergency controls are pinned to specific versions and parsed once at construction, with no runtime
re-parse path that a compromised component could use to re-enable itself. Degraded modes are
functional rather than nominal: a disabled plane must fail closed and report that it has done so.

## Performance claims

Claims such as sub-millisecond sandbox starts, fixed percentage token reduction, or multi-million-fold clearing speedups require reproducible benchmarks with hardware, workload, variance, and baseline details. They must not appear as guarantees until independently replicated.

The implementation's measured micro-benchmark is reported as a sanity bound with its container and
threading context, **not** as evidence for the routing latency target, and the paper does not restate
it here as a claim.

## Deployment stages

1. Typed local protocol adapters.
2. Deterministic replay and invariant tests.
3. Sandboxed synthetic workloads.
4. Red-team and privacy evaluation.
5. Human-approved testnet operations.
6. Independent security audit.
7. Only then consider limited production use.

Stages 1 and 2 have local implementations with committed test evidence. Stage 3 is partial: tier
selection exists, sandbox execution does not. Stages 4–7 have not been performed.

## Revision history

- **0.2 (2026-09-25)** — Added the four-plane scope table, the APPA capability algebra, risk tiers
  and the honest sandboxing boundary, the hash-chained evidence ledger and replay classes, budget
  pacing, contextual restriction, kill switches, and the implementation's measured status.
  Corrected the "Tri-Token" framing and removed the implication that tier selection is enforcement.
- **0.1 (2026-09-23)** — Initial research draft.
