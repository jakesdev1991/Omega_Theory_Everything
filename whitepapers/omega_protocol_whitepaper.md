<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# $OMEGA Protocol Whitepaper

**Version:** 0.2 research draft  
**Status:** Design specification; not an offer, investment recommendation, or live monetary system.  
**Supersedes:** 0.1 (unchanged in claim; adds pilot status, cross-plane boundaries, and the
result of the first empirical test of the research program below).

> **Implementation status.** A **Sepolia-only, valueless** `$OMEGA` pilot suite exists in
> [`../evm/`](../evm/README.md): a fixed-supply `tOMEGA` ERC-20, a non-transferable vote escrow,
> bounded time locking, an OpenZeppelin Governor + TimelockController, and a public claim-receipt
> gate. It has **not been deployed**, independently audited, or approved for mainnet or
> value-bearing use; its assets, authorities, and residual risks are in
> [`../evm/THREAT_MODEL.md`](../evm/THREAT_MODEL.md). Nothing in this document authorizes a
> deployment.

## Abstract

$OMEGA is the proposed scarce macro-governance asset of the Omega ecosystem. It coordinates protocol parameters, long-horizon security commitments, and treasury governance. It is deliberately separated from the high-velocity compute and care layers so that governance scarcity is not confused with payment liquidity or human contribution.

The protocol may use informational-geometry concepts as design inspiration. Those concepts are not, by themselves, evidence of market value, monetary stability, or a physical law governing token supply.

## Position in the four-plane model

`$OMEGA` is one of four planes, and the whitepaper's scope is narrower than the ecosystem's:

| Plane | Question it answers | Does `$OMEGA` govern it directly? |
|---|---|---|
| CARE | What care, support, and participation records exist? | No. CARE's protected rules are set by CARE governance, not by `$OMEGA` holders alone. |
| TWC | What useful work was completed? | No. TWC issuance runs on its own work-receipt policy. |
| `$OMEGA` | Who may change a protocol parameter, and under what delayed process? | Yes. |
| AMITY | Who may convert a protected entitlement into a market-facing asset? | No. AMITY market governance handles exchange, custody, and settlement parameters. |

The governing rule is from [`../docs/care-architecture-v2.md`](../docs/care-architecture-v2.md):
no governance group may unilaterally change another plane's protected rules. `$OMEGA` voting power
must never become a path around CARE's privacy, consent, or appeal guarantees.

## Role

$OMEGA may be used for:

- bounded governance participation;
- time-locked security and parameter commitments;
- treasury proposal deposits and accountability bonds;
- protocol-level delegation;
- voting on verified, published parameter changes.

$OMEGA must not be used as a proxy for a person's care, health, identity, employability, or civic worth.

## Supply and issuance

The proposed issuance curve is a parameterized bootstrap function followed by decay:

```text
issuance(t) = bootstrap(t) + reserve_adjustment(t)
```

The BNS/kilonova analogy is an explanatory metaphor, not a claim that astrophysical decay determines token value. A production policy must publish:

- maximum supply or a defensible issuance bound;
- bootstrap duration;
- decay exponent and rounding rules;
- treasury and validator allocations;
- unlock schedules;
- emergency issuance and burn conditions;
- independent supply-reconciliation tests.

No issuance formula becomes valid merely because it is written mathematically.

**Open parameter: supply cap and curve.** This paper deliberately leaves the maximum supply and
the bootstrap/decay parameters unpinned, and the release plan records that as a blocking item — the
Sepolia ERC-20 cannot be specified beyond a placeholder fixed supply until the curve is fixed
([`../launch/novel_day_one_plan.md`](../launch/novel_day_one_plan.md), §5 and risk R2). The pilot's
`tOMEGA` supply is a test constant and must not be read as the proposed issuance.

## Topological-impedance staking

A lock position may reduce liquid voting or transfer power for a defined period in exchange for a governance weight or protocol reward. The implementation should use an explicit lock function:

```text
weight = amount × duration_factor × bounded_commitment_factor
```

The term "informational freeze" describes reduced liquidity in the protocol model. It does not imply a physical change in spacetime or a guaranteed financial return.

Required controls:

- no hidden rehypothecation;
- visible lock expiry;
- emergency exit policy;
- capped governance concentration;
- delegation revocation;
- conflict-of-interest disclosure;
- timelocked parameter changes.

The pilot vote escrow is deliberately **non-transferable**, so that a lock position cannot be sold
as a separate speculative instrument before the policy above is settled.

## Governance

Every proposal must include a machine-readable change set, simulation results, affected invariants, rollback plan, and an independent review. The agentic framework may summarize evidence or run tests, but it cannot silently substitute for authorized governance.

A minimum lifecycle is:

```text
draft -> review -> simulation -> vote -> timelock -> activation -> evaluation
```

The same lifecycle applies to the CARE and TWC planes through the shared primitives in
[`../docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md) §1, so that a policy
change is identifiable by `policy_version` across all planes rather than by a document revision.

Governance changes require broader participation than any single council. Conflicted participants
recuse themselves, and no group may vote itself an expanded allocation.

## Security invariants

1. Total supply changes only through versioned issuance, burn, or reversal events.
2. Historical events are append-only.
3. Voting power cannot be transferred during a snapshot unless the policy says so explicitly.
4. A failed proposal cannot partially alter state.
5. Emergency operators have bounded, audited powers.
6. No token balance can be changed by a model output without deterministic authorization.

Additional cross-plane invariants that constrain `$OMEGA` specifically:

7. Treasury commitments to other planes are explicit, time-limited or budgeted, publicly accounted
   for, segregated from personal funds, and tested against drawdowns.
8. Expected market value of another plane's asset is never counted as reserve value.
9. No enforcement action pays a percentage to the founder, an Archangel, a reviewer, an agent, or a
   `$OMEGA` holder.

## Research program

The research work should test whether the proposed geometry-informed metrics improve governance outcomes compared with simpler baselines. Success criteria must include security, participation, concentration, latency, and distributional effects—not only price or throughput.

**First result: negative on the designed claim.** The RCOD optimizer-governor prototype and its
label-noise-recovery benchmark were built as an instance of this ask, and the specified
configuration did **not** outperform a plain baseline on the designed metric
([`../rcod/RESULTS.md`](../rcod/RESULTS.md)). Two narrower sub-findings survived, and the calibrated
variant recovered to a lower final loss while costing a longer recovery window — but the honest
summary is that the specification as written failed its own test. The paper keeps that result rather
than restating the hypothesis as a finding, and any future geometry-informed metric should be
compared against the same baseline before it is proposed as a governance parameter.

## Deployment stages

1. Formal state-machine model.
2. Local simulation with synthetic identities.
3. Testnet with valueless tokens.
4. Independent contract and economic audit.
5. Small, voluntary pilot with a sunset date.
6. Public evaluation before any expansion.

Stage 2 is in progress and stage 3 tooling exists locally but has not been run on-chain. The
Sepolia pilot covers only the mechanical parts of stage 3; it does not satisfy stages 1, 4, or 5,
and it says nothing about the unsettled supply parameters in §Supply and issuance.
