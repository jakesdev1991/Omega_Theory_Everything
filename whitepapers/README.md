<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Omega Ecosystem Whitepapers

These are research drafts and system specifications. They do not constitute an offer of securities, a promise of liquidity, a clinical protocol, or a validated scientific result.

**Index revision:** 2026-09-25. Each paper carries its own version header; the table below is the
authoritative map of what each draft does and does not currently cover.

## Whitepapers

| Document | Version | Covers | Implementation status |
|---|---|---|---|
| [`omega_protocol_whitepaper.md`](omega_protocol_whitepaper.md) | 0.2 | Macro-governance, staking, supply, and governance for `$OMEGA` | Sepolia-only, valueless `tOMEGA` pilot suite exists in [`../evm/`](../evm/README.md); not deployed, not audited |
| [`twc_whitepaper.md`](twc_whitepaper.md) | 0.1 | Token of the World Citizen: proof-of-useful-work receipts, settlement, verification, appeals, and the Devnet boundary | Solana Devnet-only, valueless `tTWC` pilot; not deployed, not audited |
| [`tokamak_domain_token_whitepaper.md`](tokamak_domain_token_whitepaper.md) | 0.3 | Legacy plasma-domain computation, telemetry boundaries, and scientific PoUW research under the historical name `TOKAMAK` | Design only; **`TOKAMAK` is retired as the Solana token identity** |
| [`care_amity_protocol_whitepaper.md`](care_amity_protocol_whitepaper.md) | 0.2 | Proof of Care, CARE Verifiers, human Archangels, stewardship progression, privacy, arbitration, and the CARE-to-AMITY exchange boundary | Local social prototype and bounded conversion/AMITY testnet scaffold exist; no value-bearing deployment authorized |
| [`lucifer_hermes_omni_bridge_whitepaper.md`](lucifer_hermes_omni_bridge_whitepaper.md) | 0.2 | Agentic routing, verification, sandboxing, audit, and governance boundaries | Header-only C++23 control-boundary implementation with an acceptance suite in [`../omni-bridge/`](../omni-bridge/README.md); host-side sandboxing and production governance separation are not implemented |

## Canonical identity and the missing TWC paper

The selected canonical Solana identity is **Token of the World Citizen (`TWC`)**, with the
visibly test-only Devnet symbol `tTWC`. The TWC paper is now [`twc_whitepaper.md`](twc_whitepaper.md). The interim implementation and integration references remain:

- [`../docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md) §3 — the work-class
  receipt protocol and verifier adapters;
- [`../solana/README.md`](../solana/README.md) — the Devnet-only issuer/verifier as built;
- [`../docs/TRADEMARKS.md`](../docs/TRADEMARKS.md) — the naming decision and its unresolved
  trademark-clearance status.

Do not read the existence of a TWC paper as an endorsement of `TWC` as a value-bearing asset, and do
not read the `TOKAMAK` draft as its specification.

## How these papers relate to the rest of the repository

The whitepapers are the *design* layer. They are deliberately less current than the executable and
machine-readable layers, and where the two disagree the executable layer plus the documents below
are authoritative:

- [`../tri_token_sovereign_economy_blueprint.md`](../tri_token_sovereign_economy_blueprint.md) —
  long-form systems spec across all planes.
- [`../docs/care-architecture-v2.md`](../docs/care-architecture-v2.md) — the current C.A.R.E. plane
  architecture (v0.2), including the invariants, weekly distribution, wallet-resource budget,
  hardship rules, role ladder, and integrity sanctions.
- [`../docs/care-policy-v2.json`](../docs/care-policy-v2.json) — the machine-readable parameter set
  for the above. Where a number appears in both a whitepaper and this file, this file governs.
- [`../launch/novel_day_one_plan.md`](../launch/novel_day_one_plan.md) — release-day decisions and
  the open risk register.
- [`../rcod/RESULTS.md`](../rcod/RESULTS.md) — the empirical test the `$OMEGA` paper's research
  program asks for, including its negative result.

## Version policy

Whitepaper versions are bumped when a change alters a claim, a parameter, an invariant, or a stated
scope — not for wording or formatting. An unchanged version number means the claims are unchanged,
not that the document has been reviewed against every commit since. The repository is maintained as
a set of separately staged artifacts; a whitepaper having a low version number is not evidence that
its subject matter is unimplemented, and an implementation existing is not evidence that a
whitepaper claim has been validated.

## License

All whitepapers in this directory, including this index, are all rights reserved and marked `LicenseRef-Omega-Product-Proprietary`. No general commercial-use license is granted here. Commercial use may be considered only under a separate signed written agreement with a percentage-based royalty; the rate and calculation base must be negotiated and specified in that agreement. See [`../docs/LICENSING.md`](../docs/LICENSING.md), including its explanation of rights already granted under the previous MIT-licensed version and of copyright's limits on protecting ideas.

## Naming decision

The ecosystem uses **CARE** and **AMITY** as two layers of one social-infrastructure system:

- **CARE** is the protected, non-market proof and verification layer.
- **AMITY** is the exchange-eligible representation of approved, auditable CARE entitlements, subject to compliance, reserves, custody, and appeal.

They must not be treated as interchangeable balances. Care evidence must never be exposed to an exchange, and an agent must never be the sole legitimacy verifier.

The full four-plane model — CARE, TWC, OMEGA, AMITY — is described in
[`../docs/care-architecture-v2.md`](../docs/care-architecture-v2.md) §3 and
[`../docs/tri-token-integration-v1.md`](../docs/tri-token-integration-v1.md) §2. A paper that says
"Tri-Token" is using the older framing; the fourth plane (AMITY) was separated out afterwards.
