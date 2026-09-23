<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# tTWC Devnet pilot threat model and operator runbook

## Scope and trust boundary

This document covers only the **valueless Solana Devnet `tTWC` pilot** built by this directory. It is not an audit, a mainnet readiness claim, a custody policy, a tokenomics specification, or a commercial launch authorization.

The deployment trusts an authorized operator to control the Devnet payer and name the treasury. It does not trust an RPC endpoint, metadata host, browser wallet, local `.env`, or a user-supplied client tier to establish asset facts without verification.

## Assets and intended invariants

| Asset / property | Required invariant |
|---|---|
| Network | RPC genesis hash equals Solana Devnet, never a host-name assumption alone |
| Mint | Standard SPL Token Program; 9 decimals; exact one-time configured supply |
| Mint authority | `None` after the one atomic transaction |
| Freeze authority | `None` from mint initialization onward |
| Treasury | The configured public address owns the canonical ATA containing the complete initial supply |
| Metadata | Canonical Metaplex metadata PDA; `Fungible`; exact name/symbol/URI; 0 seller fee; immutable |
| Off-chain metadata | HTTPS, no redirects, hash-pinned exact JSON, Devnet/valueless wording |
| Secrets | Deployer key is external or ignored, mode `0600`, never logged or written to a manifest |
| Manifest | Public-only, ignored local record; never silently overwritten |

## Controls and residual risks

| Threat | Control in this pilot | Residual risk / response |
|---|---|---|
| Mainnet or wrong-cluster broadcast | Compare `getGenesisHash` with the hard-coded Devnet hash before issuance and verification. | A compromised RPC can still deny service or lie; use an independent RPC/explorer during operator review. |
| Accidental broadcast | The command requires both `--confirm-devnet` and `TWC_DEPLOY_CONFIRM=DEVNET_TWC_PILOT`; default behavior stops after preflight. | A confirmed operator can still send an irreversible Devnet transaction. Review parameters before acknowledgement. |
| Excess or later supply | Mint, metadata, full supply mint, and mint-authority revocation are in one transaction; verifier checks supply and null authority. | A source/dependency defect could exist; this is why the pilot remains valueless and requires independent review before expansion. |
| Freeze/seizure control | The standard mint is initialized with `freezeAuthority: None`; verifier checks it. | Existing accounts can still be controlled by their own owners/delegates; no custody is supplied by this project. |
| Wrong treasury | An explicit public treasury address is required; verifier derives its ATA and checks the complete balance. | An operator can still enter an incorrect but valid address. Require out-of-band owner confirmation before broadcasting. |
| Mutable or swapped metadata | On-chain `isMutable: false`; URI must be HTTPS/no redirect; bytes are SHA-256 pinned and rechecked by verifier. | Immutable on-chain URI cannot ensure permanent hosting. Prefer a content-addressed durable host and archive the bytes. |
| Misleading pilot identity | On-chain symbol is `tTWC`; metadata must state Devnet and valueless status. | Names/tickers can still collide or be impersonated. Legal/trademark clearance is not completed here. |
| Payer secret exposure | Key file path only; no private key environment variable; strict JSON/permission checks; manifests contain public data only. | Endpoint malware, shell history, backup systems, and a compromised host remain out of scope. Use an externally held disposable Devnet signer. |
| Dependency or SDK drift | Exact package lock, Node 22+, offline instruction-plan tests, runtime audit command, Dependabot/CI coverage. | A passing dependency audit is not an audit of logic or upstream programs. Pin review and re-run the full suite before a deployment. |
| Client-side unlock abuse | No web gate is connected to this pilot. | Future gates must independently query chain data or verified events; never authorize a user from a client-supplied tier/address. |

## Required preflight

Before a Devnet transaction, an authorized operator must check all of the following:

- [ ] `git status` and source commit are reviewed; `npm ci`, `npm run check`, and `npm audit --omit=dev --audit-level=high` pass.
- [ ] `SOLANA_DEPLOYER_KEYPAIR_PATH` points to a new, external **Devnet-only** signer with owner-only file permissions.
- [ ] The payer's public address was funded only with Devnet SOL and is not a mainnet/prod treasury.
- [ ] `TWC_TREASURY_ADDRESS` has been independently confirmed by its authorized controller.
- [ ] Final metadata is public, direct HTTPS, immutable/content-addressed where possible, has been visually reviewed, and its exact JSON hash equals `TWC_METADATA_SHA256`.
- [ ] The preflight output says Devnet, correct payer/treasury/supply/metadata, and shows no existing manifest.
- [ ] A second reviewer approves the exact command and expected explorer records.

## Post-transaction verification

Run `npm run verify:devnet` using the manifest, then independently check the mint and transaction in a second Devnet RPC/explorer. Confirm every invariant in the asset table, especially `mintAuthority = None`, `freezeAuthority = None`, the full treasury balance, and immutable metadata.

Do not distribute even test tokens until that evidence is recorded.

## Failure and incident handling

### Before broadcast / preflight fails

Stop. Correct configuration, funding, metadata availability, or the RPC endpoint. Do not weaken the genesis/hash/confirmation guards.

### Transaction fails or is rejected

The issuance sequence is atomic: a failed transaction must not leave a partial pilot mint, metadata account, or supply. Preserve the error and transaction signature if one exists, inspect it in a Devnet explorer, and do not retry blindly. Re-run preflight only after understanding the failure.

### Broadcast status is unknown

Record the displayed mint and transaction signature; query independent Devnet RPC/explorers before taking another action. Do not attempt to recreate a mint with the same public identity or claim that an unknown transaction succeeded. If confirmed, create a public-only manifest only after full verification; if not confirmed, retain the incident record and begin a fresh reviewed attempt.

### Post-deploy verification fails

Treat the mint as nonconforming. Do not distribute it or represent it as this pilot. Preserve the manifest candidate, transaction signature, RPC responses, metadata bytes, and source commit for review. A correction requires a new separately reviewed Devnet pilot; no upgrade, admin recovery, freeze, or hidden mint path exists.

### Key compromise or wrong treasury

Do not attempt an on-chain recovery: authority removal intentionally prevents it. Mark the pilot compromised/noncanonical in off-chain records, cease any distribution, notify affected test participants, and review the operational failure. A new pilot requires a new disposable Devnet signer, independently verified treasury, and fresh review.

## Explicit non-goals

This pilot has no sale, price, liquidity pool, bridge, exchange listing, vesting, staking, governance claim, personal-data store, real-world redemption, custody service, or production unlock authorization. Those capabilities must not be inferred from a standard transferable SPL test token.
