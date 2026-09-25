<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# $OMEGA Sepolia pilot contracts

This directory is the **Ethereum/Sepolia leg only** of the C.A.R.E. Protocol testnet pilot. It provides a deliberately valueless, Sepolia-only `$OMEGA` implementation, a bounded lock-to-vote flow, an OpenZeppelin Governor + TimelockController stack, and an on-chain novel-claim receipt gate.

> **Not production software. Not a token sale. Not a promise of value.** The custom pilot contracts reject deployment anywhere except Ethereum Sepolia (`chainId 11155111`). They have not received an independent audit. Do not send assets, use a mainnet private key, or treat a Sepolia balance as a right to a real-world payment, product, or service.

The broader three-chain plan remains deliberately split:

| Leg | Status in this repository |
| --- | --- |
| `$OMEGA` / Ethereum | **Implemented here for a valueless Sepolia pilot** |
| Token of the World Citizen (`TWC`) / Solana | A separate Devnet-only, valueless `tTWC` SPL issuer/verifier is in [`../solana/`](../solana/); no Devnet mint has been created and no mainnet/value-bearing use is approved. |
| AMITY / Bitcoin Lightning Taproot Assets | Requires separate `litd`/`tapd` testnet infrastructure; no asset is created here |

That scope follows the staged implementation requirement in the [tri-token blueprint](../tri_token_sovereign_economy_blueprint.md) and the `$OMEGA` workstream in the [day-one launch plan](../launch/novel_day_one_plan.md).

## What is included

| Contract | Purpose | Safety boundary |
| --- | --- | --- |
| `OmegaTestToken` | Fixed-supply ERC-20/Permit/Votes token named **Omega Test Token** (`tOMEGA`) | Entire initial supply is minted once to the supplied treasury; no post-deployment mint, owner, pause, blacklist, or administrative transfer function exists. A holder may only burn their own balance. |
| `OmegaVotingEscrow` | Non-transferable `veOMEGA` voting units | The locking contract is set exactly once; its setup owner is renounced by the deployer script. Units cannot be transferred. |
| `OmegaLocking` | Locks `tOMEGA` for 7–365 days | Each position has an on-chain expiry; vote weight is bounded from 1x to 2x. The guardian may only toggle a voluntary emergency exit, never move someone else’s position or mint. |
| `OmegaGovernor` | Snapshot-based governance | Reads `veOMEGA`, not liquid `tOMEGA`; successful actions must be queued through the timelock. |
| OpenZeppelin `TimelockController` | Delayed execution | The deployment script gives proposal rights to the governor, permits public execution, then removes the deployer’s temporary admin role. |
| `OmegaNovelGate` | One on-chain pilot claim per qualifying wallet | Immutable threshold/window; holds no content key, personal data, private evidence, or off-chain entitlement. Guardian pause affects the gate only, never token transfers. |

All custom contracts have a `SepoliaTestnetOnly` construction guard. The local Hardhat network mirrors Sepolia’s chain ID so the production bytecode is exercised in tests.

## Testnet parameter choices

The script pins the following **pilot defaults** unless an explicitly documented environment variable overrides them:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| Initial supply | `1,000,000,000 tOMEGA` | Minted once to `TREASURY_ADDRESS` at construction |
| Gate threshold | `1 tOMEGA` | Balance needed at the instant a test claim is submitted |
| Gate window | deploy-time → 90 days later | Immutable test claim window |
| Minimum / maximum lock | 7 / 365 days | Visible lock interval |
| Vote multiplier | 1x → 2x | Linear, bounded duration factor; **not a yield or reward** |
| Voting delay | 1 block | Sepolia pilot setting |
| Voting period | 7,200 blocks | Approximately one day at 12-second blocks; configurable before deployment only |
| Proposal threshold | `1,000 tOMEGA` locked | Snapshot threshold for proposal creation |
| Quorum | 4% of `veOMEGA` supply | Snapshot quorum |
| Timelock delay | 86,400 seconds | One day before a passed action can execute |

These are test inputs, not a mainnet tokenomics decision. A mainnet implementation must get a separate specification, legal review, threat model, economic simulation, independent audit, and deployment decision.

## Local verification

Prerequisites: Node.js 22+ and npm. The project pins `solc-js` locally, so compiling does **not** download a compiler during CI or local build.

```bash
cd evm
npm ci
npm run compile
npm test
npm audit --omit=dev
```

The test suite covers:

- EIP-170 runtime bytecode-size bounds for every custom deployable contract;
- one-time fixed supply and voluntary burn only;
- bounded time locks, automatic initial self-delegation, and non-transferable `veOMEGA`;
- expiry and voluntary guardian-enabled emergency exit;
- threshold, duplicate-claim, and pause behavior in the gate;
- proposal → vote → timelock queue → delayed execution;
- removal of temporary deployment authority.

## Sepolia deployment

1. Create a **new, Sepolia-only** deployer wallet and fund it with faucet test ETH. Do not reuse a mainnet wallet or store a mainnet private key in this repository.
2. Choose and verify public addresses for the pilot treasury and guardian. A testnet Safe/multisig is preferred when practical.
3. Configure environment variables locally:

   ```bash
   cd evm
   cp .env.example .env
   # Edit .env locally. It is ignored by Git.
   ```

   Required values:

   ```dotenv
   DEPLOYER_PRIVATE_KEY=0x...          # Sepolia-only key
   SEPOLIA_RPC_URL=https://...
   TREASURY_ADDRESS=0x...
   GUARDIAN_ADDRESS=0x...
   ```

   Optional inputs are listed in [`.env.example`](.env.example). Amounts are decimal token amounts; times are Unix seconds. The script rejects an invalid address, invalid number, threshold larger than supply, non-Sepolia network, or a missing deployer key.

4. Run a non-broadcasting preflight first, then deploy only after review:

   ```bash
   npm ci
   npm run compile
   npm run preflight:sepolia
   npm run deploy:sepolia
   ```

5. The script writes a public-address-and-parameter manifest to `evm/deployments/sepolia.json`. It is intentionally ignored so a local deployment cannot silently become a canonical release. Archive the manifest, transaction hashes, compiler settings, source commit, and verified explorer links in the pilot record.
6. With `ETHERSCAN_API_KEY` set, submit all six contracts from that manifest for source verification:

   ```bash
   npm run verify:sepolia
   ```

### What the deployment script hands off

The deployer is intentionally temporary:

1. Deploy `tOMEGA`, `veOMEGA`, `TimelockController`, `OmegaLocking`, `OmegaGovernor`, and `OmegaNovelGate`.
2. Bind `veOMEGA` to `OmegaLocking` once, then renounce the escrow owner.
3. Give the governor Timelock `PROPOSER_ROLE`.
4. Give `DEFAULT_ADMIN_ROLE` for the gate and locking contracts to the timelock at construction.
5. Renounce the deployer’s Timelock `DEFAULT_ADMIN_ROLE` and stop if that handoff did not succeed.

`GUARDIAN_ADDRESS` has `PAUSER_ROLE` on the gate and `EMERGENCY_ROLE` on locking. Neither role can mint, seize balances, rewrite a claim, change an immutable threshold/window, or bypass the timelock.

## Before anyone uses the pilot

Use this as a minimum post-deploy drill, not as an audit substitute:

- [ ] Compare all deployed bytecode, constructor arguments, chain ID, and addresses against the generated manifest.
- [ ] Run `npm run verify:sepolia` and confirm each source on a Sepolia explorer uses the exact Solidity `0.8.26`, optimizer (200 runs), and Cancun EVM settings in `hardhat.config.js`.
- [ ] Confirm `OmegaTestToken.totalSupply() == initialSupply()` and that the treasury received the supply.
- [ ] Confirm `OmegaVotingEscrow.owner() == address(0)` and that `locker()` is the deployed locking contract.
- [ ] Confirm the deployer no longer has Timelock `DEFAULT_ADMIN_ROLE`; confirm only the governor has `PROPOSER_ROLE`.
- [ ] Distribute only valueless test tokens from the treasury; test transfer, lock, delegation, expiry withdrawal, emergency-exit enable/disable, and vote snapshots with separate wallets.
- [ ] Test a passed proposal through the full timelock delay and test gate pause/unpause with the guardian.
- [ ] Test insufficient-balance, repeated-claim, expired-window, and paused-gate paths.
- [ ] Record expected recovery contacts and a human incident path outside the chain; no private key or content key belongs in a contract, issue tracker, or deployment manifest.
- [ ] Complete independent security review and adversarial testing before expanding beyond a valueless testnet pilot.

## Security notes and known limits

The pilot-specific assets, authorities, residual risks, and required Sepolia drills are documented in [`THREAT_MODEL.md`](THREAT_MODEL.md). The summary below is not a replacement for that review.

- The contracts are intentionally **non-upgradeable**. Fixes require a new pilot deployment and an explicit migration notice.
- The gate records a public wallet address and an event. It is not suitable for sensitive care, health, identity, or evidence data.
- Balance gating is a participation receipt, not DRM: content delivered off-chain can be copied.
- Voting power is only as sybil-resistant as the test-token distribution. This pilot does not solve identity, compliance, custody, or governance-concentration risk.
- `tOMEGA` is a fixed-supply test token; `veOMEGA` is a non-transferable accounting token for governance snapshots, not a redeemable asset.
- Solana and Taproot Assets have materially different security and operational models. Do not infer their readiness from this EVM implementation.

See [`../launch/novel_day_one_plan.md`](../launch/novel_day_one_plan.md), [`../whitepapers/omega_protocol_whitepaper.md`](../whitepapers/omega_protocol_whitepaper.md), and [`../tri_token_sovereign_economy_blueprint.md`](../tri_token_sovereign_economy_blueprint.md) for the broader, deliberately staged design.
