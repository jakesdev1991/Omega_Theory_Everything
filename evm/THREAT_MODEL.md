<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# $OMEGA Sepolia pilot threat model

**Status:** Testnet design artifact. This is not an audit, legal analysis, production threat model, or claim that the controls below eliminate risk.

## Scope and non-goals

This model covers the [`evm/`](.) `$OMEGA` pilot contracts on Ethereum Sepolia only:

- `tOMEGA` fixed-supply test-token balances;
- `veOMEGA` vote-escrow balances and lock positions;
- Governor / Timelock state and administrative roles;
- public novel-claim receipt events.

It does **not** cover real money, a token sale, off-chain content delivery, custody, identity/KYC, the Solana token leg, AMITY/Taproot Assets, private C.A.R.E. evidence, wallet browser extensions, an RPC operator, or a future mainnet deployment.

## Assets and security properties

| Asset/property | Required pilot property |
| --- | --- |
| `tOMEGA` supply | Constructed once; no post-deployment mint path; supply may only fall through an authorized holder burn. |
| Lock principal | A position owner can recover their own locked `tOMEGA` after its recorded expiry; no guardian or admin can redirect it. |
| Vote power | `veOMEGA` is non-transferable, derives only from a lock, and burns when that position exits. |
| Governance action | A successful vote is queued through the TimelockController before it can execute. |
| Gate receipt | Only an address meeting the immutable balance threshold during the immutable claim window can emit one claim event. |
| Privacy boundary | No content key, private evidence, health/care data, identity profile, or secret is stored in these contracts. |
| Deployment boundary | Custom pilot contracts reject construction outside Sepolia (`chainId 11155111`). |

## Authorities and trust assumptions

| Authority | Capability | Explicit limit |
| --- | --- | --- |
| Treasury | Receives the entire fixed pilot supply, then distributes valueless test tokens | Cannot mint additional `tOMEGA`, change another holder's balance, or bypass a lock. Treasury concentration is a pilot risk. |
| Guardian | Pause/unpause the gate; toggle voluntary emergency exit | Cannot transfer `tOMEGA`, mint/burn another account's `veOMEGA`, alter claim criteria, or withdraw someone else's position. |
| Governor + timelock | Govern future operations through a vote and delay | Can be captured by concentrated or sybil-distributed test tokens; no identity claim is made. |
| Deployer | Performs one-time wiring | Must renounce Timelock admin and vote-escrow ownership; the deployment script verifies that handoff. |
| Participant | Holds, locks, withdraws, votes, or claims with their own wallet | Is responsible for their wallet and for the public nature of an on-chain address/claim event. |

The pilot assumes OpenZeppelin Contracts `5.6.1`, Ethereum Sepolia, the configured compiler, the source commit, and the deployment manifest are independently checked before use. It does not assume an RPC provider, block producer, treasury, guardian, or user is honest by default.

## Threats and controls

| Threat | Current control | Residual risk / testnet drill |
| --- | --- | --- |
| Hidden or arbitrary token mint | `OmegaTestToken` mints only in its constructor and exposes no mint role/function; tests inspect fixed supply. | Review deployed ABI and `totalSupply == initialSupply`; voluntary burning still reduces supply. |
| Admin steals or freezes user tokens | The token has no privileged transfer, pause, blacklist, or freeze path. Lock positions have no admin withdrawal path. | Contract bugs and user-approved allowances remain risks; test normal transfer/withdraw while gate is paused. |
| Vote buying / flash-loan voting | Governor reads `veOMEGA`; voting units require a 7–365-day lock and are non-transferable. | Distribution concentration and sybil wallets remain unresolved; test snapshot behavior and publish concentration metrics. |
| Vote power remains after exit | `OmegaLocking` burns the position's `veOMEGA` before returning the principal. | Check the receipt and balance after normal and emergency exits. Snapshot rules intentionally preserve historical votes for a completed snapshot. |
| Proposal bypasses delay | Governor routes successful proposals to `TimelockController`; only governor has proposer rights after handoff. | A governance majority can still schedule a malicious but delayed action. Test propose → vote → queue → delay → execute, and review every target/calldata. |
| Guardian abuse | Guardian can stop gate claims or enable voluntary early exit, but cannot change balances or claims. | Availability/censorship is still a risk. Test pause/unpause incident response and replace/revoke guardian only through governance. |
| Duplicate or under-threshold claim | `hasClaimed` records one event per address; gate checks current balance, immutable window, and immutable threshold. | One-address-one-claim is not one-person-one-claim. Test repeat, insufficient, paused, and expired paths. |
| Sensitive data on chain | Gate stores only address, balance at claim, threshold, and timestamp in an event. | Wallet addresses remain public and linkable. Never use this pilot for sensitive C.A.R.E. data. |
| Wrong network / accidental mainnet deployment | Every custom contract has a construction guard for Sepolia and script checks `eth_chainId`. | Check the chain ID and explorer before signing. The raw OpenZeppelin timelock dependency alone is not chain-guarded and is not a usable pilot without the custom suite. |
| Incorrect deployment wiring | Script writes a manifest and asserts escrow-owner and Timelock-role handoffs. | Script cannot prove external addresses are controlled by intended humans. Independently compare constructor arguments and role memberships. |
| Dependency or compiler mismatch | Exact package lock, local `solc-js` 0.8.26, compiler settings, CI test job. | Run `npm ci`, `npm run check`, `npm audit --omit=dev`, and source verification from a clean checkout. |

## Required Sepolia drills

Before publishing a pilot address or distributing test tokens, complete and record each drill:

1. **Build provenance:** clean checkout; `npm ci`; `npm run check`; runtime dependency audit; record the source commit and lockfile hash.
2. **Deployment inspection:** verify chain ID, all constructor arguments, compiler settings, and deployment transaction hashes against the generated manifest.
3. **Authority handoff:** prove `OmegaVotingEscrow.owner() == address(0)`, the deployer lacks Timelock `DEFAULT_ADMIN_ROLE`, and the governor holds Timelock `PROPOSER_ROLE`.
4. **Supply and transfer:** reconcile `initialSupply`, treasury balance, and `totalSupply`; transfer test tokens between independent wallets.
5. **Lock lifecycle:** lock a short test position, inspect voting units/delegation, show early withdrawal fails, then show expiry withdrawal burns the vote units and returns principal.
6. **Emergency path:** guardian enables then disables the emergency exit; an owner exits voluntarily; verify no other account can exit that position.
7. **Governance path:** distribute test voting power, run a benign proposal through vote, timelock delay, and execution; decode and review target calldata before execution.
8. **Gate failure modes:** test insufficient balance, qualifying claim, duplicate claim, guardian pause/unpause, and expiry using separate wallets.
9. **Incident rehearsal:** simulate guardian key loss or pause misuse; document the timelocked governance response and human communications path.
10. **Privacy check:** inspect logs and confirm no sensitive data, content key, seed phrase, or private evidence entered a transaction, issue, manifest, or public channel.

A passed drill only shows the tested behavior on that configuration. It is not a substitute for independent review, adversarial testing, audit, legal analysis, or a mainnet go/no-go decision.
