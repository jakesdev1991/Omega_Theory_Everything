<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Day-One Token-Gated Novel Launch Plan

**Status:** Planning specification — not an offer, not a deployment, not legal advice.
**Version:** 0.1
**Date:** 2026-09-23
**Depends on:** `tri_token_sovereign_economy_blueprint.md`, `whitepapers/`, `novel/`, `app/`

## 1. Objective and locked decisions

Release the novel as the flagship day-one asset of the ecosystem: on the same day the three
tokens go live on mainnet, anyone who **holds any one of the three tokens** can unlock the
novel. This covers both kinds of economy participation:

- **Investing** — holding/purchasing a token.
- **Working** — earning a token through verified useful work (PoUW), per the ecosystem spec
  ("creators may earn WCT/AMITY for approved work").

| Decision | Value |
|---|---|
| Release asset | The novel (16 chapters; manuscript held by the author; title TBD) |
| Gate mechanism | Token-gated unlock (hold to unlock) |
| $OMEGA | Ethereum (specific network **TBD** — see §10) |
| WCT | Solana (**ticker collision — see §10, critical**) |
| AMITY | Bitcoin, Lightning via Taproot Assets |
| Day-one rule | Holding ≥ published threshold of any one token unlocks the novel |

## 2. Where things stand today

Honest inventory as of this plan:

| Component | State |
|---|---|
| Novel manuscript | Complete through Ch. 16 in the author's possession; **not yet staged in repo**; title, cover, ebook formats not produced |
| Token contracts | **None exist.** No Solidity, no SPL program, no Taproot asset |
| Rust core (`rust/`) | Deterministic ledger + PoUW claim lifecycle prototype with tests; no chain integration |
| App (`app/`) | Offline, valueless mockup; no wallet connection of any kind |
| Specs/whitepapers | Tri-token blueprint + 4 whitepapers ($OMEGA, TOKAMAK, C.A.R.E./AMITY, Lucifer–Hermes) |
| Naming consistency | **Mismatches:** whitepaper says TOKAMAK but the live token is WCT; blueprint's SOV/USE/CARE set vs the live OMEGA/WCT/AMITY set |
| Audits, legal, testnet | Nothing started |

The repo's own blueprint (§13–§15) requires local prototypes → testnet pilot → audit before
value. This plan sequences the launch to respect that, or to document any risk accepted by
choosing otherwise.

## 3. Chain legs

### 3.1 $OMEGA — Ethereum (network TBD)

**Open decision (highest priority):** Ethereum mainnet vs an L2 (Base, Arbitrum, OP Mainnet,
Linea, etc.).

- Recommendation: **an L2**. The gate performs frequent balance checks and claims; L2 gas is
  negligible where mainnet gas would tax every claimant. Choose mainnet only if brand
  positioning demands the settlement layer.

Needed for this leg:

1. ERC-20 with the published supply/issuance curve from the $OMEGA whitepaper (fixed cap or
   defensible bound; the whitepaper currently leaves this parameterized — it must be pinned).
2. Governance stack (Governor + TimelockController, OpenZeppelin-style) — required by the
   whitepaper's mandate that parameter changes be delayed and observable.
3. Staking/lock contract ("topological-impedance staking"): `weight = amount ×
   duration_factor × bounded_commitment_factor`, with visible expiry and emergency exit.
4. The **gate/claim contract** (see §4).
5. Sepolia testnet pilot, then audit, then mainnet.

### 3.2 WCT — Solana

**Critical naming issue:** the ticker **WCT is already WalletConnect Token** — a major,
widely listed token on OP Mainnet with 130k+ holders. Launching a different "WCT" invites
exchange confusion, listing refusals, and potential trademark/dispute exposure. Options:

- keep the name "World Citizen Token" but adopt a distinct ticker (e.g. `WCIT`, `WORLDC`,
  `CITZ`), or
- rename the token outright.

This must be decided **before** any contract/metadata is created.

Needed for this leg:

1. SPL Token + Metaplex metadata under the final name/ticker.
2. Distribution plan: workers earn via PoUW (bridging from the Rust claim registry logic to
   on-chain issuance), investors via sale/allocations (subject to legal review).
3. Gate: signed-message verification — wallet signs a challenge, service verifies SPL balance
   ≥ threshold via RPC. Trust-minimized and simple; an on-chain program is optional.
4. Devnet pilot, audit of any on-chain program, then mainnet-beta.

### 3.3 AMITY — Bitcoin, Lightning via Taproot Assets

Feasibility (as of mid-2026): **viable and production-grade.** Taproot Assets is at v0.8;
Lightning transport has been live since v0.4 (July 2024); Tether's USDT rides Lightning via
Taproot Assets since early 2025. Reference stack: `litd` (bundled lnd + tapd), Universe
servers for asset proof/issuance history.

This is the most operationally heavy leg — you are running Bitcoin infrastructure:

1. Stand up `litd` + `tapd` (testnet first, then mainnet): node, channels, liquidity.
2. Issue the AMITY asset; publish a **Universe** server so holders can verify issuance and
   sync proofs.
3. Decide the AMITY distribution route: Lightning transfers to workers/investors (asset
   carried inside channels, routed by BTC).
4. Gate design: Bitcoin has no smart contracts, so the gate is **proof-of-holdings** — the
   wallet holding the Taproot Asset signs a message; the launch service verifies the asset
   proof (MS-SMT proof against the asset's universe/UTXO) and unlocks. Every verification
   should be logged and published as a receipt (matches the ecosystem's receipts ethos).
5. Wallet reality check: retail Taproot-Assets wallet support is still thin. Expect to
   document supported wallets explicitly, and provide a fallback verification path
   (e.g. custodial verification with published receipts) for holders who cannot sign proofs.

Start this leg first — longest lead time.

## 4. Unlock design

### 4.1 Sealed manuscript (pre-launch commitment)

The manuscript is committed to the repo **encrypted**, with a SHA-256 commitment of the
plaintext (see `novel/README.md`):

```
novel/manuscript.md.enc    # AES-256-CBC (PBKDF2) ciphertext — committed
novel/manuscript.sha256    # commitment: SHA-256 of the plaintext — committed
novel/.release-key.txt     # content key — NEVER committed; offline backup required
novel/.plaintext/          # working plaintext — never committed
```

Anyone can verify on day one that the released book is exactly the book that was sealed
before launch. The key is released only through the gates.

### 4.2 Three gates, one key

- Each chain gets an independent gate ($OMEGA claim contract; WCT signed-message
  verification; AMITY proof-of-holdings).
- Each gate, upon verifying a holder, releases **the same content key** (or a copy of the
  key encrypted to that claimant).
- A holder of any single token can therefore unlock the full novel alone. This is a hard
  requirement — a 3-of-3 Shamir split across chains would lock out single-token holders and
  is rejected for that reason.
- Optional ceremonial layer: a 3-shard Shamir split revealed simultaneously at the moment
  each chain's gate first opens (three ledgers, one mind — on-theme), with each gate still
  sufficient on its own to deliver the full key. Theater on top of the mechanism, not the
  mechanism.

### 4.3 Claim flow (user's view)

```
launch page → connect wallet (ETH / Solana / BTC) →
gate verifies holding ≥ threshold →
claim recorded (on-chain event or claim NFT + published receipt) →
content key delivered → download EPUB/PDF (optionally watermarked per claimant)
```

Claim window: e.g. 90 days from day one; one claim per wallet; thresholds published before
launch and not changeable after (or only by the documented governance path).

### 4.4 Honest limitation

Token-gating is an access ritual and a participant perk, **not DRM**. Once any claimant has
the plaintext, it can be reshared. Optional hardening: per-claimant watermarked ebook
editions; signed/numbered limited editions; keeping premium formats (illustrated, signed)
as the durable gated asset.

## 5. Eligibility — "investing or working"

Day-one simplicity: **balance-based gating**. Holding ≥ threshold of any one token at claim
time qualifies. Workers naturally hold WCT/AMITY earned through PoUW; investors hold $OMEGA
(or any of the three). No separate registry is needed for day one.

Future (post-launch): tie gating/claims to the on-chain `UsefulWorkRegistry` (blueprint §9)
so verified work itself — not just the resulting balance — can qualify a participant.

## 6. Content & IP workstream (start immediately — longest lead)

1. **Title** the novel; produce cover + EPUB/PDF; optional per-claimant watermarking.
2. **IP clearance before any commercial distribution:**
   - *Naruto* references are load-bearing plot material (Ch. 5–6, Ch. 14: Infinite
     Tsukuyomi, Madara, 211 episodes). Requires review, rewrite, or clearance.
   - *D-Wave* is a real company named in a fictional heist (Ch. 4–6). Requires review.
   - *Crime and Punishment* is public domain — fine.
   - Satoshi Nakamoto / historical bitcoin events: fiction about a pseudonym; low risk, but
     include in counsel's pass.
3. **License:** the novel stays `LicenseRef-Omega-Product-Proprietary`. Decide (with
   counsel) what license claimants receive — e.g. personal, non-commercial read license;
   whether secondary resale of "editions" is permitted.
4. **Naming/trademark checks:** WCT ticker (§3.2, critical); check existing "Amity"-named
   tokens; check the novel's eventual title.

## 7. Workstreams

| # | Workstream | State | Blocking decisions |
|---|---|---|---|
| W1 | Manuscript & IP (title, formats, clearance) | Not started | Title; Naruto/D-Wave approach |
| W2 | $OMEGA contracts (ERC-20, governor, staking, gate) | Not started | Ethereum network choice |
| W3 | WCT token (SPL, metadata, distribution) | Not started | **Ticker rename** |
| W4 | AMITY Taproot infra (litd/tapd, universe, issuance) | Not started | Ops/hosting budget |
| W5 | Gates + key ceremony (3 gates, 1 key, dry-runs) | Not started | — |
| W6 | Launch app (wallet connect + unlock page) | Not started — app is offline-only today | Which wallets to support |
| W7 | Testnet drills + audits | Not started | Audit firm selection |
| W8 | Legal & compliance (securities, ToS, privacy, claims) | Not started | Counsel engagement |
| W9 | Ops & day-0 runbook (multisig, monitoring, pause) | Not started | — |

## 8. Sequencing (relative to launch day, "T-0")

**Now → T-10 weeks — decisions & derisking**

- Decide Ethereum network (mainnet vs L2) and the WCT ticker.
- Title the novel; begin IP clearance (longest legal lead).
- Stand up AMITY testnet node (`tapd` on testnet) — derisk the heaviest leg.
- Stage and seal the manuscript (`novel/seal.sh`); back up the key offline.
- Begin $OMEGA contract development locally (per blueprint §15: local prototypes first).
- Reconcile token naming across all docs (TOKAMAK → WCT? SOV/USE/CARE vs OMEGA/WCT/AMITY).

**T-10 → T-6 — build**

- All three legs live on testnets/devnet (Sepolia, Solana devnet, Bitcoin testnet).
- Gate prototypes working end-to-end against test tokens.
- Key ceremony dry-run #1 (generate, split, store, reconstruct).

**T-6 → T-3 — harden**

- External audits (EVM contracts; any Solana program; tapd config review).
- Adversarial drills per blueprint §12 threat model (sybil claims, threshold gaming, key
  leakage, wallet spoofing).
- Launch app integration (three wallet connections, unlock flow, downloads).
- Watermarking pipeline (if adopted).

**T-3 → T-1 — stage**

- Mainnet deployments in paused/gated state; final IP sign-off.
- Publish thresholds, claim rules, wallet-support docs, and the commitment hash.
- Freeze the runbook; go/no-go review against the blueprint's acceptance criteria (§14).

**T-0 — day one**

1. $OMEGA live (deployment + initial allocations per legal plan).
2. WCT live on Solana.
3. AMITY asset issued; Lightning distribution begins.
4. Gates open; content key released through gates; claims live.
5. Announcement with verification instructions (how to prove holding on each chain).

**T+1 → — operate**

- Claims support and appeals; published receipts; metrics (claims per chain, failures,
  support load); post-mortem.

## 9. Day-0 runbook (draft skeleton)

- Owners and fallback owners for every step; time-boxed sequence; comms templates pre-written.
- **The key release is one-way.** Once the content key is public it cannot be recalled —
  rehearse the full sequence on testnet at least twice before T-0.
- Pause authority: each gate independently pausable via multisig; pause does not affect
  token transfers.
- Incident path: gate bug → pause gate (holders of other chains unaffected) → fix → re-open.
- Escape hatches: if AMITY leg fails at T-0, open a manual verification desk (signed
  messages verified by humans, receipts published) rather than delaying the whole launch.

## 10. Risks and open decisions (register)

| # | Item | Severity | Notes |
|---|---|---|---|
| R1 | **WCT ticker collides with WalletConnect Token** | Critical | Rename ticker before any metadata exists |
| R2 | Ethereum network undecided | High | Blocks all $OMEGA contract work |
| R3 | Naruto / D-Wave IP in the novel | High | Clearance or rewrite before commercial release |
| R4 | AMITY ops burden + thin retail wallet support | High | Start infra earliest; provide fallback verification |
| R5 | Securities/regulatory posture of "hold to unlock" + investor framing | High | Counsel review before comms; careful wording of "investing" in public materials |
| R6 | Gating is not DRM | Medium | Optional watermarking; treat gate as perk/ritual |
| R7 | Content key custody | Medium | Ceremony, offline backups, no single holder |
| R8 | Day-one ambition vs blueprint's audit-first phases | Medium | Either respect Phase 2 or document accepted risk explicitly |
| R9 | Token naming inconsistency across docs | Medium | TOKAMAK vs WCT; SOV/USE/CARE vs live set |
| R10 | Snapshot/claim abuse (sybil wallets just above threshold) | Medium | One-claim-per-wallet, per-token thresholds, monitoring |

## 11. Immediate next actions

1. Decide the Ethereum network for $OMEGA.
2. Decide the WCT ticker (rename strongly advised).
3. Deliver the manuscript file and run `novel/seal.sh`; verify the commitment; back up
   `novel/.release-key.txt` offline.
4. Choose the novel's title; commission the IP review of the Naruto and D-Wave passages.
5. Stand up the AMITY testnet node (longest lead item).
6. Draft the $OMEGA ERC-20 and the gate contract locally (blueprint §15 order).
7. Reconcile token naming across the whitepapers, blueprint, and app copy.
