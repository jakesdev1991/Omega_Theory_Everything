<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Day-One Token-Gated Novel Launch Plan

**Status:** Planning specification — not an offer, not a deployment, not legal advice.
**Version:** 0.3 — 2026-09-23 decisions incorporated (network, selected TWC identity, protocol name, book title, copyright workstream)
**Date:** 2026-09-23
**Depends on:** `tri_token_sovereign_economy_blueprint.md`, `whitepapers/`, `novel/`, `app/`

## 1. Objective and locked decisions

Release the novel — **"Crucible: The Satoshi Protocol"** — as the flagship day-one asset of
the ecosystem: on the same day the three tokens go live on mainnet, anyone who **holds any
one of the three tokens** can unlock the novel. This covers both kinds of economy
participation:

- **Investing** — holding/purchasing a token.
- **Working** — earning a token through verified useful work (PoUW), per the ecosystem spec
  ("creators may earn TWC/AMITY for approved work"; any future issuance remains subject to a separate reviewed policy — see §2).

### 1.1 Decision register (updated 2026-09-23)

| Decision | Value | Status |
|---|---|---|
| Release asset | **Crucible: The Satoshi Protocol** (novel; draft complete through Ch. 16, author states it is not final) | ✅ Titled |
| Gate mechanism | Token-gated unlock (hold to unlock) | ✅ Locked |
| Umbrella protocol name | **The C.A.R.E. Protocol** (whole ecosystem) | ✅ Locked |
| $OMEGA | EVM — **current pilot on Ethereum Sepolia testnet**; mainnet-day-one venue (Ethereum L1 vs EVM L2) **still open** — see §3.1 | ⚠️ Partial |
| Solana token name | **Token of the World Citizen (`TWC`)**; Devnet pilot identity is `tTWC` | ✅ Selected; Devnet-only issuer/verifier implemented locally, no mint deployed |
| AMITY | Bitcoin/Lightning with Taproot Assets | ✅ Locked |
| Day-one rule | Holding ≥ published threshold of any one token unlocks the novel | ✅ Locked |
| Copyright | Author holds copyright automatically on creation; US registration of the draft via copyright.gov eCO planned (see §6.1) | 🔄 In progress |

**Important clarification on "Sepolia":** Sepolia is an Ethereum **testnet**. It is exactly
where the $OMEGA contracts should be piloted (blueprint Phase 2), and that is now the plan
of record. But "mainnet day one" requires the token to ultimately live on an EVM mainnet
venue — whether Ethereum L1 or an EVM L2 — because Sepolia test ETH carries no value and
cannot back a real launch. The mainnet venue decision (Ethereum L1 vs Base/Arbitrum/OP-style
EVM L2) remains open; §3.1 keeps the recommendation of an L2 for cheap gate/claim interactions.

## 2. Where things stand today

Honest inventory as of this plan:

| Component | State |
|---|---|
| Novel manuscript | Draft (author states not final); file not yet delivered to the repo — staging prepared in `novel/`; title, cover, ebook formats not produced |
| Token contracts | **Ethereum pilot implemented locally:** [`../evm/`](../evm/) contains a Sepolia-only, valueless `tOMEGA` fixed-supply token, vote escrow, locking, Governor + TimelockController, and claim gate. **Solana pilot implemented locally:** [`../solana/`](../solana/) contains a Devnet-only, valueless `tTWC` standard SPL mint issuer/verifier with immutable Metaplex fungible metadata and null mint/freeze authorities. **AMITY scaffold implemented locally:** [`../amity/`](../amity/) contains a testnet-only operator/configuration scaffold, canonical holder-challenge formatting, and a non-broadcasting preflight for the future Taproot Assets leg. **None is deployed or independently audited.** No Taproot asset exists yet. |
| Rust core (`rust/`) | Deterministic ledger + PoUW claim lifecycle prototype with tests; no chain integration |
| App (`app/`) | Local-first wallet prototype with real client-side signing flows; **currently wires two release currencies into the unlock path**: `$OMEGA` (Ethereum) and `TWC` (Solana) |
| Specs/whitepapers | Tri-token blueprint + 4 whitepapers ($OMEGA, TOKAMAK, C.A.R.E./AMITY, Lucifer–Hermes) |
| Naming consistency | The ecosystem/protocol is **the C.A.R.E. Protocol**; pilot identities are **`tOMEGA`** (Ethereum Sepolia), **`tTWC`** (Solana Devnet), and AMITY remains a separate Bitcoin/Lightning workstream. The canonical future Solana identity is **Token of the World Citizen (`TWC`)**. `TOKAMAK` is retired as the Solana-token name; blueprint SOV/USE/CARE remain legacy local-simulation names. Trademark/legal clearance remains open. |
| Audits, legal, testnet | No independent audit, legal review, or public testnet deployment has started |

The repo's own blueprint (§13–§15) requires local prototypes → testnet pilot → audit before
value. This plan sequences the launch to respect that, or to document any risk accepted by
choosing otherwise.

## 3. Chain legs

### 3.1 $OMEGA — Ethereum (pilot: Sepolia; mainnet venue open)

Decided: the $OMEGA contracts are piloted on **Sepolia testnet** first. That matches
blueprint Phase 2 and is now the plan of record.

Still open (highest-priority remaining decision): the **mainnet-day-one venue** —

- Ethereum L1 (settlement-layer brand; highest gas for every gate check and claim), vs
- an L2 such as Base, Arbitrum, or OP Mainnet (near-zero gas for frequent gate/claim
  interactions).
- Recommendation remains: **an L2**, for the same reasons as v0.1 of this plan.

Needed for this leg:

1. ERC-20 with the published supply/issuance curve from the $OMEGA whitepaper (fixed cap or
   defensible bound; the whitepaper currently leaves this parameterized — it must be pinned).
2. Governance stack (Governor + TimelockController, OpenZeppelin-style) — required by the
   whitepaper's mandate that parameter changes be delayed and observable.
3. Staking/lock contract ("topological-impedance staking"): `weight = amount ×
   duration_factor × bounded_commitment_factor`, with visible expiry and emergency exit.
4. The **gate/claim contract** (see §4).
5. **Sepolia pilot** (decided) → audit → mainnet deployment (venue decision pending).

**Implementation status (2026-09-23):** the local, Sepolia-only pilot suite is in
[`../evm/`](../evm/). It uses a plainly labelled `tOMEGA` test token, is guarded against
non-Sepolia deployment, and includes fixed supply, non-transferable vote escrow, 7–365-day
bounded locking, OpenZeppelin Governor + TimelockController, and an immutable-threshold
claim-receipt gate. Its automated test suite passes locally; no Sepolia deployment, external
audit, mainnet implementation, or value-bearing use is implied. Follow the preflight and
post-deploy drills in [`../evm/README.md`](../evm/README.md) before sending any deployment
transaction.

### 3.2 Token of the World Citizen (TWC) — Solana

The selected canonical Solana identity is **Token of the World Citizen (`TWC`)**. Its
intentionally non-production Devnet identity is **`tTWC`**. The implementation in
[`../solana/`](../solana/) is a standard SPL Token Program issuer/verifier, not a bespoke
Solana program: it creates an atomic fixed-supply Devnet mint, starts with no freeze authority,
revokes the mint authority, and attaches immutable Metaplex `Fungible` metadata. It is locally
tested and deployment-script ready, but **no Devnet mint exists yet**.

Historical collision review favored TWC over the discarded WCC/WCT options; that selection is
not a trademark search, a registration claim, or permission to commercialize the name. Counsel
must complete trademark, securities/compliance, consumer-protection, and distribution review
before any value-bearing or mainnet use.

Needed for this leg:

1. Approved public Devnet treasury, externally held Devnet-only signer/funds, immutable metadata
   URI/hash, independent preflight, then a valueless Devnet drill from [`../solana/README.md`](../solana/README.md).
2. A separate reviewed distribution policy before any future issuance beyond a test pilot; no sale,
   allocation, or economic claim is implemented here.
3. Gate: signed-message verification — wallet signs a challenge, service independently verifies
   an SPL balance or claim event via RPC. A local Devnet holder-verifier helper now exists in
   [`../solana/`](../solana/) for signed-proof + balance checks against a deployed mint manifest,
   but the production-facing web/nonced challenge flow still needs to be integrated; the existing
   client-supplied tier flow is not adequate.
4. Independent security review, legal approval, and a separate mainnet specification before any
   mainnet-beta deployment.

### 3.3 AMITY — Bitcoin/Lightning with Taproot Assets

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

- Each chain gets an independent gate ($OMEGA claim contract; TWC signed-message
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
time qualifies. Workers may eventually hold TWC/AMITY earned through independently reviewed
PoUW policies; holders may use any one of the three tokens. No separate registry is needed for day one.

Future (post-launch): tie gating/claims to the on-chain `UsefulWorkRegistry` (blueprint §9)
so verified work itself — not just the resulting balance — can qualify a participant.

## 6. Content & IP workstream (start immediately — longest lead)

1. **Title:** decided — **"Crucible: The Satoshi Protocol"** (Ch. 15's CRUCIBLE reveal names
   the book). Confirm the byline (legal name "Jacob See" vs a pen name) — the eCO
   registration and title page need it.
2. Produce cover + EPUB/PDF; optional per-claimant watermarking.
3. **IP clearance before any commercial distribution:**
   - *Naruto* references are load-bearing plot material (Ch. 5–6, Ch. 14: Infinite
     Tsukuyomi, Madara, 211 episodes). Requires review, rewrite, or clearance.
   - *D-Wave* is a real company named in a fictional heist (Ch. 4–6). Requires review.
   - *Crime and Punishment* is public domain — fine.
   - Satoshi Nakamoto / historical bitcoin events: fiction about a pseudonym; low risk, but
     include in counsel's pass.
4. **License:** the novel stays `LicenseRef-Omega-Product-Proprietary`. Decide (with
   counsel) what license claimants receive — e.g. personal, non-commercial read license;
   whether secondary resale of "editions" is permitted.
5. **Naming/trademark checks:** selected Solana identity TWC (§3.2), including a full
   clearance search; check existing "Amity"-named tokens; note that a book **title is not protected by copyright** (titles
   and short phrases are excluded), so if "Crucible: The Satoshi Protocol" matters as a
   brand, that is a trademark question.

### 6.1 Copyright registration (United States)

Facts as of 2026-09 (US Copyright Office; the author is a US-based rights holder):

- **Copyright already exists.** Under US law, copyright arises automatically when an
  original work is fixed in a tangible medium — the manuscript file itself is protected the
  moment it exists. No notice, registration, or © symbol is required to own the copyright.
- **Registration is separate and worth doing.** It is required before filing an infringement
  suit, and registering before infringement (or within 3 months of publication) unlocks
  statutory damages (up to $150,000 per work for willful infringement) and attorney's fees.
- **An unpublished draft is registrable now.** The author states the book is not final; the
  Office accepts registrations of unpublished manuscripts. Common practice: register the
  draft to lock in priority, then register the final published version later — the second
  registration covers the text as published.
- **How:** copyright.gov → eCO (Electronic Copyright Office) → "Register a New Claim" →
  work type **Literary Work**, material type **Text**, publication status **unpublished**;
  upload a complete copy (PDF) as the deposit; pay the fee. Fees (Circular 4): **$45**
  single application (one work, one author, not made for hire, same claimant), **$65**
  standard, $125 paper. Processing runs roughly 1–8 months; the filing date is locked at
  submission.
- **Repo evidence:** the sealed ciphertext + SHA-256 commitment in `novel/` (with dated
  commits) supports the authorship/priority record per
  [`../docs/PROVENANCE.md`](../docs/PROVENANCE.md), but it is **not a substitute** for
  registration. "Poor man's copyright" (mailing a copy to yourself) is a myth — don't rely
  on it.
- **Scope:** registration protects the text (expression). It does not protect the title,
  the Satoshi character concept, or ideas — only the specific written expression.

To-do: file the eCO application for the current draft once the manuscript file is sealed,
and again for the final version at (or within 3 months of) day-one publication.

## 7. Workstreams

| # | Workstream | State | Blocking decisions |
|---|---|---|---|
| W1 | Manuscript & IP (title, formats, clearance) | Not started | Title; Naruto/D-Wave approach |
| W2 | $OMEGA contracts (ERC-20, governor, staking, gate) | **Implemented locally; Sepolia deployment pending** — see [`../evm/`](../evm/) | Sepolia treasury/guardian addresses, test ETH, preflight, independent audit before any expansion |
| W3 | TWC token (SPL, immutable metadata, distribution policy) | **Devnet-only `tTWC` issuer/verifier implemented locally; no mint deployed** | Authorized Devnet treasury/signer/funds/metadata, preflight, independent review; legal clearance before any mainnet/value-bearing step |
| W4 | AMITY Taproot infra (litd/tapd, universe, issuance) | **Started:** local testnet-only operator scaffold + preflight now exist in [`../amity/`](../amity/); no node, asset, or Universe server yet | Ops/hosting budget |
| W5 | Gates + key ceremony (3 gates, 1 key, dry-runs) | Not started | — |
| W6 | Launch app (wallet connect + unlock page) | **In progress:** wallet/web flow now wires `$OMEGA` and `TWC` proofs end to end, and the web app can independently check the configured Sepolia/Devnet pilot rails when manifests or equivalent env vars are present; AMITY is now operator-visible as a separate scaffold status surface only, and broader deployment hardening remains pending | Which additional wallets to support beyond MetaMask + Phantom |
| W7 | Testnet drills + audits | Not started | Audit firm selection |
| W8 | Legal & compliance (securities, ToS, privacy, claims) | Not started | Counsel engagement |
| W9 | Ops & day-0 runbook (multisig, monitoring, pause) | Not started | — |

## 8. Sequencing (relative to launch day, "T-0")

**Now → T-10 weeks — decisions & derisking**

- Decide Ethereum network (mainnet vs L2); the Solana identity is selected as TWC, while legal/trademark clearance remains open.
- Title the novel; begin IP clearance (longest legal lead).
- Stand up AMITY testnet node (`tapd` on testnet) — derisk the heaviest leg.
- Stage and seal the manuscript (`novel/seal.sh`); back up the key offline.
- Run the `$OMEGA` local test suite and Sepolia preflight/deployment drill from [`../evm/`](../evm/) (the local pilot contracts are implemented; do not skip the review gates).
- Reconcile token naming across all docs (TOKAMAK retired as Solana-token name; canonical TWC; legacy SOV/USE/CARE vs OMEGA/TWC/AMITY).

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
2. TWC live on Solana only after its separately approved mainnet specification, security review, legal clearance, and deployment decision.
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
| R1 | Historical WCT/WCC ticker collisions | Resolved for pilot identity | WCT and WCC were discarded; TWC is selected. This does not replace trademark/legal clearance (see R11). |
| R2 | $OMEGA mainnet venue undecided | High | Sepolia testnet pilot decided; mainnet-day-one venue (L1 vs L2) still blocks the final deployment path |
| R3 | Naruto / D-Wave IP in the novel | High | Clearance or rewrite before commercial release |
| R4 | AMITY ops burden + thin retail wallet support | High | Start infra earliest; provide fallback verification |
| R5 | Securities/regulatory posture of "hold to unlock" + investor framing | High | Counsel review before comms; careful wording of "investing" in public materials |
| R6 | Gating is not DRM | Medium | Optional watermarking; treat gate as perk/ritual |
| R7 | Content key custody | Medium | Ceremony, offline backups, no single holder |
| R8 | Day-one ambition vs blueprint's audit-first phases | Medium | Either respect Phase 2 or document accepted risk explicitly |
| R9 | Token naming inconsistency across docs | Medium | Canonical: C.A.R.E. Protocol umbrella; $OMEGA / TWC / AMITY; Devnet identities are `tOMEGA` / `tTWC`; TOKAMAK is retired as the Solana-token name. Reconcile remaining legacy whitepaper/app copy. |
| R10 | Snapshot/claim abuse (sybil wallets just above threshold) | Medium | One-claim-per-wallet, per-token thresholds, monitoring |
| R11 | TWC name/ticker collision and trademark risk | Medium | TWC was selected after historical comparison, but no availability conclusion is claimed. Obtain jurisdiction- and goods/services-specific trademark clearance and legal review before commercial/mainnet use. |
| R12 | Manuscript not final | Low | Seal draft now for priority evidence; re-seal final at publication; re-register copyright for the final text (§6.1) |

## 11. Immediate next actions

1. Decide the **$OMEGA mainnet venue** (Ethereum L1 vs an L2 such as Base/Arbitrum/OP) — Sepolia is already decided for the pilot.
2. Complete TWC trademark/legal clearance and the Devnet-only `tTWC` preflight in [`../solana/README.md`](../solana/README.md); deploy only after the named treasury, external Devnet signer/funds, approved metadata URI/hash, and independent review are present.
3. Deliver the manuscript file (`full-book.md` did not reach the repo on 2026-09-23 — re-attach), then run `novel/seal.sh`; verify the commitment; back up `novel/.release-key.txt` offline.
4. Confirm the byline (legal name vs pen name) and file the **eCO copyright registration for the draft** (§6.1: $45, Literary Work, unpublished).
5. Commission the IP review of the Naruto and D-Wave passages.
6. Stand up the AMITY testnet node (longest lead item).
7. Complete the `$OMEGA` Sepolia pilot preflight in [`../evm/README.md`](../evm/README.md), deploy only valueless `tOMEGA` to Sepolia, and record the drill before commissioning an independent audit.
8. Reconcile remaining legacy whitepaper and app copy (C.A.R.E. Protocol / $OMEGA / TWC / AMITY), retaining historical wording only where clearly labeled.
