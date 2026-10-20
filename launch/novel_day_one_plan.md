<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# Public Novel Release & C.A.R.E. Participation Plan

**Status:** The current manuscript draft is public and licensed under CC BY 4.0 (decision incorporated 2026-10-10). This remains a planning specification for future economy participation and does not make the text token-gated, private, or exclusive. Not an offer, deployment, or legal advice.
**Version:** 0.4 — public manuscript and licensing decision incorporated
**Date:** 2026-10-10
**Depends on:** `tri_token_sovereign_economy_blueprint.md`, `whitepapers/`, `novel/`, `app/`

## 1. Objective and locked decisions

Make the current draft of **"Crucible: The Satoshi Protocol"** freely available as public knowledge. The complete text is committed at `web/public/book/full-book.md` and released under CC BY 4.0, so reading and reuse do not require a token, wallet, or claim. The existing $OMEGA and TWC proof flows may remain as optional reader/participation demonstrations, but they do not restrict the manuscript or grant exclusive access. Any future gated edition or bonus work must be genuinely new, rights-cleared, and licensed separately.

Token ownership and verified useful work remain separate participation concepts in the broader economy; neither is a condition for reading or reusing the current book.

### 1.1 Decision register (updated 2026-10-10)

| Decision | Value | Status |
|---|---|---|
| Release asset | **Crucible: The Satoshi Protocol** (public 17-chapter draft; author states it is not final) | ✅ Public draft at `web/public/book/full-book.md` |
| Access | Public text under CC BY 4.0; proof-verified reader flow is optional and non-exclusive | ✅ Applied in this release update |
| Umbrella protocol name | **The C.A.R.E. Protocol** (whole ecosystem) | ✅ Locked |
| $OMEGA | EVM — **current pilot on Ethereum Sepolia testnet**; mainnet-day-one venue (Ethereum L1 vs EVM L2) **still open** — see §3.1 | ⚠️ Partial |
| Solana token name | **Token of the World Citizen (`TWC`)**; Devnet pilot identity is `tTWC` | ✅ Selected; Devnet-only issuer/verifier implemented locally, no mint deployed |
| AMITY | Bitcoin/Lightning with Taproot Assets | ✅ Locked |
| Day-one rule | No token threshold for reading or reusing the published text; any future token-holder benefit must be separate | ✅ Public text |
| Copyright / license | CC BY 4.0 grant for original material the rights holder is authorized to license; third-party clearance and provenance review remain open (see §6) | ✅ License stated; review open |

**Important clarification on "Sepolia":** Sepolia is an Ethereum **testnet**. It is exactly
where the $OMEGA contracts should be piloted (blueprint Phase 2), and that is now the plan
of record. But "mainnet day one" requires the token to ultimately live on an EVM mainnet
venue — whether Ethereum L1 or an EVM L2 — because Sepolia test ETH carries no value and
cannot back a real launch. The mainnet venue decision (Ethereum L1 vs Base/Arbitrum/OP-style
EVM L2) remains open; §3.1 keeps the recommendation of an L2 for cheap, optional proof interactions.

## 2. Where things stand today

Honest inventory as of this plan:

| Component | State |
|---|---|
| Novel manuscript | Public 17-chapter draft at `web/public/book/full-book.md`, licensed CC BY 4.0; author states it is not final. No ciphertext or release key protects this public copy; cover and ebook formats are not produced |
| Token contracts | **Ethereum pilot implemented locally:** [`../evm/`](../evm/) contains a Sepolia-only, valueless `tOMEGA` fixed-supply token, vote escrow, locking, Governor + TimelockController, and claim gate. **Solana pilot implemented locally:** [`../solana/`](../solana/) contains a Devnet-only, valueless `tTWC` standard SPL mint issuer/verifier with immutable Metaplex fungible metadata and null mint/freeze authorities. **AMITY scaffold implemented locally:** [`../amity/`](../amity/) contains a testnet-only operator/configuration scaffold, canonical holder-challenge formatting, and a non-broadcasting preflight for the future Taproot Assets leg. **None is deployed or independently audited.** No Taproot asset exists yet. |
| Rust core (`rust/`) | Deterministic ledger + PoUW claim lifecycle prototype with tests; no chain integration |
| App (`app/`) | Local-first wallet prototype with real client-side signing flows; **currently wires two currencies into the optional reader-proof demo**: `$OMEGA` (Ethereum) and `TWC` (Solana) |
| Specs/whitepapers | Tri-token sovereign-economy blueprint + 7 whitepaper/manifesto documents in [`../whitepapers/`](../whitepapers/): $OMEGA, **TWC**, the **TOKAMAK domain-research draft** (historical name; retired as the Solana-token identity — see its own naming note), C.A.R.E./AMITY, C.A.R.E. economy + manifesto, and Lucifer–Hermes Omni-Bridge |
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
4. Any claim-receipt contract is separate from the public novel; it does not restrict access to the CC BY 4.0 text.
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
3. Optional reader-proof demonstration: a signed message can be checked against a Devnet
   balance or claim event via RPC. The local helper in [`../solana/`](../solana/) supports that
   pilot flow, but no proof is required to read or reuse the public CC BY 4.0 manuscript.
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
4. No AMITY gate is needed for the public manuscript. If AMITY proofs are later used for a
   separate participation service, the wallet may prove Taproot Asset holdings via an MS-SMT
   proof against the asset's Universe/UTXO; document that as a separate feature, not as control
   over the CC BY 4.0 text.
5. Wallet reality check: retail Taproot-Assets wallet support is still thin. Expect to
   document supported wallets explicitly, and provide a fallback verification path
   (e.g. custodial verification with published receipts) for holders who cannot sign proofs.

Start this leg first — longest lead time.

## 4. Public access and CC BY 4.0

### 4.1 Published manuscript

The current full draft is `web/public/book/full-book.md`. It is present in public Git and in
Next.js' public asset directory, so it is available directly at `/book/full-book.md` without
wallet or proof. This is intentional. Do not describe this copy as encrypted, confidential,
exclusive, or token-gated. Running `novel/seal.sh` on a copy cannot undo the existing Git or
website disclosure.

### 4.2 License and attribution

The original literary text is released by Jacob See under the Creative Commons Attribution
4.0 International Public License (CC BY 4.0), only to the extent the licensor owns or is
authorized to license the relevant rights. CC BY 4.0 permits sharing and adaptation, including
commercial use, with appropriate attribution, a link to the license, and an indication of
changes. The license is not a public-domain dedication. See `web/public/book/LICENSE.md` and
`../LICENSES/CC-BY-4.0.txt` for the notice and full legal code.

The license does not grant third-party rights in characters, names, marks, quotations, or
other material the licensor cannot license. The Naruto and D-Wave references are still flagged
for rights review; the CC BY grant is not that review and does not imply endorsement. Consult
qualified counsel before commercial exploitation of material whose clearance is uncertain.

### 4.3 Optional proof-verified reader

The current $OMEGA and TWC proof routes may continue as optional demonstrations of wallet
signature and pilot-chain verification. They do not control access to the public manuscript,
change the CC BY terms, or confer exclusive rights. Visitors can read and reuse the text
without completing a proof. Any future token-holder content must be distinct from this draft,
rights-cleared, and clearly licensed before it is offered.

### 4.4 Honest limitation

CC BY 4.0 allows recipients to redistribute and adapt the licensed text under its terms. A
website may offer a convenient reader or proof flow, but must not imply that the public text
is technically or legally restricted to token holders. Attribution, trademark, third-party
rights, and any rights in later versions are separate questions.

## 5. Participation eligibility

There is no token threshold for reading or reusing the current manuscript. CC BY 4.0 is
available to everyone who complies with its terms. Token balances or proof-of-useful-work
receipts may support separately specified participation benefits, but they cannot be used to
revoke or narrow the public license for this text. Any future gated content must be newly
created or otherwise not already covered by this public grant.

## 6. Content, attribution & rights workstream

1. **Public release:** the current draft is already available at `web/public/book/full-book.md`
   under CC BY 4.0. Keep the source, public license notice, and direct access link together.
2. **Attribution:** the notice currently recommends “Jacob See, *Crucible: The Satoshi
   Protocol* (public draft), CC BY 4.0.” Confirm whether the site should use this name or a
   pen name before producing a cover or other final byline treatment.
3. **Third-party material:** the launch plan's review items remain open. In particular, the
   Naruto references and the D-Wave company reference require rights review, rewrite, or
   clearance before commercialization. Public availability or CC BY licensing does not
   clear those elements.
4. **Formats:** any EPUB, PDF, or other copy of this same text should carry the CC BY 4.0
   notice and preserve attribution. If a later edition contains materially new content,
   identify its version and license separately; prior CC BY copies remain usable under their
   original terms.
5. **Names and marks:** the title may be used as a book title, but CC BY 4.0 does not grant
   trademark rights in “Crucible: The Satoshi Protocol” or other project names. Trademark
   clearance remains a separate review item.

### 6.1 Copyright and provenance follow-up

The manuscript is no longer described as unpublished or sealed: it is in public Git and in
the website's public asset directory. Retain source drafts and dated repository evidence, and
record the public license notice. If the rights holder seeks formal registration or makes
ownership claims, confirm current Copyright Office requirements and the work's provenance
with qualified counsel; this document is not legal advice. The repository's rights register
still calls for review of authorship, contributors, AI-generated or adapted material, and
third-party content. See [`../docs/PROVENANCE.md`](../docs/PROVENANCE.md).

## 7. Workstreams

| # | Workstream | State | Blocking decisions |
|---|---|---|---|
| W1 | Public manuscript, attribution & IP review | **Draft public under CC BY 4.0** | Confirm final byline; review Naruto/D-Wave and other third-party material |
| W2 | $OMEGA contracts (ERC-20, governor, staking, gate) | **Implemented locally; Sepolia deployment pending** — see [`../evm/`](../evm/) | Sepolia treasury/guardian addresses, test ETH, preflight, independent audit before any expansion |
| W3 | TWC token (SPL, immutable metadata, distribution policy) | **Devnet-only `tTWC` issuer/verifier implemented locally; no mint deployed** | Authorized Devnet treasury/signer/funds/metadata, preflight, independent review; legal clearance before any mainnet/value-bearing step |
| W4 | AMITY Taproot infra (litd/tapd, universe, issuance) | **Started:** local testnet-only operator scaffold + preflight now exist in [`../amity/`](../amity/); no node, asset, or Universe server yet | Ops/hosting budget |
| W5 | Public text and optional proof-reader flow | **Text is public; no content-key ceremony** | Keep optional proof flow clearly separate from access/license |
| W6 | Optional reader proof demo | **In progress:** wallet/web flow wires `$OMEGA` and `TWC` proofs end to end and can check configured Sepolia/Devnet pilot rails; this is not required for the public manuscript. AMITY remains a separate operator-visible scaffold | Which additional wallets to support beyond MetaMask + Phantom |
| W7 | Testnet drills + audits | Not started | Audit firm selection |
| W8 | Legal & compliance (securities, ToS, privacy, claims) | Not started | Counsel engagement |
| W9 | Ops & day-0 runbook (multisig, monitoring, pause) | Not started | — |

## 8. Sequencing (relative to launch day, "T-0")

**Now → T-10 weeks — decisions & derisking**

- Decide Ethereum network (mainnet vs L2); the Solana identity is selected as TWC, while legal/trademark clearance remains open.
- Confirm the public byline and begin rights clearance (longest legal lead).
- Stand up AMITY testnet node (`tapd` on testnet) — derisk the heaviest leg.
- Keep the CC BY 4.0 notice beside the public manuscript and verify the direct public link; do not seal this already-published draft.
- Run the `$OMEGA` local test suite and Sepolia preflight/deployment drill from [`../evm/`](../evm/) (the local pilot contracts are implemented; do not skip the review gates).
- Reconcile token naming across all docs (TOKAMAK retired as Solana-token name; canonical TWC; legacy SOV/USE/CARE vs OMEGA/TWC/AMITY).

**T-10 → T-6 — build**

- All three legs live on testnets/devnet (Sepolia, Solana devnet, Bitcoin testnet).
- Optional proof-reader prototypes working end-to-end against test tokens.
- Verify that no reader setup or token proof is required for the public manuscript.

**T-6 → T-3 — harden**

- External audits (EVM contracts; any Solana program; tapd config review).
- Adversarial drills per blueprint §12 threat model (sybil claims, threshold gaming, key
  leakage, wallet spoofing).
- Harden the optional proof-reader integration and make direct manuscript/download links easy to find.
- Watermarking pipeline (if adopted).

**T-3 → T-1 — stage**

- Any approved mainnet token deployments remain paused; complete final economy/legal review before activation. Public manuscript status and CC BY 4.0 are independent of these gates.
- Publish any token-pilot thresholds and wallet-support docs separately from the book license; no manuscript key or secrecy commitment is involved.
- Freeze the runbook; go/no-go review against the blueprint's acceptance criteria (§14).

**T-0 — day one**

1. $OMEGA live (deployment + initial allocations per legal plan).
2. TWC live on Solana only after its separately approved mainnet specification, security review, legal clearance, and deployment decision.
3. AMITY asset issued; Lightning distribution begins.
4. Announce the public manuscript with its direct link, CC BY 4.0 notice, attribution guidance, and open rights-review caveat.
5. If pilot proof flows are offered, label them optional and provide verification instructions without implying exclusive access to the text.

**T+1 → — operate**

- Claims support and appeals; published receipts; metrics (claims per chain, failures,
  support load); post-mortem.

## 9. Day-0 runbook (draft skeleton)

- Owners and fallback owners for every step; time-boxed sequence; comms templates pre-written.
- **The CC BY 4.0 grant is irrevocable for compliant recipients.** Confirm the licensed paths, attribution, and rights the licensor can grant; no content key or rollback can withdraw the public permission.
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
| R5 | Securities/regulatory posture of token and investment framing | High | Counsel review before comms; never present the public novel as exclusive token-holder value |
| R6 | Public CC BY text is non-exclusive | Resolved for current draft | Keep the website and promotion honest; token proofs are optional and cannot restrict CC BY reuse |
| R7 | Third-party rights / provenance | High | Review source, authorship, and third-party references; CC BY does not clear rights the licensor does not hold |
| R8 | Day-one ambition vs blueprint's audit-first phases | Medium | Either respect Phase 2 or document accepted risk explicitly |
| R9 | Token naming inconsistency across docs | Medium | Canonical: C.A.R.E. Protocol umbrella; $OMEGA / TWC / AMITY; Devnet identities are `tOMEGA` / `tTWC`; TOKAMAK is retired as the Solana-token name. Reconcile remaining legacy whitepaper/app copy. |
| R10 | Snapshot/claim abuse (sybil wallets just above threshold) | Medium | One-claim-per-wallet, per-token thresholds, monitoring |
| R11 | TWC name/ticker collision and trademark risk | Medium | TWC was selected after historical comparison, but no availability conclusion is claimed. Obtain jurisdiction- and goods/services-specific trademark clearance and legal review before commercial/mainnet use. |
| R12 | Manuscript not final | Medium | Label revisions; recipients retain CC BY rights in each already-published version. Assess and license substantial new editions separately. |

## 11. Immediate next actions

1. Decide the **$OMEGA mainnet venue** (Ethereum L1 vs an L2 such as Base/Arbitrum/OP) — Sepolia remains the pilot network.
2. Complete TWC trademark/legal clearance and the Devnet-only `tTWC` preflight in [`../solana/README.md`](../solana/README.md); deploy only after the named treasury, external Devnet signer/funds, approved metadata URI/hash, and independent review are present.
3. Keep the public book and CC BY 4.0 notice linked together; test that `/book/full-book.md` is directly readable without a wallet and that the site does not describe the text as exclusive.
4. Confirm the displayed byline and update attribution consistently across the website and publication formats.
5. Commission the rights/provenance review of the Naruto, D-Wave, and any other third-party material. The CC BY grant does not replace this review.
6. Stand up the AMITY testnet node (longest lead item).
7. Complete the `$OMEGA` Sepolia pilot preflight in [`../evm/README.md`](../evm/README.md), deploy only valueless `tOMEGA` to Sepolia, and record the drill before commissioning an independent audit.
8. Reconcile remaining legacy whitepaper and app copy (C.A.R.E. Protocol / $OMEGA / TWC / AMITY), retaining historical wording only where clearly labeled.
