<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# Licensing policy and scope

> This is a project policy summary, not legal advice or a signed commercial agreement. Have an intellectual-property lawyer review the transition and prepare any commercial license or contributor agreement before relying on it.

## Chosen model

There is no single license for the whole repository. The root [`LICENSE`](../LICENSE) is a scope index, not a grant covering everything. Materials fall into four buckets:

| Scope | License / status | What it means |
|---|---|---|
| Science and physics: simulations, Omega Theory notes, Lean proofs, LaTeX and plain-text proof companions, `rcod/` | **MIT** | Open source. Commercial use is permitted under MIT's attribution and disclaimer terms. No royalty is charged for use of these covered files. |
| Products: `web/` (website, wallet hub, App Store storefront), `mobile-node/`, `app/`, `desktop/`, `omni-bridge/`, `cpp/`, `mcp/`, `launch/`, `novel/`, `docs/store/` | **PolyForm Noncommercial License 1.0.0** | Free to use, modify, and share for **noncommercial** purposes. **Commercial use requires a separate paid, signed written license from Jacob See** with percentage-based compensation. This repository sets no rate. |
| Crypto and C.A.R.E.: `rust/`, `evm/`, `solana/`, `amity/`, `whitepapers/`, the Tri-Token blueprint, `docs/care-architecture-v2.md`, `docs/care-policy-v2.json`, `docs/tri-token-integration-v1.md` | **Read-only, no license** (`LicenseRef-Omega-ReadOnly`) | View-only in this repository. No copying, reproduction, modification, distribution, or derivative works without prior written permission. |
| Store legal texts (`web/public/legal/`), project administration and legal notices (`LICENSE`, `LICENSES/`, `NOTICE`, `CONTRIBUTING.md`, `README.md`, `docs/` otherwise, `.github/`) | **All rights reserved / read-only** | These documents explain the policy; they do not expand the grants above or license the product architecture. |

Nostr App Store apps and licenses sit on top of this: end users run listed apps under per-key license records (kind 31335) issued under the Store Terms `omega-store-eula-1.0`; publishers list apps under a signed Publisher Agreement. See [Nostr App Store licensing](#nostr-app-store-licensing) below.

## Exact path scope

### MIT scientific materials

The MIT grant applies to these repository paths, to the extent the rights holder is authorized to license them:

- `Sim1_Emergent_Geometry.py`, `Sim2_Cosmology.py`, `Sim3_Dynamic_Scale.py`, `Sim4_Evolution.py`, `Sim5_Emergent_Gravity.py`, `Sim7_Radial_Metric.py`, and `sim6_v14_depletion.py`;
- `Omega_Theory_Laymans_Guide.md`, `Omega_Theory_v4.0_Technical.md`, and `Omega_Theory_v4.0_Radial_Metric.md`;
- all files in `lean_proofs/`, `latex_docs/`, `txt_proofs/`, and `rcod/`.

The full license is [`LICENSES/MIT.txt`](../LICENSES/MIT.txt). MIT grants no trademark rights and does not relicense third-party content or dependencies.

### PolyForm Noncommercial product materials

The product scope is `web/**` (except `web/public/legal/**`), `mobile-node/**`, `app/**`, `desktop/**`, `omni-bridge/**`, `cpp/**`, `mcp/**`, `launch/**`, `novel/**`, and `docs/store/**`. These files carry or inherit `SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0`; see [`LICENSES/PolyForm-Noncommercial-1.0.0.txt`](../LICENSES/PolyForm-Noncommercial-1.0.0.txt).

The PolyForm Noncommercial License permits any noncommercial use, including modification and redistribution with attribution. Charitable, educational, public-research, public-safety-or-health, environmental, and government institutions count as noncommercial regardless of funding. **Any commercial purpose requires a license from Jacob See.** The intended commercial route is an **advance, signed written license with percentage-based compensation**. This repository does not set a rate, royalty base (for example, gross or net receipts), minimum, term, reporting/audit terms, territory, sublicensing rights, or other deal terms. Those must be negotiated and written into the separate agreement. Do not infer commercial permission from the fact that the files are publicly viewable or downloadable.

For a commercial licensing inquiry, contact the rights holder through [github.com/jakesdev1991](https://github.com/jakesdev1991). Do not post confidential deal terms in a public issue. No email address is stated here because none has been verified for publication.

### Read-only crypto and C.A.R.E. materials

The read-only scope is `rust/**`, `evm/**`, `solana/**`, `amity/**`, `whitepapers/**`, `tri_token_sovereign_economy_blueprint.md`, `docs/care-architecture-v2.md`, `docs/care-policy-v2.json`, and `docs/tri-token-integration-v1.md`. These files carry or inherit `SPDX-License-Identifier: LicenseRef-Omega-ReadOnly`; see [`LICENSES/LicenseRef-Omega-ReadOnly.txt`](../LICENSES/LicenseRef-Omega-ReadOnly.txt).

These materials implement or specify the C.A.R.E. Protocol and the Tri-Token Sovereign Economy. They are published for transparency and review only: they may be read in place in this repository, but no license to copy, reproduce, modify, distribute, sublicense, or build on them is granted, and none is planned in this repository version. Requests for written permission go to the rights holder as above. This posture interacts with the prior-MIT history below: copies of these files already distributed under the earlier MIT notice retain the rights that notice granted.

## Nostr App Store licensing

The App Store has three layers of licensing. Each is covered by a different instrument:

| Layer | Instrument | Where |
|---|---|---|
| Store software: the storefront (`web/src/components/AppStore.tsx`, `web/src/lib/nostr-store.ts`, `web/src/lib/store-license.ts`) and the execution node (`mobile-node/`) | PolyForm Noncommercial License 1.0.0 | This document; per-file SPDX headers |
| Listed apps distributed through the store | Proprietary unless the listing's `license` tag says otherwise (`LicenseRef-Omega-Product-Proprietary`) | Per-listing metadata |
| End users running listed apps | **Store Terms of Use & End-User License**, terms ID `omega-store-eula-1.0` (limited, personal, non-transferable, revocable right to run an app through the store; no source rights) | [`web/public/legal/store-terms.md`](../web/public/legal/store-terms.md), shown at `/store/terms` |
| Third-party publishers listing apps | **Publisher Agreement** template `omega-store-publisher-1.0` (listing licence to the Store Operator, publisher warranties, percentage revenue share payable to Jacob See; rate set per publisher in Schedule A) | [`web/public/legal/publisher-agreement.md`](../web/public/legal/publisher-agreement.md). **Not in force until completed and signed** |

**Technical enforcement.** Each end-user license is a signed Nostr event (kind 31335, provisional) issued by the store key to the user's public key. It records app, tier, terms ID, expiry, status and an optional payment reference. The mobile node runs apps marked `license.access: "licensed"` only for its operator keys, or for requesters that attach a valid license from a trusted issuer. It follows newer revocations live from relays and answers missing or revoked licenses with NIP-90 `payment-required`. Apps without that marking stay operator-only. Licenses are issued and revoked with `mobile-node/issue-license.mjs`. The full specification is [`docs/store/LICENSE-PROTOCOL.md`](store/LICENSE-PROTOCOL.md).

**Payment is not wired in yet.** Licenses are currently granted by the publisher out of band (`payment` tag: `manual` or a free-form reference). The protocol leaves room for Lightning zap receipts or C.A.R.E. token payments later. No paid tier should be sold until the refund policy, governing law and other bracketed items in the Store Terms have been completed by counsel.

**Limits.** A license record proves that the store key authorized a key to run an app. It does not by itself create enforceable contract terms against an anonymous key holder. Enforceability of click-through acceptance, of the restrictions, and of the liability limits varies by jurisdiction. Have counsel review both documents before relying on them for revenue.

## Previously flagged conflicts — now resolved

- **`rcod/`** was listed as proprietary in earlier versions of this document while its files carried Apache-2.0 headers. It is research tooling for the science side and is now **MIT**, consistent with the other scientific materials. Recipients of copies distributed with the earlier Apache-2.0 headers keep the rights those headers granted; Apache-2.0 and MIT obligations are compatible in practice for downstream users.
- **`mcp/`** previously declared `"license": "MIT"` in `pyproject.toml` and `server.json` while being unlisted. It is product material and is now **PolyForm Noncommercial 1.0.0**, and its metadata has been updated to match. Recipients of copies already published under the MIT metadata keep the rights that metadata granted for those copies; the change is prospective only.

## Important limits on the royalty goal

Copyright generally protects original expression, not abstract ideas, facts, methods, systems, processes, or concepts. This repository can reserve rights in the actual text, illustrations, and code to the extent they are copyrightable and owned, but this notice alone cannot require royalties merely because someone independently implements an idea described in a public paper. Patent, trademark, contract, or other rights may have different requirements and limitations. Have counsel evaluate what can actually be licensed and enforced before promising exclusivity or royalties.

## Prior MIT license — material limit on this change

The public baseline commit [`c717dc5`](https://github.com/jakesdev1991/Omega_Theory_Everything/commit/c717dc53bd20db4a6290734f09a3db5bb93fa287) included a root MIT license alongside the science files, app, Rust prototype, whitepapers, and blueprint. The exact prior license text is retained at [`LICENSES/MIT-legacy.txt`](../LICENSES/MIT-legacy.txt).

This licensing change is prospective. It cannot retroactively withdraw the permissions already granted to recipients of copies distributed under the prior MIT-licensed version. The MIT license allowed commercial use, modification, and redistribution subject to retaining its notice; those earlier rights do not acquire a royalty because this repository now states a different policy. This means the new commercial model cannot guarantee control over use of the already-published MIT version, including the crypto materials now marked read-only. Removing files now would not revoke rights already granted. Get legal advice before representing the product materials as exclusively controlled or royalty-bearing.

## Public disclosure and patents

The repository and product materials are already publicly accessible on GitHub. Public disclosure can affect patent rights, with consequences that vary by jurisdiction and facts; this repository cannot undo disclosure. No patent-pending status is represented here. See [`PATENT-POSTURE.md`](PATENT-POSTURE.md) and [`PROVENANCE.md`](PROVENANCE.md), and consult patent counsel before any further disclosure or patent-related claim.

## Third-party material, contributions, and ownership

The grants above apply only to rights the rights holder owns or is authorized to license. They do not change third-party licenses. The provenance record is incomplete, and the local checkout contains only shallow Git history. No signed contributor license agreement is recorded in this repository. See [`PROVENANCE.md`](PROVENANCE.md) and [`../CONTRIBUTING.md`](../CONTRIBUTING.md) before accepting contributions or making ownership claims.
