<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->
# Nostr App Store: licensing index

| Document | ID | Audience | Status |
|---|---|---|---|
| [Store Terms of Use & End-User License](../../web/public/legal/store-terms.md) (rendered at `/store/terms`) | `omega-store-eula-1.0` | Everyone who uses the store or holds a license | Draft; bracketed items need counsel before paid tiers |
| [Publisher Agreement](../../web/public/legal/publisher-agreement.md) | `omega-store-publisher-1.0` | Third parties who want an app listed | Template; in force only when completed and signed |
| [License protocol](LICENSE-PROTOCOL.md) | v1 | Developers of the storefront and the node | Implemented and tested |
| [Repository licensing policy](../LICENSING.md#nostr-app-store-licensing) | — | Everyone | Store code is `LicenseRef-Omega-Product-Proprietary` |

When you change a legal text materially, give it a new ID (for example `omega-store-eula-1.1`) and update `STORE_TERMS_ID` in `web/src/lib/store-license.ts`. Also update the `terms` / `acceptedTerms` entries in `mobile-node/algorithms.json`, so the node knows which licenses remain acceptable. The web test suite checks that each document still contains the ID the code references.
