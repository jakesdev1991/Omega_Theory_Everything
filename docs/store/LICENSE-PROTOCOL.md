<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->
# Omega App Store: license protocol (v1)

This document specifies how app licenses are expressed, issued, checked and revoked over Nostr. There are two implementations, and they must stay in sync:

- storefront (displays licenses and attaches them to job requests): [`web/src/lib/store-license.ts`](../../web/src/lib/store-license.ts)
- mobile node (enforces licenses): [`mobile-node/lib/license.mjs`](../../mobile-node/lib/license.mjs), gated in [`mobile-node/lib/policy.mjs`](../../mobile-node/lib/policy.mjs)

A cross-implementation test (`web/src/lib/store-license.test.ts`, "identical license tags") pins the tag layout. The daemon integration test (`mobile-node/test/daemon.integration.test.mjs`) exercises issue → run → revoke over a real WebSocket relay.

The legal side (what a license *means*) is the Store Terms, [`web/public/legal/store-terms.md`](../../web/public/legal/store-terms.md) (`omega-store-eula-1.0`).

## 1. Listing: advertising license terms (kinds 31990 / 30017)

Directory events published by the store root key gain these tags:

| Tag | Values | Meaning |
|---|---|---|
| `["access", …]` | `operators` (default) · `licensed` | Who may run the app. Unknown values are treated as `operators` (fail closed). |
| `["terms", …]` | e.g. `omega-store-eula-1.0` | Terms version a license must record. |
| `["license", …]` | SPDX id, default `LicenseRef-Omega-Product-Proprietary` | Licence of the app itself (informational). |

The content JSON may repeat these in `content.license = { access, terms, tiers[], spdx, price }`. If a tag and the content disagree, the tag wins. `tiers` lists the license tiers the app accepts. `price` is display text only.

`mobile-node/publish-directory.mjs` generates all of this from each algorithm's `license` block in `algorithms.json`:

```json
"license": {
  "access": "licensed",
  "terms": "omega-store-eula-1.0",
  "acceptedTerms": ["omega-store-eula-1.0"],
  "tiers": ["trial", "standard", "pro"],
  "spdx": "LicenseRef-Omega-Product-Proprietary",
  "price": "Free self-test; license issued on request"
}
```

`acceptedTerms` is optional and defaults to `[terms]`. It lets older licenses keep working after a terms update. Without a `license` block, an algorithm is operator-only, which was the behavior before licensing existed.

## 2. License record (kind 31335, provisional)

A license is a **parameterized replaceable** event signed by a trusted issuer. By default the issuer is the store root key, which is normally the node's own key.

```text
kind: 31335
tags:
  ["d", "<appId>:<licenseePubkeyHex>"]        one live record per app + licensee
  ["p", "<licenseePubkeyHex>"]                lets the buyer's client subscribe (#p)
  ["app", "<appId>"]
  ["a", "31990:<issuerPubkeyHex>:<appId>"]    NIP-01 address of the listing
  ["tier", "<tier>"]                          ^[a-z0-9][a-z0-9-]{0,31}$
  ["status", "active" | "revoked"]
  ["terms", "<termsId>"]                      terms version accepted at issue time
  ["expiration", "<unix seconds>"]            optional (NIP-40)
  ["payment", "<ref>"]                        optional; manual | demo | zap:<id> | …
content: {"note": "..."} or {}
```

Kind 31335 continues the project's provisional 31331–31334 namespace (see `web/src/lib/nostr.ts`). It is not an allocated NIP kind.

## 3. Validation rules (both implementations)

A candidate license is **valid for (app A, requester R, now T)** only if all of these hold:

1. Strict NIP-01 verification: the id is recomputed from the serialized event and the schnorr signature is checked against it.
2. `kind == 31335` and the author is in the trusted-issuer set.
3. `d == "A:R"`, `p == R`, `app == A`. The app id must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`.
4. `created_at <= T + 600` (future-dated records beyond 10 minutes of skew are ignored).
5. Among valid candidates from the same issuer, the **newest** wins (`created_at`, with ties broken by the lowest id, per NIP-01).
6. The winning record has `status == active`.
7. There is no `expiration`, or `expiration > T`.
8. If the listing restricts tiers, `tier` is among them.
9. If the listing sets terms, `terms` is in `acceptedTerms`.

Failure codes: `license-required`, `license-revoked`, `license-expired`, `license-tier`, `license-terms`.

## 4. Running a licensed app (NIP-90)

The storefront attaches the newest valid license **in full** to the job request content:

```json
{ "app": "echo", "params": { … }, "license": { …signed kind 31335 event… } }
```

Gate 2 of the node's policy becomes: the requester is an operator, **or** the algorithm is `licensed` and §3 holds. Candidates are the attached license plus every newer version the node has seen on relays. The node subscribes to `{ kinds: [31335], authors: <issuers> }` whenever at least one algorithm is licensed, and keeps the newest version per `(issuer, d)` in a bounded `LicenseLedger`. That is why an old "active" record attached by a client cannot outrun a published revocation.

On failure the node never executes. It publishes a result with `["status", "payment-required"]` and content `{status, code, reason, terms}`, which the storefront shows as **license required**. The request content may exceed `maxParamsBytes` by at most 16 KiB, to leave room for the attached license.

## 5. Issuing and revoking

```bash
cd mobile-node
export LUCIFER_RELAYS=wss://nos.lol,wss://relay.damus.io LUCIFER_ROOT_SECRET=nsec1…
node issue-license.mjs issue  --app echo --to npub1buyer… --tier standard --days 30 --payment manual
node issue-license.mjs revoke --app echo --to npub1buyer…
node issue-license.mjs show   --app echo --to npub1buyer…   # sign and print only
```

The issuer key never leaves the terminal. Renewal or upgrade is simply a newer `issue`, with the same `d` tag. Revocation is a newer record with `status=revoked`.

Node configuration: `LUCIFER_LICENSE_ISSUERS` (comma-separated npub/hex) defaults to the node's own key.

## 6. Security properties and known limits

- **Forgery:** only trusted issuers' signatures count, and ids are recomputed, so tags cannot be rewritten.
- **Replay by another key:** a license only authorizes the pubkey in `p`/`d`, and the job request itself must be signed by that key.
- **Revocation freshness:** revocation is only as fast as the relays the node reads. A node that cannot see the revocation will honor the stale license until it reconnects, or until the license expires. Prefer expiring licenses (`--days`) for paid tiers.
- **Key sharing:** anyone holding the licensee's nsec can use the license. This is a contractual restriction (Store Terms §3), not a technical one.
- **Public data:** licenses and job requests are public events. They reveal which key licensed which app.
- **Rate limiting:** licensed access opens the node to non-operator keys. Per-licensee rate and concurrency limits are **not yet implemented**. Keep licensed algorithms cheap, or add limits before selling heavy tiers.
- **Payment:** not wired in yet. The `payment` tag is a reference only. A future version can let the node or an issuer bot mint licenses automatically from NIP-57 zap receipts, or from token payments, without changing the record format.
