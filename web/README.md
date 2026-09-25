<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->
# Omega web unlock app

This Next.js app is the public reading and release surface for **Genesis Block: The Satoshi
Protocol**, the home of the **Omega Wallet GUI**, and the **Economy Test Console** used for
complete offline testing of the four-plane crypto economy.

## Current wired currencies

The current wallet-to-web unlock flow is wired for **two currencies**:

- **$OMEGA** — EVM rail (current pilot: Ethereum / Sepolia)
- **TWC** — Solana / Devnet pilot rail

`AMITY` remains a separate Bitcoin/Lightning-with-Taproot workstream and is not wired into the
live unlock flow yet.

A user signs a release-day unlock statement in the wallet, the web API verifies the signature
server-side, and then the API independently checks the configured on-chain rail before serving
gated chapter prose.

## Quick start

```bash
cd web
cp .env.example .env.local   # optional overrides
npm ci
npm run dev                  # predev syncs the wallet GUI into public/omega-wallet/
```

| Script | Purpose |
|---|---|
| `npm run dev` / `npm run build` | Next dev/build; both run `scripts/sync-wallet.mjs` first |
| `npm run sync:wallet` | Re-sync the wallet GUI from `../app` into `public/omega-wallet/` |
| `npm test` | Domain + economy scenario suites (`node:test` via `tsx`) |
| `npm run test:economy` | Just the four-plane scenario suite |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run lint` | Flat-config ESLint (typescript-eslint + react-hooks) |

## Page map

| Route | What it is |
|---|---|
| `/` | Forward-facing landing page |
| `/economy` | Three-currency economy explainer + cross-plane workbench |
| `/wallet` | **Wallet GUI hub**: launch the GUI in-browser, live preview, per-file SHA-256 |
| `/wallet/download` | **Downloads**: offline bundle (ZIP + checksums + launchers), installable PWA, native build status |
| `/store/terms` | Store Terms of Use & End-User License, and the Publisher Agreement template |
| `/store` | **App Store**: static frontend over the Nostr backplane — directory from kinds 31990/30017, NIP-90 job requests to the mobile node, settlement into TWC |
| `/testnet` | **Economy Test Console**: readiness, scenario suite, faucet, action runner, audit exports, Nostr surface |
| `/care` | C.A.R.E. social prototype |
| `/novel`, `/novel/[slug]` | Token-gated reading experience |
| `/invest` | Participation and unlock tiers |
| `/mcp` | MCP hub overview |
| `/ops` | Operator status for the wired rails |

## The wallet GUI on this site

The wallet source of truth stays in [`../app`](../app) (proprietary product material). `npm run
dev`, `npm run build`, or `npm run sync:wallet` copy it into the gitignored
`public/omega-wallet/` and generate the PWA glue:

```
public/omega-wallet/
├── index.html  app.js  wallet.js  styles.css  vendor/ethers.umd.min.js   (copied from ../app)
├── manifest.webmanifest   sw.js   install.js                             (generated, installable app)
├── icon.svg  icon-192.png  icon-512.png                                  (generated)
└── BUILDINFO.json                                                          (version + hashes)
```

- **Launch**: `/omega-wallet/index.html` (linked from `/wallet` and the nav).
- **Install**: the generated manifest + service worker make it an installable, offline-capable app.
- **Download**: `GET /api/wallet/download` builds a deterministic ZIP (same GUI + `run.sh`,
  `run.command`, `run.bat`, `serve.mjs`, `CHECKSUMS.txt`, `LICENSE.txt`, `README.txt`). The ZIP's
  SHA-256 is published by `GET /api/wallet/manifest` and embedded in `CHECKSUMS.txt`, so the hash
  on the download page always matches the bytes served.
- **Native builds**: [`../desktop`](../desktop) is a Tauri v2 wrapper around this exact GUI;
  `.github/workflows/wallet-desktop.yml` builds installers on `wallet-v*` tags and published
  artifacts are listed in `desktop/releases.json` (empty today — the download page says so).
- **Mount-agnostic**: all generated PWA paths are relative, so the same bundle runs from the
  website mount or from any local port/directory.

## App Store (`/store`) and the mobile node

The store is a static frontend over the Nostr backplane managed by
[`../mobile-node`](../mobile-node):

- Directory: parameterized replaceable events **31990** (NIP-89 handler announcements, preferred)
  and **30017** (NIP-99 listings), published by `NOSTR_STORE_ROOT_NPUB`
  (`mobile-node/publish-directory.mjs` generates them from `algorithms.json`).
- Execution: **Run** signs a NIP-90 job request (kind 5000–5999, default 5001) with a browser-local
  operator key; the Termux daemon verifies it against its five-gate policy, executes an allowlisted
  algorithm in the Debian sandbox, and answers with a job result (request kind + 1000) correlated by
  `["e", requestId]` / `["p", requester]`.
- Settlement: **Settle result as TWC work** posts the verified job into the economy ledger as an
  audited work receipt.
- Demo mode: fixture listings + an in-page responder test the entire flow with zero relays.
- Strict verification everywhere: event ids are recomputed before signature checks, because
  nostr-tools' `verifyEvent` alone does not recompute the id from content/tags.
- Licensing: listings advertise `access` (`operators` | `licensed`), `terms` and SPDX `license`
  tags. Licensed apps accept any key holding a signed **kind 31335** license from the store key,
  which the storefront attaches to the job request and the node re-verifies (issuer, licensee,
  tier, terms, expiry, live revocations). Users accept the Store Terms (`omega-store-eula-1.0`,
  rendered at **`/store/terms`** from `public/legal/store-terms.md`) before obtaining a license.
  Demo mode can issue and revoke a demo license locally. Spec:
  [`../docs/store/LICENSE-PROTOCOL.md`](../docs/store/LICENSE-PROTOCOL.md).

## Economy Test Console (`/testnet`)

Complete, chain-free testing of the four-plane economy (CARE, TWC, $OMEGA, AMITY):

- **Readiness** — `GET /api/economy/health`: rail configuration, AMITY scaffold, wallet GUI sync,
  Nostr surface, test-control status, ledger snapshot.
- **Scenarios** — the suite in `src/lib/domain/scenarios.ts` covers integration slices A–G from
  `docs/tri-token-integration-v1.md` §7 plus both unlock-rail signature round-trips and the
  cross-plane double-entry invariants (10 scenarios / 43 steps). Runs in the browser console, in
  CI (`npm test`), and server-side via `POST /api/economy/scenarios`.
- **Ledger & faucet** — `POST /api/economy/faucet` grants valueless balances (test mode only);
  `POST /api/economy/reset` restores a fresh engine.
- **Action runner** — every audited mutation the shared engine accepts, as editable JSON.
- **Audit trail** — `GET /api/economy/audit` with plane/action/actor filters and CSV/NDJSON export.
- **Nostr surface** — `GET /api/social/status`: the contract the separately developed Nostr client
  plugs into (relays, publisher npub, economy event kinds 31331–31334, client requirements).

Test controls are enabled automatically outside production builds; in production they require
`ECONOMY_TEST_MODE=1`. A production build without that flag gets a fail-closed engine: no faucet,
no bootstrap voting power, no test credits.

## API routes

| Route | Method | Purpose |
|---|---|---|
| `/api/unlock/status` | GET | Rail configuration status for the novel banner |
| `/api/unlock` | POST | Verify an $OMEGA/TWC proof + on-chain rail check |
| `/api/chapter/[slug]` | POST | Verified-readers-only chapter prose |
| `/api/amity/status` | GET | AMITY scaffold visibility (not an unlock rail) |
| `/api/economy/state` | GET/POST | Shared ledger snapshot / audited economy actions |
| `/api/economy/scenarios` | GET/POST | List or run the offline scenario suite |
| `/api/economy/health` | GET | Single readiness report |
| `/api/economy/faucet` | POST | Test-mode valueless grants |
| `/api/economy/reset` | POST | Test-mode ledger reset |
| `/api/economy/audit` | GET | Filterable audit trail, JSON/CSV/NDJSON |
| `/api/wallet/manifest` | GET | Wallet versions, hashes, bundle + desktop status |
| `/api/wallet/download` | GET | Deterministic offline bundle ZIP |
| `/api/social/status` | GET | Nostr integration surface incl. store root key and DVM kinds |

## On-chain verification sources

The API routes prefer local pilot manifests when they exist. From the repo root you can inspect or
sync that operator state with:

```bash
node launch/check_unlock_rails.mjs
node launch/sync_web_unlock_env.mjs
```

The API routes prefer local pilot manifests when they exist:

- `../evm/deployments/sepolia.json`
- `../solana/deployments/twc-devnet.json`

Those files are intentionally gitignored, so local unlock verification will fail closed until a
pilot deployment manifest is present or equivalent environment variables are supplied.

### Optional environment overrides

See [`.env.example`](.env.example). Highlights:

- `$OMEGA`: `OMEGA_SEPOLIA_RPC_URL`, `OMEGA_NOVEL_GATE_ADDRESS`, `OMEGA_TOKEN_ADDRESS`, `OMEGA_SEPOLIA_MANIFEST_PATH`
- `TWC`: `SOLANA_DEVNET_RPC_URL`, `TWC_MINT_ADDRESS`, `TWC_DEVNET_MANIFEST_PATH`
- Test controls: `ECONOMY_TEST_MODE`
- Nostr: `NOSTR_RELAYS`, `NOSTR_PUBLISHER_NPUB`, `NOSTR_NIP05_DOMAIN`

`GET /api/amity/status` is separate on purpose: it reports the AMITY **testnet scaffold** state for
operator visibility without changing the fact that the live wallet/web unlock flow is still wired
for only **two currencies so far — `$OMEGA` and `TWC`**.

Both unlock routes verify the signed proof first. Then:

- `$OMEGA` checks the configured Sepolia pilot gate/token on-chain
- `TWC` checks the configured Devnet SPL mint balance on-chain

If the pilot rails are not configured, the routes fail closed with an explanatory error.
