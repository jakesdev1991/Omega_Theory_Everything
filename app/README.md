<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# Omega release wallet prototype

The application source and design in this directory (`index.html`, `app.js`, and `styles.css`) are product materials, not open-source code. They are all rights reserved under `LicenseRef-Omega-Product-Proprietary`; see [`../LICENSES/Omega-Product-Proprietary.txt`](../LICENSES/Omega-Product-Proprietary.txt) and [`../docs/LICENSING.md`](../docs/LICENSING.md).

This is an unaudited prototype, not an offline or production wallet. Wallet creation, import, local keystore encryption, and message signing run in the browser; the encrypted keystore is stored in browser local storage. After creating a wallet, its recovery phrase remains visible until you confirm an offline backup and hide it, and the app blocks another wallet creation while that phrase is unconfirmed. Local proof signing asks for the keystore password in-page and clears the field afterward. An ETH balance lookup runs only when requested and sends the selected address to public Ethereum mainnet RPC providers. If the optional reader-proof flow is submitted, this site receives the signed statement and checks the configured Sepolia or Solana Devnet rail. The GUI does not submit blockchain transactions, but it can import keys that may control real assets; do not use it with valuable funds. `$OMEGA` and `TWC` proof rails are test-network demonstrations, and `AMITY` remains a separate Bitcoin/Lightning/Taproot workstream.

Viewing the files in this public repository does not grant a right to deploy, copy, modify, distribute, or commercially exploit the app. A commercial license requires a separate signed written agreement specifying a percentage-based royalty and its calculation terms. Historical rights granted under the earlier MIT-licensed version are addressed in the licensing policy.

## Distribution through the website

This directory is the single source of truth for the wallet GUI. The forward-facing website in
[`../web`](../web) syncs these files into its gitignored `public/omega-wallet/` directory
(`npm run sync:wallet` in `web/`), serves them at `/omega-wallet/`, and packages them into the
downloadable offline bundle at `GET /api/wallet/download`. Generated PWA glue (manifest, service
worker, icons) is produced by `web/scripts/sync-wallet.mjs` and never edits the files here. The
same GUI is wrapped for native installers by [`../desktop`](../desktop).
