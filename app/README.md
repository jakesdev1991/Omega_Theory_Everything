<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Omega release wallet prototype

The application source and design in this directory (`index.html`, `app.js`, and `styles.css`) are product materials, not open-source code. They are all rights reserved under `LicenseRef-Omega-Product-Proprietary`; see [`../LICENSES/Omega-Product-Proprietary.txt`](../LICENSES/Omega-Product-Proprietary.txt) and [`../docs/LICENSING.md`](../docs/LICENSING.md).

The prototype is intentionally offline and valueless. In the current build it wires only two live unlock rails — `$OMEGA` on the EVM rail and `TWC` on Solana — while `AMITY` remains a separate Bitcoin/Lightning/Taproot workstream. Viewing the files in this public repository does not grant a right to deploy, copy, modify, distribute, or commercially exploit the app. A commercial license requires a separate signed written agreement specifying a percentage-based royalty and its calculation terms. Historical rights granted under the earlier MIT-licensed version are addressed in the licensing policy.

## Distribution through the website

This directory is the single source of truth for the wallet GUI. The forward-facing website in
[`../web`](../web) syncs these files into its gitignored `public/omega-wallet/` directory
(`npm run sync:wallet` in `web/`), serves them at `/omega-wallet/`, and packages them into the
downloadable offline bundle at `GET /api/wallet/download`. Generated PWA glue (manifest, service
worker, icons) is produced by `web/scripts/sync-wallet.mjs` and never edits the files here. The
same GUI is wrapped for native installers by [`../desktop`](../desktop).
