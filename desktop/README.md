<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Omega Wallet desktop wrapper

A thin [Tauri v2](https://v2.tauri.app) wrapper that packages the **same wallet GUI** the website
serves at `/omega-wallet/` (synced from [`../app`](../app) by
[`../web/scripts/sync-wallet.mjs`](../web/scripts/sync-wallet.mjs)) into native installers.

There is no second wallet implementation here. The wrapper embeds the synced GUI as its frontend
dist, so the desktop app and the web app are byte-identical apart from the generated PWA glue.

## Status

**Scaffold — no published binaries yet.** `desktop/releases.json` lists published artifacts; it is
currently empty, and the website's download page (`/wallet/download`) renders that state honestly
with the build-from-source instructions below. The first binaries come from
[`.github/workflows/wallet-desktop.yml`](../.github/workflows/wallet-desktop.yml) on a
`wallet-v*` tag.

## Build from source

Prerequisites: Node 22+, Rust stable, and the platform webview (WebKitGTK on Linux, WebView2 on
Windows, WKWebView on macOS), plus the Tauri v2 [platform prerequisites](https://v2.tauri.app/start/prerequisites/).

```bash
cd ../web && npm ci && npm run sync:wallet   # produces web/public/omega-wallet/
cd ../desktop && npm install
npm run build:linux     # .AppImage + .deb
npm run build:macos     # .dmg (signing/notarization optional, see tauri.conf.json)
npm run build:windows   # .msi
```

Artifacts land in `src-tauri/target/release/bundle/`.

## Publishing

1. Tag the release: `git tag wallet-v0.1.0 && git push origin wallet-v0.1.0`.
2. The workflow builds all three platforms and attaches the bundles to the GitHub release.
3. Record the published artifacts in `desktop/releases.json` (os, arch, fileName, url, sha256) so
   `/wallet/download` can link them with checksums.

## Security notes

- The webview runs with `javascript: true` and no remote URLs; everything is bundled locally.
- Keys and keystores stay in the per-app local storage profile; nothing is phoned home.
- The wrapper adds no network permissions beyond what the wallet GUI already uses for public RPC
  balance reads and (optionally) browser-wallet bridges.
