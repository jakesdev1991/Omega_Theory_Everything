# Omega web unlock app

This Next.js app is the public reading and release surface for **Genesis Block: The Satoshi Protocol**.

## Current wired currencies

The current wallet-to-web unlock flow is wired for **two currencies**:

- **$OMEGA** — EVM rail (current pilot: Ethereum / Sepolia)
- **TWC** — Solana / Devnet pilot rail

`AMITY` remains a separate Bitcoin/Lightning-with-Taproot workstream and is not wired into the live unlock flow yet.

A user signs a release-day unlock statement in the wallet, the web API verifies the signature server-side, and then the API independently checks the configured on-chain rail before serving gated chapter prose.

## On-chain verification sources

The API routes prefer local pilot manifests when they exist. From the repo root you can inspect or sync that operator state with:

```bash
node launch/check_unlock_rails.mjs
node launch/sync_web_unlock_env.mjs
```

The API routes prefer local pilot manifests when they exist:

- `../evm/deployments/sepolia.json`
- `../solana/deployments/twc-devnet.json`

Those files are intentionally gitignored, so local unlock verification will fail closed until a pilot deployment manifest is present or equivalent environment variables are supplied.

### Optional environment overrides

For `$OMEGA`:

- `OMEGA_SEPOLIA_RPC_URL`
- `OMEGA_NOVEL_GATE_ADDRESS`
- `OMEGA_TOKEN_ADDRESS`
- `OMEGA_SEPOLIA_MANIFEST_PATH`

For `TWC`:

- `SOLANA_DEVNET_RPC_URL`
- `TWC_MINT_ADDRESS`
- `TWC_DEVNET_MANIFEST_PATH`

## Local development

```bash
cd web
cp .env.example .env.local  # optional overrides
npm ci
npm run build
```

The API routes are:

- `GET /api/unlock/status`
- `GET /api/amity/status`
- `POST /api/unlock`
- `POST /api/chapter/[slug]`

The novel UI uses `GET /api/unlock/status` to show a setup banner whenever one of the two wired rails is still missing local deployment metadata. There is also an operator page at `/ops` that surfaces the same status along with the exact preflight/deploy commands for each rail.

`GET /api/amity/status` is separate on purpose: it reports the AMITY **testnet scaffold** state for operator visibility without changing the fact that the live wallet/web unlock flow is still wired for only **two currencies so far — `$OMEGA` and `TWC`**.

Both unlock routes verify the signed proof first. Then:

- `$OMEGA` checks the configured Sepolia pilot gate/token on-chain
- `TWC` checks the configured Devnet SPL mint balance on-chain

If the pilot rails are not configured, the routes fail closed with an explanatory error.
