<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# AMITY testnet scaffold

This directory is the **first local scaffold** for the AMITY leg of the C.A.R.E. Protocol. It does **not** issue a Taproot Asset, open Lightning channels, talk to `tapd`, hold keys, or claim production readiness. It is a **testnet-only operator/configuration starter** for the Bitcoin + Lightning + Taproot workstream described in the launch plan.

> **Not production software. Not a token sale. Not a promise of value.** No AMITY Taproot Asset exists from this repository today.

## What exists here now

- a strict **testnet-only** operator config parser;
- a canonical **AMITY holder unlock challenge** builder/parser for the future proof-of-holdings gate;
- a non-broadcasting **preflight** command for local operator setup;
- a local **status-manifest scaffold** generator for operator/web visibility; and
- offline tests for config, manifest, and challenge format.

This is meant to derisk the shape of the AMITY operator workflow before wiring a real `litd` / `tapd` stack into the release path.

## Local validation

```bash
cd amity
npm test
```

## Preflight

```bash
cd amity
cp .env.example .env
# fill testnet-only values locally
npm run preflight:testnet
```

The preflight validates only local configuration. It makes **no network calls**, issues **no asset**, and writes **no chain state**.

## Local manifest scaffold

```bash
cd amity
npm run manifest:testnet
```

That command writes the ignored local snapshot:

- `deployments/amity-testnet.json`

It is a **status scaffold only**. It records the AMITY testnet asset/operator shape that the future web status surface can display, but it does **not** imply issuance, proof verification, or a third live unlock rail. The wallet/web unlock flow in this repository still accepts only **two wired currencies so far: `$OMEGA` and `TWC`**.

Use `npm run manifest:testnet -- --force` to refresh an existing snapshot after local operator files change.

## Environment variables

See [`.env.example`](.env.example). The important fields are:

- `AMITY_NETWORK=testnet`
- `AMITY_ASSET_ID=<64 hex chars>`
- `AMITY_UNIVERSE_URL=https://...`
- `LITD_RPC_HOST=host:port`
- `TAPD_RPC_HOST=host:port`

Optional file paths can be supplied for TLS certs and macaroons once the real testnet nodes exist.

## Canonical holder challenge format

The future AMITY novel-unlock flow is expected to use a signed proof-of-holdings challenge. This scaffold already standardizes the challenge bytes so later wallet/server work can target a single format:

```text
OMEGA AMITY TESTNET PILOT - RELEASE-DAY NOVEL UNLOCK
Address: <holder address>
Asset ID: <taproot asset id>
Network: testnet
Universe: <universe url>
Origin: <site origin>
Nonce: <opaque nonce>
Issued At: <canonical UTC timestamp>
Purpose: Verify current AMITY testnet Taproot Asset holdings for release-day novel unlock.
```

This directory does **not yet** implement Bitcoin/Lightning/Taproot holder verification or live Universe proof validation.

## Intended next steps

1. stand up `litd` + `tapd` on **testnet**;
2. issue a valueless AMITY test asset;
3. publish a Universe endpoint;
4. add a live holder verifier that checks a signed proof against asset ownership/proofs; and
5. only then consider wiring AMITY into the web unlock app alongside `$OMEGA` and `TWC`.
