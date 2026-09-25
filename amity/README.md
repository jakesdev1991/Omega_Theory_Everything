<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# AMITY testnet scaffold

This directory is the **first local scaffold** for the AMITY leg of the C.A.R.E. Protocol. It does **not** issue a Taproot Asset, open Lightning channels, talk to `tapd`, hold keys, or claim production readiness. It is a **testnet-only operator/configuration starter** for the Bitcoin + Lightning + Taproot workstream described in the launch plan.

> **Not production software. Not a token sale. Not a promise of value.** No AMITY Taproot Asset exists from this repository today.

## What exists here now

- a strict **testnet-only** operator config parser;
- a canonical **AMITY holder unlock challenge** builder/parser for the future proof-of-holdings gate;
- a source-neutral holder-proof policy boundary with an explicitly test-only fixture adapter;
- a non-broadcasting **preflight** command for local operator setup;
- a local **status-manifest scaffold** generator for operator/web visibility; and
- offline tests for config, manifest, challenge format, and invalid/stale/wrong-asset proof cases.

This is meant to derisk the shape of the AMITY operator workflow before wiring a real `litd` / `tapd` stack into the release path.

## Local validation

```bash
cd amity
npm test
npm run test:fixture
```

`npm run test:fixture` runs a non-networking holder-claim harness. It exercises the exact testnet, asset-ID, Universe, origin, freshness, positive-balance, and signed-claim bindings through an injected fixture source. The fixture uses a real secp256k1 ECDSA test key, so tampering with the signed claim is detected. The fixture public key is not bound to a wallet address, however, so this is **not** a Taproot wallet Schnorr signature or a real asset-ownership proof.

Fixture proofs are rejected by the verifier unless `allowFixture: true` is passed explicitly. The fixture source is not used by the web app and cannot make AMITY eligible for unlocks.

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

## CARE to AMITY bootstrap policy

`lib/care-amity-policy.mjs` contains a deterministic, valueless policy calculator for the bootstrap schedule. It does not mint tokens, query prices, or inspect off-chain barter.

The current test schedule is:

- initial: 5% participant reserve dock;
- elevated: 10% dock;
- maximum: 20% dock;
- extreme: 20% dock plus a 15% supply burn, for a 35% total reduction.

The participant dock may not exceed 20%, the burn may not exceed 15%, and the combined reduction may not exceed 35%. The reserve dock and burn are applied before AMITY enters circulation, so later trading for goods cannot bypass the conversion rule. These are policy-test parameters, not a live economic guarantee or a mainnet configuration.

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

The fixture policy boundary does **not** implement Bitcoin/Lightning/Taproot ownership verification, Taproot wallet Schnorr verification, or live Universe proof validation. A successful fixture run must never be interpreted as a live holding or as an unlock authorization.

## Intended live-test path

1. stand up `litd` + `tapd` on **testnet**;
2. issue a valueless AMITY test asset and record its asset ID;
3. publish or configure a trusted Universe endpoint;
4. implement the injected source adapter using real `tapd`/Universe proof responses and verify the wallet's supported signature scheme (including Schnorr where applicable);
5. add integration tests against that adapter for invalid, stale, wrong-asset, wrong-holder, and insufficient-balance proofs; and
6. only then consider wiring AMITY into the web unlock app alongside `$OMEGA` and `TWC`.

Until those steps have real infrastructure and evidence, the manifest and web status surface must continue to report `holderVerificationReady=false` and `wiredIntoWebUnlock=false`.
