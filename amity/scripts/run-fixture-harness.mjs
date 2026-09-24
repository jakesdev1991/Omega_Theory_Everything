#!/usr/bin/env node

import assert from "node:assert/strict";

import { buildHolderUnlockChallenge } from "../lib/proof.mjs";
import { readAmityConfig } from "../lib/config.mjs";
import { createFixtureProofSource, verifyHolderProof } from "../lib/holder-verification.mjs";

const assetId = "a".repeat(64);
const issuedAt = "2026-09-24T00:00:00.000Z";
const now = Date.parse("2026-09-24T00:01:00.000Z");

const config = await readAmityConfig({
  AMITY_NETWORK: "testnet",
  AMITY_ASSET_ID: assetId,
  AMITY_UNIVERSE_URL: "https://universe.example.testnet/amity",
  AMITY_ASSET_NAME: "AMITY Test Asset",
  AMITY_UNIT_NAME: "AMITY",
  LITD_RPC_HOST: "127.0.0.1:8443",
  TAPD_RPC_HOST: "127.0.0.1:10029",
});

const challenge = buildHolderUnlockChallenge({
  address: "tb1ptestwalletaddressplaceholder",
  assetId: config.assetId,
  universeUrl: config.universeUrl,
  origin: "https://omega.example/novel",
  nonce: "fixture-nonce-123",
  issuedAt,
});

const result = await verifyHolderProof({
  challenge,
  expected: {
    network: config.network,
    assetId: config.assetId,
    universeUrl: config.universeUrl,
    origin: "https://omega.example/novel",
  },
  proofSource: createFixtureProofSource({ amount: "1000", proofId: "fixture-proof-1000" }),
  now,
  allowFixture: true,
});

assert.equal(result.status, "accepted");
assert.equal(result.source, "fixture");
assert.equal(result.claimSignatureChecked, true);
assert.equal(result.holderIdentityBound, false);
assert.equal(result.taprootAssetOwnershipChecked, false);

console.log("AMITY fixture readiness harness");
console.log("  network                 testnet");
console.log(`  asset id                ${result.assetId}`);
console.log(`  holder                  ${result.challenge.address}`);
console.log(`  amount                  ${result.amount}`);
console.log(`  proof source            ${result.source}`);
console.log(`  fixture signature       ${result.claimSignatureChecked ? "valid (ECDSA test key)" : "not checked"}`);
console.log(`  holder key binding      ${result.holderIdentityBound ? "checked" : "not checked (fixture)"}`);
console.log(`  Taproot ownership       ${result.taprootAssetOwnershipChecked ? "checked" : "not checked (fixture)"}`);
console.log("  live tapd/Universe      not connected");
console.log("\nFixture policy and signature binding passed. No network calls, issuance, or unlock wiring occurred.");
