import assert from "node:assert/strict";
import test from "node:test";

import { buildHolderUnlockChallenge, parseHolderUnlockChallenge } from "../lib/proof.mjs";
import {
  createFixtureProofSource,
  validateHolderProofEnvelope,
  verifyHolderProof,
} from "../lib/holder-verification.mjs";

const assetId = "A".repeat(64);
const universeUrl = "https://universe.example.testnet/amity";
const origin = "https://omega.example/novel";
const issuedAt = "2026-09-24T00:00:00.000Z";
const now = Date.parse("2026-09-24T00:01:00.000Z");

function challenge(overrides = {}) {
  return buildHolderUnlockChallenge({
    address: "tb1ptestwalletaddressplaceholder",
    assetId,
    universeUrl,
    origin,
    nonce: "nonce-12345678",
    issuedAt,
    ...overrides,
  });
}

function expected(overrides = {}) {
  return {
    network: "testnet",
    assetId,
    universeUrl,
    origin,
    ...overrides,
  };
}

test("fixture proof adapter verifies the complete AMITY testnet binding", async () => {
  const result = await verifyHolderProof({
    challenge: challenge(),
    expected: expected(),
    proofSource: createFixtureProofSource({ amount: "1000", proofId: "fixture-proof-1000" }),
    now,
    allowFixture: true,
  });

  assert.equal(result.status, "accepted");
  assert.equal(result.source, "fixture");
  assert.equal(result.claimSignatureChecked, true);
  assert.equal(result.holderIdentityBound, false);
  assert.equal(result.taprootAssetOwnershipChecked, false);
  assert.equal(result.amount, "1000");
  assert.equal(result.assetId, assetId.toLowerCase());
  assert.equal(result.challenge.address, "tb1ptestwalletaddressplaceholder");
});

test("fixture proofs are refused unless explicitly enabled", async () => {
  await assert.rejects(
    () =>
      verifyHolderProof({
        challenge: challenge(),
        expected: expected(),
        proofSource: createFixtureProofSource(),
        now,
      }),
    /fixture proofs are disabled/,
  );
});

test("holder verification fails closed on asset, origin, and stale challenges", async () => {
  await assert.rejects(
    () =>
      verifyHolderProof({
        challenge: challenge(),
        expected: expected({ assetId: "B".repeat(64) }),
        proofSource: createFixtureProofSource(),
        now,
        allowFixture: true,
      }),
    /asset ID does not match/,
  );

  await assert.rejects(
    () =>
      verifyHolderProof({
        challenge: challenge(),
        expected: expected({ origin: "https://attacker.example/novel" }),
        proofSource: createFixtureProofSource(),
        now,
        allowFixture: true,
      }),
    /origin does not match/,
  );

  await assert.rejects(
    () =>
      verifyHolderProof({
        challenge: challenge({ issuedAt: "2026-09-23T23:00:00.000Z" }),
        expected: expected(),
        proofSource: createFixtureProofSource(),
        now,
        allowFixture: true,
      }),
    /challenge has expired/,
  );
});

test("signed fixture claims fail closed when holdings are tampered", async () => {
  const message = challenge();
  const source = createFixtureProofSource({ amount: "1000" });
  const proof = await source.verify({ challenge: parseHolderUnlockChallenge(message) });
  proof.amount = "1000000";

  assert.throws(
    () => validateHolderProofEnvelope({ challenge: message, proof, expected: expected(), now, allowFixture: true }),
    /proof signature does not validate/,
  );
});

test("proof envelope validation rejects a non-positive amount", () => {
  const message = challenge();
  const proof = {
    version: 1,
    source: "fixture",
    acceptedBySource: true,
    signatureAlgorithm: "ecdsa-secp256k1-sha256",
    publicKeyPem: "not-a-key",
    signature: "AAAA",
    network: "testnet",
    assetId,
    holderAddress: "tb1ptestwalletaddressplaceholder",
    universeUrl,
    amount: "0",
    proofId: "fixture-proof-zero",
    challengeDigest: "0".repeat(64),
  };

  assert.throws(
    () => validateHolderProofEnvelope({ challenge: message, proof, expected: expected(), now, allowFixture: true }),
    /positive integer/,
  );
});

test("challenge builder rejects malformed asset IDs and credentialed URLs", () => {
  assert.throws(
    () => challenge({ assetId: "not-an-asset-id" }),
    /Asset ID must be a 64-character hexadecimal/,
  );

  assert.throws(
    () => challenge({ universeUrl: "https://user:password@universe.example.testnet/amity" }),
    /Universe must not contain credentials/,
  );
});
