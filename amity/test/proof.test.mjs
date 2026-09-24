import assert from "node:assert/strict";
import test from "node:test";

import { buildHolderUnlockChallenge, parseHolderUnlockChallenge } from "../lib/proof.mjs";

test("AMITY holder unlock challenges are canonicalized and parseable", () => {
  const issuedAt = "2026-09-23T22:00:00.000Z";
  const message = buildHolderUnlockChallenge({
    address: "tb1ptestwalletaddressplaceholder",
    assetId: "A".repeat(64),
    universeUrl: "https://universe.example.testnet/amity",
    origin: "https://omega.example/novel",
    nonce: "nonce-12345678",
    issuedAt,
  });

  const parsed = parseHolderUnlockChallenge(message);
  assert.equal(parsed.address, "tb1ptestwalletaddressplaceholder");
  assert.equal(parsed.assetId, "A".repeat(64));
  assert.equal(parsed.universeUrl, "https://universe.example.testnet/amity");
  assert.equal(parsed.origin, "https://omega.example/novel");
  assert.equal(parsed.nonce, "nonce-12345678");
  assert.equal(parsed.issuedAt, issuedAt);
});

test("AMITY holder unlock challenges reject malformed network and timestamps", () => {
  const message = [
    "OMEGA AMITY TESTNET PILOT - RELEASE-DAY NOVEL UNLOCK",
    "Address: tb1ptestwalletaddressplaceholder",
    `Asset ID: ${"A".repeat(64)}`,
    "Network: regtest",
    "Universe: https://universe.example.testnet/amity",
    "Origin: https://omega.example/novel",
    "Nonce: nonce-12345678",
    "Issued At: 2026-09-23T22:00:00.000Z",
    "Purpose: Verify current AMITY testnet Taproot Asset holdings for release-day novel unlock.",
  ].join("\n");

  assert.throws(() => parseHolderUnlockChallenge(message), /network must be testnet/);
});
