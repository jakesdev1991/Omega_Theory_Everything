// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import assert from "node:assert/strict";
import test from "node:test";

import bs58 from "bs58";
import { Wallet } from "ethers";
import nacl from "tweetnacl";

import {
  buildUnlockMessage,
  OMEGA_NETWORK,
  parseUnlockMessage,
  TWC_NETWORK,
  verifyUnlockProof,
} from "./unlock";

const SITE_ORIGIN = "https://omega.example";
const SIGNED_AT = "2026-10-16T12:00:00.000Z";

test("parses a canonical signed-message origin", () => {
  const wallet = Wallet.createRandom();
  const message = buildUnlockMessage({
    currency: "OMEGA",
    address: wallet.address,
    network: OMEGA_NETWORK,
    origin: SITE_ORIGIN,
    timestamp: SIGNED_AT,
  });

  assert.equal(parseUnlockMessage(message).origin, SITE_ORIGIN);
});

test("rejects non-origin URLs and non-HTTP schemes", () => {
  const wallet = Wallet.createRandom();
  for (const origin of ["null", "file:///tmp/wallet", "javascript:alert(1)", "https://omega.example/path", "https://omega.example?next=/"]) {
    const message = buildUnlockMessage({
      currency: "OMEGA",
      address: wallet.address,
      network: OMEGA_NETWORK,
      origin,
      timestamp: SIGNED_AT,
    });
    assert.throws(() => parseUnlockMessage(message), /canonical HTTP\(S\) origin/);
  }
});

test("verifies EVM proofs only for the requesting site's origin", async () => {
  const wallet = Wallet.createRandom();
  const message = buildUnlockMessage({
    currency: "OMEGA",
    address: wallet.address,
    network: OMEGA_NETWORK,
    origin: SITE_ORIGIN,
    timestamp: SIGNED_AT,
  });
  const signature = await wallet.signMessage(message);

  const verified = verifyUnlockProof({ message, signature }, SITE_ORIGIN);
  assert.equal(verified.address, wallet.address);
  assert.equal(verified.origin, SITE_ORIGIN);
  assert.equal(verified.currency, "OMEGA");
  assert.throws(
    () => verifyUnlockProof({ message, signature }, "https://attacker.example"),
    /origin does not match this site's origin/,
  );
});

test("verifies Solana proofs and enforces the same origin binding", () => {
  const keypair = nacl.sign.keyPair();
  const address = bs58.encode(keypair.publicKey);
  const message = buildUnlockMessage({
    currency: "TWC",
    address,
    network: TWC_NETWORK,
    origin: SITE_ORIGIN,
    timestamp: SIGNED_AT,
  });
  const signature = Buffer.from(nacl.sign.detached(new TextEncoder().encode(message), keypair.secretKey)).toString("base64");

  const verified = verifyUnlockProof({ message, signature }, SITE_ORIGIN);
  assert.equal(verified.address, address);
  assert.equal(verified.origin, SITE_ORIGIN);
  assert.equal(verified.currency, "TWC");
  assert.throws(
    () => verifyUnlockProof({ message, signature }, "https://attacker.example"),
    /origin does not match this site's origin/,
  );
});
