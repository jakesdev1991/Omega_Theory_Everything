// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";

import { hexToNpub } from "../lib/bech32.mjs";
import { getEventHashHex, getPublicKeyHex, npubOrHexToHex, signEvent, verifyEventStrict } from "../lib/events.mjs";

const SECRET = "0000000000000000000000000000000000000000000000000000000000000001";

test("bech32 round-trips npub and accepts hex operators", () => {
  const pubkey = getPublicKeyHex(SECRET);
  const npub = hexToNpub(pubkey);
  assert.ok(npub.startsWith("npub1"));
  assert.equal(npubOrHexToHex(npub), pubkey);
  assert.equal(npubOrHexToHex(pubkey.toUpperCase()), pubkey);
  assert.throws(() => npubOrHexToHex("nope"), /Not a pubkey/);
});

test("Signed events verify, and any rewrite fails strict verification", () => {
  const event = signEvent(
    { kind: 5001, created_at: 1767225600, tags: [["p", getPublicKeyHex(SECRET)]], content: '{"app":"echo"}' },
    SECRET,
  );
  assert.equal(verifyEventStrict(event), true);
  assert.equal(event.id, getEventHashHex(event));

  assert.equal(verifyEventStrict({ ...event, content: '{"app":"forged"}' }), false, "content rewrite");
  assert.equal(verifyEventStrict({ ...event, id: "0".repeat(64) }), false, "id rewrite");
  assert.equal(verifyEventStrict({ ...event, sig: "0".repeat(128) }), false, "signature rewrite");
  assert.equal(verifyEventStrict({ ...event, tags: [["p", "ff".repeat(32)]] }), false, "tag rewrite");
});
