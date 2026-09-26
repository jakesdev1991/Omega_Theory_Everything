// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";

import { getNostrIntegrationStatus } from "./nostr";

function setNostrEnv(env: Record<string, string | undefined>) {
  const keys = [
    "NOSTR_RELAYS",
    "NOSTR_PUBLISHER_NPUB",
    "NOSTR_NIP05_DOMAIN",
    "NOSTR_STORE_ROOT_NPUB",
  ];
  const saved: Record<string, string | undefined> = {};
  for (const key of keys) {
    saved[key] = process.env[key];
    if (env[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = env[key];
    }
  }
  return () => {
    for (const key of keys) {
      if (saved[key] === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = saved[key];
      }
    }
  };
}

test("Relay list keeps only strict wss:// entries", () => {
  const restore = setNostrEnv({
    NOSTR_RELAYS:
      "wss://ryzenvoid:8080,http://insecure.example,ws://plain.example, wss://relay2.example ,https://not-a-relay.example",
  });
  try {
    const status = getNostrIntegrationStatus();
    assert.deepEqual(status.relays, [
      "wss://ryzenvoid:8080",
      "wss://relay2.example",
    ]);
    assert.equal(status.configured, true);
    assert.equal(status.status, "ready");
  } finally {
    restore();
  }
});

test("Status is not_configured until NOSTR_RELAYS carries a wss:// entry", () => {
  const restore = setNostrEnv({
    NOSTR_RELAYS: "http://only-insecure.example,ws://still-not-wss.example",
  });
  try {
    const status = getNostrIntegrationStatus();
    assert.deepEqual(status.relays, []);
    assert.equal(status.configured, false);
    assert.equal(status.status, "not_configured");
  } finally {
    restore();
  }
});

test("Status flips not_configured -> ready when NOSTR_RELAYS is set", () => {
  const restore = setNostrEnv({});
  let before;
  try {
    before = getNostrIntegrationStatus();
    assert.equal(before.status, "not_configured");
    assert.equal(before.configured, false);
    assert.deepEqual(before.relays, []);

    process.env.NOSTR_RELAYS = "wss://ryzenvoid:8080";
    const after = getNostrIntegrationStatus();
    assert.equal(after.status, "ready");
    assert.equal(after.configured, true);
    assert.deepEqual(after.relays, ["wss://ryzenvoid:8080"]);
  } finally {
    restore();
  }
  const cleared = getNostrIntegrationStatus();
  assert.equal(cleared.status, "not_configured", "env restored between tests");
});

test("Publisher, NIP-05 domain, and store root npub surface through the status", () => {
  const restore = setNostrEnv({
    NOSTR_RELAYS: "wss://ryzenvoid:8080",
    NOSTR_PUBLISHER_NPUB: "npub1placeholder",
    NOSTR_NIP05_DOMAIN: "example.org",
    NOSTR_STORE_ROOT_NPUB: "npub1rootplaceholder",
  });
  try {
    const status = getNostrIntegrationStatus();
    assert.equal(status.publisherNpub, "npub1placeholder");
    assert.equal(status.nip05Domain, "example.org");
    assert.equal(status.storeRootNpub, "npub1rootplaceholder");
  } finally {
    restore();
  }
});
