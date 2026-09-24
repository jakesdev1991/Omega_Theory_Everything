import assert from "node:assert/strict";
import test from "node:test";

import { readAmityConfig } from "../lib/config.mjs";

test("AMITY config parses the testnet operator scaffold", async () => {
  const config = await readAmityConfig({
    AMITY_NETWORK: "testnet",
    AMITY_ASSET_ID: "A".repeat(64),
    AMITY_UNIVERSE_URL: "https://universe.example.testnet/amity",
    LITD_RPC_HOST: "127.0.0.1:8443",
    TAPD_RPC_HOST: "127.0.0.1:10029",
    AMITY_ASSET_NAME: "AMITY Test Asset",
    AMITY_UNIT_NAME: "AMITY",
  });

  assert.equal(config.network, "testnet");
  assert.equal(config.assetId, "a".repeat(64));
  assert.equal(config.universeUrl, "https://universe.example.testnet/amity");
  assert.equal(config.litdRpcHost, "127.0.0.1:8443");
  assert.equal(config.tapdRpcHost, "127.0.0.1:10029");
});

test("AMITY config rejects non-testnet networks and malformed identifiers", async () => {
  await assert.rejects(
    () =>
      readAmityConfig({
        AMITY_NETWORK: "mainnet",
        AMITY_ASSET_ID: "A".repeat(64),
        AMITY_UNIVERSE_URL: "https://universe.example.testnet/amity",
      }),
    /AMITY_NETWORK must be testnet/,
  );

  await assert.rejects(
    () =>
      readAmityConfig({
        AMITY_NETWORK: "testnet",
        AMITY_ASSET_ID: "xyz",
        AMITY_UNIVERSE_URL: "https://universe.example.testnet/amity",
      }),
    /AMITY_ASSET_ID/,
  );

  await assert.rejects(
    () =>
      readAmityConfig({
        AMITY_NETWORK: "testnet",
        AMITY_ASSET_ID: "A".repeat(64),
        AMITY_UNIVERSE_URL: "not-a-url",
      }),
    /AMITY_UNIVERSE_URL/,
  );
});
