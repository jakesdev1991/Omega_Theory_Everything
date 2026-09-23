import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveWebsocketUrl,
  parseWholeTokenSupply,
  readDeploymentConfig,
  validateMetadataUri,
  validateSha256,
} from "../lib/config.mjs";
import { DEFAULT_INITIAL_SUPPLY_TOKENS, PILOT_TOKEN_DECIMALS } from "../lib/constants.mjs";

test("the pilot supply is converted exactly to SPL base units", () => {
  const supply = parseWholeTokenSupply(DEFAULT_INITIAL_SUPPLY_TOKENS);
  assert.equal(supply.wholeTokens, 1_000_000_000n);
  assert.equal(supply.baseUnits, 1_000_000_000_000_000_000n);
  assert.equal(parseWholeTokenSupply("1", PILOT_TOKEN_DECIMALS).baseUnits, 1_000_000_000n);
});

test("supply parser rejects fractional, zero, non-canonical, and overflowing values", () => {
  for (const invalid of ["", "0", "01", "1.5", "-1", "1_000", "abc"]) {
    assert.throws(() => parseWholeTokenSupply(invalid), /TWC_INITIAL_SUPPLY/);
  }
  assert.throws(() => parseWholeTokenSupply("18446744073709551616"), /u64 supply limit/);
});

test("metadata URI constraints exclude placeholders and credentials", () => {
  assert.equal(
    validateMetadataUri("https://arweave.net/abc123"),
    "https://arweave.net/abc123",
  );
  for (const invalid of [
    "http://arweave.net/abc",
    "https://example.invalid/twc.json",
    "https://localhost/twc.json",
    "https://user:pass@arweave.net/twc.json",
  ]) {
    assert.throws(() => validateMetadataUri(invalid));
  }
});

test("RPC websocket URL is derived without assuming localhost", () => {
  assert.equal(
    deriveWebsocketUrl("https://api.devnet.solana.com"),
    "wss://api.devnet.solana.com/",
  );
  assert.equal(
    deriveWebsocketUrl("http://rpc.example.net/custom"),
    "ws://rpc.example.net/custom",
  );
});

test("deployment config canonicalizes public inputs and does not need a secret at parse time", () => {
  const config = readDeploymentConfig({
    SOLANA_DEPLOYER_KEYPAIR_PATH: "keys/devnet.json",
    SOLANA_DEVNET_RPC_URL: "https://api.devnet.solana.com",
    TWC_TREASURY_ADDRESS: "11111111111111111111111111111111",
    TWC_METADATA_URI: "https://arweave.net/metadata-transaction",
    TWC_METADATA_SHA256: "A".repeat(64),
    TWC_INITIAL_SUPPLY: "42",
  });

  assert.equal(config.metadataSha256, "a".repeat(64));
  assert.equal(config.initialSupplyBaseUnits, 42_000_000_000n);
  assert.match(config.deployerKeypairPath, /solana[\\/]keys[\\/]devnet\.json$/);
  assert.equal(validateSha256("B".repeat(64)), "b".repeat(64));
});
